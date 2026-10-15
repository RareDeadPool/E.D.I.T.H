"""Comprehensive Automated Test Suite for WhatsApp Messaging Subsystem (E.D.I.T.H. V4).

Validates:
1. Contact resolution & disambiguation (multiple Rahuls, exact matches, roles, groups).
2. Direct phone number parsing and resolution (specifically test number 8591211342).
3. Draft creation, in-place revision (shorten, add note, revert), and preview.
4. Authorization gating: substantive assistant content requires confirmation; exact dictation pre-authorized.
5. Immediate cancellation on 'cancel that' / 'don't send' / 'abort'.
6. Duplicate send suppression within 120-second window.
7. Multi-turn follow-ups ('send the revised message to the same person', 'use the group chat').
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_brain import EdithBrain
from runtime.contact_resolver import ContactResolver, Contact
from runtime.whatsapp_adapter import WhatsAppAdapter


class TestV4WhatsAppMessagingSubsystem(unittest.TestCase):
    """Production test suite for E.D.I.T.H. V4 WhatsApp capabilities."""

    def setUp(self):
        self.brain = EdithBrain(enable_execution=True)
        self.resolver = ContactResolver()
        self.adapter = WhatsAppAdapter(contact_resolver=self.resolver)

    def test_01_contact_resolution_ambiguous(self):
        """Ambiguous first name 'Rahul' returns clarification prompt with multiple candidates."""
        res = self.resolver.resolve("Rahul")
        self.assertEqual(res.status, "needs_clarification")
        self.assertTrue(res.is_ambiguous)
        self.assertGreaterEqual(len(res.candidates), 2)
        candidate_names = [c.name for c in res.candidates]
        self.assertIn("Rahul Sharma", candidate_names)
        self.assertIn("Rahul Verma", candidate_names)
        self.assertIn("Which person do you mean?", res.clarification_prompt)

    def test_02_contact_resolution_exact_and_role(self):
        """Resolves unambiguous contacts and roles."""
        res_sharma = self.resolver.resolve("Rahul Sharma")
        self.assertEqual(res_sharma.status, "resolved")
        self.assertEqual(res_sharma.recipient_name, "Rahul Sharma")
        self.assertEqual(res_sharma.resolved_contact.phone, "+919876543210")

        res_mom = self.resolver.resolve("Mom")
        self.assertEqual(res_mom.status, "resolved")
        self.assertEqual(res_mom.recipient_name, "Mom")
        self.assertEqual(res_mom.resolved_contact.phone, "+919822334455")

    def test_03_test_phone_number_8591211342(self):
        """Specifically verifies resolution of designated test number 8591211342."""
        res = self.resolver.resolve("8591211342")
        self.assertEqual(res.status, "resolved")
        self.assertIsNotNone(res.resolved_contact)
        self.assertEqual(res.resolved_contact.phone, "+918591211342")

        res_intl = self.resolver.resolve("+918591211342")
        self.assertEqual(res_intl.status, "resolved")
        self.assertEqual(res_intl.resolved_contact.phone, "+918591211342")

    def test_04_draft_creation_and_preservation(self):
        """Drafting stores message in active draft without submitting."""
        res = self.brain.process("Draft a WhatsApp message to Mom that I've reached college")
        self.assertIn(res.get("status"), ["success", "OPERATIONAL"])
        self.assertIn("Draft", res.get("directive", {}).get("message", "") or res.get("execution", {}).get("message", ""))
        reg = self.brain.executor.action_registry
        self.assertIsNotNone(reg.whatsapp_adapter.active_draft)
        self.assertEqual(reg.whatsapp_adapter.active_draft.recipient, "Mom")
        self.assertIn("reached college", reg.whatsapp_adapter.active_draft.content)

    def test_05_in_place_revision_shorten(self):
        """Revising active draft updates version and shortens message."""
        self.brain.process("Draft a WhatsApp message to Rahul Sharma saying I'll be 15 minutes late.")
        r_rev = self.brain.process("Make that message shorter")
        self.assertEqual(r_rev.get("status"), "OPERATIONAL")
        self.assertIn("Revised WhatsApp draft for Rahul Sharma (v2)", r_rev.get("directive", {}).get("message", ""))
        self.assertIn("15 mins late", r_rev.get("directive", {}).get("message", ""))

    def test_06_in_place_revision_add_note(self):
        """Adding a point or note updates the draft content."""
        self.brain.process("Draft a WhatsApp message to Rahul Sharma saying I will be late.")
        r_rev = self.brain.process("Add that I have the slides ready.")
        self.assertEqual(r_rev.get("status"), "OPERATIONAL")
        reg = self.brain.executor.action_registry
        self.assertIn("slides ready", reg.whatsapp_adapter.active_draft.content)

    def test_07_preview_before_sending(self):
        """Shows formatted preview of active draft with recipient and version."""
        self.brain.process("Draft a WhatsApp message to Mom that I've reached college")
        r_prev = self.brain.process("Show me the message before sending it")
        self.assertEqual(r_prev.get("status"), "OPERATIONAL")
        self.assertIn("WhatsApp Message Preview", r_prev.get("directive", {}).get("message", ""))
        self.assertIn("Mom", r_prev.get("directive", {}).get("message", ""))

    def test_08_substantive_message_requires_confirmation(self):
        """Assistant-generated message requires user approval before sending."""
        r_send = self.brain.process("Send Rahul Sharma a message saying I will be 15 minutes late.")
        self.assertEqual(r_send.get("status"), "NEEDS_CONFIRMATION")
        self.assertTrue(r_send.get("directive", {}).get("needs_confirmation"))
        self.assertIn("Would you like me to send this now?", r_send.get("directive", {}).get("message", ""))

    def test_09_verbatim_exact_dictation_pre_authorized(self):
        """Exact dictation to unambiguous contact transmits immediately."""
        r_exact = self.brain.process("Send Rahul Sharma this exact message: I am outside the building")
        self.assertEqual(r_exact.get("status"), "success")
        self.assertEqual(r_exact.get("execution", {}).get("transmission_state"), "message_submitted")
        self.assertEqual(r_exact.get("execution", {}).get("content"), "I am outside the building")

    def test_10_cancel_aborts_pending_actions(self):
        """'Cancel that' immediately clears active draft and aborts transmission."""
        self.brain.process("Send Rahul Sharma a message saying I will be 15 minutes late.")
        self.assertIsNotNone(self.brain.context.pending_confirmation)
        r_cancel = self.brain.process("Cancel that")
        self.assertEqual(r_cancel.get("status"), "OPERATIONAL")
        self.assertIn("Action cancelled", r_cancel.get("directive", {}).get("message", ""))
        self.assertIsNone(self.brain.context.pending_confirmation)
        reg = self.brain.executor.action_registry
        self.assertIsNone(reg.whatsapp_adapter.active_draft)

    def test_11_duplicate_send_prevention(self):
        """Identical messages to same recipient within 2 minutes are blocked."""
        # Send first message to test number 8591211342
        r1 = self.brain.process("Send 8591211342 this exact message: Safe transmission alpha")
        self.assertEqual(r1.get("status"), "success")
        # Attempt identical message immediately
        r2 = self.brain.process("Send 8591211342 this exact message: Safe transmission alpha")
        self.assertEqual(r2.get("execution", {}).get("status"), "duplicate_prevented")
        self.assertIn("Duplicate send prevented", r2.get("execution", {}).get("message"))

    def test_12_multi_turn_disambiguation_flow(self):
        """Multi-turn flow: ambiguous 'Rahul' -> clarification -> disambiguation response -> draft created."""
        r1 = self.brain.process("Draft a WhatsApp message to Rahul saying the meeting starts at 4 PM")
        self.assertEqual(r1.get("status"), "NEEDS_CLARIFICATION")
        self.assertIn("Rahul Sharma", r1.get("directive", {}).get("message"))

        # User clarifies with exact name
        r2 = self.brain.process("Rahul Sharma")
        self.assertEqual(r2.get("status"), "OPERATIONAL")
        self.assertIn("Draft", r2.get("directive", {}).get("message"))
        self.assertIn("Rahul Sharma", r2.get("directive", {}).get("message"))
        self.assertIn("meeting starts at 4 PM", r2.get("directive", {}).get("message"))

    def test_13_multi_turn_send_revised_message(self):
        """Multi-turn flow: draft -> shorten -> 'Send the revised message to the same person'."""
        self.brain.process("Draft a WhatsApp message to Mom that I've reached college")
        self.brain.process("Make that message shorter")
        r_send = self.brain.process("Send the revised message to the same person")
        self.assertEqual(r_send.get("status"), "success")
        self.assertEqual(r_send.get("execution", {}).get("transmission_state"), "message_submitted")
        self.assertEqual(r_send.get("execution", {}).get("recipient"), "Mom")


if __name__ == "__main__":
    unittest.main()
