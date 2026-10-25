"""Intelligent Contact and Group Resolution Subsystem for E.D.I.T.H. V4.

Extracts recipient names, groups, roles, and message content from natural language.
Handles partial matching, typos, alias mapping, role resolution (e.g. Mom, Professor),
group vs direct chat discrimination, multi-turn follow-ups, and unambiguous clarification.
"""

import os
import re
import json
import difflib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class Contact:
    """Represents a direct contact or group."""
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None           # e.g., "mom", "dad", "professor", "recruiter", "manager"
    is_group: bool = False
    aliases: List[str] = field(default_factory=list)
    platform: str = "all"                # whatsapp, email, all

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contact":
        return cls(**data)


@dataclass
class ContactResolutionResult:
    """Outcome of resolving a contact reference from natural language."""
    status: str                         # "resolved", "needs_clarification", "not_found", "empty"
    resolved_contact: Optional[Contact] = None
    recipient_name: Optional[str] = None
    is_group: bool = False
    is_ambiguous: bool = False
    candidates: List[Contact] = field(default_factory=list)
    clarification_prompt: Optional[str] = None
    extracted_text: Optional[str] = None
    action_type: str = "send"           # "send", "draft", "edit", "preview"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "resolved_contact": self.resolved_contact.to_dict() if self.resolved_contact else None,
            "recipient_name": self.recipient_name,
            "is_group": self.is_group,
            "is_ambiguous": self.is_ambiguous,
            "candidates": [c.to_dict() for c in self.candidates],
            "clarification_prompt": self.clarification_prompt,
            "extracted_text": self.extracted_text,
            "action_type": self.action_type,
        }


class ContactResolver:
    """Production contact repository, extraction engine, and disambiguator."""

    DEFAULT_CONTACTS = [
        Contact(name="Test Recipient", phone="+918591211342", role="test", aliases=["test", "test recipient", "8591211342", "+918591211342", "tester"]),
        Contact(name="Rahul Sharma", phone="+919876543210", role="colleague", aliases=["rahul", "rahul sharma"]),
        Contact(name="Rahul Verma", phone="+919811223344", role="college friend", aliases=["rahul verma"]),
        Contact(name="Mom", phone="+919822334455", role="mom", aliases=["mother", "mummy", "maa"]),
        Contact(name="Dad", phone="+919833445566", role="dad", aliases=["father", "papa"]),
        Contact(name="Professor Smith", email="smith@university.edu", role="professor", aliases=["prof", "professor", "prof smith"]),
        Contact(name="Recruiter Jane", email="jane.recruiter@techcorp.com", role="recruiter", aliases=["recruiter", "jane"]),
        Contact(name="Project Core Group", is_group=True, aliases=["project group", "the project group", "project team", "core team"]),
        Contact(name="College Study Group", is_group=True, aliases=["study group", "college group", "class group"]),
        Contact(name="Family Group", is_group=True, aliases=["family", "home group"]),
    ]

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "contacts.json"
        )
        self.contacts: List[Contact] = []
        self._load_contacts()

    def _load_contacts(self):
        """Load contacts from local storage or initialize with defaults."""
        if os.path.isfile(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.contacts = [Contact.from_dict(d) for d in data]
                    return
            except Exception:
                pass
        
        # Initialize default contacts
        self.contacts = list(self.DEFAULT_CONTACTS)
        self.save_contacts()

    def save_contacts(self):
        """Save contact repository to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in self.contacts], f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def add_contact(self, contact: Contact):
        """Add or update a contact in the repository."""
        for i, c in enumerate(self.contacts):
            if c.name.lower() == contact.name.lower():
                self.contacts[i] = contact
                self.save_contacts()
                return
        self.contacts.append(contact)
        self.save_contacts()

    def extract_intent_and_entities(self, query: str) -> Dict[str, Any]:
        """
        Parses casual natural language into action mode (draft vs send vs edit vs preview),
        recipient name/group, and intended message content.
        """
        q = query.strip()
        q_low = q.lower()

        # 1. Determine Action Type
        action_type = "send"
        if any(w in q_low for w in ["cancel that", "abort", "cancel the message", "don't send that"]):
            action_type = "cancel"
        elif any(w in q_low for w in ["keep it as a draft", "don't send it yet", "keep as draft", "just keep it as a draft"]):
            action_type = "keep_draft"
        elif any(w in q_low for w in [
            "show me the message", "preview", "show draft", "show the message",
            "show me the message first", "show the message first", "let me see the message"
        ]):
            action_type = "preview"
        elif any(w in q_low for w in [
            "send the message we just drafted", "send the revised version", "send the revised message",
            "send the message", "send it", "send message", "send that"
        ]):
            action_type = "send"
        elif any(w in q_low for w in [
            "make that message shorter", "make it shorter", "make the message sound more polite",
            "make it more polite", "make that shorter", "make it more formal", "make it more professional",
            "add that", "add a note", "add a sentence", "correct the message"
        ]) or bool(re.search(r"\brevise\b", q_low)):
            action_type = "edit"
        elif any(w in q_low for w in ["draft", "compose", "write a draft", "prepare a message"]):
            action_type = "draft"
        elif any(w in q_low for w in ["send", "tell", "message", "text", "whatsapp", "ping"]):
            action_type = "send"

        recipient_candidate = None
        extracted_text = None
        is_exact = False

        q_work = re.sub(r"^(?:can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|please\s+)", "", q, flags=re.IGNORECASE).strip()

        # 2. Extract Exact Quoted Dictation (Preserves inner instructions like "open the browser")
        # e.g., Tell Rahul 'open the browser when you reach home'
        m_quoted = re.search(
            r"^(?:send\s+(?:a\s+)?(?:whatsapp\s+)?(?:message|text)?\s*(?:to\s+)?|message\s+(?:to\s+)?|text\s+(?:to\s+)?|whatsapp\s+(?:to\s+|message\s+to\s+)?|tell\s+|notify\s+|inform\s+|ping\s+)(?:the\s+|my\s+)?([a-zA-Z0-9_\-\+]+(?:\s+[a-zA-Z0-9_\-\+]+)?)(?:\s+(?:on|via)\s+whatsapp)?\s*['\"](.+?)['\"]$",
            q_work,
            re.IGNORECASE
        )
        if m_quoted:
            recipient_candidate = m_quoted.group(1).strip()
            extracted_text = m_quoted.group(2).strip()
            is_exact = True

        # 2b. Exact Dictation with explicit marker
        # e.g., "Send Rahul this exact message: I will be late"
        if not extracted_text:
            m_exact = re.search(
                r"^(?:send|text|message)\s+(?:(?:to\s+)?([a-zA-Z0-9_\-\s]+?)\s+)?(?:this\s+exact\s+message|exact\s+text(?:\s+i\s+dictated)?|exactly\s+this)(?:\s+to\s+([a-zA-Z0-9_\-\s]+?))?\s*:\s*(.+)$",
                q_work,
                re.IGNORECASE
            )
            if m_exact:
                recipient_candidate = (m_exact.group(1) or m_exact.group(2) or "").strip() or None
                extracted_text = m_exact.group(3).strip()
                is_exact = True
            else:
                m_exact2 = re.search(r"(?:this\s+exact\s+message|exact\s+text(?:\s+i\s+dictated)?|exactly\s+this)\s*(?:to\s+(.+?))?\s*:\s*(.+)$", q_work, re.IGNORECASE)
                if m_exact2:
                    if m_exact2.group(1):
                        recipient_candidate = m_exact2.group(1).strip()
                    extracted_text = m_exact2.group(2).strip()
                    is_exact = True

        # 3. Pattern: "Send [a] [whatsapp] message to <RECIPIENT> saying/that/asking/about <TEXT>"
        # e.g. "Send a message to my project group saying the meeting starts at 4 PM."
        if not extracted_text:
            m_send_to_saying = re.search(
                r"^(?:send\s+(?:a\s+)?(?:whatsapp\s+)?(?:message|text|note)?\s*(?:to\s+)?|message\s+(?:to\s+)?|text\s+(?:to\s+)?|whatsapp\s+(?:to\s+|message\s+to\s+)?|tell\s+|notify\s+|inform\s+|ping\s+)(?:the\s+|my\s+)?([a-zA-Z0-9_\-\+]+(?:\s+[a-zA-Z0-9_\-\+]+)?)(?:\s+(?:on|via)\s+whatsapp)?\s+(?:saying|that|asking|about|with\s+text)\s+(.+)$",
                q_work,
                re.IGNORECASE
            )
            if m_send_to_saying:
                recipient_candidate = m_send_to_saying.group(1).strip()
                extracted_text = m_send_to_saying.group(2).strip()

        # 4. Pattern: "Draft/Write a (polite) message to <RECIPIENT> saying/asking/about <TEXT>"
        if not extracted_text:
            m_draft = re.search(
                r"^(?:draft|write|compose)\s+(?:a\s+)?(?:polite\s+|formal\s+|quick\s+|short\s+)?(?:whatsapp\s+)?(?:message|text|email|note)?\s+to\s+([a-zA-Z0-9_\-\s]+?)\s+(?:saying|asking|about|that)\s+(.+)$",
                q_work,
                re.IGNORECASE
            )
            if m_draft:
                action_type = "draft"
                recipient_candidate = m_draft.group(1).strip()
                extracted_text = m_draft.group(2).strip()

        # 5. Pattern: "Send <RECIPIENT> a message saying/that <TEXT>"
        if not extracted_text:
            m_send_a = re.search(
                r"^(?:send|text)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:a\s+)?(?:whatsapp\s+)?(?:message|text)?\s*(?:saying|that)\s+(.+)$",
                q_work,
                re.IGNORECASE
            )
            if m_send_a:
                recipient_candidate = m_send_a.group(1).strip()
                extracted_text = m_send_a.group(2).strip()

        # 6. Pattern: "WhatsApp/Ping/Tell/Notify/Message <RECIPIENT> [on WhatsApp] saying/that/about <TEXT>"
        # e.g. "WhatsApp Rahul that I'm on my way." or "Tell Rahul on WhatsApp I'll be there in ten minutes."
        if not extracted_text:
            m_wa_tell = re.search(
                r"^(?:tell|notify|inform|ping|whatsapp|message)\s+(?:the\s+|my\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s+(?:on|via)\s+whatsapp)?\s+(?:saying\s+|that\s+|about\s+)(.+)$",
                q_work,
                re.IGNORECASE
            )
            if m_wa_tell:
                cand_recip = m_wa_tell.group(1).strip()
                cand_text = m_wa_tell.group(2).strip()
                # Ignore if cand_recip is generic
                if cand_recip.lower() not in ["the message", "it", "that", "browser"]:
                    recipient_candidate = cand_recip
                    extracted_text = cand_text

        # 7. Pattern: "No, send it to <RECIPIENT> instead"
        if not recipient_candidate:
            m_redir = re.search(r"^(?:no,?\s+)?send\s+it\s+to\s+([a-zA-Z0-9_\-\s]+?)\s+instead$", q_work, re.IGNORECASE)
            if m_redir:
                action_type = "send"
                recipient_candidate = m_redir.group(1).strip()
                extracted_text = None

        # 8. Pattern: "Send <TEXT> to <RECIPIENT>"
        if not extracted_text:
            m_send_to = re.search(
                r"^send\s+(?:the\s+)?(?:revised\s+|active\s+)?(?:message|draft|text)?\s*(.+?)\s+to\s+([a-zA-Z0-9_\-\s]+)$",
                q_work,
                re.IGNORECASE
            )
            if m_send_to:
                cand_text = m_send_to.group(1).strip()
                cand_recip = m_send_to.group(2).strip()
                if not cand_text or cand_text.lower() in ["message", "a message", "draft", "a draft", "revised message", "that", "the message"]:
                    extracted_text = None
                else:
                    extracted_text = cand_text
                recipient_candidate = cand_recip

        # 8b. Pattern: "Search (for) / Find <RECIPIENT> on WhatsApp [and message/tell him/her/them [saying/that <TEXT>]]"
        if not recipient_candidate:
            m_search_wa = re.search(
                r"^(?:search\s+(?:for\s+)?|find\s+)([a-zA-Z0-9_\-\+]+(?:\s+[a-zA-Z0-9_\-\+]+)?)\s+(?:on|in)\s+whatsapp(?:\s+and\s+(?:message|text|tell)\s+(?:him|her|them)?(?:\s+(?:saying|that|about)\s+(.+))?)?$",
                q_work,
                re.IGNORECASE
            )
            if m_search_wa:
                action_type = "send"
                recipient_candidate = m_search_wa.group(1).strip()
                extracted_text = m_search_wa.group(2).strip() if m_search_wa.group(2) else None

        # 9. Simple Direct Form: "message/text/whatsapp/send message to <RECIPIENT> [<TEXT>]"
        # e.g. "message rahul", "text mom", "whatsapp rahul", "message rahul hello", "send message to rahul"
        if not recipient_candidate:
            m_simple = re.search(
                r"^(?:send\s+(?:a\s+)?(?:whatsapp\s+)?(?:message|text)?\s*(?:to\s+)?|message\s+(?:to\s+)?|text\s+(?:to\s+)?|whatsapp\s+(?:to\s+|message\s+to\s+)?|ping\s+)(?:the\s+|my\s+)?([a-zA-Z0-9_\-\+]+)(?:\s+(.+))?$",
                q_work,
                re.IGNORECASE
            )
            if m_simple:
                cand_rec = m_simple.group(1).strip()
                cand_extra = m_simple.group(2).strip() if m_simple.group(2) else None
                # Check if cand_extra starts with second word of recipient name (e.g. 'Sharma')
                if cand_extra:
                    parts = cand_extra.split(" ", 1)
                    two_w = f"{cand_rec} {parts[0]}".lower()
                    known_matches = [c.name.lower() for c in self.contacts]
                    if two_w in known_matches or two_w in ["project group", "test recipient"]:
                        cand_rec = f"{cand_rec} {parts[0]}"
                        cand_extra = parts[1].strip() if len(parts) > 1 else None
                recipient_candidate = cand_rec
                extracted_text = cand_extra

        # 10. Clean up recipient candidate
        if recipient_candidate:
            recipient_candidate = re.sub(r"^(?:a\s+message\s+to\s+|message\s+to\s+|to\s+)", "", recipient_candidate, flags=re.IGNORECASE).strip()
            recipient_candidate = re.sub(r"^(?:the\s+)?(?:contact\s+|person\s+|group\s+)?", "", recipient_candidate, flags=re.IGNORECASE).strip()
            recipient_candidate = re.sub(r"^(?:my\s+)", "", recipient_candidate, flags=re.IGNORECASE).strip()
            # Remove trailing words like 'on whatsapp', 'via whatsapp'
            recipient_candidate = re.sub(r"\s+(?:on|via|through)\s+whatsapp.*$", "", recipient_candidate, flags=re.IGNORECASE).strip()
            if recipient_candidate.lower() in ["a", "an", "the", "a recipient", "someone", "recipient", "person", "contact", "anybody"]:
                recipient_candidate = None

        # 11. Clean up extracted text
        if extracted_text:
            # Strip outer quotes if present
            extracted_text = re.sub(r'^["\']|["\']$', '', extracted_text).strip()

        return {
            "action_type": action_type,
            "recipient_raw": recipient_candidate,
            "message_text": extracted_text,
            "is_exact": is_exact,
        }

    def resolve(
        self,
        recipient_query: Optional[str] = None,
        context: Optional[Any] = None,
        raw_query: Optional[str] = None
    ) -> ContactResolutionResult:
        """
        Unambiguously resolve a recipient query against contacts, context, and follow-ups.
        """
        recip_term = (recipient_query or "").strip()
        q_text = (raw_query or "").strip().lower()

        # 1. Check for Contextual References / Pronouns
        # e.g., "send it to the same person", "send to him", "send to her", "the selected contact"
        is_same_person = any(phrase in (recip_term.lower() or q_text) for phrase in [
            "same person", "same contact", "selected contact", "same recipient", "him", "her", "them"
        ])

        if (not recip_term or is_same_person) and context and hasattr(context, "last_person") and context.last_person:
            # Resolve from current task context
            prior_name = context.last_person
            c = self.get_exact_contact(prior_name)
            if c:
                return ContactResolutionResult(
                    status="resolved",
                    resolved_contact=c,
                    recipient_name=c.name,
                    is_group=c.is_group,
                    candidates=[c]
                )
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=Contact(name=prior_name),
                recipient_name=prior_name,
                is_group=False
            )

        # 2. Check for Follow-Up Corrections
        # e.g., "not that Rahul", "find the other Rahul", "use the group chat", "send it to Rahul instead"
        if "use the group chat" in q_text or "use the group" in q_text:
            # Filter groups
            groups = [c for c in self.contacts if c.is_group]
            if len(groups) == 1:
                return ContactResolutionResult(
                    status="resolved",
                    resolved_contact=groups[0],
                    recipient_name=groups[0].name,
                    is_group=True,
                    candidates=groups
                )
            elif len(groups) > 1:
                opts = [f"{i+1}) {g.name}" for i, g in enumerate(groups)]
                return ContactResolutionResult(
                    status="needs_clarification",
                    is_ambiguous=True,
                    is_group=True,
                    candidates=groups,
                    clarification_prompt=f"Which group chat would you like to use?\n  " + "\n  ".join(opts)
                )

        m_other = re.search(r"\b(?:not\s+that|the\s+other|other|second)\s+([a-zA-Z]+)", q_text)
        if m_other or "the other rahul" in q_text or "not that rahul" in q_text:
            name_term = m_other.group(1).lower() if m_other else "rahul"
            prior_name = getattr(context, "last_person", None) if context else None
            cand_objs = []
            if context and hasattr(context, "pending_clarification") and context.pending_clarification:
                raw_cands = context.pending_clarification.get("candidates", [])
                for rc in raw_cands:
                    c_name = rc.get("name") if isinstance(rc, dict) else str(rc)
                    c_obj = self.get_exact_contact(c_name)
                    if c_obj and c_obj not in cand_objs:
                        cand_objs.append(c_obj)
            all_matches = cand_objs if cand_objs else [c for c in self.contacts if name_term in c.name.lower()]
            other_matches = [c for c in all_matches if c.name != prior_name]
            if not other_matches and len(all_matches) > 1:
                other_matches = [all_matches[1]]
            if len(other_matches) == 1:
                return ContactResolutionResult(
                    status="resolved",
                    resolved_contact=other_matches[0],
                    recipient_name=other_matches[0].name,
                    is_group=other_matches[0].is_group,
                    candidates=other_matches
                )
            elif len(other_matches) > 1:
                opts = [f"{i+1}) {c.name} ({c.role or 'Contact'})" for i, c in enumerate(other_matches)]
                return ContactResolutionResult(
                    status="needs_clarification",
                    is_ambiguous=True,
                    candidates=other_matches,
                    clarification_prompt=f"Which {name_term.capitalize()} did you mean?\n  " + "\n  ".join(opts)
                )

        if not recip_term:
            return ContactResolutionResult(
                status="empty",
                clarification_prompt="Who would you like me to send this message to?"
            )

        # 2b. Direct Phone Number Recognition (e.g. 8591211342 or +918591211342)
        clean_digits = re.sub(r"[^\d+]", "", recip_term)
        digits_only = re.sub(r"\D", "", recip_term)
        if len(digits_only) >= 10:
            for c in self.contacts:
                if c.phone and re.sub(r"\D", "", c.phone).endswith(digits_only[-10:]):
                    return ContactResolutionResult(
                        status="resolved",
                        resolved_contact=c,
                        recipient_name=c.name,
                        is_group=False,
                        candidates=[c]
                    )
            phone_formatted = clean_digits if clean_digits.startswith("+") else f"+91{digits_only[-10:]}"
            direct_contact = Contact(name=recip_term, phone=phone_formatted, role="direct_number")
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=direct_contact,
                recipient_name=recip_term,
                is_group=False,
                candidates=[direct_contact]
            )

        # 2c. Direct Email Address Recognition (e.g. user@example.com)
        if "@" in recip_term and "." in recip_term:
            email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", recip_term)
            if email_match:
                e_addr = email_match.group(0).lower()
                for c in self.contacts:
                    if c.email and c.email.lower() == e_addr:
                        return ContactResolutionResult(
                            status="resolved",
                            resolved_contact=c,
                            recipient_name=c.name,
                            is_group=False,
                            candidates=[c]
                        )
                # Generate human-readable recipient name from local-part
                prefix = e_addr.split("@")[0].replace(".", " ").replace("_", " ").title()
                direct_email = Contact(name=prefix, email=e_addr, role="direct_email")
                return ContactResolutionResult(
                    status="resolved",
                    resolved_contact=direct_email,
                    recipient_name=prefix,
                    is_group=False,
                    candidates=[direct_email]
                )

        # 3. Match against Contact Repository
        clean_target = recip_term.lower()

        # Exact match on full name
        for c in self.contacts:
            if c.name.lower() == clean_target:
                return ContactResolutionResult(
                    status="resolved",
                    resolved_contact=c,
                    recipient_name=c.name,
                    is_group=c.is_group,
                    candidates=[c]
                )

        # Token / First-name check across all contacts (e.g. "Rahul" -> "Rahul Sharma" & "Rahul Verma")
        token_matches = []
        for c in self.contacts:
            c_tokens = [t.lower() for t in re.split(r"[\s_\-]+", c.name) if t]
            if clean_target in c_tokens:
                token_matches.append(c)

        if len(token_matches) > 1:
            opts = [f"{i+1}) {c.name} ({c.role or 'Contact'})" for i, c in enumerate(token_matches)]
            return ContactResolutionResult(
                status="needs_clarification",
                is_ambiguous=True,
                candidates=token_matches,
                clarification_prompt=(
                    f"I found multiple people named '{recip_term}':\n  " +
                    "\n  ".join(opts) +
                    f"\nWhich person do you mean?"
                )
            )
        elif len(token_matches) == 1:
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=token_matches[0],
                recipient_name=token_matches[0].name,
                is_group=token_matches[0].is_group,
                candidates=token_matches
            )

        # Exact match on aliases or role
        exact_alias_matches = []
        for c in self.contacts:
            if clean_target == (c.role or "").lower() or clean_target in [a.lower() for a in c.aliases]:
                exact_alias_matches.append(c)

        if len(exact_alias_matches) == 1:
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=exact_alias_matches[0],
                recipient_name=exact_alias_matches[0].name,
                is_group=exact_alias_matches[0].is_group,
                candidates=exact_alias_matches
            )
        elif len(exact_alias_matches) > 1:
            # Ambiguity among alias matches (e.g. multiple "colleagues" or multiple "Rahuls")
            opts = [f"{i+1}) {c.name} ({c.role or 'Contact'})" for i, c in enumerate(exact_alias_matches)]
            return ContactResolutionResult(
                status="needs_clarification",
                is_ambiguous=True,
                candidates=exact_alias_matches,
                clarification_prompt=(
                    f"I found multiple contacts matching '{recip_term}':\n  " +
                    "\n  ".join(opts) +
                    f"\nWhich person would you like to message?"
                )
            )

        # Partial substring match on names
        partial_matches = []
        for c in self.contacts:
            if clean_target in c.name.lower():
                partial_matches.append(c)

        if len(partial_matches) == 1:
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=partial_matches[0],
                recipient_name=partial_matches[0].name,
                is_group=partial_matches[0].is_group,
                candidates=partial_matches
            )
        elif len(partial_matches) > 1:
            # Multiple matches for partial name (e.g. "Rahul" -> "Rahul Sharma", "Rahul Verma")
            opts = [f"{i+1}) {c.name} ({c.role or 'Contact'})" for i, c in enumerate(partial_matches)]
            return ContactResolutionResult(
                status="needs_clarification",
                is_ambiguous=True,
                candidates=partial_matches,
                clarification_prompt=(
                    f"I found multiple people named '{recip_term}':\n  " +
                    "\n  ".join(opts) +
                    f"\nWhich person do you mean?"
                )
            )

        # Fuzzy match for typos / minor misspellings (e.g. "Raahul", "Adity")
        fuzzy_matches = []
        for c in self.contacts:
            name_ratio = difflib.SequenceMatcher(None, clean_target, c.name.lower()).ratio()
            alias_ratios = [difflib.SequenceMatcher(None, clean_target, a.lower()).ratio() for a in c.aliases]
            best = max([name_ratio] + alias_ratios, default=0.0)
            if best >= 0.78:
                fuzzy_matches.append((best, c))

        if fuzzy_matches:
            fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
            top_best, top_contact = fuzzy_matches[0]
            # Check if second match is nearly identical in score
            if len(fuzzy_matches) > 1 and (top_best - fuzzy_matches[1][0]) < 0.10:
                cand_list = [c for _, c in fuzzy_matches[:3]]
                opts = [f"{i+1}) {c.name}" for i, c in enumerate(cand_list)]
                return ContactResolutionResult(
                    status="needs_clarification",
                    is_ambiguous=True,
                    candidates=cand_list,
                    clarification_prompt=f"Did you mean one of these contacts?\n  " + "\n  ".join(opts)
                )
            return ContactResolutionResult(
                status="resolved",
                resolved_contact=top_contact,
                recipient_name=top_contact.name,
                is_group=top_contact.is_group,
                candidates=[top_contact]
            )

        # If completely unknown, return unverified raw name without fabricating details
        is_grp = "group" in clean_target or "team" in clean_target
        return ContactResolutionResult(
            status="not_found",
            recipient_name=recip_term.title(),
            is_group=is_grp,
            clarification_prompt=f"'{recip_term}' was not found in your saved contacts. Would you like to message this contact name on WhatsApp anyway?"
        )

    def get_exact_contact(self, name: str) -> Optional[Contact]:
        """Retrieve contact by exact name."""
        name_low = name.strip().lower()
        for c in self.contacts:
            if c.name.lower() == name_low:
                return c
        return None
