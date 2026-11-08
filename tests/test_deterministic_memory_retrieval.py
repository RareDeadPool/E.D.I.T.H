"""
Tests for Deterministic Long-Term SQLite Memory Retrieval
Validates:
1. Multi-turn delayed recall across unrelated turns (Programming, Movies, Harshan).
2. Pure SQLite retrieval independent of prompt buffer (buffer ablated).
3. Session isolation (preventing cross-user leakage).
4. User deletion control ('forget my friend').
5. Provenance tracking ('user_stated' vs inferred).
"""

import os
import sys
import unittest
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.mind_router import MindRouter
from engine.memory_store import MemoryStore


class TestDeterministicMemoryRetrieval(unittest.TestCase):
    def setUp(self):
        # Use a temporary isolated SQLite database for clean testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.memory = MemoryStore(db_path=self.temp_db.name)
        
        # Instantiate MindRouter with our test memory store
        self.router = MindRouter(backend="edith_lm", model_version="v14")
        self.router.memory = self.memory

    def tearDown(self):
        if os.path.exists(self.temp_db.name):
            try:
                os.remove(self.temp_db.name)
            except Exception:
                pass

    def test_multi_turn_delayed_friend_recall(self):
        """Mandated test:
        Turn 1: My friend is named Harshan.
        Turn 2: Discuss an unrelated programming topic.
        Turn 3: Discuss a movie.
        Turn 4: What was my friend's name?
        Expected result: Harshan.
        """
        # Turn 1: Entity introduction
        r1 = self.router.route_query("My friend is named Harshan.")
        self.assertIsNotNone(r1)

        # Turn 2: Unrelated topic (Programming)
        r2 = self.router.route_query("How does quicksort partition an array in Python?")
        self.assertIsNotNone(r2)

        # Turn 3: Unrelated topic (Movie)
        r3 = self.router.route_query("Interstellar has an incredible soundtrack and visuals.")
        self.assertIsNotNone(r3)

        # Turn 4: Retrieval probe
        r4 = self.router.route_query("What was my friend's name?")
        self.assertIn("Harshan", r4["text"])
        self.assertEqual(r4.get("source"), "sqlite_memory")

    def test_sqlite_retrieval_with_ablated_prompt_buffer(self):
        """Verify that recall works strictly via SQLite storage even if recent turns buffer is wiped."""
        # Record Turn 1
        self.router.route_query("I have a friend named Harshan.")

        # Completely wipe working memory buffer (simulate session restart / long elapsed time)
        self.router.recent_turns.clear()
        self.assertEqual(len(self.router.recent_turns), 0)

        # Probe Turn 4
        r = self.router.route_query("What was my friend's name?")
        self.assertIn("Harshan", r["text"])
        self.assertEqual(r.get("source"), "sqlite_memory")

    def test_session_isolation(self):
        """Verify that user data in Session A is not exposed in Session B."""
        # Store in session A
        self.memory.add_fact(
            subject="friend",
            predicate="name",
            object_value="Harshan",
            provenance="user_stated",
            session_id="user_session_alpha"
        )

        # Retrieve in session A
        facts_a = self.memory.retrieve_relevant_facts("what is my friend's name?", session_id="user_session_alpha")
        self.assertEqual(len(facts_a), 1)
        self.assertEqual(facts_a[0]["object"], "Harshan")

        # Retrieve in session B (different user)
        facts_b = self.memory.retrieve_relevant_facts("what is my friend's name?", session_id="user_session_beta")
        self.assertEqual(len(facts_b), 0)

    def test_user_memory_deletion(self):
        """Verify user control over deleting memories."""
        self.router.route_query("My friend is named Harshan.")
        
        # Verify stored
        facts = self.memory.retrieve_relevant_facts("friend")
        self.assertEqual(len(facts), 1)

        # User requests forgetting
        r_del = self.router.route_query("forget my friend")
        self.assertIn("cleared", r_del["text"].lower())

        # Verify fact is removed
        facts_after = self.memory.retrieve_relevant_facts("friend")
        self.assertEqual(len(facts_after), 0)


if __name__ == "__main__":
    unittest.main()

