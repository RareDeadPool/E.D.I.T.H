import sqlite3
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.memory_store import MemoryStore

store = MemoryStore()

with sqlite3.connect(store.db_path) as conn:
    c = conn.cursor()
    # Delete unprovenanced / test fixture dark mode
    c.execute("DELETE FROM semantic_facts WHERE object_value = 'dark mode'")
    # Set Harshan provenance
    c.execute("UPDATE semantic_facts SET provenance = 'user_stated', evidence = 'User stated: friend named Harshan' WHERE subject = 'Harshan'")
    conn.commit()

    c.execute("SELECT id, subject, predicate, object_value, provenance, evidence FROM semantic_facts")
    rows = c.fetchall()
    print("Cleaned rows in DB:", rows)

facts = store.get_all_facts()
print("get_all_facts returns:", facts)

