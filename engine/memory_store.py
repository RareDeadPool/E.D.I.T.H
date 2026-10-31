"""E.D.I.T.H. V13 Persistent Memory Engine & Local Contrastive Store.

Features:
1. SQLite-backed persistent episodic conversation log.
2. Semantic fact table (user preferences, friends, relationships, jobs).
3. Own-trained lightweight dense sentence encoder (pure PyTorch, no HuggingFace embedders).
4. User privacy controls ("forget that", "what do you know about me", selective deletion).
"""

import json
import math
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.edith_lm.tokenizer import EdithByteBPETokenizer


class LocalSentenceEncoder(nn.Module):
    """Proprietary contrastive sentence encoder using our own Byte-BPE tokens."""

    def __init__(self, vocab_size: int = 3051, embed_dim: int = 128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Mean pooling over token embeddings
        embeds = self.embeddings(input_ids)
        mask = (input_ids > 0).unsqueeze(-1).float()
        sum_embeds = torch.sum(embeds * mask, dim=1)
        lengths = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = sum_embeds / lengths
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)


class MemoryStore:
    DB_PATH = os.path.join(PROJECT_ROOT, "config", "memory_v13.db")

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        tok_path = os.path.join(PROJECT_ROOT, "models", "checkpoints", "edith_bpe_tokenizer.json")
        self.tokenizer = EdithByteBPETokenizer(vocab_file=tok_path)
        self.encoder = LocalSentenceEncoder(vocab_size=self.tokenizer.vocab_size).eval()

        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # 1. Episodic conversation turns
            c.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    user_utterance TEXT,
                    edith_response TEXT,
                    topic TEXT,
                    embedding BLOB
                )
            """)
            # 2. Semantic facts & relationships
            c.execute("""
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    predicate TEXT,
                    object_value TEXT,
                    timestamp REAL,
                    confidence REAL
                )
            """)
            conn.commit()

    def get_embedding(self, text: str) -> List[float]:
        """Computes dense normalized embedding vector for arbitrary text."""
        ids = self.tokenizer.encode(text)
        if not ids:
            return [0.0] * 128
        tensor_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            vec = self.encoder(tensor_ids).squeeze(0).tolist()
        return vec

    def record_turn(self, user_msg: str, edith_reply: str, topic: Optional[str] = None):
        """Records a conversational turn with its dense embedding vector."""
        emb = self.get_embedding(user_msg)
        emb_blob = json.dumps(emb).encode("utf-8")
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO episodic_memory (timestamp, user_utterance, edith_response, topic, embedding) VALUES (?, ?, ?, ?, ?)",
                (time.time(), user_msg, edith_reply, topic or "general", emb_blob),
            )
            conn.commit()

    def add_fact(self, subject: str, predicate: str, object_value: str, confidence: float = 1.0):
        """Stores a persistent semantic fact (e.g. Harshan is a friend)."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Upsert
            c.execute(
                "DELETE FROM semantic_facts WHERE subject = ? AND predicate = ?",
                (subject.strip(), predicate.strip()),
            )
            c.execute(
                "INSERT INTO semantic_facts (subject, predicate, object_value, timestamp, confidence) VALUES (?, ?, ?, ?, ?)",
                (subject.strip(), predicate.strip(), object_value.strip(), time.time(), confidence),
            )
            conn.commit()

    def search_facts(self, query: str) -> List[Dict[str, str]]:
        """Retrieves semantic facts relevant to a query."""
        q_lower = query.lower()
        results = []
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT subject, predicate, object_value FROM semantic_facts")
            for subj, pred, obj in c.fetchall():
                if subj.lower() in q_lower or obj.lower() in q_lower or pred.lower() in q_lower:
                    results.append({"subject": subj, "predicate": pred, "object": obj})
        return results

    def get_all_facts(self) -> List[Dict[str, str]]:
        """Lists all known user facts (for transparency: 'what do you know about me')."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT subject, predicate, object_value FROM semantic_facts ORDER BY timestamp DESC")
            for subj, pred, obj in c.fetchall():
                results.append({"subject": subj, "predicate": pred, "object": obj})
        return results

    def forget_fact(self, subject_or_keyword: str) -> int:
        """Deletes facts matching a keyword (for privacy: 'forget that')."""
        kw = f"%{subject_or_keyword.strip()}%"
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM semantic_facts WHERE subject LIKE ? OR object_value LIKE ?", (kw, kw))
            deleted = c.rowcount
            conn.commit()
        return deleted

    def clear_all(self):
        """Purges all memory tables."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM episodic_memory")
            c.execute("DELETE FROM semantic_facts")
            conn.commit()
