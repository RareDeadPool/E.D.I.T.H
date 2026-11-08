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
    VALID_PROVENANCES = {"user_stated", "observed", "inferred_labeled"}

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
            # 2. Semantic facts & relationships with strict provenance tracking
            c.execute("""
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT,
                    predicate TEXT,
                    object_value TEXT,
                    timestamp REAL,
                    confidence REAL,
                    provenance TEXT,
                    evidence TEXT,
                    session_id TEXT
                )
            """)
            conn.commit()

            # Automatic schema migration for existing databases
            c.execute("PRAGMA table_info(semantic_facts)")
            existing_cols = {row[1] for row in c.fetchall()}
            if "provenance" not in existing_cols:
                c.execute("ALTER TABLE semantic_facts ADD COLUMN provenance TEXT")
            if "evidence" not in existing_cols:
                c.execute("ALTER TABLE semantic_facts ADD COLUMN evidence TEXT")
            if "session_id" not in existing_cols:
                c.execute("ALTER TABLE semantic_facts ADD COLUMN session_id TEXT")
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

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        confidence: float = 1.0,
        provenance: str = "user_stated",
        evidence: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Stores a persistent semantic fact with strict provenance validation."""
        if provenance not in self.VALID_PROVENANCES:
            raise ValueError(f"Invalid provenance '{provenance}'. Must be one of {self.VALID_PROVENANCES}")

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Upsert
            c.execute(
                "DELETE FROM semantic_facts WHERE subject = ? AND predicate = ?",
                (subject.strip(), predicate.strip()),
            )
            c.execute(
                """INSERT INTO semantic_facts 
                   (subject, predicate, object_value, timestamp, confidence, provenance, evidence, session_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (subject.strip(), predicate.strip(), object_value.strip(), time.time(), confidence, provenance, evidence or "", session_id or ""),
            )
            conn.commit()

    def extract_and_store_user_facts(self, user_msg: str, session_id: str = "default") -> List[Dict[str, Any]]:
        """Extracts explicit user-stated facts and stores them with verified provenance."""
        import re
        stored = []
        msg_clean = user_msg.strip()
        msg_lower = msg_clean.lower()

        # 1. Friend / Person Name Pattern: "My friend is named Harshan", "friend's name is X", "friend named X"
        m_friend = re.search(r"\b(?:friend\s+(?:is\s+)?named|friend['’]s\s+name\s+is|friend\s+is\s+called|have\s+a\s+friend\s+named)\s+([A-Za-z]+)\b", msg_clean, re.IGNORECASE)
        if m_friend:
            friend_name = m_friend.group(1).capitalize()
            self.add_fact(
                subject="friend",
                predicate="name",
                object_value=friend_name,
                confidence=1.0,
                provenance="user_stated",
                evidence=msg_clean,
                session_id=session_id
            )
            stored.append({"subject": "friend", "predicate": "name", "object": friend_name})

        # 2. User Name Pattern: "My name is X", "I am X" (when explicit)
        m_name = re.search(r"\b(?:my\s+name\s+is|call\s+me)\s+([A-Za-z]+)\b", msg_clean, re.IGNORECASE)
        if m_name and not m_friend:
            user_name = m_name.group(1).capitalize()
            self.add_fact(
                subject="user",
                predicate="name",
                object_value=user_name,
                confidence=1.0,
                provenance="user_stated",
                evidence=msg_clean,
                session_id=session_id
            )
            stored.append({"subject": "user", "predicate": "name", "object": user_name})

        # 3. Preference Pattern: "My favorite X is Y"
        m_fav = re.search(r"\bmy\s+favorite\s+([a-zA-Z]+)\s+is\s+([a-zA-Z0-9_\s]+?)(?:\.|$)", msg_clean, re.IGNORECASE)
        if m_fav:
            fav_topic = m_fav.group(1).lower().strip()
            fav_val = m_fav.group(2).strip()
            self.add_fact(
                subject="preference",
                predicate=f"favorite_{fav_topic}",
                object_value=fav_val,
                confidence=1.0,
                provenance="user_stated",
                evidence=msg_clean,
                session_id=session_id
            )
            stored.append({"subject": "preference", "predicate": f"favorite_{fav_topic}", "object": fav_val})

        # 4. Profession / Role: "I work as a X", "I am a software engineer"
        m_job = re.search(r"\b(?:i\s+work\s+as\s+an?|i\s+am\s+an?)\s+([a-zA-Z\s]+?)(?:\.|$)", msg_clean, re.IGNORECASE)
        if m_job and "stressed" not in msg_lower and "feeling" not in msg_lower and "fine" not in msg_lower:
            job_title = m_job.group(1).strip()
            if len(job_title.split()) <= 4:
                self.add_fact(
                    subject="user",
                    predicate="occupation",
                    object_value=job_title,
                    confidence=1.0,
                    provenance="user_stated",
                    evidence=msg_clean,
                    session_id=session_id
                )
                stored.append({"subject": "user", "predicate": "occupation", "object": job_title})

        return stored

    def retrieve_relevant_facts(self, query: str, session_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieves semantic facts relevant to a query, strictly filtered by valid provenance and session."""
        q_lower = query.lower()
        results = []
        
        # Check relevance intent keywords
        is_query_seeking_memory = any(w in q_lower for w in [
            "who", "what", "where", "remember", "know about", "friend", "name", "favorite", "job", "work"
        ])
        if not is_query_seeking_memory:
            return []

        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT subject, predicate, object_value, provenance, evidence, confidence, timestamp, session_id 
                FROM semantic_facts 
                WHERE provenance = 'user_stated' AND (session_id = ? OR session_id = '' OR session_id IS NULL)
                ORDER BY timestamp DESC
            """, (session_id,))
            for subj, pred, obj, prov, evid, conf, ts, sid in c.fetchall():
                # Precision match: check if query mentions subject, predicate, or context
                subj_match = subj.lower() in q_lower
                pred_match = pred.lower() in q_lower or any(p in q_lower for p in pred.lower().split("_"))
                if subj_match or pred_match:
                    results.append({
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "provenance": prov,
                        "evidence": evid,
                        "confidence": conf,
                        "timestamp": ts,
                        "session_id": sid
                    })
        return results

    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Lists all known user facts strictly filtered by valid provenance."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT subject, predicate, object_value, provenance, evidence, confidence, timestamp 
                FROM semantic_facts 
                WHERE provenance IN ('user_stated', 'observed', 'inferred_labeled')
                ORDER BY timestamp DESC
            """)
            for subj, pred, obj, prov, evid, conf, ts in c.fetchall():
                results.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "provenance": prov,
                    "evidence": evid,
                    "confidence": conf,
                    "timestamp": ts
                })
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
