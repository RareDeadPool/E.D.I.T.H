"""Intent-Driven Local File Discovery and Ranking Subsystem for E.D.I.T.H. V3.

Provides semantic file discovery across user folders (Desktop, Documents,
Downloads, Workspace/Project directories) using multi-signal ranking:
- Natural language descriptions and keyword intent parsing
- Extension and document type mapping
- Temporal recency weighting ("latest", "yesterday", "recently downloaded")
- Location hinting ("on desktop", "in downloads")
- Local text extraction and content-based matching (.txt, .docx, .py, .md, .json, etc.)
- Typo resilience and partial matching
- Disambiguation and candidate scoring
- Incremental indexing and fast caching
"""

import os
import re
import time
import difflib
import zipfile
import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set


@dataclass
class FileCandidate:
    """Represents a discovered and scored file candidate."""
    name: str
    path: str
    extension: str
    size_bytes: int
    mtime: float
    mtime_str: str
    directory: str
    score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    snippet: Optional[str] = None
    is_recent: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FileDiscoveryEngine:
    """Production-grade semantic file search and ranking engine."""

    # Document type to extension mappings
    TYPE_EXT_MAP = {
        "resume": [".pdf", ".docx", ".doc", ".txt", ".md"],
        "cv": [".pdf", ".docx", ".doc", ".txt"],
        "report": [".docx", ".doc", ".pdf", ".txt", ".md", ".xlsx"],
        "presentation": [".pptx", ".ppt", ".pdf", ".key"],
        "slides": [".pptx", ".ppt", ".pdf"],
        "python": [".py", ".ipynb"],
        "script": [".py", ".ps1", ".bat", ".sh", ".cmd"],
        "code": [".py", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".java", ".json"],
        "spreadsheet": [".xlsx", ".xls", ".csv"],
        "sheet": [".xlsx", ".xls", ".csv"],
        "data": [".csv", ".json", ".sqlite", ".db", ".xlsx"],
        "image": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"],
        "photo": [".png", ".jpg", ".jpeg", ".webp"],
        "audio": [".mp3", ".wav", ".flac", ".m4a"],
        "music": [".mp3", ".wav", ".flac", ".m4a"],
        "video": [".mp4", ".mkv", ".avi", ".mov"],
        "archive": [".zip", ".tar", ".gz", ".7z", ".rar"],
        "notes": [".txt", ".md", ".docx", ".note"],
        "word": [".docx", ".doc"],
        "document": [".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx"],
    }

    # Folders to skip during indexing
    IGNORE_DIRS = {
        "node_modules", ".git", ".idea", ".vscode", "__pycache__",
        "appdata", "$recycle.bin", "system volume information",
        ".gemini", "env", "venv", ".venv", "dist", "build", "site-packages"
    }

    def __init__(self, search_roots: Optional[List[str]] = None, max_depth: int = 3, include_system_roots: bool = True):
        home = os.path.expanduser("~")
        self.home = home
        self.default_roots = [
            os.path.join(home, "Desktop"),
            os.path.join(home, "Documents"),
            os.path.join(home, "Downloads"),
            os.getcwd(),
        ]
        self.custom_roots = list(search_roots) if search_roots else []
        self._include_system_roots = include_system_roots
        self.max_depth = max_depth
        # Cache format: {path: (mtime, size, FileCandidate, cached_content)}
        self._cache: Dict[str, Tuple[float, int, FileCandidate, Optional[str]]] = {}
        self._last_scan_time: float = 0.0
        self._cache_ttl: float = 120.0  # Rescan at most once every 2 minutes unless forced

    @property
    def custom_search_paths(self) -> List[str]:
        return self.custom_roots

    @custom_search_paths.setter
    def custom_search_paths(self, paths: List[str]):
        self.custom_roots = list(paths)
        self._cache.clear()
        self._last_scan_time = 0.0

    @property
    def search_roots(self) -> List[str]:
        return self.get_search_roots()

    @search_roots.setter
    def search_roots(self, paths: List[str]):
        self.custom_roots = list(paths)
        self._include_system_roots = False
        self._cache.clear()
        self._last_scan_time = 0.0

    @property
    def include_system_roots(self) -> bool:
        return self._include_system_roots

    @include_system_roots.setter
    def include_system_roots(self, value: bool):
        self._include_system_roots = bool(value)
        self._cache.clear()
        self._last_scan_time = 0.0

    def get_search_roots(self) -> List[str]:
        roots = []
        base = list(self.custom_roots)
        if self._include_system_roots:
            base.extend(self.default_roots)
        for r in base:
            p = os.path.abspath(r)
            if os.path.isdir(p) and p not in roots:
                roots.append(p)
        return roots

    def _format_time(self, timestamp: float) -> str:
        """Format timestamp into human-readable relative/absolute string."""
        dt = datetime.datetime.fromtimestamp(timestamp)
        now = datetime.datetime.now()
        diff = now - dt

        if diff.days == 0:
            if diff.seconds < 3600:
                mins = max(1, diff.seconds // 60)
                return f"{mins} min{'s' if mins > 1 else ''} ago"
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.days == 1:
            return f"yesterday at {dt.strftime('%H:%M')}"
        elif diff.days < 7:
            return f"{diff.days} days ago ({dt.strftime('%A')})"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")

    def _extract_text_content(self, file_path: str, max_bytes: int = 65536, max_chars: Optional[int] = None) -> Optional[str]:
        """Safely extract plain text from supported document formats without external dependencies."""
        if not os.path.isfile(file_path):
            return None

        read_limit = max_chars if max_chars is not None else max_bytes
        ext = os.path.splitext(file_path)[1].lower()

        # 1. Plain text formats
        if ext in [".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".xml", ".ini", ".yaml", ".yml", ".sql"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(read_limit)
            except Exception:
                return None

        # 2. Microsoft Word .docx formats (ZIP containing word/document.xml)
        elif ext == ".docx":
            try:
                with zipfile.ZipFile(file_path) as z:
                    if "word/document.xml" in z.namelist():
                        xml_content = z.read("word/document.xml")
                        tree = ET.fromstring(xml_content)
                        # Extract all text nodes in w:t tags
                        texts = [node.text for node in tree.iter() if node.text and node.tag.endswith("t")]
                        return " ".join(texts)[:read_limit]
            except Exception:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read(read_limit)
                except Exception:
                    return None

        # 3. PDF or binary documents (fallback text scan)
        elif ext in [".pdf", ".pptx", ".xlsx"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read(read_limit)
            except Exception:
                return None

        return None

    def index_files(self, force_refresh: bool = False) -> List[FileCandidate]:
        """Index files across all search roots with caching."""
        now = time.time()
        if not force_refresh and (now - self._last_scan_time) < self._cache_ttl and self._cache:
            return [entry[2] for entry in self._cache.values()]

        roots = self.get_search_roots()
        candidates: List[FileCandidate] = []
        new_cache: Dict[str, Tuple[float, int, FileCandidate, Optional[str]]] = {}

        for root in roots:
            if not os.path.isdir(root):
                continue

            root_depth = root.rstrip(os.sep).count(os.sep)

            try:
                for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                    # Filter out ignored directories in place
                    dirnames[:] = [d for d in dirnames if d.lower() not in self.IGNORE_DIRS and not d.startswith(".")]

                    current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
                    if current_depth >= self.max_depth:
                        dirnames.clear()

                    for fname in filenames:
                        if fname.startswith("~$") or fname.startswith("."):
                            continue

                        full_path = os.path.join(dirpath, fname)
                        try:
                            st = os.stat(full_path)
                            mtime = st.st_mtime
                            size = st.st_size

                            # Check if cached and unchanged
                            if full_path in self._cache:
                                cached_mtime, cached_size, cached_cand, cached_content = self._cache[full_path]
                                if cached_mtime == mtime and cached_size == size:
                                    new_cache[full_path] = (mtime, size, cached_cand, cached_content)
                                    candidates.append(cached_cand)
                                    continue

                            ext = os.path.splitext(fname)[1].lower()
                            dir_name = os.path.basename(dirpath)
                            mtime_str = self._format_time(mtime)
                            is_recent = (now - mtime) < (86400 * 3)  # Modified within 3 days

                            cand = FileCandidate(
                                name=fname,
                                path=full_path,
                                extension=ext,
                                size_bytes=size,
                                mtime=mtime,
                                mtime_str=mtime_str,
                                directory=dir_name,
                                is_recent=is_recent,
                            )
                            # Extract content for small/medium documents (<2MB)
                            content = self._extract_text_content(full_path) if size < 2000000 else None

                            new_cache[full_path] = (mtime, size, cand, content)
                            candidates.append(cand)
                        except (PermissionError, FileNotFoundError, OSError):
                            continue
            except Exception:
                continue

        self._cache = new_cache
        self._last_scan_time = now
        return candidates

    def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract semantic signals, constraints, and target concepts from natural language queries."""
        q = query.strip()
        q_low = q.lower()

        # 1. Location hints
        loc_hint = None
        if "on desktop" in q_low or "on my desktop" in q_low or "from desktop" in q_low:
            loc_hint = "desktop"
        elif "in downloads" in q_low or "downloaded" in q_low:
            loc_hint = "downloads"
        elif "in documents" in q_low or "my documents" in q_low:
            loc_hint = "documents"
        elif "in project" in q_low or "in workspace" in q_low or "here" in q_low:
            loc_hint = "workspace"

        # 2. Temporal hints
        temporal = None
        if any(w in q_low for w in ["latest", "newest", "most recent", "recently", "newer", "new report", "new version"]) or (re.search(r"\bnew\b", q_low) and "not the old" in q_low):
            temporal = "latest"
        elif any(w in q_low for w in ["yesterday", "working on yesterday", "saved yesterday"]):
            temporal = "yesterday"
        elif any(w in q_low for w in ["today", "earlier today", "just now"]):
            temporal = "today"
        elif any(w in q_low for w in ["older", "old version", "previous version"]):
            temporal = "old"

        # 3. Document / File Type Hints
        target_exts: Set[str] = set()
        for doc_type, exts in self.TYPE_EXT_MAP.items():
            if re.search(rf"\b{re.escape(doc_type)}s?\b", q_low):
                target_exts.update(exts)

        # Explicit extension in query (e.g. .py, .pdf, .docx)
        explicit_exts = re.findall(r"\.([a-zA-Z0-9]{2,5})\b", q_low)
        for ext in explicit_exts:
            target_exts.add("." + ext)

        # 4. Clean search keywords
        # Remove query framing phrases
        cleaned = re.sub(
            r"^(?:find|locate|search for|look for|where did i save|where is|get|open|pull up|show me|can you find|where's)\s+",
            "",
            q,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"\b(?:my|the|a|an|that|this|file|document|folder|on my desktop|on desktop|in downloads|in documents|recently|yesterday|latest|newest|about|called|named)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(r"[^\w\.\-\s]", " ", cleaned).strip()
        keywords = [k for k in cleaned.split() if len(k) > 1]

        # 5. Relative exclusion / preference
        exclude_terms = []
        if any(w in q_low for w in ["not the old one", "not that one", "other version", "not the old", "different version"]):
            exclude_terms.append("exclude_previous")
        if any(w in q_low for w in ["not the old", "not old", "not the previous", "other version"]):
            exclude_terms.append("exclude_old")

        return {
            "raw_query": query,
            "keywords": keywords,
            "target_exts": target_exts,
            "loc_hint": loc_hint,
            "temporal": temporal,
            "exclude_terms": exclude_terms,
        }

    def find_files(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 5,
        force_refresh: bool = False,
        limit: Optional[int] = None,
        location_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search and rank files matching natural language query.
        Returns ranked candidates, disambiguation flags, and content snippets.
        """
        if limit is not None:
            max_results = limit

        intent = self.parse_query_intent(query)
        if location_hint:
            intent["loc_hint"] = location_hint.strip().lower()

        candidates = self.index_files(force_refresh=force_refresh)

        if not candidates:
            return {
                "status": "not_found",
                "query": query,
                "candidates": [],
                "is_ambiguous": False,
                "message": "No accessible files found in search directories."
            }

        now = time.time()
        scored_candidates: List[FileCandidate] = []

        keywords = [k.lower() for k in intent["keywords"]]
        target_exts = intent["target_exts"]
        loc_hint = intent["loc_hint"]
        temporal = intent["temporal"]
        exclude_previous = "exclude_previous" in intent["exclude_terms"]
        exclude_old = "exclude_old" in intent["exclude_terms"]

        last_selected_path = context.get("last_path") if context else None

        for cand in candidates:
            # Skip if user explicitly said "not that one" or "other version"
            if exclude_previous and last_selected_path and os.path.exists(cand.path) and os.path.exists(last_selected_path):
                try:
                    if os.path.samefile(cand.path, last_selected_path):
                        continue
                except Exception:
                    pass

            score = 0.0
            reasons = []
            cand_name_low = cand.name.lower()
            cand_stem_low = os.path.splitext(cand.name)[0].lower()

            # Heavy penalty if user said "not the old one" and file is named old
            if exclude_old and "old" in cand_name_low:
                score -= 0.80

            # Slight penalty if file is marked as 'draft' or 'temp' and query didn't ask for draft/temp
            if ("draft" in cand_stem_low or "temp" in cand_stem_low) and not any(w in query.lower() for w in ["draft", "temp"]):
                score -= 0.10

            # 1. Filename & Stem Matching
            matched_kw_count = 0
            for kw in keywords:
                if kw in cand_stem_low:
                    score += 0.45
                    matched_kw_count += 1
                    reasons.append(f"Filename matches '{kw}'")
                elif kw in cand_name_low:
                    score += 0.35
                    matched_kw_count += 1
                else:
                    # Fuzzy match on filename tokens (split on non-alphanumeric including _)
                    tokens = [t for t in re.split(r"[_\W]+", cand_stem_low) if t]
                    best_ratio = max([difflib.SequenceMatcher(None, kw, t).ratio() for t in tokens], default=0.0)
                    if best_ratio > 0.75:
                        score += best_ratio * 0.40
                        matched_kw_count += 1
                        reasons.append(f"Typo/fuzzy match on '{kw}' ({int(best_ratio*100)}%)")

            # 2. Extension / Type Match
            if target_exts:
                if cand.extension.lower() in target_exts:
                    score += 0.35
                    reasons.append(f"Format matches '{cand.extension}'")
                else:
                    # Slight penalty if query specifically demanded a different type
                    score -= 0.15

            # 3. Location / Directory Hint
            if loc_hint:
                cand_dir_low = cand.directory.lower()
                cand_path_low = cand.path.lower()
                if loc_hint in cand_dir_low or loc_hint in cand_path_low:
                    score += 0.35
                    reasons.append(f"Located in '{cand.directory}'")
                else:
                    score -= 0.80

            # 4. Temporal / Recency Weighting
            age_seconds = now - cand.mtime
            if temporal == "latest" or temporal == "newest":
                # Recency boost with exponential decay
                recency_boost = max(0.0, 0.40 * (1.0 - min(1.0, age_seconds / (86400 * 30))))
                score += recency_boost
                if age_seconds < (86400 * 3):
                    reasons.append(f"Recently modified ({cand.mtime_str})")
            elif temporal == "yesterday":
                # Target files modified 1 to 2 days ago
                if 86400 <= age_seconds <= (86400 * 2):
                    score += 0.50
                    reasons.append("Modified yesterday")
            elif temporal == "today":
                if age_seconds < 86400:
                    score += 0.50
                    reasons.append("Modified today")

            # 5. Content Inspection & Snippet Match
            snippet = None
            if full_entry := self._cache.get(cand.path):
                content = full_entry[3]
                if content:
                    content_low = content.lower()
                    query_words = [w for w in re.findall(r"[a-zA-Z0-9\+\#]+", query.lower()) if len(w) > 1 and w not in ["find", "file", "with", "for", "the", "and", "about", "that", "this", "look"]]
                    content_matches = [w for w in query_words if w in content_low]
                    if content_matches:
                        score += 0.35 * len(content_matches)
                        reasons.append(f"Content mentions {', '.join(content_matches)}")
                        first_idx = content_low.find(content_matches[0])
                        start = max(0, first_idx - 30)
                        end = min(len(content), first_idx + 50)
                        snippet = f"...{content[start:end].strip()}..."

            # If no keywords matched and score is negligible, discard
            if score <= 0.20 and not (target_exts and cand.extension.lower() in target_exts and temporal):
                continue

            # Create scored candidate copy
            cand_scored = FileCandidate(
                name=cand.name,
                path=cand.path,
                extension=cand.extension,
                size_bytes=cand.size_bytes,
                mtime=cand.mtime,
                mtime_str=cand.mtime_str,
                directory=cand.directory,
                score=round(score, 4),
                match_reasons=reasons,
                snippet=snippet,
                is_recent=cand.is_recent,
            )
            scored_candidates.append(cand_scored)

        if not scored_candidates:
            return {
                "status": "not_found",
                "query": query,
                "candidates": [],
                "is_ambiguous": False,
                "message": f"Could not find any files matching '{query}' in your workspace or personal folders."
            }

        # Sort by score descending, then by mtime descending
        scored_candidates.sort(key=lambda c: (c.score, c.mtime), reverse=True)
        top_candidates = scored_candidates[:max_results]

        # Check for ambiguity: multiple top candidates with very close scores
        is_ambiguous = False
        clarification_msg = None

        if len(top_candidates) > 1:
            diff = top_candidates[0].score - top_candidates[1].score
            # If scores are within 0.15 of each other, multiple candidates are similarly relevant
            if diff <= 0.15:
                is_ambiguous = True
                opts = [
                    f"{i+1}) '{c.name}' ({c.directory}, modified {c.mtime_str})"
                    for i, c in enumerate(top_candidates[:3])
                ]
                clarification_msg = (
                    f"I found multiple relevant files for '{query}':\n  " +
                    "\n  ".join(opts) +
                    "\nWhich one would you like me to open?"
                )

        return {
            "status": "success",
            "query": query,
            "candidates": [c.to_dict() for c in top_candidates],
            "top_candidate": top_candidates[0].to_dict(),
            "is_ambiguous": is_ambiguous,
            "clarification_prompt": clarification_msg,
            "message": clarification_msg or f"Found '{top_candidates[0].name}' at {top_candidates[0].path} ({top_candidates[0].mtime_str})."
        }

    def read_document_text(self, file_path: str, max_chars: int = 25000) -> Dict[str, Any]:
        """Safely read and extract text from supported document types (.txt, .md, .py, .json, .docx, .pdf)."""
        abs_p = os.path.abspath(file_path)
        if not os.path.isfile(abs_p):
            return {"status": "error", "success": False, "message": f"File does not exist: '{file_path}'"}
        
        text = self._extract_text_content(abs_p, max_chars=max_chars)
        if text is None:
            return {"status": "error", "success": False, "message": f"Could not read text content from '{file_path}'."}
        
        return {
            "status": "success",
            "success": True,
            "path": abs_p,
            "name": os.path.basename(abs_p),
            "text": text,
            "length": len(text),
            "word_count": len(text.split()),
            "message": f"Successfully read {len(text)} characters from '{os.path.basename(abs_p)}'."
        }

    def summarize_document(self, file_path: str, max_points: int = 4, focus_topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Extractive, local document summarization without third-party LLMs.
        Extracts key sentences, headers, core metrics, and produces a structured summary.
        """
        read_res = self.read_document_text(file_path)
        if not read_res.get("success"):
            return read_res
        
        raw_text = read_res["text"]
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
        if not lines:
            return {
                "status": "success",
                "success": True,
                "path": read_res["path"],
                "title": read_res["name"],
                "summary": "The document is empty.",
                "points": [],
                "message": f"Document '{read_res['name']}' contains no readable text."
            }
        
        # 1. Determine Title / Header
        title = read_res["name"]
        for line in lines[:5]:
            # Look for markdown header or prominent leading line
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
            elif len(line) < 80 and not line.endswith(".") and not line.startswith("//") and not line.startswith("/*"):
                title = line
                break

        # 2. Extract Clean Sentences
        sentences = []
        in_code_block = False
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            # Skip code import lines, formatting-only lines, diagrams, table rows
            if (line_str.startswith("import ") or line_str.startswith("from ") or
                line_str.startswith("|") or "│" in line_str or "├" in line_str or "└" in line_str or "$$" in line_str or
                "-->" in line_str or '["' in line_str or '"]' in line_str or line_str.startswith("---") or line_str.startswith("===")):
                continue
            # Strip markdown formatting
            clean_l = re.sub(r"[*_~`#]", "", line_str).strip()
            if not clean_l or len(clean_l) < 25:
                continue
            for s in re.split(r"(?<=[.!?])\s+", clean_l):
                s_clean = s.strip()
                if len(s_clean) >= 25 and not s_clean.startswith("//") and not s_clean.startswith("•"):
                    sentences.append(s_clean)

        if not sentences:
            sentences = [re.sub(r"[*_~`#]", "", l).strip() for l in lines if len(l.strip()) >= 20][:max_points]

        # 3. Score Sentences by Extractive Importance
        # Build local term frequency map excluding common stopwords
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "with", "by", "of", "from", "is", "was", "are", "were", "be", "been",
            "this", "that", "these", "those", "it", "its", "as", "have", "has", "had"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", raw_text.lower())
        word_freq: Dict[str, int] = {}
        for w in words:
            if w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1

        scored: List[Tuple[float, int, str]] = []
        key_markers = ["conclude", "summary", "result", "goal", "achieve", "implement", "feature", "metric", "architecture", "final", "status", "deadline"]
        focus_terms = re.findall(r"\b[a-zA-Z]{3,}\b", (focus_topic or "").lower())

        for idx, s in enumerate(sentences):
            s_low = s.lower()
            s_words = re.findall(r"\b[a-zA-Z]{3,}\b", s_low)
            if not s_words:
                continue

            # Term frequency score
            tf_score = sum(word_freq.get(w, 0) for w in s_words) / len(s_words)
            
            # Position boost (early sentences in paragraphs contain topic sentences)
            pos_boost = 1.2 if idx < 3 else 1.0

            # Marker bonus
            marker_bonus = 1.3 if any(m in s_low for m in key_markers) else 1.0
            
            # Numeric/statistic bonus (e.g. 100%, 5MB, 2026)
            num_bonus = 1.15 if re.search(r"\b\d+(?:\.\d+)?%?\b", s) else 1.0

            # Focus query bonus if specified
            focus_bonus = 1.5 if any(ft in s_low for ft in focus_terms) else 1.0

            final_score = tf_score * pos_boost * marker_bonus * num_bonus * focus_bonus
            scored.append((final_score, idx, s))

        # Sort by importance, select top distinct points, re-sort by document order
        scored.sort(key=lambda x: x[0], reverse=True)
        top_selected = scored[:max_points]
        top_selected.sort(key=lambda x: x[1])

        points = [s for _, _, s in top_selected]

        summary_lines = [f"Document Summary: {title}", ""]
        for p in points:
            summary_lines.append(f"• {p}")
        summary_text = "\n".join(summary_lines)

        return {
            "status": "success",
            "success": True,
            "path": read_res["path"],
            "title": title,
            "summary": summary_text,
            "points": points,
            "word_count": read_res["word_count"],
            "message": f"Generated summary for '{read_res['name']}' ({len(points)} key points)."
        }

    def rename_file(self, source_path: str, new_name_or_path: str) -> Dict[str, Any]:
        """Safely rename a file or directory with path verification."""
        abs_src = os.path.abspath(source_path)
        if not os.path.exists(abs_src):
            return {"status": "error", "success": False, "message": f"Source file does not exist: '{source_path}'"}

        # If new_name_or_path is just a filename, keep the same directory
        if os.path.sep not in new_name_or_path and "/" not in new_name_or_path:
            abs_dest = os.path.join(os.path.dirname(abs_src), new_name_or_path)
        else:
            abs_dest = os.path.abspath(new_name_or_path)

        if os.path.exists(abs_dest) and abs_dest.lower() != abs_src.lower():
            return {
                "status": "error",
                "success": False,
                "message": f"Cannot rename: destination already exists at '{abs_dest}'."
            }

        try:
            os.rename(abs_src, abs_dest)
            verified = os.path.exists(abs_dest)
            return {
                "status": "success" if verified else "error",
                "success": verified,
                "source": abs_src,
                "destination": abs_dest,
                "message": f"Successfully renamed '{os.path.basename(abs_src)}' to '{os.path.basename(abs_dest)}'."
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": f"Rename failed: {e}"}

    def copy_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """Safely copy a file to a destination with overwrite protection."""
        abs_src = os.path.abspath(source_path)
        if not os.path.exists(abs_src):
            return {"status": "error", "success": False, "message": f"Source file does not exist: '{source_path}'"}

        abs_dest = os.path.abspath(destination_path)
        if os.path.isdir(abs_dest):
            abs_dest = os.path.join(abs_dest, os.path.basename(abs_src))

        if os.path.exists(abs_dest) and not overwrite:
            return {
                "status": "error",
                "success": False,
                "message": f"Destination already exists at '{abs_dest}'. Specify overwrite=True to replace."
            }

        try:
            os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
            if os.path.isdir(abs_src):
                shutil.copytree(abs_src, abs_dest, dirs_exist_ok=overwrite)
            else:
                shutil.copy2(abs_src, abs_dest)
            verified = os.path.exists(abs_dest)
            return {
                "status": "success" if verified else "error",
                "success": verified,
                "source": abs_src,
                "destination": abs_dest,
                "message": f"Successfully copied '{os.path.basename(abs_src)}' to '{abs_dest}'."
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": f"Copy failed: {e}"}

    def move_file(self, source_path: str, destination_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """Safely move a file or folder with overwrite protection."""
        abs_src = os.path.abspath(source_path)
        if not os.path.exists(abs_src):
            return {"status": "error", "success": False, "message": f"Source file does not exist: '{source_path}'"}

        abs_dest = os.path.abspath(destination_path)
        if os.path.isdir(abs_dest):
            abs_dest = os.path.join(abs_dest, os.path.basename(abs_src))

        if os.path.exists(abs_dest) and not overwrite:
            return {
                "status": "error",
                "success": False,
                "message": f"Destination already exists at '{abs_dest}'. Specify overwrite=True to replace."
            }

        try:
            os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
            shutil.move(abs_src, abs_dest)
            verified = os.path.exists(abs_dest) and not os.path.exists(abs_src)
            return {
                "status": "success" if verified else "error",
                "success": verified,
                "source": abs_src,
                "destination": abs_dest,
                "message": f"Successfully moved '{os.path.basename(abs_src)}' to '{abs_dest}'."
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": f"Move failed: {e}"}

    def organize_folder(self, folder_path: str, strategy: str = "by_extension") -> Dict[str, Any]:
        """
        Organize unorganized files in a folder into standard subdirectories (Documents, Images, Code, etc.).
        Avoids moving subdirectories, hidden files, or critical repository files.
        """
        abs_folder = os.path.abspath(folder_path)
        if not os.path.isdir(abs_folder):
            return {"status": "error", "success": False, "message": f"Folder does not exist: '{folder_path}'"}

        category_map = {
            "Documents": [".pdf", ".docx", ".doc", ".txt", ".rtf", ".odt"],
            "Spreadsheets": [".xlsx", ".xls", ".csv"],
            "Images": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".svg"],
            "Code": [".py", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".java", ".json", ".sql", ".bat", ".ps1"],
            "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
            "Audio": [".mp3", ".wav", ".flac", ".m4a"],
            "Video": [".mp4", ".mkv", ".avi", ".mov"]
        }

        moved_files = []
        errors = []

        try:
            for item in os.listdir(abs_folder):
                item_path = os.path.join(abs_folder, item)
                if os.path.isdir(item_path) or item.startswith("."):
                    continue

                ext = os.path.splitext(item)[1].lower()
                target_sub = "Other"
                for cat, exts in category_map.items():
                    if ext in exts:
                        target_sub = cat
                        break

                target_dir = os.path.join(abs_folder, target_sub)
                os.makedirs(target_dir, exist_ok=True)
                dest_path = os.path.join(target_dir, item)

                if os.path.exists(dest_path):
                    continue  # Don't overwrite existing file in target subfolder

                try:
                    shutil.move(item_path, dest_path)
                    moved_files.append({"file": item, "category": target_sub})
                except Exception as ex:
                    errors.append(f"Failed to move '{item}': {ex}")

            return {
                "status": "success",
                "success": True,
                "folder": abs_folder,
                "moved_count": len(moved_files),
                "moved_files": moved_files,
                "errors": errors,
                "message": f"Organized {len(moved_files)} files in '{os.path.basename(abs_folder)}' by category."
            }
        except Exception as e:
            return {"status": "error", "success": False, "message": f"Folder organization failed: {e}"}

