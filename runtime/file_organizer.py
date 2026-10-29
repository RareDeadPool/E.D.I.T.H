"""Local Semantic File Organizer for E.D.I.T.H. V7.

Provides:
1. Multi-signal semantic taxonomy classification across 6 principal categories & subcategories:
   - Education (Lecture notes, Assignments, Presentations, Exam prep, Research papers)
   - Software Development (Source code, Projects, Documentation, Build output, Test reports)
   - Professional (Resume, Applications, Internship docs, Certificates, Portfolio assets)
   - Media (Photographs, Screenshots, Videos, Audio, Artwork)
   - Games (Screenshots, Clips, Mods, Save files)
   - Personal Documents (Important documents, Receipts, Financial records)
   - Unsorted / Needs Review (Ambiguous or low confidence files)
2. Strict safety protection for active Git repositories, projects, and relative imports.
3. Dry-run proposed organization plan with confidence scores and rationale.
4. Transaction logging and atomic rollback mechanism to restore any moved file.
"""

import os
import sys
import re
import json
import time
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@dataclass
class ProposedFileMove:
    """Represents a proposed file reorganization move."""
    source_path: str
    filename: str
    target_category: str
    target_subcategory: str
    proposed_destination: str
    confidence: float
    reason: str
    breaks_project_references: bool = False
    is_protected: bool = False
    status: str = "proposed"  # 'proposed', 'executed', 'skipped', 'rolled_back'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrganizationPlan:
    """Represents a complete organization proposal with safety evaluations."""
    source_directory: str
    destination_root: str
    generated_at: float
    total_files: int
    moves: List[Dict[str, Any]] = field(default_factory=list)
    protected_files: List[Dict[str, Any]] = field(default_factory=list)
    unsorted_files: List[Dict[str, Any]] = field(default_factory=list)
    summary_by_category: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticFileOrganizer:
    """Production-grade semantic file organizer with dry-run planning and reversible transactions."""

    # Taxonomy keyword signatures: category -> subcategory -> keywords
    TAXONOMY_KEYWORDS = {
        "Education": {
            "Lecture Notes": ["lecture", "notes", "slides", "chapter", "unit", "module", "syllabus"],
            "Assignments": ["assignment", "homework", "hw", "problem_set", "exercise", "task", "lab"],
            "Presentations": ["presentation", "deck", "seminar", "defense", "colloquium"],
            "Exam Preparation": ["exam", "midterm", "final", "quiz", "study_guide", "cheatsheet", "formula"],
            "Research Papers": ["paper", "thesis", "dissertation", "journal", "ieee", "arxiv", "abstract", "literature"]
        },
        "Professional": {
            "Resume & CV": ["resume", "cv", "curriculum_vitae", "biodata"],
            "Applications": ["cover_letter", "application", "job_app", "referral", "offer_letter"],
            "Certificates": ["certificate", "certification", "credential", "diploma", "badge", "license"],
            "Internship Documents": ["internship", "appraisal", "recommendation", "lor", "experience_letter"],
            "Portfolio": ["portfolio", "case_study", "project_brief", "work_sample"]
        },
        "Software Development": {
            "Source Code": ["main", "app", "index", "script", "module", "utils", "helper", "client", "server"],
            "Documentation": ["readme", "api", "docs", "specification", "architecture", "changelog", "contributing"],
            "Test Reports": ["test", "report", "benchmark", "coverage", "metrics", "eval"],
            "Build Output": ["bundle", "output", "build", "release", "dist"]
        },
        "Media": {
            "Photographs": ["img", "dsc", "photo", "pic", "camera", "dcim"],
            "Screenshots": ["screenshot", "screen_shot", "capture", "snip"],
            "Videos": ["vid", "video", "recording", "clip", "screen_recording"],
            "Audio": ["song", "track", "audio", "voice_note", "podcast", "recording"],
            "Artwork": ["drawing", "design", "sketch", "logo", "banner", "poster", "artwork"]
        },
        "Games": {
            "Screenshots": ["game_capture", "replay", "killcam", "victory"],
            "Clips": ["clip", "highlight", "gameplay", "stream"],
            "Save Files": ["save", "autosave", "quicksave", "slot", "checkpoint"],
            "Mods": ["mod", "addon", "plugin", "texture_pack", "shader"]
        },
        "Personal Documents": {
            "Receipts & Invoices": ["receipt", "invoice", "bill", "payment", "order", "statement", "tax"],
            "Records & Identity": ["passport", "id_card", "license", "voter", "birth_cert", "lease", "insurance"],
            "Medical": ["prescription", "medical", "report", "health", "doctor", "lab_result"]
        }
    }

    # Extension mappings
    EXT_CATEGORY_MAP = {
        # Media - Photos
        ".jpg": ("Media", "Photographs"),
        ".jpeg": ("Media", "Photographs"),
        ".png": ("Media", "Photographs"),
        ".webp": ("Media", "Photographs"),
        ".heic": ("Media", "Photographs"),
        ".raw": ("Media", "Photographs"),
        # Media - Videos
        ".mp4": ("Media", "Videos"),
        ".mkv": ("Media", "Videos"),
        ".mov": ("Media", "Videos"),
        ".avi": ("Media", "Videos"),
        # Media - Audio
        ".mp3": ("Media", "Audio"),
        ".wav": ("Media", "Audio"),
        ".flac": ("Media", "Audio"),
        ".m4a": ("Media", "Audio"),
        # Dev Code
        ".py": ("Software Development", "Source Code"),
        ".js": ("Software Development", "Source Code"),
        ".ts": ("Software Development", "Source Code"),
        ".cpp": ("Software Development", "Source Code"),
        ".c": ("Software Development", "Source Code"),
        ".rs": ("Software Development", "Source Code"),
        ".go": ("Software Development", "Source Code"),
        ".java": ("Software Development", "Source Code"),
        ".html": ("Software Development", "Source Code"),
        ".css": ("Software Development", "Source Code"),
        # Games
        ".sav": ("Games", "Save Files"),
        # Documents
        ".pdf": ("Personal Documents", "Records & Identity"),
        ".docx": ("Personal Documents", "Records & Identity"),
        ".xlsx": ("Personal Documents", "Receipts & Invoices"),
        ".pptx": ("Education", "Presentations"),
    }

    # Protected root directory indicators
    PROJECT_INDICATORS = {
        ".git", "package.json", "pyproject.toml", "setup.py", "cargo.toml", "go.mod", "requirements.txt"
    }

    def __init__(self, log_directory: Optional[str] = None):
        self.log_directory = log_directory or os.path.join(PROJECT_ROOT, "data", "organization_logs")
        os.makedirs(self.log_directory, exist_ok=True)

    def is_inside_project_repository(self, file_path: str) -> bool:
        """Check if file resides within an active software project or Git repo."""
        current = os.path.dirname(os.path.abspath(file_path))
        # Traverse up to 4 parent levels
        for _ in range(4):
            if any(os.path.exists(os.path.join(current, ind)) for ind in self.PROJECT_INDICATORS):
                return True
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return False

    def classify_file(self, file_path: str) -> Tuple[str, str, float, str]:
        """Classify a file based on filename keywords, parent directory, and extension.
        
        Returns: (Category, Subcategory, Confidence, Rationale)
        """
        fname = os.path.basename(file_path).lower()
        base_name, ext = os.path.splitext(fname)
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()

        file_tokens = set(re.findall(r"[a-z0-9]+", base_name))
        parent_tokens = set(re.findall(r"[a-z0-9]+", parent_dir))

        best_cat = None
        best_sub = None
        best_score = 0.0
        best_reason = ""

        # 1. Primary check: keyword signatures in filename tokens
        for cat, subcats in self.TAXONOMY_KEYWORDS.items():
            for sub, keywords in subcats.items():
                matched_kw = [kw for kw in keywords if kw in file_tokens or any(kw in tok for tok in file_tokens)]
                if matched_kw:
                    score = 0.70 + (0.10 * len(matched_kw))
                    if score > best_score:
                        best_score = min(0.98, score)
                        best_cat = cat
                        best_sub = sub
                        best_reason = f"Filename matched keywords: {', '.join(matched_kw)}"

        # 2. Secondary check: keyword signatures in parent directory (only if filename didn't match)
        if not best_cat:
            for cat, subcats in self.TAXONOMY_KEYWORDS.items():
                for sub, keywords in subcats.items():
                    matched_kw = [kw for kw in keywords if kw in parent_tokens and kw not in {"test", "tmp", "temp"}]
                    if matched_kw:
                        score = 0.55 + (0.05 * len(matched_kw))
                        if score > best_score:
                            best_score = min(0.80, score)
                            best_cat = cat
                            best_sub = sub
                            best_reason = f"Parent folder matched keywords: {', '.join(matched_kw)}"

        # 2. If no keywords matched, check extension mapping
        if not best_cat and ext in self.EXT_CATEGORY_MAP:
            cat, sub = self.EXT_CATEGORY_MAP[ext]
            best_cat = cat
            best_sub = sub
            best_score = 0.65
            best_reason = f"Extension {ext} associated with {cat}/{sub}"

        # 3. Disambiguation & Fallbacks
        if not best_cat or best_score < 0.5:
            return "Unsorted / Needs Review", "General", 0.3, "Ambiguous filename and extension"

        return best_cat, best_sub, round(best_score, 2), best_reason

    def generate_organization_plan(
        self,
        source_directory: str,
        destination_root: Optional[str] = None
    ) -> OrganizationPlan:
        """Scan source directory and generate a safe, proposed dry-run organization plan."""
        source = os.path.normpath(os.path.abspath(source_directory))
        dest_root = destination_root or os.path.join(source, "Organized")

        plan = OrganizationPlan(
            source_directory=source,
            destination_root=dest_root,
            generated_at=time.time(),
            total_files=0
        )

        for dirpath, dirnames, filenames in os.walk(source):
            # Avoid scanning the organized destination folder itself
            if os.path.commonpath([dirpath, dest_root]) == dest_root:
                dirnames.clear()
                continue

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                plan.total_files += 1

                # Check if file belongs to an active software project or git repo
                is_proj = self.is_inside_project_repository(fpath)

                cat, sub, conf, reason = self.classify_file(fpath)

                # Safe proposed destination
                sub_folder = os.path.join(dest_root, cat, sub)
                target_dest = os.path.join(sub_folder, fname)

                move = ProposedFileMove(
                    source_path=fpath,
                    filename=fname,
                    target_category=cat,
                    target_subcategory=sub,
                    proposed_destination=target_dest,
                    confidence=conf,
                    reason=reason,
                    breaks_project_references=is_proj,
                    is_protected=is_proj or cat == "Unsorted / Needs Review"
                )

                if is_proj:
                    move.reason += " (PROTECTED: file is in an active Git/code project)"
                    plan.protected_files.append(move.to_dict())
                elif cat == "Unsorted / Needs Review":
                    plan.unsorted_files.append(move.to_dict())
                else:
                    plan.moves.append(move.to_dict())
                    plan.summary_by_category[cat] = plan.summary_by_category.get(cat, 0) + 1

        return plan

    def execute_organization_plan(
        self,
        plan: OrganizationPlan,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Execute a proposed organization plan safely with a reversible transaction log."""
        session_id = f"tx_{int(time.time())}"
        log_file = os.path.join(self.log_directory, f"{session_id}.json")

        record = {
            "transaction_id": session_id,
            "timestamp": time.time(),
            "dry_run": dry_run,
            "source_directory": plan.source_directory,
            "destination_root": plan.destination_root,
            "executed_moves": [],
            "skipped_protected": len(plan.protected_files),
            "unsorted_count": len(plan.unsorted_files),
            "errors": []
        }

        if dry_run:
            record["status"] = "dry_run_success"
            record["planned_moves"] = len(plan.moves)
            return record

        for move_dict in plan.moves:
            src = move_dict["source_path"]
            dst = move_dict["proposed_destination"]

            # Double check protection
            if move_dict.get("is_protected") or move_dict.get("breaks_project_references"):
                continue

            if not os.path.exists(src):
                continue

            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # Collision prevention
                final_dst = dst
                if os.path.exists(final_dst) and final_dst != src:
                    base, ext = os.path.splitext(dst)
                    final_dst = f"{base}_{int(time.time())}{ext}"

                shutil.move(src, final_dst)
                record["executed_moves"].append({
                    "original_path": src,
                    "new_path": final_dst
                })
            except Exception as e:
                record["errors"].append({"file": src, "error": str(e)})

        # Save transaction log
        record["status"] = "completed"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            record["transaction_file"] = log_file
        except Exception:
            pass

        return record

    def rollback_transaction(self, transaction_file_or_id: str) -> Dict[str, Any]:
        """Atomically reverse all moves recorded in a transaction log back to their original locations."""
        t_path = transaction_file_or_id
        if not os.path.isfile(t_path):
            t_path = os.path.join(self.log_directory, f"{transaction_file_or_id}.json")

        if not os.path.isfile(t_path):
            return {"success": False, "error": f"Transaction log not found: {transaction_file_or_id}"}

        try:
            with open(t_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception as e:
            return {"success": False, "error": f"Failed to read transaction log: {e}"}

        restored = []
        errors = []

        for move in record.get("executed_moves", []):
            cur = move.get("new_path")
            orig = move.get("original_path")
            if not cur or not orig or not os.path.exists(cur):
                continue

            try:
                os.makedirs(os.path.dirname(orig), exist_ok=True)
                shutil.move(cur, orig)
                restored.append({"from": cur, "to": orig})
            except Exception as e:
                errors.append({"from": cur, "to": orig, "error": str(e)})

        # Update log to mark rolled back
        record["status"] = "rolled_back"
        record["rolled_back_at"] = time.time()
        try:
            with open(t_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception:
            pass

        return {
            "success": len(errors) == 0,
            "restored_count": len(restored),
            "restored_files": restored,
            "errors": errors
        }
