"""Safe Storage Intelligence and Cleanup Subsystem for E.D.I.T.H. V7.

Phase A: Storage Audit (Capacity, large directories, large files, duplicates by SHA-256,
         caches, build artifacts, installers, logs).
Phase B: Multi-signal Importance Assessment (Type, location, recency, project membership,
         hash verification, user declarations).
Phase C: Strict 4-Tier Risk Categorization:
         - Category A: Low Risk (disposable caches, reproducible build output, temporary logs).
         - Category B: Review Recommended (old downloads, installers, duplicate archives).
         - Category C: Personal or Potentially Important (photos, videos, documents, game saves - NEVER auto-deleted).
         - Category D: Protected (OS files, app data, credentials, databases, git repositories).
Phase D: Reversible Cleanup & Quarantine (dry-run analysis, space savings, quarantine with rollback manifest).
Phase E: Learning & User Preference Persistence (data/storage_preferences.json).
"""

import os
import sys
import json
import time
import shutil
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class StorageItem:
    """Represents a discovered file or directory candidate for analysis."""
    path: str
    name: str
    size_bytes: int
    size_mb: float
    mtime: float
    extension: str
    category: str  # 'A', 'B', 'C', 'D'
    risk_label: str  # 'Low Risk', 'Review Recommended', 'Personal / Important', 'Protected'
    reason: str
    is_duplicate: bool = False
    content_hash: Optional[str] = None
    suggested_action: str = "keep"  # 'quarantine', 'review', 'keep'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StorageAuditReport:
    """Structured report generated from a storage audit."""
    scanned_root: str
    scan_timestamp: float
    disk_total_gb: float
    disk_free_gb: float
    disk_used_percent: float
    total_scanned_files: int
    total_scanned_bytes: int
    potential_savings_bytes: int
    potential_savings_mb: float
    largest_files: List[Dict[str, Any]] = field(default_factory=list)
    largest_directories: List[Dict[str, Any]] = field(default_factory=list)
    category_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    candidates_for_cleanup: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StorageAnalyzer:
    """Safe, reversible storage intelligence and cleanup analyzer."""

    # Category A: Low risk disposable caches & build outputs
    DISPOSABLE_CACHE_NAMES = {
        "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
        ".tox", ".parcel-cache", "thumbs.db", "desktop.ini"
    }
    DISPOSABLE_CACHE_DIRS = {
        "node_modules", "dist", "build", ".next", ".nuxt", "bin", "obj"
    }
    DISPOSABLE_EXTENSIONS = {
        ".tmp", ".temp", ".log", ".bak", ".swp", ".pyc", ".pyo"
    }

    # Category B: Review recommended (Installers & archives)
    INSTALLER_EXTENSIONS = {
        ".msi", ".iso", ".img", ".vmdk"
    }
    ARCHIVE_EXTENSIONS = {
        ".zip", ".tar", ".gz", ".7z", ".rar", ".bz2", ".xz"
    }

    # Category C: Personal or potentially important (NEVER auto-delete)
    PERSONAL_EXTENSIONS = {
        # Images & Photos
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".raw", ".cr2", ".nef", ".heic",
        # Videos & Clips
        ".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv",
        # Audio & Music
        ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg",
        # Documents
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".odt", ".txt", ".md",
        # Creative & Project Assets
        ".psd", ".ai", ".blend", ".svg", ".prproj", ".aep",
        # Game Saves
        ".sav", ".save", ".dat"
    }

    # Category D: Protected directories and system markers
    PROTECTED_DIR_NAMES = {
        "windows", "system32", "program files", "program files (x86)",
        "appdata", ".git", ".svn", ".hg", "system volume information",
        "$recycle.bin", ".ssh", ".gnupg", ".aws", ".azure"
    }

    def __init__(self, preferences_path: Optional[str] = None):
        self.preferences_path = preferences_path or os.path.join(
            PROJECT_ROOT, "data", "storage_preferences.json"
        )
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict[str, Any]:
        """Load user-declared preferences and folder exclusions."""
        default_prefs = {
            "protected_folders": [],
            "active_projects": [],
            "disposable_folders": [],
            "never_move_folders": [],
            "custom_cleanup_rules": []
        }
        if os.path.isfile(self.preferences_path):
            try:
                with open(self.preferences_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    default_prefs.update(data)
            except Exception:
                pass
        return default_prefs

    def save_preferences(self, preferences: Dict[str, Any]) -> bool:
        """Persist user storage preferences."""
        self.preferences.update(preferences)
        try:
            os.makedirs(os.path.dirname(self.preferences_path), exist_ok=True)
            with open(self.preferences_path, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, indent=2)
            return True
        except Exception:
            return False

    def mark_active_project(self, folder_path: str):
        """Mark a folder as an active project (protects code, prevents aggressive cleanup)."""
        norm = os.path.normpath(os.path.abspath(folder_path))
        if norm not in self.preferences["active_projects"]:
            self.preferences["active_projects"].append(norm)
            self.save_preferences(self.preferences)

    def mark_important_personal_folder(self, folder_path: str):
        """Mark a folder as important personal files (Category C, strictly protected)."""
        norm = os.path.normpath(os.path.abspath(folder_path))
        if norm not in self.preferences["protected_folders"]:
            self.preferences["protected_folders"].append(norm)
            self.save_preferences(self.preferences)

    @staticmethod
    def compute_sha256(filepath: str, max_bytes: int = 10 * 1024 * 1024) -> Optional[str]:
        """Compute SHA-256 content hash of a file (up to max_bytes for speed)."""
        try:
            h = hashlib.sha256()
            bytes_read = 0
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
                    bytes_read += len(chunk)
                    if max_bytes and bytes_read >= max_bytes:
                        break
            return h.hexdigest()
        except Exception:
            return None

    def classify_file(self, path: str, size: int, mtime: float) -> Tuple[str, str, str, str]:
        """Classify candidate into Category A, B, C, or D with reason and suggested action."""
        norm_path = os.path.normpath(os.path.abspath(path)).lower()
        fname = os.path.basename(norm_path)
        ext = os.path.splitext(fname)[1].lower()
        now = time.time()
        age_days = (now - mtime) / 86400.0

        # Check explicit user declared protections first
        for prot in self.preferences.get("protected_folders", []):
            if os.path.normpath(prot).lower() in norm_path:
                return "D", "Protected", "Explicit user declared protected folder", "keep"

        for act in self.preferences.get("active_projects", []):
            if os.path.normpath(act).lower() in norm_path:
                # Inside active project: only clear low-risk caches (__pycache__, build)
                if any(c in norm_path.split(os.sep) for c in self.DISPOSABLE_CACHE_NAMES) or ext in self.DISPOSABLE_EXTENSIONS:
                    return "A", "Low Risk", "Disposable cache or log in active project", "quarantine"
                return "D", "Protected", "File in active user project", "keep"

        # Check Category D: System / AppData / Protected Directories
        path_parts = norm_path.split(os.sep)
        # Note: AppData\Local\Temp is a temporary disposable location, not protected AppData
        is_temp_dir = "temp" in path_parts or "tmp" in path_parts
        for part in path_parts:
            if part in self.PROTECTED_DIR_NAMES:
                if part == "appdata" and is_temp_dir:
                    continue  # Allow Temp directories inside AppData\Local\Temp
                return "D", "Protected", f"Inside protected system or application directory: {part}", "keep"

        # Check Category A: Low Risk Caches & Logs
        if any(part in self.DISPOSABLE_CACHE_NAMES for part in path_parts) or ext in self.DISPOSABLE_EXTENSIONS:
            return "A", "Low Risk", "Reproducible temporary cache or log file", "quarantine"

        if any(part in self.DISPOSABLE_CACHE_DIRS for part in path_parts):
            return "A", "Low Risk", "Generated project build output / package directory", "quarantine"

        # Check Category B: Review Recommended (Installers, Archives, Downloads)
        if ext in self.INSTALLER_EXTENSIONS or (ext == ".exe" and any(k in fname for k in ["setup", "install", "installer"])):
            return "B", "Review Recommended", f"Old software installer ({ext})", "review"

        if ext in self.ARCHIVE_EXTENSIONS:
            if age_days > 30:
                return "B", "Review Recommended", f"Old archive file (>30 days old)", "review"
            return "B", "Review Recommended", f"Archive file ({ext})", "review"

        # Check Category C: Personal or Potentially Important
        if ext in self.PERSONAL_EXTENSIONS:
            return "C", "Personal / Important", f"Personal photograph, document, video, or creative asset ({ext})", "keep"

        # Source code files outside recognized caches
        if ext in {".py", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".java", ".json", ".rs", ".go"}:
            return "C", "Personal / Important", f"Software source code or data file ({ext})", "keep"

        # Default fallback: Low confidence / ambiguous
        return "B", "Review Recommended", "Unclassified file format", "review"

    def audit_storage(
        self,
        target_dir: Optional[str] = None,
        max_depth: int = 4,
        check_duplicates: bool = True
    ) -> StorageAuditReport:
        """Perform a comprehensive storage audit of the target directory or user profile."""
        root = target_dir or os.path.expanduser("~")
        root = os.path.normpath(os.path.abspath(root))

        # Query disk space via psutil
        total_gb, free_gb, used_pct = 0.0, 0.0, 0.0
        if psutil:
            try:
                du = psutil.disk_usage(root)
                total_gb = round(du.total / (1024 ** 3), 2)
                free_gb = round(du.free / (1024 ** 3), 2)
                used_pct = round(du.percent, 1)
            except Exception:
                pass

        total_files = 0
        total_bytes = 0
        potential_savings_bytes = 0

        all_items: List[StorageItem] = []
        dir_sizes: Dict[str, int] = {}
        hash_map: Dict[str, List[StorageItem]] = {}

        root_depth = root.count(os.sep)

        for dirpath, dirnames, filenames in os.walk(root):
            cur_depth = dirpath.count(os.sep) - root_depth
            if cur_depth > max_depth:
                dirnames.clear()
                continue

            # Don't recurse into protected system folders relative to root
            rel_parts = [p.lower() for p in os.path.relpath(dirpath, root).split(os.sep) if p and p != "."]
            if any(p in self.PROTECTED_DIR_NAMES for p in rel_parts):
                dirnames.clear()
                continue

            current_dir_bytes = 0

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    stat = os.stat(fpath)
                    fsize = stat.st_size
                    fmtime = stat.st_mtime
                except (OSError, PermissionError):
                    continue

                total_files += 1
                total_bytes += fsize
                current_dir_bytes += fsize

                cat, risk, reason, action = self.classify_file(fpath, fsize, fmtime)

                item = StorageItem(
                    path=fpath,
                    name=fname,
                    size_bytes=fsize,
                    size_mb=round(fsize / (1024 * 1024), 2),
                    mtime=fmtime,
                    extension=os.path.splitext(fname)[1].lower(),
                    category=cat,
                    risk_label=risk,
                    reason=reason,
                    suggested_action=action
                )

                if cat == "A":
                    potential_savings_bytes += fsize

                all_items.append(item)

                # For duplicate checking (only for files > 64KB)
                if check_duplicates and fsize > 65536 and cat in ("A", "B", "C"):
                    h = self.compute_sha256(fpath, max_bytes=2 * 1024 * 1024)
                    if h:
                        item.content_hash = h
                        hash_map.setdefault(h, []).append(item)

            dir_sizes[dirpath] = current_dir_bytes

        # Aggregate largest files
        sorted_files = sorted(all_items, key=lambda x: x.size_bytes, reverse=True)
        largest_files = [x.to_dict() for x in sorted_files[:15]]

        # Aggregate largest directories
        sorted_dirs = sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)
        largest_dirs = [
            {"path": dpath, "size_mb": round(dsize / (1024 * 1024), 2)}
            for dpath, dsize in sorted_dirs[:10]
        ]

        # Category summaries
        cat_summary: Dict[str, Dict[str, Any]] = {
            "A": {"label": "Category A - Low Risk (Caches/Builds)", "count": 0, "size_mb": 0.0},
            "B": {"label": "Category B - Review Recommended (Archives/Installers)", "count": 0, "size_mb": 0.0},
            "C": {"label": "Category C - Personal / Important (Photos/Docs/Saves)", "count": 0, "size_mb": 0.0},
            "D": {"label": "Category D - Protected (System/Projects)", "count": 0, "size_mb": 0.0},
        }

        for item in all_items:
            cat_summary[item.category]["count"] += 1
            cat_summary[item.category]["size_mb"] += item.size_mb

        for c in cat_summary.values():
            c["size_mb"] = round(c["size_mb"], 2)

        # Detect duplicates
        duplicate_groups = []
        for chash, items in hash_map.items():
            if len(items) > 1:
                for dupe in items[1:]:
                    dupe.is_duplicate = True
                duplicate_groups.append({
                    "hash": chash,
                    "count": len(items),
                    "wasted_mb": round((len(items) - 1) * items[0].size_mb, 2),
                    "files": [x.path for x in items]
                })

        candidates_for_cleanup = [
            x.to_dict() for x in all_items if x.suggested_action == "quarantine"
        ]

        return StorageAuditReport(
            scanned_root=root,
            scan_timestamp=time.time(),
            disk_total_gb=total_gb,
            disk_free_gb=free_gb,
            disk_used_percent=used_pct,
            total_scanned_files=total_files,
            total_scanned_bytes=total_bytes,
            potential_savings_bytes=potential_savings_bytes,
            potential_savings_mb=round(potential_savings_bytes / (1024 * 1024), 2),
            largest_files=largest_files,
            largest_directories=largest_dirs,
            category_summary=cat_summary,
            duplicates=duplicate_groups,
            candidates_for_cleanup=candidates_for_cleanup
        )

    def quarantine_candidates(
        self,
        candidates: List[Dict[str, Any]],
        dry_run: bool = True,
        quarantine_root: Optional[str] = None
    ) -> Dict[str, Any]:
        """Safely move cleanup candidates to an isolated quarantine folder with a rollback manifest.
        
        NEVER permanently deletes any file. Never touches Category C or D files.
        """
        target_root = quarantine_root or os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
            "EDITH", "Quarantine", f"session_{int(time.time())}"
        )

        manifest = {
            "session_id": f"quarantine_{int(time.time())}",
            "created_at": time.time(),
            "dry_run": dry_run,
            "quarantine_folder": target_root,
            "items": [],
            "total_bytes_saved": 0,
            "errors": []
        }

        # Filter: ONLY Category A candidates are eligible for automated quarantine
        safe_candidates = [
            c for c in candidates
            if c.get("category") == "A" and c.get("suggested_action") == "quarantine"
        ]

        if dry_run:
            for c in safe_candidates:
                manifest["items"].append({
                    "original_path": c["path"],
                    "quarantine_path": os.path.join(target_root, os.path.basename(c["path"])),
                    "size_bytes": c.get("size_bytes", 0),
                    "reason": c.get("reason", "Low risk disposable cache")
                })
                manifest["total_bytes_saved"] += c.get("size_bytes", 0)
            manifest["status"] = "dry_run_completed"
            return manifest

        # Execution mode
        os.makedirs(target_root, exist_ok=True)
        for c in safe_candidates:
            orig = c["path"]
            if not os.path.exists(orig):
                continue
            try:
                dest_name = f"{int(time.time())}_{os.path.basename(orig)}"
                dest_path = os.path.join(target_root, dest_name)
                shutil.move(orig, dest_path)
                manifest["items"].append({
                    "original_path": orig,
                    "quarantine_path": dest_path,
                    "size_bytes": c.get("size_bytes", 0),
                    "reason": c.get("reason", "")
                })
                manifest["total_bytes_saved"] += c.get("size_bytes", 0)
            except Exception as e:
                manifest["errors"].append({"path": orig, "error": str(e)})

        # Save manifest
        manifest_path = os.path.join(target_root, "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

        manifest["status"] = "quarantine_executed"
        manifest["manifest_path"] = manifest_path
        return manifest

    @staticmethod
    def restore_quarantine(manifest_path: str) -> Dict[str, Any]:
        """Restore all quarantined files from a manifest back to their original paths."""
        if not os.path.isfile(manifest_path):
            return {"success": False, "error": f"Manifest not found: {manifest_path}"}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return {"success": False, "error": f"Failed to read manifest: {e}"}

        restored = []
        errors = []

        for item in manifest.get("items", []):
            qpath = item.get("quarantine_path")
            orig = item.get("original_path")
            if not qpath or not orig or not os.path.exists(qpath):
                continue

            try:
                os.makedirs(os.path.dirname(orig), exist_ok=True)
                shutil.move(qpath, orig)
                restored.append(orig)
            except Exception as e:
                errors.append({"quarantine_path": qpath, "original_path": orig, "error": str(e)})

        return {
            "success": len(errors) == 0,
            "restored_count": len(restored),
            "restored_files": restored,
            "errors": errors
        }
