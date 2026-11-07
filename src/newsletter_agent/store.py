"""Private storage: caches, checkpoints, and reports, all owner-only."""

import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config


class InventoryStore:
    """Files for one data folder. Normal runs use ~/.newsletter-agent/inventory."""

    def __init__(self, directory: Path | None = None, checkpoint_name: str = "checkpoint.json"):
        self.directory = directory or config.inventory_dir()
        self.reports_directory = self.directory / "reports"
        self.checkpoint_name = checkpoint_name
        for folder in (self.directory, self.reports_directory):
            if folder.is_symlink():
                raise ValueError(f"Refusing to use a symlink for private data: {folder}")
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(folder, 0o700)
        self._lock_file = None
        self.metadata = self._load_json("metadata_cache.json", {})
        self.classifications = self._load_json("classification_cache.json", {})

    def lock(self):
        """Stop two runs (say, a scheduled briefing and a manual inventory) sharing the caches."""
        self._lock_file = open(self.directory / ".lock", "w")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another newsletter-agent run is using this data. Try again when it finishes.") from None

    def unlock(self):
        if self._lock_file:
            self._lock_file.close()
            self._lock_file = None

    def _load_json(self, name: str, default):
        path = self.directory / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"Warning: ignoring unreadable file {name}; it will be rebuilt.", file=sys.stderr)
            return default

    def _save_json(self, name: str, data):
        config.write_private_file(self.directory / name, json.dumps(data, ensure_ascii=False, indent=1))

    def save_metadata(self):
        self._save_json("metadata_cache.json", self.metadata)

    def save_classifications(self):
        self._save_json("classification_cache.json", self.classifications)

    def load_checkpoint(self) -> dict | None:
        return self._load_json(self.checkpoint_name, None)

    def save_checkpoint(self, state_dict: dict):
        self._save_json(self.checkpoint_name, state_dict)

    def save_report(self, report: dict, text: str, html: str) -> list[Path]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        paths = []
        for suffix, content in (("json", json.dumps(report, ensure_ascii=False, indent=1)), ("md", text), ("html", html)):
            path = self.reports_directory / f"inventory-{stamp}.{suffix}"
            config.write_private_file(path, content)
            paths.append(path)
        return paths


def latest_report_path(suffix: str = "json") -> Path | None:
    """The newest saved inventory report, or None if there is none yet."""
    reports = sorted((config.inventory_dir() / "reports").glob(f"inventory-*.{suffix}"))
    return reports[-1] if reports else None


def load_latest_report() -> dict | None:
    path = latest_report_path()
    return json.loads(path.read_text(encoding="utf-8")) if path else None
