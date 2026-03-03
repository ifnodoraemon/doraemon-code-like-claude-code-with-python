"""
Spec Manager - Specification-driven development workflow.

Manages the lifecycle of spec sessions through phases:
  DRAFT → REVIEW → EXECUTE → COMPLETE

Generates and tracks 3 structured documents:
  spec.md     - System design specification
  tasks.md    - Task breakdown (- [ ] T1: Title)
  checklist.md - Verification checklist (- [ ] C1: Check)
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SpecPhase(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    EXECUTE = "execute"
    COMPLETE = "complete"


@dataclass
class SpecTask:
    id: str          # e.g. "T1"
    title: str
    status: str = "pending"  # pending | in_progress | done | skipped


@dataclass
class SpecCheckItem:
    id: str          # e.g. "C1"
    description: str
    checked: bool = False


@dataclass
class SpecSession:
    id: str
    name: str
    description: str
    phase: SpecPhase
    spec_dir: str        # stored as string for JSON serialization
    created_at: float
    updated_at: float
    tasks: list[SpecTask] = field(default_factory=list)
    checklist: list[SpecCheckItem] = field(default_factory=list)
    revision_count: int = 0

    @property
    def spec_path(self) -> Path:
        return Path(self.spec_dir)


# ── Spec document filenames ──────────────────────────────────
SPEC_DOCS = ("spec.md", "tasks.md", "checklist.md")


class SpecManager:
    """
    Manages spec session lifecycle and document persistence.

    Storage layout:
        .doraemon/specs/<name>/
            meta.json
            spec.md
            tasks.md
            checklist.md
    """

    SPECS_ROOT = ".doraemon/specs"

    def __init__(self, root: str | Path | None = None):
        self._root = Path(root) if root else Path.cwd() / self.SPECS_ROOT
        self._session: SpecSession | None = None

    # ── Properties ────────────────────────────────────────────

    @property
    def session(self) -> SpecSession | None:
        return self._session

    @property
    def phase(self) -> SpecPhase | None:
        return self._session.phase if self._session else None

    @property
    def is_active(self) -> bool:
        return self._session is not None and self._session.phase != SpecPhase.COMPLETE

    # ── Session lifecycle ─────────────────────────────────────

    def create_spec(self, description: str) -> SpecSession:
        """Create a new spec session from a user description."""
        spec_id = os.urandom(4).hex()
        name = self._slugify(description)
        now = time.time()

        spec_dir = self._root / name
        spec_dir.mkdir(parents=True, exist_ok=True)

        self._session = SpecSession(
            id=spec_id,
            name=name,
            description=description,
            phase=SpecPhase.DRAFT,
            spec_dir=str(spec_dir),
            created_at=now,
            updated_at=now,
        )
        self._save_meta()
        logger.info(f"Created spec session: {spec_id} ({name})")
        return self._session

    def advance_phase(self, to_phase: SpecPhase) -> None:
        """Advance the session to a new phase."""
        if not self._session:
            raise RuntimeError("No active spec session")
        self._session.phase = to_phase
        self._session.updated_at = time.time()
        self._save_meta()
        logger.info(f"Spec {self._session.id} → {to_phase.value}")

    def abort(self) -> None:
        """Abort the current spec session."""
        if self._session:
            logger.info(f"Spec {self._session.id} aborted")
            self._session = None

    # ── Document I/O ──────────────────────────────────────────

    def write_spec_doc(self, filename: str, content: str) -> Path:
        """Write a spec document and auto-parse if tasks.md or checklist.md."""
        if not self._session:
            raise RuntimeError("No active spec session")
        if filename not in SPEC_DOCS:
            raise ValueError(f"Invalid spec doc: {filename}. Must be one of {SPEC_DOCS}")

        path = self._session.spec_path / filename
        path.write_text(content, encoding="utf-8")
        self._session.updated_at = time.time()

        # Auto-parse structured documents
        if filename == "tasks.md":
            self._session.tasks = self._parse_tasks(content)
        elif filename == "checklist.md":
            self._session.checklist = self._parse_checklist(content)

        self._save_meta()
        return path

    def read_spec_doc(self, filename: str) -> str | None:
        """Read a spec document, returning None if not found."""
        if not self._session:
            return None
        path = self._session.spec_path / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_all_spec_content(self) -> str:
        """Concatenate all 3 spec documents for context injection."""
        parts = []
        for doc in SPEC_DOCS:
            content = self.read_spec_doc(doc)
            if content:
                parts.append(f"## {doc}\n\n{content}")
        return "\n\n---\n\n".join(parts) if parts else ""

    # ── Task & checklist updates ──────────────────────────────

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update a task's status and sync tasks.md on disk."""
        if not self._session:
            return False
        valid_statuses = ("pending", "in_progress", "done", "skipped")
        if status not in valid_statuses:
            return False

        for task in self._session.tasks:
            if task.id == task_id:
                task.status = status
                self._session.updated_at = time.time()
                self._sync_tasks_md()
                self._save_meta()
                return True
        return False

    def check_item(self, item_id: str, checked: bool = True) -> bool:
        """Mark a checklist item as checked/unchecked and sync checklist.md."""
        if not self._session:
            return False

        for item in self._session.checklist:
            if item.id == item_id:
                item.checked = checked
                self._session.updated_at = time.time()
                self._sync_checklist_md()
                self._save_meta()
                return True
        return False

    # ── Progress tracking ─────────────────────────────────────

    def get_progress(self) -> dict:
        """Return progress summary."""
        if not self._session:
            return {"tasks_done": 0, "tasks_total": 0,
                    "checks_done": 0, "checks_total": 0, "percent": 0}

        tasks_done = sum(1 for t in self._session.tasks if t.status == "done")
        tasks_total = len(self._session.tasks)
        checks_done = sum(1 for c in self._session.checklist if c.checked)
        checks_total = len(self._session.checklist)

        total = tasks_total + checks_total
        done = tasks_done + checks_done
        percent = int(done / total * 100) if total > 0 else 0

        return {
            "tasks_done": tasks_done,
            "tasks_total": tasks_total,
            "checks_done": checks_done,
            "checks_total": checks_total,
            "percent": percent,
        }

    def check_draft_complete(self) -> bool:
        """Check if all 3 spec documents exist; if so, transition to REVIEW."""
        if not self._session or self._session.phase != SpecPhase.DRAFT:
            return False

        all_exist = all(
            (self._session.spec_path / doc).exists() for doc in SPEC_DOCS
        )
        if all_exist:
            self.advance_phase(SpecPhase.REVIEW)
            return True
        return False

    # ── List / resume ─────────────────────────────────────────

    def list_specs(self) -> list[dict]:
        """List all spec sessions found on disk."""
        specs = []
        if not self._root.exists():
            return specs

        for meta_path in sorted(self._root.glob("*/meta.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                specs.append({
                    "id": data["id"],
                    "name": data["name"],
                    "description": data["description"],
                    "phase": data["phase"],
                    "updated_at": data["updated_at"],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return specs

    def resume_spec(self, name_or_id: str) -> SpecSession | None:
        """Resume a previously saved spec session by name or id."""
        if not self._root.exists():
            return None

        for meta_path in self._root.glob("*/meta.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                if data.get("id") == name_or_id or data.get("name") == name_or_id:
                    self._session = self._load_session(data)
                    logger.info(f"Resumed spec: {self._session.id} ({self._session.name})")
                    return self._session
            except (json.JSONDecodeError, KeyError):
                continue
        return None

    # ── Internal: parsing ─────────────────────────────────────

    def _parse_tasks(self, content: str) -> list[SpecTask]:
        """Parse tasks from markdown content. Format: - [ ] T1: Title"""
        tasks = []
        # Match: - [ ] T1: title  or  - [x] T1: title
        pattern = re.compile(r"^-\s+\[([ xX])\]\s+(T\d+):\s*(.+)$", re.MULTILINE)
        for match in pattern.finditer(content):
            check_char, task_id, title = match.groups()
            status = "done" if check_char.lower() == "x" else "pending"
            tasks.append(SpecTask(id=task_id, title=title.strip(), status=status))
        return tasks

    def _parse_checklist(self, content: str) -> list[SpecCheckItem]:
        """Parse checklist from markdown content. Format: - [ ] C1: Description"""
        items = []
        pattern = re.compile(r"^-\s+\[([ xX])\]\s+(C\d+):\s*(.+)$", re.MULTILINE)
        for match in pattern.finditer(content):
            check_char, item_id, desc = match.groups()
            checked = check_char.lower() == "x"
            items.append(SpecCheckItem(id=item_id, description=desc.strip(), checked=checked))
        return items

    # ── Internal: sync markdown on disk ───────────────────────

    def _sync_tasks_md(self) -> None:
        """Rewrite tasks.md to reflect current task statuses."""
        if not self._session:
            return
        path = self._session.spec_path / "tasks.md"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")

        for task in self._session.tasks:
            check = "x" if task.status == "done" else " "
            # Replace the checkbox for this task ID
            content = re.sub(
                rf"^(-\s+\[)[ xX](\]\s+{re.escape(task.id)}:)",
                rf"\g<1>{check}\g<2>",
                content,
                flags=re.MULTILINE,
            )
        path.write_text(content, encoding="utf-8")

    def _sync_checklist_md(self) -> None:
        """Rewrite checklist.md to reflect current check states."""
        if not self._session:
            return
        path = self._session.spec_path / "checklist.md"
        if not path.exists():
            return
        content = path.read_text(encoding="utf-8")

        for item in self._session.checklist:
            check = "x" if item.checked else " "
            content = re.sub(
                rf"^(-\s+\[)[ xX](\]\s+{re.escape(item.id)}:)",
                rf"\g<1>{check}\g<2>",
                content,
                flags=re.MULTILINE,
            )
        path.write_text(content, encoding="utf-8")

    # ── Internal: persistence ─────────────────────────────────

    def _save_meta(self) -> None:
        """Persist session state to meta.json."""
        if not self._session:
            return
        path = self._session.spec_path / "meta.json"
        data = {
            "id": self._session.id,
            "name": self._session.name,
            "description": self._session.description,
            "phase": self._session.phase.value,
            "spec_dir": self._session.spec_dir,
            "created_at": self._session.created_at,
            "updated_at": self._session.updated_at,
            "tasks": [asdict(t) for t in self._session.tasks],
            "checklist": [asdict(c) for c in self._session.checklist],
            "revision_count": self._session.revision_count,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_session(self, data: dict) -> SpecSession:
        """Reconstruct a SpecSession from meta.json data."""
        return SpecSession(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            phase=SpecPhase(data["phase"]),
            spec_dir=data["spec_dir"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            tasks=[SpecTask(**t) for t in data.get("tasks", [])],
            checklist=[SpecCheckItem(**c) for c in data.get("checklist", [])],
            revision_count=data.get("revision_count", 0),
        )

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filesystem-safe slug."""
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug[:60] or "unnamed-spec"
