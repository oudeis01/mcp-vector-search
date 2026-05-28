"""Shared project registry for docs-search and code-search MCP servers.

Each MCP server gets its own SQLite-backed registry under platformdirs user-data.
Projects are identified by a `slug` (logical name) and tied to an `abs_path`
(canonical project root). Both are UNIQUE: re-registering either is rejected so
agents cannot silently overwrite or duplicate an existing project.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "mcp-vector-search"


class RegistryError(Exception):
    """Base class for registry errors."""


class ProjectExistsError(RegistryError):
    """Raised when create() would collide with an existing slug or abs_path."""


class ProjectNotRegisteredError(RegistryError):
    """Raised when a tool call references a slug that is not in the registry."""


@dataclass(frozen=True)
class Project:
    slug: str
    abs_path: str
    description: str | None
    created_at: int
    last_indexed_at: int | None


def _data_root(tool: str) -> Path:
    """Return the per-tool data directory. Created on first access."""
    base = Path(user_data_dir(APP_NAME, appauthor=False)) / tool
    base.mkdir(parents=True, exist_ok=True)
    return base


class ProjectRegistry:
    """SQLite-backed project registry. One instance per tool ('docs' or 'code')."""

    def __init__(self, tool: str, db_path: Path | None = None) -> None:
        self.tool = tool
        self.data_dir = _data_root(tool) if db_path is None else db_path.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (self.data_dir / "registry.db")
        # check_same_thread=False so fastmcp's worker threads can use the same
        # connection. Tools are invoked serially per session, and SQLite's own
        # file-level locking serializes the rare writes, so a shared connection
        # is safe for this single-process registry.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                slug TEXT PRIMARY KEY,
                abs_path TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at INTEGER NOT NULL,
                last_indexed_at INTEGER
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _normalize_path(path: str) -> str:
        return os.path.realpath(os.path.expanduser(path))

    def first_boot_marker(self, name: str = ".nuked") -> Path:
        return self.data_dir / name

    def create(
        self,
        slug: str,
        abs_path: str,
        description: str | None = None,
    ) -> Project:
        slug = slug.strip()
        if not slug:
            raise RegistryError("slug must be a non-empty string")
        normalized = self._normalize_path(abs_path)
        if not os.path.isabs(normalized):
            raise RegistryError(f"abs_path must be absolute: {abs_path}")
        now = int(time.time())
        try:
            self._conn.execute(
                "INSERT INTO projects (slug, abs_path, description, created_at) VALUES (?, ?, ?, ?)",
                (slug, normalized, description, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            existing = self._lookup_collision(slug, normalized)
            raise ProjectExistsError(
                f"Cannot create project '{slug}' at {normalized}: collides with {existing}"
            ) from e
        return Project(
            slug=slug,
            abs_path=normalized,
            description=description,
            created_at=now,
            last_indexed_at=None,
        )

    def _lookup_collision(self, slug: str, abs_path: str) -> str:
        rows = self._conn.execute(
            "SELECT slug, abs_path FROM projects WHERE slug = ? OR abs_path = ?",
            (slug, abs_path),
        ).fetchall()
        if not rows:
            return "unknown row"
        parts = [f"slug='{r['slug']}' abs_path='{r['abs_path']}'" for r in rows]
        return "; ".join(parts)

    def get(self, slug: str) -> Project:
        row = self._conn.execute(
            "SELECT * FROM projects WHERE slug = ?", (slug,)
        ).fetchone()
        if row is None:
            raise ProjectNotRegisteredError(
                f"Project '{slug}' is not registered. "
                f"Call create_project(slug, abs_path, description?) first."
            )
        return _row_to_project(row)

    def list_all(self) -> list[Project]:
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY slug"
        ).fetchall()
        return [_row_to_project(r) for r in rows]

    def resolve_from_path(self, path: str) -> Project | None:
        """Return the deepest project whose abs_path is a prefix of `path`."""
        target = self._normalize_path(path)
        rows = self._conn.execute("SELECT * FROM projects").fetchall()
        best: Project | None = None
        for r in rows:
            base = r["abs_path"]
            if target == base or target.startswith(base.rstrip("/") + "/"):
                p = _row_to_project(r)
                if best is None or len(p.abs_path) > len(best.abs_path):
                    best = p
        return best

    def update_description(self, slug: str, description: str | None) -> Project:
        self.get(slug)
        self._conn.execute(
            "UPDATE projects SET description = ? WHERE slug = ?",
            (description, slug),
        )
        self._conn.commit()
        return self.get(slug)

    def touch_indexed(self, slug: str) -> None:
        self.get(slug)
        self._conn.execute(
            "UPDATE projects SET last_indexed_at = ? WHERE slug = ?",
            (int(time.time()), slug),
        )
        self._conn.commit()

    def delete(self, slug: str) -> Project:
        project = self.get(slug)
        self._conn.execute("DELETE FROM projects WHERE slug = ?", (slug,))
        self._conn.commit()
        return project

    def close(self) -> None:
        self._conn.close()


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        slug=row["slug"],
        abs_path=row["abs_path"],
        description=row["description"],
        created_at=row["created_at"],
        last_indexed_at=row["last_indexed_at"],
    )
