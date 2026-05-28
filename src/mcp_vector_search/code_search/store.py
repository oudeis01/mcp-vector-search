"""Project-aware glue between the registry and the project-rag binary.

Every public function here forces `project=slug` (and `path=registry.abs_path`
where applicable) onto the underlying call, so the project-rag binary cannot
see or operate on data outside the requested project.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from platformdirs import user_data_dir

from .._rag_runtime import call_tool as _rag_call_tool
from .._registry import (
    Project,
    ProjectExistsError,
    ProjectNotRegisteredError,
    ProjectRegistry,
)

_registry: ProjectRegistry | None = None

# project-rag default LanceDB location on Linux (matches its PlatformPaths
# resolver). Used only by the first-boot orphan nuke.
_LANCEDB_PATH = Path(user_data_dir("project-rag", appauthor=False)) / "lancedb"


def _get_registry() -> ProjectRegistry:
    global _registry
    if _registry is None:
        _registry = ProjectRegistry("code")
        _maybe_nuke_orphans(_registry)
    return _registry


def _maybe_nuke_orphans(registry: ProjectRegistry) -> None:
    """On first boot, drop the entire project-rag LanceDB directory.

    The user opted in to nuking test data when introducing project isolation.
    Guarded by a marker file inside the registry data dir so this only runs
    once per machine.
    """
    marker = registry.first_boot_marker()
    if marker.exists():
        return
    try:
        if _LANCEDB_PATH.exists():
            shutil.rmtree(_LANCEDB_PATH)
            print(
                f"[code-search] nuked orphan LanceDB at {_LANCEDB_PATH}",
                file=sys.stderr,
            )
        marker.write_text(str(int(time.time())))
    except Exception as e:
        print(f"[code-search] orphan nuke failed: {e}", file=sys.stderr)


def _project_to_dict(p: Project) -> dict:
    return {
        "project": p.slug,
        "abs_path": p.abs_path,
        "description": p.description,
        "created_at": p.created_at,
        "last_indexed_at": p.last_indexed_at,
    }


def _parse(payload: str) -> dict | list | str:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


async def _call(tool: str, args: dict) -> dict | list | str:
    raw = await _rag_call_tool(tool, args)
    return _parse(raw)


def create_project(slug: str, abs_path: str, description: str | None = None) -> dict:
    registry = _get_registry()
    p = registry.create(slug, abs_path, description)
    return _project_to_dict(p)


def list_projects() -> list[dict]:
    registry = _get_registry()
    return [_project_to_dict(p) for p in registry.list_all()]


def resolve_project_from_path(path: str) -> dict | None:
    registry = _get_registry()
    p = registry.resolve_from_path(path)
    return None if p is None else _project_to_dict(p)


def update_project_description(slug: str, description: str | None) -> dict:
    registry = _get_registry()
    p = registry.update_description(slug, description)
    return _project_to_dict(p)


def _require(slug: str) -> Project:
    return _get_registry().get(slug)


def index_codebase(
    project: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size: int = 1_048_576,
) -> dict | list | str:
    p = _require(project)
    args: dict = {
        "path": p.abs_path,
        "project": p.slug,
        "max_file_size": max_file_size,
    }
    if include_patterns:
        args["include_patterns"] = include_patterns
    if exclude_patterns:
        args["exclude_patterns"] = exclude_patterns
    result = asyncio.run(_call("index_codebase", args))
    _get_registry().touch_indexed(project)
    return result


def query_codebase(
    project: str,
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    hybrid: bool = False,
) -> dict | list | str:
    p = _require(project)
    args = {
        "query": query,
        "project": p.slug,
        "path": p.abs_path,
        "limit": limit,
        "min_score": min_score,
        "hybrid": hybrid,
    }
    return asyncio.run(_call("query_codebase", args))


def search_by_filters(
    project: str,
    query: str,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    path_patterns: list[str] | None = None,
    limit: int = 10,
    min_score: float = 0.0,
    hybrid: bool = False,
) -> dict | list | str:
    p = _require(project)
    args: dict = {
        "query": query,
        "project": p.slug,
        "path": p.abs_path,
        "limit": limit,
        "min_score": min_score,
        "hybrid": hybrid,
    }
    if file_extensions:
        args["file_extensions"] = file_extensions
    if languages:
        args["languages"] = languages
    if path_patterns:
        args["path_patterns"] = path_patterns
    return asyncio.run(_call("search_by_filters", args))


def search_git_history(
    project: str,
    query: str,
    branch: str | None = None,
    max_commits: int = 10,
    limit: int = 10,
    min_score: float = 0.0,
    author: str | None = None,
    since: str | None = None,
    until: str | None = None,
    file_pattern: str | None = None,
) -> dict | list | str:
    p = _require(project)
    args: dict = {
        "query": query,
        "project": p.slug,
        "path": p.abs_path,
        "max_commits": max_commits,
        "limit": limit,
        "min_score": min_score,
    }
    if branch:
        args["branch"] = branch
    if author:
        args["author"] = author
    if since:
        args["since"] = since
    if until:
        args["until"] = until
    if file_pattern:
        args["file_pattern"] = file_pattern
    return asyncio.run(_call("search_git_history", args))


def find_definition(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    p = _require(project)
    return asyncio.run(
        _call(
            "find_definition",
            {
                "project": p.slug,
                "file_path": file_path,
                "line": line,
                "column": column,
            },
        )
    )


def find_references(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    p = _require(project)
    return asyncio.run(
        _call(
            "find_references",
            {
                "project": p.slug,
                "file_path": file_path,
                "line": line,
                "column": column,
            },
        )
    )


def get_call_graph(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    p = _require(project)
    return asyncio.run(
        _call(
            "get_call_graph",
            {
                "project": p.slug,
                "file_path": file_path,
                "line": line,
                "column": column,
            },
        )
    )


def get_statistics(project: str) -> dict:
    """Return project-rag statistics.

    project-rag's get_statistics is database-wide, not per-project, so we tag
    the response with per_project=False and pass the requested slug through
    for context. (Tracking per-project counts requires a future enhancement
    upstream or a wrapper-side count.)
    """
    _require(project)
    raw = asyncio.run(_call("get_statistics", {}))
    if isinstance(raw, dict):
        raw["per_project"] = False
        raw["requested_project"] = project
        return raw
    return {"per_project": False, "requested_project": project, "raw": raw}


def delete_project(project: str) -> dict:
    """Remove the project from the registry.

    project-rag does not expose a per-project clear, so vectors tagged with
    this project remain in LanceDB until clear_all_indexes() is called. The
    registry deletion alone is enough to make subsequent project-aware calls
    refuse to touch the orphaned vectors.
    """
    registry = _get_registry()
    try:
        p = registry.delete(project)
    except ProjectNotRegisteredError:
        return {"success": False, "message": f"Project '{project}' is not registered."}
    return {
        "success": True,
        "message": (
            f"Project '{project}' removed from registry. Orphaned vectors in "
            f"LanceDB remain until clear_all_indexes() is called."
        ),
        "abs_path": p.abs_path,
    }


def clear_all_indexes(confirm: bool = False) -> dict | list | str:
    if not confirm:
        return {
            "error": "ConfirmationRequired",
            "message": "Pass confirm=True to wipe ALL indexed data (every project).",
        }
    return asyncio.run(_call("clear_index", {}))


__all__ = [
    "ProjectExistsError",
    "ProjectNotRegisteredError",
    "clear_all_indexes",
    "create_project",
    "delete_project",
    "find_definition",
    "find_references",
    "get_call_graph",
    "get_statistics",
    "index_codebase",
    "list_projects",
    "query_codebase",
    "resolve_project_from_path",
    "search_by_filters",
    "search_git_history",
    "update_project_description",
]
