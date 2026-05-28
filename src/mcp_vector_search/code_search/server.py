"""code-search-mcp: project-isolated wrapper around project-rag.

Forces every search/index/navigation tool to specify a registered project. The
underlying project-rag binary still operates on a single global LanceDB table
filtered by a `project` column, but this wrapper guarantees the caller cannot
omit or spoof the filter.
"""

from fastmcp import FastMCP

from .._registry import ProjectExistsError, ProjectNotRegisteredError
from . import store

mcp = FastMCP("code-search")


def _wrap(call, *args, **kwargs):
    """Translate registry errors into a structured response the agent can read."""
    try:
        return call(*args, **kwargs)
    except ProjectNotRegisteredError as e:
        return {"error": "ProjectNotRegisteredError", "message": str(e)}
    except ProjectExistsError as e:
        return {"error": "ProjectExistsError", "message": str(e)}


@mcp.tool()
def create_project(
    slug: str,
    abs_path: str,
    description: str | None = None,
) -> dict:
    """Register a new codebase project. Required before any index/query call.

    Identifies the project by both `slug` (logical name) and `abs_path` (the
    canonical codebase root). Both must be unique. Re-registering either is
    rejected to prevent two agents from silently sharing a project.

    Args:
        slug: Project identifier used in subsequent calls (e.g. 'mcp-vector-search').
        abs_path: Absolute filesystem path to the codebase root.
        description: Optional one-line summary that helps later agents understand
                     what this project is.
    """
    return _wrap(store.create_project, slug, abs_path, description)


@mcp.tool()
def list_projects() -> list[dict]:
    """List all registered code projects with their metadata."""
    return store.list_projects()


@mcp.tool()
def resolve_project_from_path(path: str) -> dict | None:
    """Find the registered project whose abs_path contains the given path.

    Returns the deepest match or None. Call this when onboarding into a new
    working directory to discover whether the codebase is already indexed.
    """
    return store.resolve_project_from_path(path)


@mcp.tool()
def update_project_description(slug: str, description: str | None) -> dict:
    """Update or clear a project's description. abs_path is immutable; to
    relocate a project, delete and re-create it."""
    return _wrap(store.update_project_description, slug, description)


@mcp.tool()
def index_codebase(
    project: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_file_size: int = 1_048_576,
) -> dict | list | str:
    """Index a registered codebase. The `path` is resolved from the registry,
    so the agent cannot index data outside the project's declared root.

    Args:
        project: Registered project slug.
        include_patterns: Glob patterns to include.
        exclude_patterns: Glob patterns to exclude (e.g. ['**/node_modules/**']).
        max_file_size: Skip files larger than this (default 1 MiB).
    """
    return _wrap(
        store.index_codebase, project, include_patterns, exclude_patterns, max_file_size
    )


@mcp.tool()
def query_codebase(
    project: str,
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    hybrid: bool = False,
) -> dict | list | str:
    """Semantic search within a registered codebase.

    The wrapper forces both `project` and `path` filters on the underlying
    call, so results never leak across projects.

    Args:
        project: Registered project slug.
        query: Natural-language or code query.
        limit: Maximum results.
        min_score: Score threshold. 0.0 disables filtering. Dense cosine ranges
                   0.0-1.0; hybrid RRF scores are rank-based.
        hybrid: True = vector + BM25 fusion. False = dense-only.
    """
    return _wrap(store.query_codebase, project, query, limit, min_score, hybrid)


@mcp.tool()
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
    """Advanced search with file extension / language / path filters."""
    return _wrap(
        store.search_by_filters,
        project,
        query,
        file_extensions,
        languages,
        path_patterns,
        limit,
        min_score,
        hybrid,
    )


@mcp.tool()
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
    """Semantic search across the project's git commit history."""
    return _wrap(
        store.search_git_history,
        project,
        query,
        branch,
        max_commits,
        limit,
        min_score,
        author,
        since,
        until,
        file_pattern,
    )


@mcp.tool()
def find_definition(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    """Find the definition of the symbol at the given location."""
    return _wrap(store.find_definition, project, file_path, line, column)


@mcp.tool()
def find_references(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    """Find references to the symbol at the given location."""
    return _wrap(store.find_references, project, file_path, line, column)


@mcp.tool()
def get_call_graph(
    project: str, file_path: str, line: int, column: int
) -> dict | list | str:
    """Get the call graph rooted at the function at the given location."""
    return _wrap(store.get_call_graph, project, file_path, line, column)


@mcp.tool()
def get_statistics(project: str) -> dict | list | str:
    """Return project-rag database statistics.

    Note: project-rag's statistics are database-wide, not per-project. The
    response is tagged with `per_project=False`.
    """
    return _wrap(store.get_statistics, project)


@mcp.tool()
def delete_project(project: str) -> dict:
    """Remove a project from the registry.

    Note: project-rag does not expose per-project clear, so vectors remain in
    LanceDB until `clear_all_indexes` is run. After this call the project
    becomes inaccessible through the wrapper.
    """
    return store.delete_project(project)


@mcp.tool()
def clear_all_indexes(confirm: bool = False) -> dict | list | str:
    """Wipe ALL indexed data from project-rag (every project). Requires confirm=True."""
    return store.clear_all_indexes(confirm)


def main() -> None:
    # Pre-warm registry so the orphan nuke runs at startup, not on first call.
    store._get_registry()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
