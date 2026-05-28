from fastmcp import FastMCP

from .._registry import ProjectExistsError, ProjectNotRegisteredError
from . import config
from . import ingester

mcp = FastMCP("docs-search")


@mcp.tool()
def create_project(
    slug: str,
    abs_path: str,
    description: str | None = None,
) -> dict:
    """Register a new document project. Must be called before any ingest/search.

    Identifies the project by both `slug` (logical name) and `abs_path` (canonical
    project root). Re-registering either is rejected so an agent cannot silently
    overwrite an existing project.

    Args:
        slug: Project identifier used in subsequent calls (e.g. 'research-papers').
        abs_path: Absolute filesystem path that anchors this project. Used by
                  resolve_project_from_path so other agents can discover the
                  right project from their working directory.
        description: Optional one-line summary that helps later agents understand
                     what is indexed under this project.
    """
    try:
        return ingester.create_project(slug, abs_path, description)
    except ProjectExistsError as e:
        return {"error": "ProjectExistsError", "message": str(e)}


@mcp.tool()
def list_projects() -> list[dict]:
    """List all registered document projects with their metadata and chunk counts."""
    return ingester.list_projects()


@mcp.tool()
def resolve_project_from_path(path: str) -> dict | None:
    """Find the registered project whose abs_path contains the given path.

    Returns the deepest match (most-specific project) or None. Use this on
    onboarding to determine which project corresponds to the agent's working
    directory before calling ingest/search.
    """
    return ingester.resolve_project_from_path(path)


@mcp.tool()
def update_project_description(slug: str, description: str | None) -> dict:
    """Update or clear a project's description. abs_path is immutable; to
    relocate a project, delete and re-create it."""
    try:
        return ingester.update_project_description(slug, description)
    except ProjectNotRegisteredError as e:
        return {"error": "ProjectNotRegisteredError", "message": str(e)}


@mcp.tool()
def ingest_documents(
    project: str,
    directory: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict:
    """Index documents from a directory into a previously-registered project.
    Performs a full re-index (drops and recreates the collection).

    Args:
        project: Registered project slug. Call create_project first if missing.
        directory: Absolute or relative path to the directory to index. May
                   differ from the project's abs_path (e.g. ingest /shared/papers
                   into a 'research' project anchored at ~/research).
        include_patterns: Glob patterns to include (e.g. ['**/*.md', '**/*.txt']).
                          Empty list means include all files.
        exclude_patterns: Glob patterns to exclude (e.g. ['drafts/**']).
    """
    try:
        return ingester.ingest(
            project=project,
            directory=directory,
            include_patterns=include_patterns or [],
            exclude_patterns=exclude_patterns or [],
        )
    except ProjectNotRegisteredError as e:
        return {"error": "ProjectNotRegisteredError", "message": str(e)}


@mcp.tool()
def search_documents(
    project: str,
    query: str,
    limit: int = config.DEFAULT_LIMIT,
    min_score: float = config.DEFAULT_MIN_SCORE,
    hybrid: bool = config.DEFAULT_HYBRID,
) -> list[dict] | dict:
    """Search indexed documents in a registered project using semantic search.

    Args:
        project: Registered project slug. Errors if not registered.
        query: Natural language search query (multilingual supported via bge-m3).
        limit: Maximum number of results to return.
        min_score: Minimum score threshold. 0.0 disables filtering (recommended default).
                   Note: hybrid RRF scores cap around 0.033; dense cosine scores range 0.0-1.0.
                   Use 0.0 first, inspect score distribution, then tighten if needed.
        hybrid: True = dense + sparse with RRF fusion. False (default) = dense-only cosine search.
                Dense-only is the default because sparse BM42 caused semantic false positives;
                pass hybrid=True for keyword/identifier-heavy queries.
    """
    try:
        return ingester.search(
            project=project, query=query, limit=limit, min_score=min_score, hybrid=hybrid
        )
    except ProjectNotRegisteredError as e:
        return {"error": "ProjectNotRegisteredError", "message": str(e)}


@mcp.tool()
def delete_project(project: str) -> dict:
    """Delete a project: drops its Qdrant collection and removes the registry row.

    Args:
        project: Project slug to delete.
    """
    return ingester.delete_project(project)


def main() -> None:
    # Pre-warm the embedding model so the first tool call has no cold-start delay.
    ingester._get_embedder()
    # Initialize the registry early so first-boot orphan nuke runs immediately
    # rather than on the first project-aware tool call.
    ingester._get_registry()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
