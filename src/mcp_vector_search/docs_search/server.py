from fastmcp import FastMCP

from . import config
from . import ingester

mcp = FastMCP("docs-search")


@mcp.tool()
def ingest_documents(
    project: str,
    directory: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict:
    """Index documents from a directory into a project collection.
    Performs a full re-index (drops and recreates the collection).

    Args:
        project: Project slug identifier (e.g. 'my-papers').
        directory: Absolute or relative path to the directory to index.
        include_patterns: Glob patterns to include (e.g. ['**/*.md', '**/*.txt']).
                          Empty list means include all files.
        exclude_patterns: Glob patterns to exclude (e.g. ['drafts/**']).
    """
    return ingester.ingest(
        project=project,
        directory=directory,
        include_patterns=include_patterns or [],
        exclude_patterns=exclude_patterns or [],
    )


@mcp.tool()
def search_documents(
    project: str,
    query: str,
    limit: int = config.DEFAULT_LIMIT,
    min_score: float = config.DEFAULT_MIN_SCORE,
    hybrid: bool = config.DEFAULT_HYBRID,
) -> list[dict]:
    """Search indexed documents using semantic search.

    Args:
        project: Project slug to search within.
        query: Natural language search query (multilingual supported via bge-m3).
        limit: Maximum number of results to return.
        min_score: Minimum score threshold. 0.0 disables filtering (recommended default).
                   Note: hybrid RRF scores cap around 0.033; dense cosine scores range 0.0-1.0.
                   Use 0.0 first, inspect score distribution, then tighten if needed.
        hybrid: True = dense + sparse with RRF fusion. False (default) = dense-only cosine search.
                Dense-only is the default because sparse BM42 caused semantic false positives;
                pass hybrid=True for keyword/identifier-heavy queries.
    """
    return ingester.search(
        project=project, query=query, limit=limit, min_score=min_score, hybrid=hybrid
    )


@mcp.tool()
def list_projects() -> list[dict]:
    """List all indexed document projects with their point counts."""
    return ingester.list_projects()


@mcp.tool()
def delete_project(project: str) -> dict:
    """Delete all indexed data for a project.

    Args:
        project: Project slug to delete.
    """
    return ingester.delete_project(project)


def main() -> None:
    # Pre-warm the embedding model so the first tool call has no cold-start delay.
    ingester._get_embedder()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
