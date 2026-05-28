"""CLI wrapper for project-rag MCP server.

Communicates with the project-rag binary via MCP JSON-RPC over stdio.

This is a raw passthrough: it does NOT enforce the project registry. For
project-isolated access (the recommended path for agents) use code-search-cli /
code-search-mcp instead.
"""

import asyncio
import json
from typing import Annotated

import typer

from .._rag_runtime import BinaryNotFoundError, call_tool, check_binary

app = typer.Typer(help="project-rag-cli: raw passthrough to project-rag (no project enforcement).")


def _check_binary() -> None:
    try:
        check_binary()
    except BinaryNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


async def _call_tool(tool: str, args: dict) -> str:
    return await call_tool(tool, args)


@app.command()
def index(
    directory: Annotated[str, typer.Option("--dir", "-d", help="Directory to index")],
    project: Annotated[str | None, typer.Option("--project", "-p", help="Project slug")] = None,
    include: Annotated[list[str] | None, typer.Option("--include", "-i", help="Include glob patterns")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", "-e", help="Exclude glob patterns")] = None,
    max_file_size: Annotated[int, typer.Option("--max-file-size", help="Max file size in bytes")] = 1_048_576,
) -> None:
    """Index a codebase directory using project-rag."""
    _check_binary()
    args: dict = {"path": directory, "max_file_size": max_file_size}
    if project:
        args["project"] = project
    if include:
        args["include_patterns"] = include
    if exclude:
        args["exclude_patterns"] = exclude

    typer.echo(f"Indexing '{directory}'" + (f" as project '{project}'" if project else "") + "...")
    output = asyncio.run(_call_tool("index_codebase", args))
    try:
        parsed = json.loads(output)
        typer.echo(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        typer.echo(output)


@app.command()
def query(
    q: Annotated[str, typer.Argument(help="Search query")],
    project: Annotated[str | None, typer.Option("--project", "-p", help="Project slug filter")] = None,
    directory: Annotated[str | None, typer.Option("--dir", "-d", help="Path filter")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    # Dense cosine scores cap around 0.5 and hybrid RRF scores around 0.03, so the old 0.7
    # default returned nothing. 0.0 disables filtering; let the caller inspect and tighten.
    min_score: Annotated[float, typer.Option("--min-score")] = 0.0,
    # Dense-only by default so scores are interpretable cosine similarities; pass --hybrid
    # to add BM25 keyword fusion for identifier/symbol queries.
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid")] = False,
) -> None:
    """Search an indexed codebase using project-rag."""
    _check_binary()
    args: dict = {"query": q, "limit": limit, "min_score": min_score, "hybrid": hybrid}
    if project:
        args["project"] = project
    if directory:
        args["path"] = directory

    output = asyncio.run(_call_tool("query_codebase", args))
    try:
        parsed = json.loads(output)
        results = parsed.get("results", [])
        if not results:
            typer.echo("No results found.")
            return
        for i, r in enumerate(results, 1):
            typer.echo(f"\n[{i}] score={r['score']:.4f}  {r['file_path']}:{r['start_line']}-{r['end_line']}  ({r['language']})")
            typer.echo(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))
    except (json.JSONDecodeError, KeyError):
        typer.echo(output)


@app.command()
def stats(
    project: Annotated[str | None, typer.Option("--project", "-p")] = None,
) -> None:
    """Show index statistics from project-rag."""
    _check_binary()
    output = asyncio.run(_call_tool("get_statistics", {}))
    try:
        typer.echo(json.dumps(json.loads(output), indent=2))
    except json.JSONDecodeError:
        typer.echo(output)
