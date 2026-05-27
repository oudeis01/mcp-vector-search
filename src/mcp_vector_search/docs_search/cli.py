import json
from typing import Annotated

import typer

from . import config
from . import ingester

app = typer.Typer(help="docs-search: manage and query document index projects.")


@app.command()
def ingest(
    project: Annotated[str, typer.Option("--project", "-p", help="Project slug")],
    directory: Annotated[str, typer.Option("--dir", "-d", help="Directory to index")],
    include: Annotated[list[str] | None, typer.Option("--include", "-i", help="Include glob patterns")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", "-e", help="Exclude glob patterns")] = None,
) -> None:
    """Index documents from a directory into a project. Performs full re-index."""
    typer.echo(f"Indexing '{directory}' into project '{project}'...")
    result = ingester.ingest(
        project=project,
        directory=directory,
        include_patterns=include or [],
        exclude_patterns=exclude or [],
    )
    typer.echo(json.dumps(result, indent=2))


@app.command()
def query(
    project: Annotated[str, typer.Option("--project", "-p", help="Project slug")],
    q: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 5,
    min_score: Annotated[float, typer.Option("--min-score")] = config.DEFAULT_MIN_SCORE,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid")] = config.DEFAULT_HYBRID,
) -> None:
    """Search a project's indexed documents."""
    results = ingester.search(
        project=project, query=q, limit=limit, min_score=min_score, hybrid=hybrid
    )
    if not results:
        typer.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] score={r['score']:.4f}  {r['file_path']} (chunk {r['chunk_index']})")
        typer.echo(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))


@app.command(name="list")
def list_cmd() -> None:
    """List all indexed projects."""
    projects = ingester.list_projects()
    if not projects:
        typer.echo("No projects indexed.")
        return
    for p in projects:
        typer.echo(f"  {p['project']:30s}  {p['points_count']} points")


@app.command()
def delete(
    project: Annotated[str, typer.Option("--project", "-p", help="Project slug")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Delete all indexed data for a project."""
    if not yes:
        typer.confirm(f"Delete all data for project '{project}'?", abort=True)
    result = ingester.delete_project(project)
    typer.echo(result["message"])
