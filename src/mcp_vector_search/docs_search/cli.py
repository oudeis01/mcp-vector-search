import json
from typing import Annotated

import typer

from .._registry import ProjectExistsError, ProjectNotRegisteredError
from . import config
from . import ingester

app = typer.Typer(help="docs-search: manage and query document index projects.")
project_app = typer.Typer(help="Manage project registrations.")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    slug: Annotated[str, typer.Argument(help="Project slug (identifier)")],
    abs_path: Annotated[str, typer.Argument(help="Absolute project root path")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
) -> None:
    """Register a new project."""
    try:
        result = ingester.create_project(slug, abs_path, description)
        typer.echo(json.dumps(result, indent=2))
    except ProjectExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@project_app.command("list")
def project_list() -> None:
    """List all registered projects."""
    projects = ingester.list_projects()
    if not projects:
        typer.echo("No projects registered.")
        return
    for p in projects:
        desc = p.get("description") or ""
        typer.echo(
            f"  {p['project']:24s}  points={p.get('points_count', 0):>6}  {p['abs_path']}  {desc}"
        )


@project_app.command("resolve")
def project_resolve(
    path: Annotated[str, typer.Argument(help="Path to resolve to a project")],
) -> None:
    """Find the project whose abs_path contains the given path."""
    p = ingester.resolve_project_from_path(path)
    if p is None:
        typer.echo("No matching project.")
        raise typer.Exit(1)
    typer.echo(json.dumps(p, indent=2))


@project_app.command("describe")
def project_describe(
    slug: Annotated[str, typer.Argument(help="Project slug")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
) -> None:
    """Update or clear a project's description."""
    try:
        result = ingester.update_project_description(slug, description)
        typer.echo(json.dumps(result, indent=2))
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def ingest(
    project: Annotated[str, typer.Option("--project", "-p", help="Registered project slug")],
    directory: Annotated[str, typer.Option("--dir", "-d", help="Directory to index")],
    include: Annotated[list[str] | None, typer.Option("--include", "-i", help="Include glob patterns")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", "-e", help="Exclude glob patterns")] = None,
) -> None:
    """Index documents from a directory into a registered project. Performs full re-index."""
    typer.echo(f"Indexing '{directory}' into project '{project}'...")
    try:
        result = ingester.ingest(
            project=project,
            directory=directory,
            include_patterns=include or [],
            exclude_patterns=exclude or [],
        )
        typer.echo(json.dumps(result, indent=2))
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def query(
    project: Annotated[str, typer.Option("--project", "-p", help="Registered project slug")],
    q: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 5,
    min_score: Annotated[float, typer.Option("--min-score")] = config.DEFAULT_MIN_SCORE,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid")] = config.DEFAULT_HYBRID,
) -> None:
    """Search a registered project's indexed documents."""
    try:
        results = ingester.search(
            project=project, query=q, limit=limit, min_score=min_score, hybrid=hybrid
        )
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if not results:
        typer.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        typer.echo(f"\n[{i}] score={r['score']:.4f}  {r['file_path']} (chunk {r['chunk_index']})")
        typer.echo(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))


@app.command(name="list")
def list_cmd() -> None:
    """List all registered projects (alias for `docs-search project list`)."""
    project_list()


@app.command()
def delete(
    project: Annotated[str, typer.Option("--project", "-p", help="Registered project slug")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Delete a project: drops the Qdrant collection and the registry row."""
    if not yes:
        typer.confirm(f"Delete project '{project}' (collection + registry)?", abort=True)
    result = ingester.delete_project(project)
    typer.echo(result["message"])
