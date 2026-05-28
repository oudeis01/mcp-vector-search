"""Terminal CLI for code-search-mcp: project-isolated codebase search."""

import json
from typing import Annotated

import typer

from .._registry import ProjectExistsError, ProjectNotRegisteredError
from . import store

app = typer.Typer(help="code-search-cli: project-isolated codebase search via project-rag.")
project_app = typer.Typer(help="Manage codebase project registrations.")
app.add_typer(project_app, name="project")


@project_app.command("create")
def project_create(
    slug: Annotated[str, typer.Argument(help="Project slug")],
    abs_path: Annotated[str, typer.Argument(help="Absolute codebase root path")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
) -> None:
    """Register a new codebase project."""
    try:
        result = store.create_project(slug, abs_path, description)
        typer.echo(json.dumps(result, indent=2))
    except ProjectExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@project_app.command("list")
def project_list() -> None:
    """List all registered codebase projects."""
    projects = store.list_projects()
    if not projects:
        typer.echo("No projects registered.")
        return
    for p in projects:
        desc = p.get("description") or ""
        typer.echo(f"  {p['project']:24s}  {p['abs_path']}  {desc}")


@project_app.command("resolve")
def project_resolve(
    path: Annotated[str, typer.Argument(help="Path to resolve")],
) -> None:
    """Find the project whose abs_path contains the given path."""
    p = store.resolve_project_from_path(path)
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
        result = store.update_project_description(slug, description)
        typer.echo(json.dumps(result, indent=2))
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@project_app.command("delete")
def project_delete(
    slug: Annotated[str, typer.Argument(help="Project slug")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Delete a project from the registry."""
    if not yes:
        typer.confirm(f"Remove '{slug}' from the registry?", abort=True)
    result = store.delete_project(slug)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def index(
    project: Annotated[str, typer.Option("--project", "-p", help="Registered project slug")],
    include: Annotated[list[str] | None, typer.Option("--include", "-i")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", "-e")] = None,
    max_file_size: Annotated[int, typer.Option("--max-file-size")] = 1_048_576,
) -> None:
    """Index the registered codebase (path comes from the registry)."""
    try:
        result = store.index_codebase(project, include, exclude, max_file_size)
        typer.echo(json.dumps(result, indent=2) if not isinstance(result, str) else result)
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def query(
    q: Annotated[str, typer.Argument(help="Search query")],
    project: Annotated[str, typer.Option("--project", "-p")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    min_score: Annotated[float, typer.Option("--min-score")] = 0.0,
    hybrid: Annotated[bool, typer.Option("--hybrid/--no-hybrid")] = False,
) -> None:
    """Search a registered codebase."""
    try:
        result = store.query_codebase(project, q, limit, min_score, hybrid)
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if isinstance(result, dict):
        results = result.get("results", [])
        if not results:
            typer.echo("No results found.")
            return
        for i, r in enumerate(results, 1):
            typer.echo(
                f"\n[{i}] score={r['score']:.4f}  {r['file_path']}:{r['start_line']}-{r['end_line']}  ({r.get('language', '?')})"
            )
            content = r.get("content", "")
            typer.echo(content[:300] + ("..." if len(content) > 300 else ""))
    else:
        typer.echo(result)


@app.command()
def stats(
    project: Annotated[str, typer.Option("--project", "-p")],
) -> None:
    """Show project-rag statistics (database-wide; see per_project flag in response)."""
    try:
        result = store.get_statistics(project)
        typer.echo(json.dumps(result, indent=2) if not isinstance(result, str) else result)
    except ProjectNotRegisteredError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
