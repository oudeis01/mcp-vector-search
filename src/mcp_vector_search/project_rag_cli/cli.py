"""CLI wrapper for project-rag MCP server.

Communicates with the project-rag binary via MCP JSON-RPC over stdio.
"""

import asyncio
import glob
import json
import os
import shutil
from typing import Annotated

import typer
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = typer.Typer(help="project-rag-cli: trigger project-rag indexing and search from the terminal.")

_BINARY = "project-rag"

def _find_ort_cuda_dir() -> str | None:
    """Locate the directory holding ort's CUDA execution provider shared library.

    ort-sys downloads the ONNX Runtime into a per-target, per-hash cache directory, so
    the path is not stable across machines or versions. Honor an explicit override, then
    fall back to scanning the cache for the directory that actually contains the .so.
    """
    override = os.environ.get("PROJECT_RAG_ORT_LIB_DIR")
    if override:
        return override if os.path.isdir(override) else None
    base = os.path.expanduser("~/.cache/ort.pyke.io/dfbin")
    matches = glob.glob(os.path.join(base, "*", "*", "libonnxruntime_providers_cuda.so"))
    return os.path.dirname(matches[0]) if matches else None


def _build_env() -> dict:
    # Default RUST_LOG to off, but respect an explicit value if the caller set one.
    env = {**os.environ}
    env.setdefault("RUST_LOG", "off")

    lib_dirs = []
    ort_cuda_dir = _find_ort_cuda_dir()
    if ort_cuda_dir:
        lib_dirs.append(ort_cuda_dir)
    cuda_lib = os.environ.get("CUDA_LIB_DIR", "/opt/cuda/targets/x86_64-linux/lib")
    if os.path.isdir(cuda_lib):
        lib_dirs.append(cuda_lib)

    # If neither is found we leave LD_LIBRARY_PATH untouched; the binary still runs on CPU.
    combined = ":".join(filter(None, [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]))
    if combined:
        env["LD_LIBRARY_PATH"] = combined
    return env

_ENV = _build_env()


def _check_binary() -> None:
    if not shutil.which(_BINARY):
        typer.echo(f"Error: '{_BINARY}' not found in PATH. Build and install it first.", err=True)
        raise typer.Exit(1)


async def _call_tool(tool: str, args: dict) -> str:
    params = StdioServerParameters(command=_BINARY, args=["serve"], env=_ENV)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            # result.content is a list of TextContent / other content blocks
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)


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
