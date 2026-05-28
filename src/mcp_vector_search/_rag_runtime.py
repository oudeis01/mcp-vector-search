"""Shared runtime helpers for spawning the `project-rag` binary as an MCP server.

Both `project_rag_cli` (raw passthrough) and `code_search` (project-isolated
wrapper) shell out to the same Rust binary; this module centralizes binary
discovery, LD_LIBRARY_PATH composition for CUDA, and the stdio MCP client call.
"""

from __future__ import annotations

import glob
import os
import shutil

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BINARY = "project-rag"


def find_ort_cuda_dir() -> str | None:
    """Locate the directory holding ort's CUDA execution provider shared library.

    ort-sys downloads the ONNX Runtime into a per-target, per-hash cache directory,
    so the path is not stable across machines or versions. Honor an explicit
    override, then fall back to scanning the cache for the directory that
    actually contains the .so.
    """
    override = os.environ.get("PROJECT_RAG_ORT_LIB_DIR")
    if override:
        return override if os.path.isdir(override) else None
    base = os.path.expanduser("~/.cache/ort.pyke.io/dfbin")
    matches = glob.glob(os.path.join(base, "*", "*", "libonnxruntime_providers_cuda.so"))
    return os.path.dirname(matches[0]) if matches else None


def build_env() -> dict:
    """Build the env for spawning project-rag: silences logs, injects CUDA libs."""
    env = {**os.environ}
    env.setdefault("RUST_LOG", "off")

    lib_dirs = []
    ort_cuda_dir = find_ort_cuda_dir()
    if ort_cuda_dir:
        lib_dirs.append(ort_cuda_dir)
    cuda_lib = os.environ.get("CUDA_LIB_DIR", "/opt/cuda/targets/x86_64-linux/lib")
    if os.path.isdir(cuda_lib):
        lib_dirs.append(cuda_lib)

    combined = ":".join(filter(None, [*lib_dirs, env.get("LD_LIBRARY_PATH", "")]))
    if combined:
        env["LD_LIBRARY_PATH"] = combined
    return env


class BinaryNotFoundError(RuntimeError):
    pass


def check_binary() -> None:
    if not shutil.which(BINARY):
        raise BinaryNotFoundError(
            f"'{BINARY}' not found in PATH. Build and install it first."
        )


async def call_tool(tool: str, args: dict, env: dict | None = None) -> str:
    """Spawn project-rag, invoke a single MCP tool, return the joined text content.

    A new process is spawned per call. The binary's startup cost is non-trivial
    (model load + CUDA init) but acceptable for the wrapper's use case; if it
    becomes a bottleneck we can switch to a long-lived session.
    """
    check_binary()
    params = StdioServerParameters(command=BINARY, args=["serve"], env=env or build_env())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
