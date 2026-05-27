# mcp-vector-search

Two local semantic-search tools that speak the [Model Context Protocol](https://modelcontextprotocol.io):

- **docs-search** — a FastMCP server that indexes documents into [Qdrant](https://qdrant.tech) and serves hybrid (dense + sparse) semantic search. Multilingual via `BAAI/bge-m3`.
- **project-rag-cli** — a thin CLI wrapper that drives the [`project-rag`](https://github.com/oudeis01/project-rag) binary (codebase indexing and semantic code search) over MCP stdio.

The two are independent: `docs-search` is self-contained, and `project-rag-cli` only shells out to the `project-rag` binary by name.

## Requirements

- Python >= 3.12 ([uv](https://github.com/astral-sh/uv) recommended)
- A running Qdrant instance (for `docs-search`) — defaults to `http://localhost:6333`
- The `project-rag` binary on `PATH` (for `project-rag-cli`) — build it from the
  [fork](https://github.com/oudeis01/project-rag)
- Optional: an NVIDIA GPU + CUDA for accelerated embedding

## Install

```bash
uv sync
```

This exposes three entry points:

| Command | Purpose |
|---------|---------|
| `docs-search` | CLI for the document index (ingest / query / list / delete) |
| `docs-search-mcp` | the docs-search MCP server (stdio) |
| `project-rag-cli` | CLI wrapper around the `project-rag` binary |

## docs-search

Start Qdrant, then:

```bash
# Index a folder of documents into a project
docs-search ingest --project my-papers --dir ./papers --include "**/*.md" --include "**/*.txt"

# Query (dense-only by default; pass --hybrid for keyword fusion)
docs-search query --project my-papers "attention mechanism in transformers"
```

Notes:
- Dense embeddings use `BAAI/bge-m3` (1024d, multilingual) on GPU when available.
- Sparse embeddings use BM42. Hybrid fusion is RRF; because RRF scores are rank-based
  (cap ~0.03) while dense cosine scores range 0–1, `min_score` defaults to `0.0`.
- Duplicate chunks (same content across files) are de-duplicated at ingest time.

## project-rag-cli

Requires the `project-rag` binary on `PATH`.

```bash
# Index a codebase
project-rag-cli index --dir ./my-repo --exclude "**/vendor/**" --exclude "**/node_modules/**"

# Search (dense-only by default)
project-rag-cli query "websocket connection handler"
```

### CUDA / GPU notes

The wrapper adds the ONNX Runtime CUDA provider directory and the CUDA libraries to
`LD_LIBRARY_PATH` so the `project-rag` binary can use the GPU. Paths are auto-detected,
with environment overrides:

- `PROJECT_RAG_ORT_LIB_DIR` — directory containing `libonnxruntime_providers_cuda.so`
  (auto-detected under `~/.cache/ort.pyke.io/dfbin/` otherwise)
- `CUDA_LIB_DIR` — CUDA runtime libraries (default `/opt/cuda/targets/x86_64-linux/lib`)

If neither is found the binary still runs on CPU.

## License

MIT — see [LICENSE](LICENSE).
