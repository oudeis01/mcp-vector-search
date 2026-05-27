# mcp-vector-search

Two local semantic-search tools that speak the [Model Context Protocol](https://modelcontextprotocol.io):

- **docs-search** — a FastMCP server that indexes documents into [Qdrant](https://qdrant.tech) and serves hybrid (dense + sparse) semantic search. Multilingual via `BAAI/bge-m3`.
- **project-rag** — a Rust binary that indexes codebases and serves semantic code search. This repo ships a CLI wrapper (`project-rag-cli`) for terminal use; for MCP, connect to the binary directly.

The two are independent: `docs-search` is self-contained, and `project-rag-cli` only shells out to the `project-rag` binary by name.

## Requirements

- Python >= 3.12 ([uv](https://github.com/astral-sh/uv) recommended)
- A running Qdrant instance (for `docs-search`) — defaults to `http://localhost:6333`
- The `project-rag` binary on `PATH` (for code search) — build it from the [fork](https://github.com/oudeis01/project-rag)
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

---

## MCP Setup

Both tools expose MCP servers over stdio. Configure them in your agent of choice.

### Claude Code

Add to `.claude/settings.json` (project) or `~/.claude/settings.json` (global):

```json
{
  "mcpServers": {
    "docs-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-vector-search", "docs-search-mcp"]
    },
    "project-rag": {
      "command": "project-rag",
      "args": ["serve"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global):

```json
{
  "mcpServers": {
    "docs-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-vector-search", "docs-search-mcp"]
    },
    "project-rag": {
      "command": "project-rag",
      "args": ["serve"]
    }
  }
}
```

### opencode

Add to `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "docs-search": {
      "type": "local",
      "command": ["uv", "run", "--project", "/path/to/mcp-vector-search", "docs-search-mcp"],
      "enabled": true
    },
    "project-rag": {
      "type": "local",
      "command": ["project-rag", "serve"],
      "enabled": true
    }
  }
}
```

> **Note:** Replace `/path/to/mcp-vector-search` with the absolute path to this repo. The `project-rag` binary must be on `PATH`.

---

## docs-search

Embedding model: `BAAI/bge-m3` (1024d, multilingual). Dense search uses cosine similarity; hybrid adds BM42 sparse with RRF fusion.

### CLI

```bash
# Index a folder of documents into a project
docs-search ingest --project my-papers --dir ./papers --include "**/*.md" --include "**/*.txt"

# Query (dense-only by default)
docs-search query --project my-papers "attention mechanism in transformers"

# Hybrid search (keyword + semantic fusion)
docs-search query --project my-papers --hybrid "attention mechanism in transformers"

# List all indexed projects
docs-search list

# Delete a project's index
docs-search delete --project my-papers
```

### MCP tools

| Tool | Parameters |
|------|-----------|
| `ingest_documents` | `project` (str), `directory` (str), `include_patterns` (list[str], optional), `exclude_patterns` (list[str], optional) |
| `search_documents` | `project` (str), `query` (str), `limit` (int, default 10), `min_score` (float, default 0.0), `hybrid` (bool, default false) |
| `list_projects` | — |
| `delete_project` | `project` (str) |

### Options reference

| Option | Default | Notes |
|--------|---------|-------|
| `limit` | `10` | Max results returned |
| `min_score` | `0.0` | Score threshold. 0.0 disables filtering. See score guide below. |
| `hybrid` | `false` | `true` = dense + BM42 sparse with RRF fusion. Use for keyword/identifier queries. |
| `include_patterns` | all files | Glob patterns, e.g. `["**/*.md", "**/*.txt"]` |
| `exclude_patterns` | none | Glob patterns, e.g. `["drafts/**"]` |

Chunks are SHA256-deduplicated at ingest time; identical content across multiple files is indexed only once.

### Score guide

Dense cosine scores range `0.0–1.0`. Hybrid RRF scores are rank-based and cap around `0.033` regardless of semantic similarity. Start with `min_score=0.0`, inspect the distribution, then tighten if needed. Do not mix thresholds across modes.

---

## project-rag-cli

Embedding model: `jinaai/jina-embeddings-v2-base-code` (768d, code-specific). Requires the `project-rag` binary on `PATH`.

The `project-rag-cli` commands are for terminal use. For AI agents, connect directly to `project-rag serve` as an MCP server (see [MCP Setup](#mcp-setup)).

### CLI

```bash
# Index a codebase
project-rag-cli index --dir ./my-repo --exclude "**/vendor/**" --exclude "**/node_modules/**"

# Search (dense-only by default)
project-rag-cli query "websocket connection handler"

# Hybrid search (adds BM25 keyword fusion)
project-rag-cli query --hybrid "websocket connection handler"

# Show index statistics
project-rag-cli stats
```

### MCP tools

These are the tools exposed by `project-rag serve` and proxied by `project-rag-cli`:

| Tool | Parameters |
|------|-----------|
| `index_codebase` | `path` (str), `project` (str, optional), `include_patterns` (list[str], optional), `exclude_patterns` (list[str], optional), `max_file_size` (int, default 1048576) |
| `query_codebase` | `query` (str), `project` (str, optional), `path` (str, optional), `limit` (int, default 10), `min_score` (float, default 0.0), `hybrid` (bool, default false) |
| `get_statistics` | — |

### CUDA / GPU notes

The CLI wrapper auto-discovers CUDA libraries and injects them into `LD_LIBRARY_PATH` before launching the binary. Paths can be overridden:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROJECT_RAG_ORT_LIB_DIR` | auto-detected under `~/.cache/ort.pyke.io/dfbin/` | Directory containing `libonnxruntime_providers_cuda.so` |
| `CUDA_LIB_DIR` | `/opt/cuda/targets/x86_64-linux/lib` | CUDA runtime libraries |

If neither is found the binary still runs on CPU.

---

## Environment variables

| Variable | Tool | Default | Purpose |
|----------|------|---------|---------|
| `QDRANT_URL` | docs-search | `http://localhost:6333` | Qdrant instance URL |
| `PROJECT_RAG_ORT_LIB_DIR` | project-rag-cli | auto-detected | ORT CUDA provider library directory |
| `CUDA_LIB_DIR` | project-rag-cli | `/opt/cuda/targets/x86_64-linux/lib` | CUDA runtime library directory |
| `RUST_LOG` | project-rag-cli | `off` | Log level for the `project-rag` binary |

---

## License

MIT — see [LICENSE](LICENSE).
