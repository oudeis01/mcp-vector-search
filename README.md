# mcp-vector-search

Two local, project-isolated semantic-search tools that speak the [Model Context Protocol](https://modelcontextprotocol.io):

- **docs-search** -- a FastMCP server that indexes documents into [Qdrant](https://qdrant.tech) and serves hybrid (dense + sparse) semantic search. Multilingual via `BAAI/bge-m3`.
- **code-search** -- a FastMCP server that wraps the `project-rag` binary and enforces a project registry, so an agent working in codebase A never sees results from codebase B.

Both servers refuse to index or search anything that is not first registered as a project (slug + absolute path). The registry is the mechanism that keeps unrelated indexes from polluting an agent's context.

## Requirements

- Python >= 3.12 ([uv](https://github.com/astral-sh/uv) recommended)
- Docker (for the Qdrant container that `docs-search` talks to)
- The `project-rag` binary on `PATH` (for code search) -- build it from the [fork](https://github.com/oudeis01/project-rag)
- Optional: an NVIDIA GPU + CUDA for accelerated embedding

### Why a fork of project-rag?

`code-search` and `project-rag-cli` depend on [oudeis01/project-rag](https://github.com/oudeis01/project-rag), a fork of [Brainwires/project-rag](https://github.com/Brainwires/project-rag). Two changes in the fork are intentional design decisions that diverge from upstream:

| Change | Upstream | This fork | Reason |
|--------|----------|-----------|--------|
| Default embedding model | `all-MiniLM-L6-v2` (384d, general-purpose) | `jinaai/jina-embeddings-v2-base-code` (768d, code-specific) | MiniLM produced 0.014-0.016 scores with no discrimination; jina produces 0.44-0.60 with real semantic ranking on code |
| GPU acceleration | Silent CPU fallback via `error_on_failure=false` | Explicit CUDA ExecutionProvider registration | Eliminates silent GPU failure; 2x+ indexing speedup on CUDA hardware |

Two additional changes are bug fixes being submitted upstream as PRs: tracing logs redirected to stderr (stdout is reserved for MCP JSON-RPC), and glob-based file exclusion patterns replacing broken substring matching.

## Install

```bash
uv sync
```

This exposes five entry points:

| Command | Purpose |
|---------|---------|
| `docs-search` | CLI for the document index (project / ingest / query / list / delete) |
| `docs-search-mcp` | the docs-search MCP server (stdio) |
| `code-search-cli` | CLI for project-isolated code search (project / index / query / stats) |
| `code-search-mcp` | the code-search MCP server (stdio); wraps `project-rag` with project enforcement |
| `project-rag-cli` | raw passthrough to the `project-rag` binary; no project enforcement, kept as an escape hatch |

## Initial setup

### 1. Bring up Qdrant as a systemd user service

`docs-search` requires a running Qdrant instance. Run it via docker compose under a systemd user unit so it survives reboots:

```bash
ln -s "$PWD/infra/qdrant/qdrant.service" ~/.config/systemd/user/qdrant.service
systemctl --user daemon-reload
systemctl --user enable --now qdrant.service
curl -s http://localhost:6333/ | head -3
```

Data lives at `~/.local/share/qdrant/` (XDG). To boot the service without an active login, enable lingering once: `sudo loginctl enable-linger $USER`.

### 2. Register your first project

```bash
docs-search project create research-papers ~/papers --description "ICML 2025 reading list"
code-search-cli project create my-app ~/code/my-app --description "Production Rust service"
```

---

## MCP Setup

Both servers expose MCP over stdio. Configure them in your agent of choice.

### Claude Code

Add to `.claude/settings.json` (project) or `~/.claude/settings.json` (global):

```json
{
  "mcpServers": {
    "docs-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-vector-search", "docs-search-mcp"]
    },
    "code-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-vector-search", "code-search-mcp"]
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
    "code-search": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-vector-search", "code-search-mcp"]
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
    "code-search": {
      "type": "local",
      "command": ["uv", "run", "--project", "/path/to/mcp-vector-search", "code-search-mcp"],
      "enabled": true
    }
  }
}
```

> **Note:** Replace `/path/to/mcp-vector-search` with the absolute path to this repo. The `project-rag` binary must be on `PATH`.

---

## Project lifecycle

Both MCP servers require every index/search/navigation call to specify a previously registered project. A project is identified by:

- `slug` -- short logical identifier (e.g. `my-app`)
- `abs_path` -- absolute filesystem path that anchors the project
- `description` (optional) -- one line that helps future agents understand the project

Both `slug` and `abs_path` are UNIQUE. Re-registering either is rejected.

### Project-management tools (exposed by both servers)

| Tool | Behavior |
|------|----------|
| `create_project(slug, abs_path, description?)` | Register. Errors on slug or abs_path collision. |
| `list_projects()` | All projects with metadata. docs-search also returns Qdrant point counts. |
| `resolve_project_from_path(path)` | Deepest registered project whose abs_path contains `path`, or null. Use on onboarding to discover whether the working directory is already indexed. |
| `update_project_description(slug, description)` | Mutate description. `abs_path` is immutable; to relocate, delete and re-create. |
| `delete_project(slug)` | Remove from registry. docs-search also drops the Qdrant collection; code-search leaves the LanceDB chunks in place (project-rag has no per-project clear) but they become inaccessible through the wrapper. |

### Onboarding flow for an agent

```
1. resolve_project_from_path(cwd) -> {slug, abs_path, description} or null
2. If null: list_projects() to inspect options, then either
     - create_project(slug, abs_path, description), or
     - tell the user the codebase needs indexing
3. Pass the resolved slug into ingest_documents / index_codebase / query_* / etc.
```

Registries live at `~/.local/share/mcp-vector-search/{docs,code}/registry.db`.

### First-boot data nuke

The first time each MCP server starts after install, it drops the existing test data (Qdrant `docs__*` collections for docs-search; the entire project-rag LanceDB directory for code-search). A marker file inside the registry data dir guards the operation so it runs only once.

---

## docs-search

Embedding model: `BAAI/bge-m3` (1024d, multilingual). Dense search uses cosine similarity; hybrid adds BM42 sparse with RRF fusion.

### CLI

```bash
# Register a project (one-time per project)
docs-search project create my-papers ~/papers --description "ICML 2025"

# Index a directory into the registered project (re-index drops & recreates)
docs-search ingest --project my-papers --dir ~/papers --include "**/*.md" --include "**/*.txt"

# Query (dense-only by default)
docs-search query --project my-papers "attention mechanism in transformers"

# Hybrid search (keyword + semantic fusion)
docs-search query --project my-papers --hybrid "attention mechanism in transformers"

# Manage projects
docs-search project list
docs-search project resolve ~/papers/sub/file.md
docs-search project describe my-papers --description "updated note"
docs-search delete --project my-papers
```

### MCP tools

| Tool | Parameters |
|------|-----------|
| `create_project` | `slug`, `abs_path`, `description?` |
| `list_projects` | -- |
| `resolve_project_from_path` | `path` |
| `update_project_description` | `slug`, `description` |
| `ingest_documents` | `project`, `directory`, `include_patterns?`, `exclude_patterns?` |
| `search_documents` | `project`, `query`, `limit=10`, `min_score=0.0`, `hybrid=false` |
| `delete_project` | `project` |

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

Dense cosine scores range `0.0` to `1.0`. Hybrid RRF scores are rank-based and cap around `0.033` regardless of semantic similarity. Start with `min_score=0.0`, inspect the distribution, then tighten if needed. Do not mix thresholds across modes.

---

## code-search

Embedding model: `jinaai/jina-embeddings-v2-base-code` (768d, code-specific). Wraps the `project-rag` binary (must be on `PATH`). Every call is forced to specify a registered project, and the project's `abs_path` is auto-injected so the wrapper cannot be tricked into indexing or searching outside the declared codebase root.

### CLI

```bash
# Register a project
code-search-cli project create my-app ~/code/my-app --description "Rust service"

# Index (path is taken from the registry, not the CLI args)
code-search-cli index --project my-app --exclude "**/target/**" --exclude "**/node_modules/**"

# Search (dense-only by default; --hybrid adds BM25)
code-search-cli query --project my-app "websocket connection handler"

# Stats (database-wide; see per_project flag in response)
code-search-cli stats --project my-app

# Manage projects
code-search-cli project list
code-search-cli project resolve ~/code/my-app/src/main.rs
code-search-cli project delete my-app -y
```

### MCP tools

13 tools total: 4 for registry management + 9 wrapped from `project-rag`. Every wrapped tool requires `project` and the wrapper forces both the `project` filter and `path` (= the registry's abs_path) on the underlying call.

| Tool | Parameters |
|------|-----------|
| `create_project` | `slug`, `abs_path`, `description?` |
| `list_projects` | -- |
| `resolve_project_from_path` | `path` |
| `update_project_description` | `slug`, `description` |
| `index_codebase` | `project`, `include_patterns?`, `exclude_patterns?`, `max_file_size=1048576` |
| `query_codebase` | `project`, `query`, `limit=10`, `min_score=0.0`, `hybrid=false` |
| `search_by_filters` | `project`, `query`, `file_extensions?`, `languages?`, `path_patterns?`, `limit=10`, `min_score=0.0`, `hybrid=false` |
| `search_git_history` | `project`, `query`, `branch?`, `max_commits=10`, `limit=10`, `min_score=0.0`, `author?`, `since?`, `until?`, `file_pattern?` |
| `find_definition` | `project`, `file_path`, `line`, `column` |
| `find_references` | `project`, `file_path`, `line`, `column` |
| `get_call_graph` | `project`, `file_path`, `line`, `column` |
| `get_statistics` | `project` (response tagged `per_project=false`; project-rag's stats are database-wide) |
| `delete_project` | `project` |
| `clear_all_indexes` | `confirm=false` -- wipes ALL indexed data across every project. Requires `confirm=true`. |

### CUDA / GPU notes

`code-search` and `project-rag-cli` auto-discover CUDA libraries and inject them into `LD_LIBRARY_PATH` before spawning the binary. Paths can be overridden:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROJECT_RAG_ORT_LIB_DIR` | auto-detected under `~/.cache/ort.pyke.io/dfbin/` | Directory containing `libonnxruntime_providers_cuda.so` |
| `CUDA_LIB_DIR` | `/opt/cuda/targets/x86_64-linux/lib` | CUDA runtime libraries |

If neither is found the binary still runs on CPU.

---

## project-rag-cli (escape hatch)

A thin passthrough to the `project-rag` binary with no project enforcement. Useful when you need to probe raw global statistics, run `clear_index`, or test the binary directly. For agent use, prefer `code-search-mcp`.

```bash
project-rag-cli index --dir ./my-repo
project-rag-cli query "websocket handler"
project-rag-cli stats
```

---

## Environment variables

| Variable | Tool | Default | Purpose |
|----------|------|---------|---------|
| `QDRANT_URL` | docs-search | `http://localhost:6333` | Qdrant instance URL |
| `PROJECT_RAG_ORT_LIB_DIR` | code-search, project-rag-cli | auto-detected | ORT CUDA provider library directory |
| `CUDA_LIB_DIR` | code-search, project-rag-cli | `/opt/cuda/targets/x86_64-linux/lib` | CUDA runtime library directory |
| `RUST_LOG` | code-search, project-rag-cli | `off` | Log level for the `project-rag` binary |

---

## License

MIT -- see [LICENSE](LICENSE).
