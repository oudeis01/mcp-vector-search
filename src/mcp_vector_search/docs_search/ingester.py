import hashlib
import sys
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .._registry import (
    Project,
    ProjectExistsError,
    ProjectNotRegisteredError,
    ProjectRegistry,
)
from . import config
from .chunker import chunk_text
from .embedder import Embedder

_embedder: Embedder | None = None
_registry: ProjectRegistry | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_registry() -> ProjectRegistry:
    global _registry
    if _registry is None:
        _registry = ProjectRegistry("docs")
        _maybe_nuke_orphans(_registry)
    return _registry


def _maybe_nuke_orphans(registry: ProjectRegistry) -> None:
    """On first boot, drop all docs__* Qdrant collections (test-only data).

    A marker file inside the registry data dir guards against re-runs. If
    Qdrant is unreachable the marker is NOT written, so the nuke is retried
    next time the registry is initialized.
    """
    marker = registry.first_boot_marker()
    if marker.exists():
        return
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        for c in client.get_collections().collections:
            if c.name.startswith(config.COLLECTION_PREFIX):
                client.delete_collection(c.name)
                print(f"[docs-search] nuked orphan collection: {c.name}", file=sys.stderr)
        marker.write_text(str(int(time.time())))
    except Exception as e:
        print(f"[docs-search] orphan nuke deferred (Qdrant unreachable: {e})", file=sys.stderr)


def _collection_name(project: str) -> str:
    return f"{config.COLLECTION_PREFIX}{project}"


def _matches_patterns(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root)
    return any(rel.match(p) or path.match(p) for p in patterns)


def _scan_files(directory: str, include_patterns: list[str], exclude_patterns: list[str]) -> list[Path]:
    root = Path(directory).resolve()
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exclude_patterns and _matches_patterns(p, root, exclude_patterns):
            continue
        if include_patterns and not _matches_patterns(p, root, include_patterns):
            continue
        files.append(p)
    return files


def _ensure_collection(client: QdrantClient, name: str) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=config.DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )


def _project_to_dict(p: Project, points_count: int | None = None) -> dict:
    out = {
        "project": p.slug,
        "abs_path": p.abs_path,
        "description": p.description,
        "created_at": p.created_at,
        "last_indexed_at": p.last_indexed_at,
    }
    if points_count is not None:
        out["points_count"] = points_count
    return out


def create_project(slug: str, abs_path: str, description: str | None = None) -> dict:
    registry = _get_registry()
    p = registry.create(slug, abs_path, description)
    return _project_to_dict(p, points_count=0)


def list_projects() -> list[dict]:
    registry = _get_registry()
    projects = registry.list_all()
    if not projects:
        return []
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        existing = {c.name for c in client.get_collections().collections}
        counts: dict[str, int] = {}
        for p in projects:
            name = _collection_name(p.slug)
            if name in existing:
                info = client.get_collection(name)
                counts[p.slug] = info.points_count or 0
            else:
                counts[p.slug] = 0
    except Exception:
        counts = {p.slug: 0 for p in projects}
    return [_project_to_dict(p, counts.get(p.slug, 0)) for p in projects]


def resolve_project_from_path(path: str) -> dict | None:
    registry = _get_registry()
    p = registry.resolve_from_path(path)
    if p is None:
        return None
    return _project_to_dict(p)


def update_project_description(slug: str, description: str | None) -> dict:
    registry = _get_registry()
    p = registry.update_description(slug, description)
    return _project_to_dict(p)


def ingest(
    project: str,
    directory: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> dict:
    registry = _get_registry()
    registry.get(project)

    client = QdrantClient(url=config.QDRANT_URL)
    collection = _collection_name(project)
    _ensure_collection(client, collection)

    files = _scan_files(directory, include_patterns, exclude_patterns)
    embedder = _get_embedder()

    total_chunks = 0
    total_files = 0
    skipped_duplicates = 0
    seen_hashes: set[str] = set()
    start = time.time()

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        chunks = chunk_text(text)
        if not chunks:
            continue

        unique_chunks = []
        for c in chunks:
            h = hashlib.sha256(c.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                skipped_duplicates += 1
                continue
            seen_hashes.add(h)
            unique_chunks.append(c)
        chunks = unique_chunks
        if not chunks:
            continue

        dense_vecs, sparse_vecs = embedder.embed_batch(chunks)
        points = []
        for i, (chunk, dv, sv) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dv.tolist(),
                        "sparse": SparseVector(
                            indices=sv.indices.tolist(),
                            values=sv.values.tolist(),
                        ),
                    },
                    payload={
                        "text": chunk,
                        "file_path": str(file_path.relative_to(Path(directory).resolve())),
                        "chunk_index": i,
                        "project": project,
                        "indexed_at": int(time.time()),
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)
        total_chunks += len(points)
        total_files += 1

    duration_ms = int((time.time() - start) * 1000)
    registry.touch_indexed(project)
    return {
        "project": project,
        "files_indexed": total_files,
        "chunks_created": total_chunks,
        "duplicate_chunks_skipped": skipped_duplicates,
        "duration_ms": duration_ms,
    }


def search(
    project: str,
    query: str,
    limit: int = config.DEFAULT_LIMIT,
    min_score: float = config.DEFAULT_MIN_SCORE,
    hybrid: bool = config.DEFAULT_HYBRID,
) -> list[dict]:
    from qdrant_client.models import FusionQuery, Fusion, Prefetch

    registry = _get_registry()
    registry.get(project)

    client = QdrantClient(url=config.QDRANT_URL)
    collection = _collection_name(project)
    embedder = _get_embedder()

    (dense_vecs, sparse_vecs) = embedder.embed_batch([query])
    dv = dense_vecs[0]
    sv = sparse_vecs[0]

    if hybrid:
        results = client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dv.tolist(), using="dense", limit=limit * 2),
                Prefetch(
                    query=SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
        )
    else:
        results = client.query_points(
            collection_name=collection,
            query=dv.tolist(),
            using="dense",
            limit=limit,
            score_threshold=min_score if min_score > 0 else None,
        )

    return [
        {
            "score": r.score,
            "text": r.payload.get("text", ""),
            "file_path": r.payload.get("file_path", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
        }
        for r in results.points
    ]


def delete_project(project: str) -> dict:
    registry = _get_registry()
    try:
        registry.get(project)
    except ProjectNotRegisteredError:
        return {"success": False, "message": f"Project '{project}' is not registered."}

    client = QdrantClient(url=config.QDRANT_URL)
    collection = _collection_name(project)
    existing = {c.name for c in client.get_collections().collections}
    collection_dropped = collection in existing
    if collection_dropped:
        client.delete_collection(collection)
    registry.delete(project)
    return {
        "success": True,
        "message": f"Project '{project}' deleted.",
        "collection_dropped": collection_dropped,
    }


__all__ = [
    "create_project",
    "delete_project",
    "ingest",
    "list_projects",
    "resolve_project_from_path",
    "search",
    "update_project_description",
    "ProjectExistsError",
    "ProjectNotRegisteredError",
]
