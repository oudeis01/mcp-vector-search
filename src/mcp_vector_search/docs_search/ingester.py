import hashlib
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

from . import config
from .chunker import chunk_text
from .embedder import Embedder

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _collection_name(project: str) -> str:
    return f"{config.COLLECTION_PREFIX}{project}"


def _matches_patterns(path: Path, root: Path, patterns: list[str]) -> bool:
    # Path.match() supports ** recursive globs (Python 3.12+).
    # We test against both the full relative path and the filename alone
    # so that plain patterns like "*.md" work without a leading "**/" prefix.
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


def ingest(
    project: str,
    directory: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> dict:
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

        # Drop chunks whose content we've already indexed this run. Duplicate files in
        # the corpus would otherwise flood results with identical chunks.
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


def list_projects(client: QdrantClient | None = None) -> list[dict]:
    if client is None:
        client = QdrantClient(url=config.QDRANT_URL)
    collections = client.get_collections().collections
    projects = []
    for c in collections:
        if not c.name.startswith(config.COLLECTION_PREFIX):
            continue
        slug = c.name[len(config.COLLECTION_PREFIX):]
        info = client.get_collection(c.name)
        projects.append({
            "project": slug,
            "collection": c.name,
            "points_count": info.points_count,
        })
    return projects


def delete_project(project: str) -> dict:
    client = QdrantClient(url=config.QDRANT_URL)
    collection = _collection_name(project)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return {"success": False, "message": f"Project '{project}' not found."}
    client.delete_collection(collection)
    return {"success": True, "message": f"Project '{project}' deleted."}
