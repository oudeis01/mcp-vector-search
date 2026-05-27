import torch
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from . import config


class Embedder:
    def __init__(self) -> None:
        self._dense: SentenceTransformer | None = None
        self._sparse: SparseTextEmbedding | None = None

    @property
    def dense(self) -> SentenceTransformer:
        if self._dense is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._dense = SentenceTransformer(config.DENSE_MODEL, device=device)
            self._dense.max_seq_length = config.MAX_SEQ_LENGTH
        return self._dense

    @property
    def sparse(self) -> SparseTextEmbedding:
        if self._sparse is None:
            # BM42 is lightweight (MiniLM-L6 based); CPU is fine
            self._sparse = SparseTextEmbedding(config.SPARSE_MODEL)
        return self._sparse

    def embed_batch(self, texts: list[str]) -> tuple[list, list]:
        dense_vecs = self.dense.encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sparse_vecs = list(self.sparse.embed(texts, batch_size=config.BATCH_SIZE))
        return list(dense_vecs), sparse_vecs
