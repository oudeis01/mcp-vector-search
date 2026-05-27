import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_PREFIX = "docs__"
DENSE_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_LIMIT = 10
# RRF fusion scores cap around 0.033 (sum of 1/(k+rank) with k=60); 0.0 = no filter, let caller decide.
DEFAULT_MIN_SCORE = 0.0
# Dense-only by default: BM42 sparse caused semantic false positives in quality tests.
# Callers can still pass hybrid=True for keyword/identifier-style queries.
DEFAULT_HYBRID = False
BATCH_SIZE = 8
MAX_SEQ_LENGTH = 512
