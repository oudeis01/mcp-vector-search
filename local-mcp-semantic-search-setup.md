# 로컬 MCP 시맨틱 서치 셋업 가이드

> 대형 코드베이스 & 리서치 논문 탐색을 위한 로컬 RAG + MCP 환경 구성

---

## 목적

LLM이 대형 코드베이스나 다수의 연구 논문을 다룰 때 발생하는 두 가지 근본 문제를 해결한다.

1. **컨텍스트 한계**: 모든 파일을 프롬프트에 넣을 수 없다
2. **토큰 낭비**: 관련 없는 내용까지 컨텍스트에 포함되면 비용·정확도 모두 저하된다

해결 방향은 **필요한 청크만 정밀하게 검색해서 컨텍스트에 주입**하는 것이다. MCP(Model Context Protocol)를 통해 Claude Code / Claude Desktop 같은 클라이언트가 로컬 벡터 DB에 직접 쿼리할 수 있도록 구성한다.

```
LLM Client (Claude Code 등)
    │  MCP (stdio / SSE)
    ▼
MCP Server  ←→  Vector DB (Qdrant / LanceDB)
    │
    ├── Embedding Model (bge-m3, etc.)
    ├── Sparse Index   (BM42 / BM25)
    └── Reranker       (bge-reranker-v2-m3)
```

---

## 두 가지 사용 시나리오

| | 자연어 문서 (논문, md, txt) | 코드베이스 |
|---|---|---|
| **청킹 전략** | 문단 / sliding window | Tree-sitter AST (함수·클래스 단위) |
| **임베딩** | BAAI/bge-m3 (Dense) | BAAI/bge-m3 또는 코드 특화 모델 |
| **Sparse** | BM42 | BM25 (심볼명 exact match 필수) |
| **리랭커** | BGE-Reranker-v2-m3 | 선택적 |
| **벡터 DB** | Qdrant (Docker) | LanceDB (임베디드) 또는 Qdrant |
| **MCP 서버** | mcp-server-qdrant | Code-Index-MCP / project-rag |

---

## 공통 전제 조건

```
Python  >= 3.11
Node.js >= 18     (Code-Index-MCP 사용 시)
Rust    >= 1.75   (project-rag 사용 시)
Docker            (Qdrant 사용 시)
uv                (Python 패키지 관리 권장)
```

```bash
# uv 설치
curl -Ls https://astral.sh/uv/install.sh | sh
```

---

## Case 1 — 자연어 문서 (논문 / md / txt)

### 1-1. Qdrant 실행 (Docker)

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

헬스 체크:
```bash
curl http://localhost:6333/healthz
# {"title":"qdrant - vector search engine","version":"..."}
```

### 1-2. 컬렉션 생성 (Dense + Sparse 하이브리드)

```python
# setup_collection.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    SparseVectorParams, SparseIndexParams
)

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="papers",
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)
print("컬렉션 생성 완료")
```

```bash
uv run setup_collection.py
```

### 1-3. 임베딩 모델 준비

```bash
uv pip install fastembed sentence-transformers
```

```python
# embed.py
from fastembed import TextEmbedding, SparseTextEmbedding

dense_model  = TextEmbedding("BAAI/bge-m3")           # Dense  1024d
sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")  # BM42

def embed(texts: list[str]):
    dense  = list(dense_model.embed(texts))
    sparse = list(sparse_model.embed(texts))
    return dense, sparse
```

> **모델 선택 메모**
> - `BAAI/bge-m3`: 100개 언어 지원, 긴 문서(최대 8192 토큰)에 강함. 논문처럼 전문 용어가 많은 도메인에 적합
> - `Qdrant/bm42-all-minilm-l6-v2-attentions`: attention 기반 BM42. 기존 BM25보다 단어 가중치가 문맥을 반영

### 1-4. 문서 인덱싱

```python
# ingest_papers.py
import glob, uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from embed import embed

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "papers"
CHUNK_SIZE  = 800   # 토큰 기준 (단어 수로 근사)
CHUNK_OVERLAP = 100

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
    return chunks

def ingest(file_path: str):
    text = Path(file_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    dense_vecs, sparse_vecs = embed(chunks)

    points = []
    for i, (chunk, dv, sv) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense":  dv.tolist(),
                "sparse": SparseVector(indices=sv.indices.tolist(),
                                       values=sv.values.tolist())
            },
            payload={
                "text":   chunk,
                "source": file_path,
                "chunk":  i
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"[{file_path}] {len(points)} 청크 인덱싱 완료")

if __name__ == "__main__":
    for fp in glob.glob("papers/**/*.{md,txt}", recursive=True):
        ingest(fp)
```

```bash
uv run ingest_papers.py
```

### 1-5. 리랭커 설정 (선택)

```bash
uv pip install transformers torch
```

```python
# rerank.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_ID = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

def rerank(query: str, passages: list[str], top_k=5) -> list[tuple[float, str]]:
    pairs = [[query, p] for p in passages]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True,
                           return_tensors="pt", max_length=512)
        scores = model(**inputs).logits.squeeze(-1).tolist()
    ranked = sorted(zip(scores, passages), reverse=True)
    return ranked[:top_k]
```

### 1-6. mcp-server-qdrant 설치 및 연결

```bash
# Claude Code에 MCP 서버 등록
claude mcp add papers-search \
  -e QDRANT_URL="http://localhost:6333" \
  -e COLLECTION_NAME="papers" \
  -e EMBEDDING_MODEL="BAAI/bge-m3" \
  -- uvx mcp-server-qdrant
```

또는 `claude_desktop_config.json` 수동 편집:

```json
{
  "mcpServers": {
    "papers-search": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL":       "http://localhost:6333",
        "COLLECTION_NAME":  "papers",
        "EMBEDDING_MODEL":  "BAAI/bge-m3"
      }
    }
  }
}
```

---

## Case 2 — 코드베이스

코드는 문단 단위 청킹이 아니라 **AST(추상 구문 트리) 기반 청킹**이 필수다.
함수 하나를 통째로 하나의 청크로 만들어야 LLM이 의미 있는 단위를 검색할 수 있다.

### 옵션 A — Code-Index-MCP (Node.js, 48개 언어)

```bash
# 전역 설치
npm install -g code-index-mcp

# Claude Code에 등록
claude mcp add code-index \
  -e VOYAGE_AI_API_KEY="<key>" \   # 시맨틱 서치 원할 경우. 없으면 BM25만 동작
  -- code-index-mcp
```

주요 기능:
- Tree-sitter 48개 언어 파싱
- BM25 + 시맨틱 하이브리드 서치
- 파일 변경 실시간 감지 (sub-100ms)
- Git 연동 자동 인덱스 업데이트
- 로컬 `.indexes/` 에 저장

### 옵션 B — project-rag (Rust, 외부 의존성 없음)

```bash
# 빌드
git clone https://github.com/Brainwires/project-rag
cd project-rag
cargo build --release

# Claude Code에 등록
claude mcp add project-rag \
  -- ./target/release/project-rag
```

주요 기능:
- Tree-sitter AST 청킹 (12개 언어)
- FastEmbed + LanceDB (외부 서버 불필요)
- Vector + BM25 하이브리드 서치
- **Git 커밋 히스토리 시맨틱 검색**
- 멀티 프로젝트 동시 쿼리
- Qdrant 선택 가능 (`.env`에서 전환)

```bash
# project-rag .env 예시
LANCEDB_PATH=./.lancedb          # 기본 임베디드 DB
# QDRANT_URL=http://localhost:6333  # Qdrant로 전환 시 주석 해제
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 옵션 C — LanceDB MCP (Python, 경량)

```bash
uv pip install lancedb-mcp

claude mcp add lancedb-code \
  -e LANCEDB_PATH="./.code-db" \
  -- uvx lancedb-mcp
```

Tree-sitter chunker + LanceDB 조합으로 Python 단독 환경에서 사용하기 좋다.

---

## 두 케이스 동시 운용 예시

논문(Qdrant)과 코드(LanceDB)를 별도 MCP 서버로 등록해 Claude Code가 컨텍스트에 따라 각각 쿼리하도록 구성한다.

```json
{
  "mcpServers": {
    "papers-search": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL":      "http://localhost:6333",
        "COLLECTION_NAME": "papers",
        "EMBEDDING_MODEL": "BAAI/bge-m3"
      }
    },
    "code-search": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL":             "http://localhost:6333",
        "COLLECTION_NAME":        "codebase",
        "EMBEDDING_MODEL":        "sentence-transformers/all-MiniLM-L6-v2",
        "TOOL_STORE_DESCRIPTION": "Store code snippets with descriptions",
        "TOOL_FIND_DESCRIPTION":  "Search for relevant code snippets"
      }
    }
  }
}
```

> **컬렉션 분리 이유**: 논문(bge-m3 1024d)과 코드(MiniLM 384d)는 임베딩 차원이 다르므로 반드시 별도 컬렉션으로 관리해야 한다.

---

## 운용 팁

### 인덱싱 전략

| 상황 | 권장 |
|---|---|
| 최초 전체 인덱싱 | `ingest_papers.py` 배치 실행 |
| 파일 추가/수정 | Code-Index-MCP 실시간 감지 활용 |
| 대량 논문 추가 | Qdrant `upsert` (중복 ID 자동 갱신) |

### 청크 사이즈 튜닝

```
논문/긴 문서:   CHUNK_SIZE=800,  OVERLAP=100
짧은 메모/md:   CHUNK_SIZE=300,  OVERLAP=50
코드 (AST):     함수 단위 그대로. 512 토큰 초과 시 50 토큰 overlap으로 분할
```

### 검색 품질 향상

1. **Hybrid Search 비율 조정**: Qdrant `prefetch` + `fusion: RRF` 로 dense/sparse 결과 병합
2. **리랭커**: top-20 후보 → reranker → top-5 반환하면 정밀도 대폭 향상
3. **메타데이터 필터링**: `source`, `year`, `language` 등 페이로드 필드로 검색 범위 축소

```python
# Qdrant 하이브리드 검색 예시 (prefetch + RRF)
from qdrant_client.models import Prefetch, FusionQuery, Fusion

results = client.query_points(
    collection_name="papers",
    prefetch=[
        Prefetch(query=dense_vec,  using="dense",  limit=20),
        Prefetch(query=sparse_vec, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=5
)
```

---

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| MCP 서버 연결 안 됨 | Qdrant Docker 미실행 | `docker ps`로 컨테이너 확인 |
| 검색 결과 없음 | 컬렉션 비어 있음 | `ingest_papers.py` 실행 여부 확인 |
| 임베딩 차원 불일치 | 모델 변경 후 기존 컬렉션 사용 | 컬렉션 삭제 후 재생성 |
| 느린 인덱싱 | CPU 전용 실행 | `--cpu-execution-provider` 옵션 또는 GPU 환경 구성 |
| `fastembed` 모델 다운로드 느림 | 최초 실행 시 모델 캐시 없음 | `~/.cache/fastembed/` 경로 확인, 네트워크 대기 |

---

## 참고 링크

- [Qdrant 공식 MCP 서버](https://github.com/qdrant/mcp-server-qdrant)
- [Code-Index-MCP](https://lobehub.com/mcp/viperjuice-code-index-mcp)
- [project-rag (Rust)](https://github.com/Brainwires/project-rag)
- [BAAI/bge-m3 모델 카드](https://huggingface.co/BAAI/bge-m3)
- [Qdrant Hybrid Search 문서](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [FastEmbed](https://github.com/qdrant/fastembed)
