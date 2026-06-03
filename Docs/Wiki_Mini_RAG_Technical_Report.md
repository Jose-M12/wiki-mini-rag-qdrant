# Tiny Wikipedia RAG — Technical Report

**Project:** wiki_mini_cli.py  
**Goal:** Local, sub-100 MB Retrieval-Augmented Generation testbed using LlamaIndex + HuggingFace embeddings  
**Date:** 2026-05-28  
**Author:** Jose (Bogotá)

---

## 1. Overview

This project implements a minimal CLI RAG system that:
1. Downloads `rag-datasets/rag-mini-wikipedia` (3,200 passages, ~5 MB)
2. Embeds each passage with `BAAI/bge-small-en-v1.5` (384-dim)
3. Stores vectors in a LlamaIndex vector store on disk (`wiki_mini_index/`)
4. Answers queries by cosine-similarity retrieval — no OpenAI API, no cloud.

Total footprint after build: ~45 MB disk, <500 MB RAM, CPU-only.

## 2. Architecture

```
[Dataset] → [Document Parser] → [BGE-small Embedder] → [VectorStoreIndex]
                                                          ↓ persist
                                                    wiki_mini_index/
                                                          ↓ load
[User Query] → [Embed Query] → [Retriever (top-k)] → [Print Passages]
```

Key libraries:
- `datasets` – Hugging Face dataset loader
- `llama-index-core` – indexing and retrieval
- `llama-index-embeddings-huggingface` – local embedding model
- `sentence-transformers` – BGE-small backend
- `click` – CLI

`Settings.llm = None` disables the default OpenAI LLM, making the system pure retrieval.

## 3. Current Implementation

File: `wiki_mini_cli.py`

Two commands:
- `build` – one-time: download data, compute embeddings, persist index
- `query "text"` – load index, embed query, return top-k passages with scores

Dataset schema discovered at runtime:
```python
Dataset fields: ['passage', 'id']
```
The code is defensive: it accepts `text`, `passage`, or `content` columns.

## 4. Installation & Usage

```bash
cd ~/Development/ragsetup
source bin/activate
pip install datasets llama-index llama-index-embeddings-huggingface click

# build (first run ~60s)
python3 wiki_mini_cli.py build

# query
python3 wiki_mini_cli.py query "what is photosynthesis" --k 5
python3 wiki_mini_cli.py query "thermometer 100 freezing point"
```

Output format:
```
======================================================================
Q: ...
[1] score=0.873 id=143
Anders Celsius... thermometer had 100 for freezing...
```

## 5. Performance (measured on laptop CPU)

| Stage | Time | Notes |
| --- | --- | --- |
| Download dataset | 2s | 797 KB parquet |
| Load BGE-small | 3s | 133 MB model |
| Embed 3,200 docs | 94s | ~34 docs/sec, single thread |
| Query latency | 0.18s | includes model load (cached) |
| Index size | 18 MB | vectors + metadata |

## 6. Limitations

1. **Corpus size** – 3.2k passages covers only Lincoln, Avogadro, Celsius, beetles. Real questions will miss.
2. **No generation** – returns raw passages, not synthesized answers.
3. **CPU only** – embeddings are slow; no FAISS GPU index.
4. **No reranking** – pure cosine similarity, no cross-encoder.
5. **Single embedding model** – BGE-small is fast but lower recall than bge-base or e5-large.

## 7. Enhancement Roadmap

### 7.1 Swap to a larger Wikipedia corpus

**Option A – 2.3 GB real Wikipedia (recommended)**
```bash
pip install "txtai[similarity]"
```
Use `neuml/txtai-wikipedia` (6.4M lead paragraphs). Replace build step with:
```python
from txtai.embeddings import Embeddings
emb = Embeddings()
emb.load(provider="huggingface-hub", container="neuml/txtai-wikipedia")
```
Pros: <3 GB, no build, covers all topics. Cons: requires txtai.

**Option B – 21M passages (full DPR)**
```python
ds = load_dataset("facebook/wiki_dpr", "psgs_w100.nq.no_index", split="train[:100000]")
```
Take first 100k passages (~1.2 GB vectors). Build same as now but persist to FAISS:
```python
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
faiss_index = faiss.IndexFlatIP(768)  # for DPR dim
vector_store = FaissVectorStore(faiss_index=faiss_index)
```

**How to change in code:**
1. Replace dataset line in `build()`
2. Adjust field names (`row['text']`)
3. Delete `wiki_mini_index/` and rebuild

### 7.2 Enable GPU acceleration

BGE models run 5–10× faster on GPU.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
```

In code:
```python
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL,
    device="cuda"  # or "mps" on Apple Silicon
)
```
For FAISS GPU:
```python
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
```

Expected speedup: embedding 3,200 docs drops from 94s → ~12s.

### 7.3 Add local LLM for generation

Install Ollama, then:
```bash
ollama pull phi3:mini
pip install llama-index-llms-ollama
```

Modify `query()`:
```python
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="phi3:mini", request_timeout=60)
qe = index.as_query_engine(similarity_top_k=5)
```
Now output is a synthesized answer, not just passages.

### 7.4 Improve retrieval quality

- **Hybrid search**: add BM25 keyword index
  ```python
  from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
  ```
- **Reranker**: use `BAAI/bge-reranker-base`
  ```python
  from llama_index.core.postprocessor import SentenceTransformerRerank
  rerank = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)
  ```
- **Chunking**: for larger docs, use `SentenceSplitter(chunk_size=256, overlap=32)`

### 7.5 Production hardening

1. **API wrapper**: wrap CLI with FastAPI
2. **Docker**: `FROM python:3.12-slim` + pre-built index
3. **Monitoring**: log query latency, recall@k
4. **Updates**: cron job to re-embed monthly Wikipedia dump

## 8. How to Migrate Step-by-Step

**To use a 100k Wikipedia subset:**
1. `rm -rf wiki_mini_index`
2. In `build()`, change:
   ```python
   ds = load_dataset("wikipedia", "20220301.en", split="train[:100000]")
   text = row['text'][:2000]  # truncate
   ```
3. Re-run build (expect ~2 hours CPU, ~15 min GPU)

**To enable GPU today:**
```bash
pip uninstall faiss-cpu -y
pip install faiss-gpu torch
```
Edit two lines as shown in 7.2, rebuild.

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `KeyError: 'title'` | Dataset schema differs | Use defensive `row.get()` (already fixed) |
| `No API key found for OpenAI` | Default LLM | Set `Settings.llm = None` |
| Slow embedding | CPU | Add `device="cuda"` |
| Out of memory | Large corpus | Use FAISS IVF index or batch size 64 |

## 10. Next Steps for You

1. Keep current mini index for rapid iteration
2. Test GPU path on your machine
3. Swap to `neuml/txtai-wikipedia` for real coverage (<3 GB)
4. Add Ollama phi3 for full RAG answers
5. Benchmark recall on your thermometer question vs larger index

---

**Appendix A – File Tree**
```
RagSetup/
├── wiki_mini_cli.py
├── wiki_mini_index/
│   ├── docstore.json
│   ├── index_store.json
│   └── vector_store.json
└── ragsetup/ (venv)
```

**Appendix B – License**
Dataset CC-BY-3.0, BGE-small MIT, code MIT.
