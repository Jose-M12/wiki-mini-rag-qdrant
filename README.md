# wiki-mini-rag-qdrant

**Local CPU‑only vector search over 3,200 Wikipedia passages.**

Uses LlamaIndex + Qdrant + BGE‑small embeddings. No cloud, no GPU, no API keys.

## Quick Start

### Prerequisites
- Ubuntu 24.04 (or similar)
- Docker
- Python 3.12

### 1. Clone & set up virtualenv
```bash
git clone https://github.com/Jose-M12/wiki-mini-rag-qdrant.git
cd wiki-mini-rag-qdrant
source ragsetup/bin/activate
```

### 2. Install dependencies (hash‑pinned, secure)
```bash
python -m pip install --require-hashes -r requirements.txt
```

### 3. Build the index (~90 seconds on CPU)
```bash
chmod +x run_qdrant.sh
./run_qdrant.sh build
```

### 4. Query
```bash
./run_qdrant.sh query "what is photosynthesis" --k 3
```

## Usage

```bash
# Default (top‑3 passages)
./run_qdrant.sh query "thermometer 100 freezing point"

# Increase results
./run_qdrant.sh query "Avogadro's number" --k 5

# Keep Qdrant running for multiple queries
./run_qdrant.sh --keep-running query "test"
```

## How it works

- `run_qdrant.sh` starts a Qdrant Docker container on `127.0.0.1:6333` only when needed, stops it after each command.
- `wiki_mini_cli.py` embeds passages and retrieves the most similar ones to your question.
- All data is stored in `qdrant_data/` (persistent, secure permissions).

## Files

```
wiki-mini-rag-qdrant/
├── wiki_mini_cli.py       # CLI: build + query
├── run_qdrant.sh          # Docker orchestration
├── requirements.in        # Direct dependencies
├── requirements.txt       # Hash‑pinned lock file
├── qdrant_data/           # Vector database (persistent)
└── logs/                  # Rotating logs
```

## Troubleshooting

**Qdrant unreachable** → Ensure Docker is running, then use `./run_qdrant.sh` (never run the Python script directly).

**Permission denied (Docker)** → Add your user to the `docker` group: `sudo usermod -aG docker $USER` (log out/in).

**Slow build** → Embedding 3,200 passages on CPU takes ~90s. That's normal.

**Module not found** → Re‑run `pip install --require-hashes -r requirements.txt`.

## License

MIT – see `LICENSE`.
```

This version gets a new Ubuntu user from zero to a working query in four short steps. Let me know if you'd like any further trimming.
