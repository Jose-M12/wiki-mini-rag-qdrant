#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import click
from datasets import load_dataset
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ---------------------------------------------------------------------------
# Configuration (constants)
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "wiki_mini"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
DATA_MARKER = Path("wiki_mini_index.built")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "wiki_mini.log"
MAX_LOG_BYTES = 1_048_576  # 1 MiB
LOG_BACKUP_COUNT = 2


def setup_logging() -> logging.Logger:
    """Configure rotating file + console logging.

    Returns:
        Root logger for the application.
    """
    LOG_DIR.mkdir(mode=0o700, exist_ok=True)

    logger = logging.getLogger("wiki_mini")
    logger.setLevel(logging.DEBUG)

    # File handler – detailed logs
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(fh)

    # Console handler – user‑friendly output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


logger = setup_logging()


def _check_qdrant_health(client: QdrantClient) -> None:
    """Verify that the Qdrant instance is reachable.

    Raises:
        RuntimeError: If the health check fails.
    """
    try:
        health = client.health()
        if health.status != "ok":
            raise RuntimeError(f"Qdrant health status: {health.status}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
            "Is the container running? Use run_qdrant.sh"
        ) from exc


@click.group()
def cli() -> None:
    """Tiny Wikipedia RAG with Qdrant – local, CPU‑only, secure."""


@cli.command()
def build() -> None:
    """Download data, embed passages, and store vectors in Qdrant."""
    if DATA_MARKER.exists():
        click.echo("✅ Index already built (marker file found).")
        return

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _check_qdrant_health(client)

    # Load dataset
    logger.info("Loading rag-mini-wikipedia dataset…")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    split = "train" if "train" in ds else list(ds.keys())[0]
    corpus = ds[split]
    logger.info(f"Dataset loaded: {len(corpus)} passages.")

    # Prepare documents
    docs: List[Document] = []
    for i, row in enumerate(corpus):
        text = row.get("passage") or row.get("text") or ""
        doc_id = str(row.get("id", i))
        docs.append(Document(text=text, metadata={"id": doc_id}))

    # Embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None  # retrieval‑only

    # Create/recreate Qdrant collection
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Existing collection deleted.")
    except Exception:
        pass  # collection did not exist

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,  # bge‑small embedding dimension
            distance=Distance.COSINE,
        ),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Building index (this may take a few minutes on CPU)…")
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True
    )

    # Atomically create marker file with strict owner‑only permissions
    with DATA_MARKER.open("w", opener=lambda p, flags: os.open(p, flags, 0o600)) as f:
        f.write("qdrant")
    logger.info(f"Marker written → {DATA_MARKER}")
    click.echo("✅ Index built and stored in Qdrant.")


@cli.command()
@click.argument("question", nargs=-1)
@click.option("--k", default=3, help="Number of passages to retrieve")
def query(question: Tuple[str, ...], k: int) -> None:
    """Retrieve top‑k passages for QUESTION (no generation)."""
    q = " ".join(question).strip()
    if not q:
        return

    if not DATA_MARKER.exists():
        click.echo("❌ No index found. Run 'build' first.")
        return

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    _check_qdrant_health(client)

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(q)

    click.echo("\n" + "=" * 70)
    click.echo(f"Q: {q}")
    click.echo("=" * 70)
    for i, node in enumerate(nodes, 1):
        score = node.score if node.score is not None else 0.0
        doc_id = node.metadata.get("id", "?")
        click.echo(f"\n[{i}] score={score:.3f}  id={doc_id}")
        click.echo(node.text[:500].replace("\n", " "))
    click.echo("=" * 70)


if __name__ == "__main__":
    cli()
