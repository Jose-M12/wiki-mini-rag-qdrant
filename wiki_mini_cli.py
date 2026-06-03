#!/usr/bin/env python3
import click, os
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DATA_DIR = "wiki_mini_index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

@click.group()
def cli():
    """Tiny Wikipedia RAG (<100MB)"""
    pass

@cli.command()
def build():
    if os.path.exists(DATA_DIR):
        click.echo(f"Index already exists at./{DATA_DIR}")
        return
    click.echo("Loading rag-mini-wikipedia...")
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    split = 'train' if 'train' in ds else list(ds.keys())[0]
    corpus = ds[split]
    click.echo(f"Dataset fields: {list(corpus[0].keys())}")

    docs = []
    for i, row in enumerate(corpus):
        text = row.get('text') or row.get('passage') or ""
        doc_id = str(row.get('id', i))
        docs.append(Document(text=text, metadata={"id": doc_id}))

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None # <--- disable OpenAI
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    index.storage_context.persist(persist_dir=DATA_DIR)
    click.echo(f"✅ Built and saved to./{DATA_DIR}")

@cli.command()
@click.argument('question', nargs=-1)
@click.option('--k', default=3)
def query(question, k):
    q = " ".join(question).strip()
    if not q:
        return
    if not os.path.exists(DATA_DIR):
        click.echo("❌ No index. Run build first")
        return

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = None # <--- critical

    storage_context = StorageContext.from_defaults(persist_dir=DATA_DIR)
    index = load_index_from_storage(storage_context)

    # use retriever directly, no LLM synthesis
    retriever = index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(q)

    click.echo("\n" + "="*70)
    click.echo(f"Q: {q}")
    click.echo("="*70)
    for i, node in enumerate(nodes, 1):
        click.echo(f"\n[{i}] score={node.score:.3f} id={node.metadata.get('id')}")
        click.echo(node.text[:500].replace('\n',' '))
    click.echo("="*70)

if __name__ == '__main__':
    cli()
