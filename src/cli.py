"""
easy-rag CLI
Usage:
  python src/cli.py ingest --file data/sample_docs/my_doc.txt
  python src/cli.py ingest --folder data/sample_docs
  python src/cli.py ask "What is RAG?"
  python src/cli.py chat
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import RAGPipeline


def get_pipeline() -> RAGPipeline:
    provider = os.getenv("RAG_PROVIDER", "openai")
    llm_model = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
    store_path = os.getenv("RAG_STORE_PATH", "vector_store.json")

    if provider == "anthropic":
        llm_model = os.getenv("RAG_LLM_MODEL", "claude-3-5-haiku-20241022")
        embed_model = os.getenv("RAG_EMBED_MODEL", "voyage-3")

    print(f"🔧 Provider: {provider} | LLM: {llm_model} | Embed: {embed_model}")
    return RAGPipeline(
        provider=provider,
        llm_model=llm_model,
        embed_model=embed_model,
        store_path=store_path,
    )


def cmd_ingest(args):
    rag = get_pipeline()
    if args.file:
        path = Path(args.file)
        if path.suffix.lower() == ".pdf":
            rag.ingest_pdf(args.file)
        else:
            rag.ingest_file(args.file)
    elif args.folder:
        exts = args.extensions.split(",") if args.extensions else [".txt", ".md"]
        rag.ingest_folder(args.folder, extensions=exts)
    elif args.text:
        rag.ingest_text(args.text, source="cli-input")
    else:
        print("❌ Provide --file, --folder, or --text")


def cmd_ask(args):
    rag = get_pipeline()
    result = rag.ask(args.question)
    print("\n" + "="*60)
    print("🤖 Answer:\n")
    print(result["answer"])
    print("\n📚 Sources:", ", ".join(result["sources"]))
    print("="*60)


def cmd_chat(args):
    rag = get_pipeline()
    print("\n💬 RAG Chat Mode (type 'exit' to quit)\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        result = rag.ask(q)
        print(f"\n🤖 {result['answer']}")
        print(f"   📚 Sources: {', '.join(result['sources'])}\n")


def cmd_stats(args):
    from rag_engine import SimpleVectorStore
    import json
    store_path = os.getenv("RAG_STORE_PATH", "vector_store.json")
    store = SimpleVectorStore()
    store.load(store_path)
    sources = {}
    for doc in store.documents:
        src = doc["metadata"].get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\n📊 Vector Store Stats ({store_path})")
    print(f"   Total chunks: {len(store.documents)}")
    print("   Sources:")
    for src, count in sorted(sources.items()):
        print(f"     • {src}: {count} chunks")


def main():
    parser = argparse.ArgumentParser(
        prog="easy-rag",
        description="🔍 Easy RAG — Ingest documents, ask questions.",
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Add documents to the knowledge base")
    p_ingest.add_argument("--file", help="Path to a single file (.txt, .md, .pdf)")
    p_ingest.add_argument("--folder", help="Path to a folder of documents")
    p_ingest.add_argument("--text", help="Raw text string to ingest")
    p_ingest.add_argument("--extensions", help="Comma-separated extensions (default: .txt,.md)")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a one-off question")
    p_ask.add_argument("question", help="Your question")

    # chat
    sub.add_parser("chat", help="Interactive chat mode")

    # stats
    sub.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
