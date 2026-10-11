import argparse
import sys
from loguru import logger

from src.utils.logging import setup_logging
from src.config import settings
from src.embeddings.embedder import LocalEmbedder
from src.database.vector_store import VectorStore, VALID_RETRIEVAL_MODES
from src.retrieval.retriever import Retriever
from src.generation.generator import get_generator


def print_citations(results: list[dict]):
    """Helper to cleanly print source citations from retrieval hits."""
    if not results:
        return

    print("\n" + "=" * 40)
    print("CITATIONS:")
    for i, hit in enumerate(results, 1):
        meta = hit.get("metadata", {})
        pdf = meta.get("source_pdf", "Unknown")
        page_start = meta.get("page_number", "?")
        page_end = meta.get("page_end", page_start)
        page_label = (
            f"{page_start}-{page_end}"
            if page_end is not None and page_end != page_start
            else f"{page_start}"
        )
        score = hit.get("score", 0.0)
        print(f"[{i}] {pdf} (Page {page_label}) - Score: {score:.3f}")
    print("=" * 40 + "\n")


def execute_query(query: str, retriever: Retriever, generator, mode: str = "hybrid_rerank"):
    """Executes a single RAG pipeline query with Two-Tier Abstention Gate."""
    print(f"\n🔍 Searching for context to answer: '{query}' (mode={mode})...")

    outcome = retriever.retrieve(query, initial_k=20, final_k=4, mode=mode)

    if outcome["abstained"]:
        reason = outcome["abstention_reason"]
        print(f"🛑 Abstained before LLM generation (reason={reason}).")
        print("\n" + "─" * 60)
        print("ANSWER: \n")
        print("I cannot answer this based on the provided documents.")
        print("─" * 60)
        return

    print("🤖 Generating answer...")
    answer = generator.generate(query, outcome["context"])

    print("\n" + "─" * 60)
    print("ANSWER: \n")
    print(answer)
    print("─" * 60)

    if "cannot answer this based on the provided documents" not in answer.lower():
        print_citations(outcome["results"])


def main():
    arg_parser = argparse.ArgumentParser(description="Query the RAG Pipeline")
    arg_parser.add_argument("-q", "--query", type=str, help="Single query string.")
    arg_parser.add_argument(
        "-p",
        "--provider",
        type=str,
        help="LLM Provider override ('lmstudio', 'ollama', or 'gemini'). Defaults to config.",
    )
    arg_parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="hybrid_rerank",
        choices=sorted(VALID_RETRIEVAL_MODES),
        help="Retrieval mode: vector_only, bm25_only, hybrid_rrf, or hybrid_rerank.",
    )

    args = arg_parser.parse_args()

    setup_logging()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("Initializing Embedder and Vector Store (may take a moment)...")
    embedder = LocalEmbedder()
    try:
        vector_store = VectorStore(embedder)
    except Exception as e:
        print(f"Error connecting to Vector Store: {e}")
        print("Ensure you have ingested PDFs first using ingest.py --save")
        sys.exit(1)

    retriever = Retriever(vector_store)

    provider = args.provider or settings.llm_provider
    try:
        generator = get_generator(provider)
    except Exception as e:
        print(f"Error initializing generator for '{provider}': {e}")
        sys.exit(1)

    print(f"\n✅ Ready! Connected to Vector Store and LLM: {provider.upper()} (mode={args.mode})")

    if args.query:
        execute_query(args.query, retriever, generator, mode=args.mode)
    else:
        print("Type 'exit' or 'quit' to close.")
        while True:
            try:
                user_q = input("\nQuery > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "\\q"]:
                    break

                execute_query(user_q, retriever, generator, mode=args.mode)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
