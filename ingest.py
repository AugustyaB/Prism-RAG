#!/usr/bin/env python
"""CLI entry point for the RAG data ingestion pipeline."""

import argparse
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import settings
from src.utils.logging import setup_logging
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: Ingest PDFs, extract text (with OCR), and semantically chunk."
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to a single PDF file or a directory containing PDFs."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Embed and save the ingested chunks to the Vector DB."
    )
    
    args = parser.parse_args()
    target_path = Path(args.target)
    save_to_db = args.save
    
    setup_logging()
    settings.ensure_dirs()
    
    pipeline = IngestionPipeline()
    
    vector_store = None
    if save_to_db:
        from src.embeddings.embedder import LocalEmbedder
        from src.database.vector_store import VectorStore
        print("Initializing Embedding Model and Vector DB...")
        embedder = LocalEmbedder()
        vector_store = VectorStore(embedder)
    
    if target_path.is_file():
        if target_path.suffix.lower() != ".pdf":
            print("Error: Target file is not a PDF.", file=sys.stderr)
            sys.exit(1)
        
        result = pipeline.process_pdf(target_path)
        _print_summary(result)
        
        if save_to_db and vector_store:
            vector_store.add_document(result)
            print(f"[OK] Saved {result.source_pdf} to Vector DB.")
        
    elif target_path.is_dir():
        results = pipeline.process_directory(target_path)
        print(f"\n--- Batch Run Summary ---")
        print(f"Processed {len(results)} files.")
        for res in results:
            _print_summary(res)
            if save_to_db and vector_store:
                vector_store.add_document(res)
        
        if save_to_db:
            print(f"[OK] Saved all batch documents to Vector DB.")
    else:
        print(f"Error: Path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)


def _print_summary(result):
    from src.ingestion.classifier import PageType
    print(f"\n[OK] Result for: {result.source_pdf}")
    print(f"  Pages:      {result.total_pages}")
    print(f"  Classified: {result.classification_summary[PageType.TEXT]} Text, "
          f"{result.classification_summary[PageType.IMAGE]} Image, "
          f"{result.classification_summary[PageType.MIXED]} Mixed")
    print(f"  Chunks:     {result.total_chunks}")
    if result.errors:
        print(f"  Errors:     {len(result.errors)}")


if __name__ == "__main__":
    main()
