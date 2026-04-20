from src.embeddings.embedder import LocalEmbedder
from src.database.vector_store import VectorStore
import sys

try:
    print("Initializing embedder...")
    embedder = LocalEmbedder()
    print("Connecting to DB...")
    v = VectorStore(embedder)
    
    count = v.client.count("rag_documents").count
    print(f"Total points in collection: {count}")
    
    query = "Does the short sentence have punctuation?"
    print(f"Embedding query: '{query}'")
    query_emb = embedder.embed_text(query)
    print(f"Query embedding dimension: {len(query_emb)}")
    
    print("Calling Qdrant search...")
    res = v.client.search(
        collection_name="rag_documents",
        query_vector=query_emb,
        limit=5,
        with_payload=True
    )
    print(f"Search results count: {len(res)}")
    
    if res:
        for i, r in enumerate(res):
            print(f"[{i}] Score: {r.score} | Payload Keys: {list(r.payload.keys())}")
except Exception as e:
    print(f"FATAL ERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()

import os
os._exit(0) # bypass Qdrant background thread import error on exit
