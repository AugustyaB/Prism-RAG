from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from pathlib import Path

from src.config import settings
from src.embeddings.embedder import LocalEmbedder
from src.database.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import get_generator
from src.ingestion.pipeline import IngestionPipeline

app = FastAPI(title="RAG Pipeline API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold singletons
embedder = None
vector_store = None
retriever = None
ingestion_pipeline = None

@app.on_event("startup")
def startup_event():
    global embedder, vector_store, retriever, ingestion_pipeline
    print("Initializing embedder, database, and pipeline...")
    settings.ensure_dirs()
    embedder = LocalEmbedder()
    vector_store = VectorStore(embedder)
    retriever = Retriever(vector_store)
    ingestion_pipeline = IngestionPipeline()
    print("Startup complete.")

class QueryRequest(BaseModel):
    query: str
    provider: str | None = None

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # Save the uploaded file temporarily
    file_path = Path("data/raw") / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Processing uploaded PDF: {file.filename}")
        result = ingestion_pipeline.process_pdf(file_path)
        vector_store.add_document(result)
        return {
            "success": True, 
            "message": f"Successfully ingested {file.filename}",
            "chunks_added": result.total_chunks,
            "pages_processed": result.total_pages
        }
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_pipeline(req: QueryRequest):
    provider = req.provider or settings.llm_provider
    try:
        generator = get_generator(provider)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Generator init failed: {str(e)}")

    context, raw_results = retriever.retrieve_context(req.query, initial_k=20, final_k=4)
    if not context:
        return {"answer": "No matching context found in the vector database.", "citations": []}
        
    answer = generator.generate(req.query, context)
    
    citations = []
    for hit in raw_results:
        meta = hit.get("metadata", {})
        citations.append({
            "pdf": meta.get("source_pdf", "Unknown"),
            "page": meta.get("page_number", "?"),
            "score": round(hit.get("score", 0.0), 3)
        })
        
    return {"answer": answer, "citations": citations}

# Create static directory if it doesn't exist so mounting won't crash
os.makedirs("src/api/static", exist_ok=True)
app.mount("/", StaticFiles(directory="src/api/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)
