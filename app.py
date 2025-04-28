from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import time
from typing import List, Dict, Any, Optional
import logging
from embedding_functions import OllamaEmbeddingFunction
from ollamarag import OllamaRAG
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="OllamaRAG API",
    description="A FastAPI wrapper for PDF-based RAG using Ollama and ChromaDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize RAG system with default models
rag = OllamaRAG(llm_model="tinyllama", embedding_model="mxbai-embed-large")
rag.start_keep_alive_thread()

# Directory to store uploaded PDFs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    top_k: int = 5

class IngestRequest(BaseModel):
    collection: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 500
    max_workers: int = 4

class SwitchCollectionRequest(BaseModel):
    collection: str

class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str

class Collection(BaseModel):
    name: str
    document_count: int

class CollectionListResponse(BaseModel):
    collections: List[Collection]
    active_collection: Optional[str] = None

class StatsResponse(BaseModel):
    collection_name: str
    document_count: int
    llm_model: str
    embedding_model: str
    context_window: int
    source_documents: Optional[int] = None

class QueryResponse(BaseModel):
    response: str
    relevant_chunks: List[Dict[str, Any]]
    response_time: float

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and Ollama server are running."""
    return {
        "status": "ok",
        "llm_model": rag.llm_model,
        "embedding_model": rag.embedding_model
    }

@app.post("/ingest", status_code=202)
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Query(1000, ge=100, le=5000),
    chunk_overlap: int = Query(500, ge=0, le=2000),
    max_workers: int = Query(4, ge=1, le=16),
    collection: Optional[str] = None
):
    """
    Ingest a PDF document into the RAG system.
    This operation is performed asynchronously.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Get collection name
    if collection:
        collection_name = collection
    else:
        collection_name = os.path.splitext(file.filename)[0].replace(" ", "_")
    
    # Process PDF in background
    background_tasks.add_task(
        ingest_pdf_background,
        file_path, 
        collection_name,
        chunk_size, 
        chunk_overlap, 
        max_workers
    )
    
    return {
        "status": "processing",
        "message": f"PDF ingestion started for {file.filename}",
        "collection": collection_name
    }

async def ingest_pdf_background(
    file_path: str, 
    collection_name: str,
    chunk_size: int, 
    chunk_overlap: int, 
    max_workers: int
):
    """Background task to ingest a PDF."""
    try:
        # Set collection name from filename if needed
        rag.current_pdf = os.path.basename(file_path)
        
        # Ingest the PDF
        rag.ingest_pdf(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_workers=max_workers
        )
        
        logger.info(f"Successfully ingested {file_path} into collection {collection_name}")
    except Exception as e:
        logger.error(f"Error ingesting PDF {file_path}: {e}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    Returns the response and relevant chunks used to generate it.
    """
    # Switch collection if specified
    if request.collection:
        try:
            # Update current_pdf (needed for find_similar_chunks to work)
            rag.current_pdf = f"{request.collection}.pdf"
            rag.get_or_create_collection(request.collection)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found: {str(e)}")
    
    if not rag.current_pdf:
        raise HTTPException(status_code=400, detail="No collection active. Please ingest a PDF first or specify a collection.")
    
    # Generate response
    start_time = time.time()
    
    try:
        # Find relevant chunks
        relevant_chunks = rag.find_similar_chunks(request.query, top_k=request.top_k)
        
        if not relevant_chunks:
            return JSONResponse(
                status_code=404,
                content={"error": "No relevant information found. Please ingest a PDF document first."}
            )
        
        # Generate response
        response = rag.generate_response(request.query, top_k=request.top_k)
        response_time = time.time() - start_time
        
        # Format relevant chunks for the response
        formatted_chunks = []
        for chunk in relevant_chunks:
            formatted_chunks.append({
                "text": chunk["text"][:200] + "...",  # Truncate for response
                "source": os.path.basename(chunk["source"]),
                "similarity": chunk["similarity"]
            })
        
        return {
            "response": response,
            "relevant_chunks": formatted_chunks,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all available collections in ChromaDB."""
    try:
        collections = rag.chroma_client.list_collections()
        
        collection_list = []
        for collection in collections:
            try:
                count = collection.count()
                collection_list.append({"name": collection.name, "document_count": count})
            except Exception as e:
                logger.error(f"Error getting count for collection {collection.name}: {e}")
                collection_list.append({"name": collection.name, "document_count": -1})
        
        active_collection = None
        if rag.current_pdf:
            active_collection = os.path.splitext(rag.current_pdf)[0].replace(" ", "_")
        
        return {
            "collections": collection_list,
            "active_collection": active_collection
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.post("/collections/switch")
async def switch_collection(request: SwitchCollectionRequest):
    """Switch to a different collection."""
    try:
        # Check if collection exists
        collections = rag.chroma_client.list_collections()
        collection_names = [coll.name for coll in collections]
        
        if request.collection not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{request.collection}' not found")
        
        # Switch to the collection
        rag.get_or_create_collection(request.collection)
        
        # Update current_pdf (needed for find_similar_chunks to work)
        rag.current_pdf = f"{request.collection}.pdf"
        
        return {"status": "success", "collection": request.collection}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error switching to collection: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the current collection."""
    if not rag.current_pdf:
        raise HTTPException(status_code=400, detail="No collection currently active")
    
    collection_name = os.path.splitext(rag.current_pdf)[0].replace(" ", "_")
    
    try:
        # Get collection
        collection = rag.get_or_create_collection(collection_name)
        
        # Get collection count
        collection_count = collection.count()
        
        # Build stats response
        stats = {
            "collection_name": collection_name,
            "document_count": collection_count,
            "llm_model": rag.llm_model,
            "embedding_model": rag.embedding_model,
            "context_window": rag.context_window
        }
        
        # Try to get source document count
        try:
            collection_metadata = collection.get()
            if collection_metadata and "metadatas" in collection_metadata:
                sources = set()
                for metadata in collection_metadata["metadatas"]:
                    if "source" in metadata:
                        sources.add(metadata["source"])
                
                stats["source_documents"] = len(sources)
        except Exception as e:
            logger.warning(f"Error getting source documents: {e}")
        
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/config")
async def update_config(
    llm_model: Optional[str] = Body(None),
    embedding_model: Optional[str] = Body(None),
    context_window: Optional[int] = Body(None)
):
    """Update the RAG system configuration."""
    try:
        if llm_model:
            # Check if model exists before changing
            response = requests.get(f"{rag.base_url}/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            if f"{llm_model}:latest" not in available_models:
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"Model '{llm_model}' not available in Ollama. Please pull it first."}
                )
            
            rag.llm_model = llm_model
        
        if embedding_model:
            # Check if model exists before changing
            response = requests.get(f"{rag.base_url}/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            if f"{embedding_model}:latest" not in available_models:
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"Model '{embedding_model}' not available in Ollama. Please pull it first."}
                )
            
            rag.embedding_model = embedding_model
            # Recreate the embedding function
            rag.ollama_ef = OllamaEmbeddingFunction(
                base_url=rag.base_url,
                model_name=rag.embedding_model
            )
        
        if context_window:
            if context_window < 512 or context_window > 32768:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Context window must be between 512 and 32768"}
                )
            rag.context_window = context_window
        
        # Preload models after changes
        if llm_model or embedding_model:
            rag._preload_models()
        
        return {
            "status": "success",
            "config": {
                "llm_model": rag.llm_model,
                "embedding_model": rag.embedding_model,
                "context_window": rag.context_window
            }
        }
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)