import os
import requests
import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
import time
import sys
import threading
import re
from datetime import datetime
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from io import BytesIO
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function for ChromaDB that uses Ollama."""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        results = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text,
                    "keep_alive": "5m"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Error getting embeddings: {response.text}")
            
            results.append(response.json()["embedding"])
        
        return results


class OllamaRAG:
    
    def __init__(self, llm_model="tinyllama", embedding_model="mxbai-embed-large"):
        """
        Initialize the RAG system with specified models and ChromaDB.
        
        Args:
            llm_model: The Ollama LLM model to use for generation
            embedding_model: The Ollama embedding model to use
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.current_pdf = None
        self.model_loaded = False
        self.context_window = 2048  # Default context window size, adjust for your model
        self.base_url = "http://localhost:11434/api"
        
        # Create a directory for ChromaDB persistence
        os.makedirs("chroma_db", exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        
        # Custom Ollama embedding function for ChromaDB
        self.ollama_ef = OllamaEmbeddingFunction(
            base_url=self.base_url,
            model_name=self.embedding_model
        )
        
        # Verify that the models are available in Ollama
        self._verify_models()
        
        # Preload models
        self._preload_models()
    
    def _verify_models(self) -> Dict[str, bool]:
        """
        Verify that the required models are available.
        
        Returns:
            Dictionary containing model availability status
        """
        try:
            response = requests.get(f"{self.base_url}/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            results = {
                "llm_available": self.llm_model in available_models,
                "embedder_available": self.embedding_model in available_models,
                "missing_models": []
            }
            
            if not results["llm_available"]:
                results["missing_models"].append(self.llm_model)
            if not results["embedder_available"]:
                results["missing_models"].append(self.embedding_model)
                
            return results
            
        except Exception as e:
            return {
                "error": f"Error connecting to Ollama server: {e}",
                "llm_available": False,
                "embedder_available": False,
                "missing_models": [self.llm_model, self.embedding_model]
            }
    
    def _preload_models(self) -> Dict[str, Any]:
        """
        Preload models into memory to keep them hot for fast inference.
        
        Returns:
            Dictionary with preload status information
        """
        results = {
            "embedding_model_loaded": False,
            "llm_model_loaded": False,
            "embedding_load_time": None,
            "llm_load_time": None,
            "errors": []
        }
        
        # Define a simple prompt for model loading
        warmup_prompt = "Hello, this is a warmup prompt to load the model into memory."
        
        # Preload embedding model
        try:
            start_time = time.time()
            # Just do a single embedding to load the model
            _ = self.get_embedding("This is a test.")
            results["embedding_load_time"] = time.time() - start_time
            results["embedding_model_loaded"] = True
        except Exception as e:
            results["errors"].append(f"Failed to preload embedding model: {e}")
        
        # Preload LLM model
        try:
            start_time = time.time()
            # Send a simple prompt to load the model
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.llm_model,
                    "prompt": warmup_prompt,
                    "stream": False,
                    "keep_alive": "5m"  # Keep model loaded for 5 minutes
                }
            )
            
            # Check if we got a successful response
            if response.status_code == 200:
                results["llm_load_time"] = time.time() - start_time
                results["llm_model_loaded"] = True
                self.model_loaded = True
            else:
                results["errors"].append(f"Failed to preload LLM model. Status code: {response.status_code}")
        except Exception as e:
            results["errors"].append(f"Failed to preload LLM model: {e}")
            
        return results
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings for a piece of text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of embedding values
        """
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text,
                "keep_alive": "5m"  # Keep model loaded for 5 minutes
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting embeddings: {response.text}")
        
        return response.json()["embedding"]
    
    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get a collection or create it if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            chromadb.Collection object
        """
        # First check if collection already exists
        try:
            collection_names = self.chroma_client.list_collections()
            
            if collection_name in collection_names:
                return self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            st.error(f"Error checking existing collections: {e}")
        
        # If collection doesn't exist, create a new one
        try:
            # Try to get the collection first
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                return collection
            except Exception:
                # Collection doesn't exist, so create it
                collection = self.chroma_client.create_collection(name=collection_name)
                return collection
        except Exception as e:
            # Try with custom embedding function
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.ollama_ef
                )
                return collection
            except Exception as e2:
                # Last resort: try with no arguments
                try:
                    collection = self.chroma_client.create_collection(name=collection_name)
                    return collection
                except Exception as e3:
                    raise Exception("All collection creation methods failed!")
                    
    def _process_chunk(self, chunk_info: Dict) -> Dict:
        """Process a single chunk (for parallel processing)"""
        chunk, pdf_path, chunk_num, total_chunks = chunk_info["chunk"], chunk_info["pdf_path"], chunk_info["chunk_num"], chunk_info["total_chunks"]
        
        try:
            # We don't need to compute embeddings here, ChromaDB will handle it
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}chunk{chunk_num}",
                "success": True
            }
        except Exception as e:
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}chunk{chunk_num}",
                "success": False, 
                "error": str(e)
            }
    
    def ingest_pdf(self, pdf_file, chunk_size: int = 1000, chunk_overlap: int = 200, 
                   max_workers: int = 4, progress_callback=None) -> Dict[str, Any]:
        """
        Ingest a PDF document, chunk it, and store in ChromaDB.
        
        Args:
            pdf_file: Uploaded PDF file (or path to PDF file)
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            max_workers: Maximum number of parallel workers for processing
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with ingestion statistics
        """
        start_time = time.time()
        
        # For Streamlit uploaded file
        if hasattr(pdf_file, 'name') and hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            pdf_path = pdf_file.name
            # Create a BytesIO object from the bytes
            from io import BytesIO
            pdf_file_obj = BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file_obj)
        else:
            # Regular file path
            pdf_path = pdf_file
            reader = PyPDF2.PdfReader(pdf_path)
        
        self.current_pdf = os.path.basename(pdf_path)
        collection_name = os.path.splitext(self.current_pdf)[0].replace(" ", "_")
        
        # Get or create a collection for this PDF
        collection = self.get_or_create_collection(collection_name)
        
        # Extract text from PDF
        text = ""
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            text += page.extract_text() + " "
            if progress_callback:
                progress_callback("text_extraction", i+1, total_pages)
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Only keep chunks with meaningful content
                chunks.append(chunk)
        
        # Prepare chunk information for parallel processing
        chunk_infos = [
            {"chunk": chunk, "pdf_path": pdf_path, "chunk_num": i+1, "total_chunks": len(chunks)}
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        successful_chunks = 0
        processed_chunks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk_idx = {executor.submit(self._process_chunk, chunk_info): i for i, chunk_info in enumerate(chunk_infos)}
            
            for future in concurrent.futures.as_completed(future_to_chunk_idx):
                result = future.result()
                if result["success"]:
                    processed_chunks.append(result)
                    successful_chunks += 1
                if progress_callback:
                    progress_callback("chunk_processing", successful_chunks, len(chunks))
        
        # Add chunks to ChromaDB collection
        batch_size = 100  # ChromaDB recommends batching for better performance
        batches_total = (len(processed_chunks) + batch_size - 1) // batch_size
        batches_completed = 0
        
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i:i+batch_size]
            
            # Prepare batch for ChromaDB
            ids = [chunk["id"] for chunk in batch]
            texts = [chunk["text"] for chunk in batch]
            metadatas = [{"source": chunk["source"]} for chunk in batch]
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            batches_completed += 1
            if progress_callback:
                progress_callback("db_insertion", batches_completed, batches_total)
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        return {
            "collection_name": collection_name,
            "total_chunks": len(chunks),
            "successful_chunks": successful_chunks,
            "success_rate": successful_chunks / len(chunks) * 100 if chunks else 0,
            "total_time": total_time,
            "time_per_chunk": total_time / len(chunks) if chunks else 0
        }
    
    def find_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find chunks most similar to the query using ChromaDB.
        
        Args:
            query: The query text
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks and their metadata
        """
        if not self.current_pdf:
            return []
            
        collection_name = os.path.splitext(self.current_pdf)[0].replace(" ", "_")
        
        try:
            collection = self.get_or_create_collection(collection_name)
        except Exception as e:
            st.error(f"Error accessing collection: {e}")
            return []
            
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "text": doc,
                    "source": results["metadatas"][0][i]["source"],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def _keep_models_alive(self) -> None:
        """Background thread to keep models loaded in memory."""
        while True:
            time.sleep(240)  # Send keep-alive every 4 minutes (since timeout is 5m)
            try:
                # Keep LLM alive
                requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": "keep alive",
                        "stream": False,
                        "keep_alive": "5m"
                    }
                )
                
                # Keep embedding model alive
                requests.post(
                    f"{self.base_url}/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": "keep alive",
                        "keep_alive": "5m"
                    }
                )
            except Exception:
                # Silently ignore errors during keep-alive
                pass
    
    def start_keep_alive_thread(self) -> None:
        """Start a background thread to keep models loaded."""
        keep_alive_thread = threading.Thread(target=self._keep_models_alive, daemon=True)
        keep_alive_thread.start()
    
    def generate_response(self, query: str, system_prompt: str = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Generate a response to the query using the LLM and retrieved context.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt to guide the LLM
            top_k: Number of relevant chunks to include in context
            
        Returns:
            Dictionary with the response and retrieval information
        """
        # Find relevant chunks
        start_time = time.time()
        relevant_chunks = self.find_similar_chunks(query, top_k=top_k)
        retrieval_time = time.time() - start_time
        
        if not relevant_chunks:
            return {
                "response": "No relevant information found. Please ingest a PDF document first.",
                "relevant_chunks": [],
                "total_time": 0,
                "retrieval_time": 0,
                "generation_time": 0
            }
        
        # Create context from retrieved chunks
        context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain relevant information, say you don't know."
            )
        
        # Prepare the prompt with context and query
        prompt = f"""Context information is below.
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the following question:
{query}
"""
        
        # Ensure we don't exceed the context window
        if len(prompt) > self.context_window:
            # Truncate context to fit within context window
            max_context_length = self.context_window - len(query) - 200  # Leave room for query and other text
            context_truncated = context[:max_context_length] + "..."
            prompt = f"""Context information is below (truncated to fit context window).
---------------------
{context_truncated}
---------------------

Given the context information and not prior knowledge, answer the following question:
{query}
"""
        
        # Send the request to Ollama
        generation_start = time.time()
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "keep_alive": "5m"  # Keep model loaded for 5 minutes
            }
        )
        
        if response.status_code != 200:
            return {
                "response": f"Error generating response: {response.text}",
                "relevant_chunks": relevant_chunks,
                "total_time": time.time() - start_time,
                "retrieval_time": retrieval_time,
                "generation_time": 0
            }
        
        response_text = response.json()["response"]
        generation_time = time.time() - generation_start
        
        return {
            "response": response_text,
            "relevant_chunks": relevant_chunks,
            "total_time": time.time() - start_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections in ChromaDB.
        
        Returns:
            List of dictionaries with collection information
        """
        results = []
        try:
            # In v0.6.0, list_collections() returns strings, not objects
            collection_names = self.chroma_client.list_collections()
            
            for name in collection_names:
                col = self.chroma_client.get_collection(name=name)
                count = col.count()
                results.append({
                    "name": name,
                    "documents": count
                })
                
            return results
        except Exception as e:
            st.error(f"Error listing collections: {e}")
            return []

    def switch_collection(self, collection_name: str) -> bool:
        """
        Switch to a different collection.
        
        Args:
            collection_name: Name of the collection to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_names = self.chroma_client.list_collections()
            
            if collection_name not in collection_names:
                return False
            
            # Set current PDF name based on collection
            self.current_pdf = f"{collection_name}.pdf"
            return True
        except Exception as e:
            st.error(f"Error switching collections: {e}")
            return False


# Streamlit UI
def main():
    st.set_page_config(
        page_title="OllamaRAG Explorer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🤖 OllamaRAG with ChromaDB")
    st.markdown("A local PDF Question-Answering System using Ollama and ChromaDB")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        # Default models
        default_llm = "tinyllama"
        default_embedder = "mxbai-embed-large"
        
        with st.spinner("Initializing OllamaRAG system..."):
            st.session_state.rag = OllamaRAG(
                llm_model=default_llm,
                embedding_model=default_embedder
            )
            st.session_state.rag.start_keep_alive_thread()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    
    rag = st.session_state.rag
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model settings
        with st.expander("Model Settings", expanded=False):
            llm_model = st.text_input("LLM Model", value=rag.llm_model)
            embedding_model = st.text_input("Embedding Model", value=rag.embedding_model)
            context_window = st.slider("Context Window Size", 512, 8192, rag.context_window, step=512)
            
            if st.button("Update Models"):
                with st.spinner("Updating models..."):
                    rag.llm_model = llm_model
                    rag.embedding_model = embedding_model
                    rag.context_window = context_window
                    
                    # Check model availability
                    model_status = rag._verify_models()
                    if model_status.get("missing_models"):
                        missing = ", ".join(model_status["missing_models"])
                        st.warning(f"The following models are not available: {missing}")
                        st.markdown(f"Pull them using: `ollama pull {missing}`")
                    else:
                        st.success("Models updated successfully!")
                        # Reload models
                        preload_status = rag._preload_models()
                        if preload_status["errors"]:
                            st.warning("\n".join(preload_status["errors"]))
        
        # Collection management
        with st.expander("Collections", expanded=True):
            collections = rag.list_collections()
            
            if collections:
                collection_names = [c["name"] for c in collections]
                selected_collection = st.selectbox(
                    "Select Collection", 
                    collection_names,
                    index=collection_names.index(st.session_state.current_collection) if st.session_state.current_collection in collection_names else 0
                )
                
                if st.button("Use Selected Collection"):
                    if rag.switch_collection(selected_collection):
                        st.session_state.current_collection = selected_collection
                        st.success(f"Switched to collection: {selected_collection}")
                    else:
                        st.error(f"Failed to switch to collection: {selected_collection}")
            else:
                st.info("No collections available. Upload a PDF to create one.")
        
        # PDF upload
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, step=50)
            max_workers = st.slider("Max Workers", 1, 8, 4)
            
            if st.button("Process PDF"):
                # Create progress bars
                text_progress = st.progress(0, "Extracting text...")
                chunk_progress = st.progress(0, "Processing chunks...")
                db_progress = st.progress(0, "Inserting into database...")
                
                # Progress callback function
                def update_progress(stage, current, total):
                    progress = current / total
                    if stage == "text_extraction":
                        text_progress.progress(progress, f"Extracting text: {current}/{total} pages")
                    elif stage == "chunk_processing":
                        chunk_progress.progress(progress, f"Processing chunks: {current}/{total}")
                    elif stage == "db_insertion":
                        db_progress.progress(progress, f"Database insertion: {current}/{total} batches")
                
                # Ingest PDF
                with st.spinner("Processing PDF..."):
                    try:
                        result = rag.ingest_pdf(
                            uploaded_file,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            max_workers=max_workers,
                            progress_callback=update_progress
                        )
                        
                        st.session_state.current_collection = result["collection_name"]
                        
                        # Display stats
                        st.success("PDF processed successfully!")
                        st.write("### Processing Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chunks Created", f"{result['total_chunks']}")
                            st.metric("Success Rate", f"{result['success_rate']:.1f}%")
                        with col2:
                            st.metric("Total Time", f"{result['total_time']:.2f}s")
                            st.metric("Time per Chunk", f"{result['time_per_chunk']:.2f}s")
                    
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
    
    # Main content area - Chat interface
    st.header("Chat with your PDF")
    
    # Display current collection
    if st.session_state.current_collection:
        st.info(f"Current collection: {st.session_state.current_collection}")
    else:
        st.warning("No collection selected. Please upload a PDF or select a collection from the sidebar.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display context for assistant messages
            if message["role"] == "assistant" and "context" in message:
                # Find this code in the chat history section
                with st.expander("View retrieved context"):
                    for i, chunk in enumerate(message["context"]):
                        st.markdown(f"**Source:** {os.path.basename(chunk['source'])} | **Similarity:** {chunk['similarity']:.3f}")
                        st.text_area(f"Context {i+1}", chunk["text"], height=100, key=f"history_context_{len(st.session_state.chat_history)}_{i}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                generation_result = rag.generate_response(prompt, top_k=3)
            
            response_placeholder.markdown(generation_result["response"])
            
            # Show timings and context
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Time", f"{generation_result['total_time']:.2f}s")
            col2.metric("Retrieval Time", f"{generation_result['retrieval_time']:.2f}s")
            col3.metric("Generation Time", f"{generation_result['generation_time']:.2f}s")
            
            # Find this code in the response section
            with st.expander("View retrieved context"):
                for i, chunk in enumerate(generation_result["relevant_chunks"]):
                    st.markdown(f"**Source:** {os.path.basename(chunk['source'])} | **Similarity:** {chunk['similarity']:.3f}")
                    st.text_area(f"Context {i+1}", chunk["text"], height=100, key=f"response_context_{len(st.session_state.chat_history)}_{i}")
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": generation_result["response"],
            "context": generation_result["relevant_chunks"]
        })

if __name__ == "__main__":
    main()