import os
import requests
import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple
import concurrent.futures
import time
import sys
import threading
import re
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import chromadb

from embedding_functions import OllamaEmbeddingFunction

console = Console()

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
        self.context_window = 4096  # Default context window size, adjust for your model
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
        
        # Display welcome banner
        self._display_welcome_banner()
        
        # Verify that the models are available in Ollama
        self._verify_models()
        
        # Preload models
        self._preload_models()
    
    def _display_welcome_banner(self) -> None:
        """Display a welcome banner with system information."""
        console.print(Panel.fit(
            "[bold blue]OllamaRAG with ChromaDB[/bold blue] - [italic]Local PDF Question-Answering System[/italic]",
            border_style="blue",
            box=box.DOUBLE
        ))
        
        # Display datetime
        now = datetime.now()
        console.print(f"[dim]Session started: {now.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        console.print()
    
    def _verify_models(self) -> None:
        """Verify that the required models are available."""
        with console.status("[bold green]Checking available models...[/bold green]"):
            try:
                response = requests.get(f"{self.base_url}/tags")
                available_models = [model["name"] for model in response.json()["models"]]
                
                missing_models = []
                if f"{self.llm_model}:latest" not in available_models:
                    missing_models.append(f"{self.llm_model}:latest")
                if f"{self.embedding_model}:latest" not in available_models:
                    missing_models.append(f"{self.embedding_model}:latest")
                
                if missing_models:
                    console.print("[bold yellow]Warning:[/bold yellow] The following models are not available:")
                    for model in missing_models:
                        console.print(f"  • [yellow]{model}[/yellow]")
                    
                    console.print("\nPlease pull these models using:")
                    for model in missing_models:
                        console.print(f"  [cyan]ollama pull {model}[/cyan]")
                else:
                    console.print("[bold green]✓[/bold green] All required models are available")
            except Exception as e:
                console.print(f"[bold red]Error connecting to Ollama server:[/bold red] {e}")
                console.print("[yellow]Make sure the Ollama server is running with [bold]ollama serve[/bold][/yellow]")
                sys.exit(1)
    
    def _preload_models(self) -> None:
        """Preload models into memory to keep them hot for fast inference."""
        console.print("[bold]Preloading models into memory...[/bold]")
        
        # Define a simple prompt for model loading
        warmup_prompt = "Hello, this is a warmup prompt to load the model into memory."
        
        # Preload embedding model
        try:
            with console.status(f"[bold green]Loading embedding model '{self.embedding_model}'...[/bold green]"):
                start_time = time.time()
                # Just do a single embedding to load the model
                _ = self.get_embedding("This is a test.")
                duration = time.time() - start_time
                console.print(f"[green]✓[/green] Embedding model loaded in [bold]{duration:.2f}s[/bold]")
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] Failed to preload embedding model: {e}")
        
        # Preload LLM model
        try:
            with console.status(f"[bold green]Loading LLM model '{self.llm_model}'...[/bold green]"):
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
                    duration = time.time() - start_time
                    console.print(f"[green]✓[/green] LLM model loaded in [bold]{duration:.2f}s[/bold]")
                    self.model_loaded = True
                else:
                    console.print(f"[bold yellow]Warning:[/bold yellow] Failed to preload LLM model. Status code: {response.status_code}")
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] Failed to preload LLM model: {e}")
    
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
                console.print(f"[blue]Using existing collection:[/blue] {collection_name}")
                return self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            console.print(f"[dim]Error checking existing collections: {str(e)}[/dim]")
        
        # If collection doesn't exist, create a new one
        console.print(f"[bold blue]Creating fresh collection:[/bold blue] {collection_name}")
        
        try:
            # Try to get the collection first
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                console.print(f"[green]✓[/green] Using existing collection")
                return collection
            except Exception:
                # Collection doesn't exist, so create it
                collection = self.chroma_client.create_collection(name=collection_name)
                console.print(f"[green]✓[/green] Created collection with default embeddings")
                return collection
        except Exception as e:
            console.print(f"[bold red]Error creating collection with default embeddings:[/bold red] {e}")
            
            # Try with custom embedding function
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.ollama_ef
                )
                console.print(f"[green]✓[/green] Created collection with custom embeddings")
                return collection
            except Exception as e2:
                console.print(f"[bold red]Error creating collection with custom embeddings:[/bold red] {e2}")
                
                # Last resort: try with no arguments
                try:
                    collection = self.chroma_client.create_collection(name=collection_name)
                    console.print(f"[green]✓[/green] Created basic collection")
                    return collection
                except Exception as e3:
                    console.print(f"[bold red]All collection creation methods failed![/bold red]")
                    raise e3
                
    def _process_chunk(self, chunk_info: Dict) -> Dict:
        """Process a single chunk (for parallel processing)"""
        chunk, pdf_path, chunk_num, total_chunks = chunk_info["chunk"], chunk_info["pdf_path"], chunk_info["chunk_num"], chunk_info["total_chunks"]
        
        try:
            # We don't need to compute embeddings here, ChromaDB will handle it
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}_chunk_{chunk_num}",
                "success": True
            }
        except Exception as e:
            return {
                "text": chunk, 
                "source": pdf_path, 
                "id": f"{os.path.basename(pdf_path)}_chunk_{chunk_num}",
                "success": False, 
                "error": str(e)
            }
    
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 500, 
                   max_workers: int = 4) -> None:
        """
        Ingest a PDF document, chunk it, and store in ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            max_workers: Maximum number of parallel workers for processing
        """
        self.current_pdf = os.path.basename(pdf_path)
        collection_name = os.path.splitext(self.current_pdf)[0].replace(" ", "_")

        
        console.print(f"\n[bold blue]Ingesting PDF:[/bold blue] [italic]{pdf_path}[/italic]")
        start_time = time.time()
        
        # Get or create a collection for this PDF
        collection = self.get_or_create_collection(collection_name)
        
        # Extract text from PDF
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                total_pages = len(reader.pages)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Extracting text[/bold blue]"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("Page {task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("[blue]Extracting...", total=total_pages)
                    for i, page in enumerate(reader.pages):
                        text += page.extract_text() + " "
                        progress.update(task, advance=1)
                
                console.print("[green]✓[/green] Text extraction complete")
        except Exception as e:
            console.print(f"[bold red]Error reading PDF:[/bold red] {e}")
            return
        
        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:  # Only keep chunks with meaningful content
                chunks.append(chunk)
        
        console.print(f"[green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks from the PDF")
        
        # Prepare chunk information for parallel processing
        chunk_infos = [
            {"chunk": chunk, "pdf_path": pdf_path, "chunk_num": i+1, "total_chunks": len(chunks)}
            for i, chunk in enumerate(chunks)
        ]
        
        # Process chunks in parallel
        console.print("[bold]Processing chunks in parallel...[/bold]")
        successful_chunks = 0
        processed_chunks = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing chunks[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Chunk {task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[blue]Processing...", total=len(chunks))
            
            # Process chunks in parallel with progress updates
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk_idx = {executor.submit(self._process_chunk, chunk_info): i for i, chunk_info in enumerate(chunk_infos)}
                
                for future in concurrent.futures.as_completed(future_to_chunk_idx):
                    result = future.result()
                    if result["success"]:
                        processed_chunks.append(result)
                        successful_chunks += 1
                    progress.update(task, advance=1)
        
        # Add chunks to ChromaDB collection
        with console.status("[bold green]Adding chunks to ChromaDB...[/bold green]"):
            batch_size = 100  # ChromaDB recommends batching for better performance
            
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
        
        # Display summary
        total_time = time.time() - start_time
        
        stats_table = Table.grid(padding=(0, 1))
        stats_table.add_row("[bold]Collection name:", f"[blue]{collection_name}[/blue]")
        stats_table.add_row("[bold]Successful chunks:", f"[green]{successful_chunks}[/green]/[blue]{len(chunks)}[/blue]")
        stats_table.add_row("[bold]Success rate:", f"[green]{successful_chunks/len(chunks)*100:.1f}%[/green]")
        stats_table.add_row("[bold]Total time:", f"[blue]{total_time:.2f}s[/blue]")
        stats_table.add_row("[bold]Time per chunk:", f"[blue]{total_time/len(chunks):.2f}s[/blue]")
        
        console.print(Panel(stats_table, title="[bold]Ingestion Summary[/bold]", border_style="green"))
    
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
            console.print(f"[bold red]Error accessing collection:[/bold red] {e}")
            return []
            
        with console.status("[bold green]Finding relevant context...[/bold green]"):
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
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "model": self.llm_model,
                        "prompt": "keep alive",
                        "stream": False,
                        "keep_alive": "5m"
                    }
                )
                
                # Keep embedding model alive
                response = requests.post(
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
        console.print("[dim]Started background thread to keep models loaded[/dim]")
    
    def generate_response(self, query: str, system_prompt: str = None, top_k: int = 5) -> str:
        """
        Generate a response to the query using the LLM and retrieved context.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt to guide the LLM
            top_k: Number of relevant chunks to include in context
            
        Returns:
            The generated response
        """
        # Find relevant chunks with increased top_k for better context
        relevant_chunks = self.find_similar_chunks(query, top_k=top_k)
        
        if not relevant_chunks:
            return "No relevant information found. Please ingest a PDF document first."
        
        # Sort chunks by similarity score in descending order
        relevant_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Create context from retrieved chunks with chunk markers
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_text = chunk["text"].strip()
            # Add chunk markers to help the model distinguish between sources
            context_parts.append(f"[Document {i+1}] {chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Improved system prompt with better instructions
        if not system_prompt:
            system_prompt = (
                "You are a precise and helpful research assistant. Your task is to answer the user's question "
                "based solely on the provided context information. Follow these guidelines:\n"
                "1. Only use information explicitly stated in the context.\n"
                "2. If the context doesn't contain the answer, say 'Based on the provided information, I cannot answer this question.'\n"
                "3. Do not introduce external knowledge or assumptions.\n"
                "4. If information is partially available, provide what you can find and acknowledge the limitations.\n"
                "5. If the user asks for a specific format (list, table, etc.), follow that format.\n"
                "6. Be concise but thorough. Aim for comprehensive accuracy.\n"
                "7. If the context contains conflicting information, acknowledge the conflict and present both perspectives.\n"
                "8. Cite the document numbers ([Document X]) when referring to specific information."
            )
        
        # Improved prompt template with query-focused instructions
        prompt = f"""## Context Information
    {context}

    ## Question
    {query}

    ## Instructions
    - Answer the question thoroughly based ONLY on the context provided above.
    - If you need to make an inference from the context, clearly indicate it as such.
    - Format your answer clearly and logically.
    - Include relevant details from the context to support your answer.
    - Cite specific document sections when appropriate using [Document X] notation.
    """
        
        # Ensure we don't exceed the context window
        if len(prompt) > self.context_window:
            # More strategic truncation - keep high similarity chunks intact
            current_length = len(prompt)
            target_length = self.context_window - 500  # Leave room for other parts
            
            # Start removing lower-ranked chunks until we fit
            while current_length > target_length and len(context_parts) > 1:
                # Remove the last (lowest similarity) chunk
                removed_chunk = context_parts.pop()
                # Regenerate the context
                context = "\n\n".join(context_parts)
                # Recalculate the prompt
                prompt = f"""## Context Information
    {context}

    ## Question
    {query}

    ## Instructions
    - Answer the question thoroughly based ONLY on the context provided above.
    - If you need to make an inference from the context, clearly indicate it as such.
    - Format your answer clearly and logically.
    - Include relevant details from the context to support your answer.
    - Cite specific document sections when appropriate using [Document X] notation.

    Note: Some less relevant context was removed to fit within the model's context window.
    """
                current_length = len(prompt)
        
        start_time = time.time()
        
        # Send the request to Ollama with temperature control
        with console.status("[bold green]Generating response with LLM...[/bold green]") as status:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "temperature": 0.2,  # Lower temperature for more factual responses
                    "top_p": 0.9,        # Better quality generation
                    "keep_alive": "5m"   # Keep model loaded for 5 minutes
                }
            )
            
            if response.status_code != 200:
                return f"Error generating response: {response.text}"
            
            response_text = response.json()["response"]
            
            # Post-process the response
            response_text = self._post_process_response(response_text)
            
            # Record stats
            total_time = time.time() - start_time
            
            # Display retrieval statistics
            retrieval_table = Table(box=box.SIMPLE)
            retrieval_table.add_column("Similarity", style="cyan", justify="center")
            retrieval_table.add_column("Source", style="green")
            retrieval_table.add_column("Preview", style="blue")
            
            for i, chunk in enumerate(relevant_chunks):
                preview = chunk["text"][:50].replace("\n", " ") + "..."
                retrieval_table.add_row(
                    f"{chunk['similarity']:.3f}",
                    os.path.basename(chunk["source"]),
                    preview
                )
            
            console.print("\n[bold]Retrieval Details:[/bold]", style="yellow")
            console.print(retrieval_table)
            console.print(f"[dim]Response generated in {total_time:.2f} seconds[/dim]")
        
        return response_text

    def _post_process_response(self, text: str) -> str:
        """
        Post-process the model's response for better formatting and clarity.
        
        Args:
            text: The raw response text from the LLM
            
        Returns:
            Processed response text
        """
        # Remove any preamble like "Based on the context provided,"
        text = re.sub(r"^(Based on (the|your) (context|information|document|documents) provided,?\s*)", "", text)
        
        # Remove redundant document citations if they're overly repetitive
        doc_citation_count = len(re.findall(r"\[Document \d+\]", text))
        if doc_citation_count > 10:  # If there are too many citations
            # Try to make citations more concise
            text = re.sub(r"(\[Document \d+\])[,\s]*(\[Document \d+\])", r"\1,\2", text)
        
        # Make sure the answer is properly formatted in Markdown
        if "```" in text and not text.strip().startswith("```"):
            text = text.replace("```", "\n```")
        
        # Ensure the answer has appropriate paragraph breaks
        if len(text) > 200 and "\n\n" not in text:
            # Add paragraph breaks for readability
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 3:
                # Group sentences into paragraphs (3-4 sentences per paragraph)
                paragraphs = []
                for i in range(0, len(sentences), 3):
                    paragraph = " ".join(sentences[i:i+3])
                    paragraphs.append(paragraph)
                text = "\n\n".join(paragraphs)
        
        return text.strip()