import os
import requests
import json
import PyPDF2
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
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
from rich.markdown import Markdown
from rich import box
import chromadb
from chromadb.utils import embedding_functions

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
                if self.llm_model not in available_models:
                    missing_models.append(self.llm_model)
                if self.embedding_model not in available_models:
                    missing_models.append(self.embedding_model)
                
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
    
    # Replace the get_or_create_collection method with this:
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
    
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, 
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
    
    def generate_response(self, query: str, system_prompt: str = None, top_k: int = 3) -> str:
        """
        Generate a response to the query using the LLM and retrieved context.
        
        Args:
            query: The user query
            system_prompt: Optional system prompt to guide the LLM
            top_k: Number of relevant chunks to include in context
            
        Returns:
            The generated response
        """
        # Find relevant chunks
        relevant_chunks = self.find_similar_chunks(query, top_k=top_k)
        
        if not relevant_chunks:
            return "No relevant information found. Please ingest a PDF document first."
        
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
        
        start_time = time.time()
        
        # Send the request to Ollama
        with console.status("[bold green]Generating response with LLM...[/bold green]") as status:
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
                return f"Error generating response: {response.text}"
            
            response_text = response.json()["response"]
            
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


def display_help():
    """Display help information."""
    help_panel = Panel(
        """[bold]Available Commands:[/bold]
        
• [cyan]<question>[/cyan] - Ask any question about the document
• [cyan]exit[/cyan], [cyan]quit[/cyan] - End the session
• [cyan]list[/cyan] - List available collections in ChromaDB
• [cyan]use <collection>[/cyan] - Switch to a different collection
• [cyan]help[/cyan] - Display this help message
• [cyan]stats[/cyan] - Display knowledge base statistics
• [cyan]clear[/cyan] - Clear the terminal screen""",
        title="[bold blue]OllamaRAG with ChromaDB Help[/bold blue]",
        border_style="blue"
    )
    console.print(help_panel)


def display_stats(rag):
    """Display knowledge base statistics."""
    if not rag.current_pdf:
        console.print("[yellow]No collection currently selected.[/yellow]")
        return
    
    collection_name = os.path.splitext(rag.current_pdf)[0]
    
    try:
        collection = rag.get_or_create_collection(collection_name)
        collection_stats = collection.count()
        
        stats_table = Table(title="ChromaDB Collection Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Collection", collection_name)
        stats_table.add_row("Document Count", str(collection_stats))
        stats_table.add_row("LLM Model", rag.llm_model)
        stats_table.add_row("Embedding Model", rag.embedding_model)
        
        console.print(stats_table)
    except Exception as e:
        console.print(f"[bold red]Error getting collection statistics:[/bold red] {e}")


def list_collections(rag):
    """List all available collections in ChromaDB."""
    try:
        collections = rag.chroma_client.get_collection()
        
        if not collections:
            console.print("[yellow]No collections found in ChromaDB.[/yellow]")
            return
        
        table = Table(title="Available Collections", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Documents", style="green", justify="right")
        
        for collection in collections:
            col = rag.chroma_client.get_collection(collection.name)
            count = col.count()
            table.add_row(collection.name, str(count))
        
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing collections:[/bold red] {e}")


def switch_collection(rag, collection_name):
    """Switch to a different collection."""
    try:
        collections = [c.name for c in rag.chroma_client.list_collections()]
        
        if collection_name not in collections:
            console.print(f"[bold red]Collection '{collection_name}' not found.[/bold red]")
            return
        
        # Set current PDF name based on collection
        rag.current_pdf = f"{collection_name}.pdf"
        
        collection = rag.get_or_create_collection(collection_name)
        count = collection.count()
        console.print(f"[bold green]Switched to collection '[blue]{collection_name}[/blue]' with [cyan]{count}[/cyan] documents.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error switching collections:[/bold red] {e}")


def main():
    parser = argparse.ArgumentParser(description="PDF-based RAG using Ollama and ChromaDB")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to ingest")
    parser.add_argument("--collection", type=str, help="ChromaDB collection to use")
    parser.add_argument("--llm", type=str, default="tinyllama", help="LLM model to use")
    parser.add_argument("--embedder", type=str, default="mxbai-embed-large", help="Embedding model to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for processing")
    parser.add_argument("--context-window", type=int, default=2048, help="Context window size for the LLM")
    
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = OllamaRAG(llm_model=args.llm, embedding_model=args.embedder)
    
    # Set context window size
    rag.context_window = args.context_window
    
    # Start the keep-alive thread
    rag.start_keep_alive_thread()
    
    # Either use existing collection or ingest a new PDF
    if args.collection:
        switch_collection(rag, args.collection)
    elif args.pdf:
        rag.ingest_pdf(args.pdf, max_workers=args.workers)
    else:
        console.print("[bold yellow]No PDF or collection specified. Use --pdf to ingest a document or --collection to use an existing one.[/bold yellow]")
        collections = rag.chroma_client.list_collections()
        if collections:
            console.print("[green]Available collections:[/green]")
            for collection in collections:
                console.print(f"  • [blue]{collection.name}[/blue]")
    
    # Display system ready message
    console.print(Panel.fit(
        f"[bold green]System ready with model:[/bold green] [blue]{args.llm}[/blue]\n"
        "[italic]Type [bold cyan]help[/bold cyan] for available commands[/italic]",
        border_style="green"
    ))
    
    # Interactive query loop
    while True:
        try:
            query = console.input("\n[bold blue]>[/bold blue] ")
            
            # Check for special commands
            if query.lower() in ['exit', 'quit']:
                console.print("[yellow]Exiting...[/yellow]")
                break
            elif query.lower() == 'help':
                display_help()
                continue
            elif query.lower() == 'stats':
                display_stats(rag)
                continue
            elif query.lower() == 'list':
                list_collections(rag)
                continue
            elif query.lower().startswith('use '):
                collection_name = query.lower()[4:].strip()
                switch_collection(rag, collection_name)
                continue
            elif query.lower() == 'clear':
                console.clear()
                continue
            elif not query.strip():
                continue
            
            # Generate and display response
            start_time = time.time()
            response = rag.generate_response(query)
            total_time = time.time() - start_time
            
            # Display the response in a nice format
            console.print(Panel(
                Markdown(response),
                title="[bold]Response[/bold]",
                border_style="green",
                expand=False
            ))
            
            console.print(f"[dim]Response time: {total_time:.2f}s[/dim]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation canceled by user[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Program terminated by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)