from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import Dict, Any

console = Console()

def display_help() -> None:
    """Display available commands and help information."""
    help_table = Table(title="Available Commands", box=box.SIMPLE, show_header=True)
    help_table.add_column("Command", style="cyan", no_wrap=True)
    help_table.add_column("Description", style="green")
    
    help_table.add_row("help", "Display this help message")
    help_table.add_row("exit, quit", "Exit the program")
    help_table.add_row("stats", "Display statistics about the current collection")
    help_table.add_row("list", "List all available collections")
    help_table.add_row("use <collection>", "Switch to the specified collection")
    help_table.add_row("clear", "Clear the console screen")
    help_table.add_row("<query text>", "Ask a question about the loaded document")
    
    console.print(help_table)
    
    # Additional usage information
    console.print("\n[bold]Usage Tips:[/bold]")
    console.print("• To ingest a PDF: [cyan]--pdf /path/to/document.pdf[/cyan]")
    console.print("• To change the LLM model: [cyan]--llm <model_name>[/cyan]")
    console.print("• To change embedding model: [cyan]--embedder <model_name>[/cyan]")

def display_stats(rag_system) -> None:
    """
    Display statistics about the current collection.
    
    Args:
        rag_system: The OllamaRAG instance
    """
    if not rag_system.current_pdf:
        console.print("[yellow]No collection currently active. Use --pdf to ingest a document or 'use <collection>' to select one.[/yellow]")
        return
    
    collection_name = None
    if rag_system.current_pdf:
        collection_name = rag_system.current_pdf.split('.')[0].replace(" ", "_")
    
    try:
        # Get collection
        collection = rag_system.get_or_create_collection(collection_name)
        
        # Get collection info
        collection_info = collection.count()
        
        # Display stats
        stats_table = Table(title=f"Collection Statistics: [blue]{collection_name}[/blue]", box=box.SIMPLE)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Document Count", str(collection_info))
        stats_table.add_row("LLM Model", rag_system.llm_model)
        stats_table.add_row("Embedding Model", rag_system.embedding_model)
        stats_table.add_row("Context Window", str(rag_system.context_window))
        
        # Additional collection metadata if available
        try:
            collection_metadata = collection.get()
            if collection_metadata and "metadatas" in collection_metadata:
                sources = set()
                for metadata in collection_metadata["metadatas"]:
                    if "source" in metadata:
                        sources.add(metadata["source"])
                
                stats_table.add_row("Source Documents", str(len(sources)))
        except Exception:
            # Ignore errors when fetching metadata
            pass
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[bold red]Error retrieving collection statistics:[/bold red] {e}")

def list_collections(rag_system) -> None:
    """
    List all available collections in ChromaDB.
    
    Args:
        rag_system: The OllamaRAG instance
    """
    try:
        collections = rag_system.chroma_client.list_collections()
        
        if not collections:
            console.print("[yellow]No collections found. Use --pdf to ingest a document first.[/yellow]")
            return
        
        # Create a table to display collections
        table = Table(title="Available Collections", box=box.SIMPLE)
        table.add_column("Name", style="cyan")
        table.add_column("Document Count", style="green", justify="right")
        
        # Add rows for each collection
        for collection in collections:
            try:
                count = collection.count()
                table.add_row(collection.name, str(count))
            except Exception:
                table.add_row(collection.name, "Error")
        
        console.print(table)
        
        # Show which collection is active
        if rag_system.current_pdf:
            active_collection = rag_system.current_pdf.split('.')[0].replace(" ", "_")
            console.print(f"\n[green]Active collection:[/green] [blue]{active_collection}[/blue]")
        
    except Exception as e:
        console.print(f"[bold red]Error listing collections:[/bold red] {e}")

def switch_collection(rag_system, collection_name: str) -> None:
    """
    Switch to a different collection.
    
    Args:
        rag_system: The OllamaRAG instance
        collection_name: Name of the collection to switch to
    """
    try:
        # Check if collection exists
        collections = rag_system.chroma_client.list_collections()
        collection_names = [coll.name for coll in collections]
        
        if collection_name not in collection_names:
            console.print(f"[bold red]Collection '{collection_name}' not found.[/bold red]")
            console.print("[yellow]Available collections:[/yellow]")
            for name in collection_names:
                console.print(f"  • [blue]{name}[/blue]")
            return
        
        # Switch to the collection
        rag_system.get_or_create_collection(collection_name)
        
        # Update current_pdf (needed for find_similar_chunks to work)
        # This is a bit of a hack but works with the current code structure
        rag_system.current_pdf = f"{collection_name}.pdf"
        
        console.print(f"[green]Switched to collection:[/green] [blue]{collection_name}[/blue]")
        
        # Show some stats about the collection
        collection = rag_system.get_or_create_collection(collection_name)
        count = collection.count()
        console.print(f"[dim]Collection contains {count} document chunks[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error switching collections:[/bold red] {e}")