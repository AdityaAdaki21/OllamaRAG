import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
from datetime import datetime
import time

from ollamarag import OllamaRAG
from ui_utils import display_help, display_stats, list_collections, switch_collection

console = Console()

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