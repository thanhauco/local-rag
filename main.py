"""
Local RAG System - CLI Utility

A command-line interface for interacting with the Local RAG System.
Supports document ingestion, querying, and interactive mode.
"""

import sys
import logging
from typing import Optional
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.markdown import Markdown

from src.pipeline import create_pipeline
from src.config import config

# Initialize rich console
console = Console()
logger = logging.getLogger("rag_cli")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Local RAG System - No Cloud Platforms, No IDE Magic."""
    pass


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--chunk-size", type=int, help="Override chunk size")
@click.option("--chunk-overlap", type=int, help="Override chunk overlap")
def ingest(source: str, chunk_size: Optional[int], chunk_overlap: Optional[int]):
    """Ingest documents from a file or directory."""
    console.print(f"[bold blue]Ingesting documents from:[/bold blue] {source}")
    
    pipeline = create_pipeline()
    try:
        with console.status("[bold green]Processing documents..."):
            stats = pipeline.ingest(
                source, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        
        table = Table(title="Ingestion Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        
        table.add_row("Documents Loaded", str(stats.documents_loaded))
        table.add_row("Chunks Created", str(stats.chunks_created))
        table.add_row("Vectors Indexed", str(stats.vectors_indexed))
        
        console.print(table)
        console.print("[bold green]Success![/bold green] Documents are ready for querying.")
        
    except Exception as e:
        console.print(f"[bold red]Error during ingestion:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--top-k", type=int, default=config.retrieval.top_k, help="Number of sources to retrieve")
def query(question: str, top_k: int):
    """Ask a question based on indexed documents."""
    pipeline = create_pipeline()
    pipeline.retrieval_manager.top_k = top_k
    
    try:
        with console.status("[bold green]Thinking..."):
            result = pipeline.query(question)
        
        console.print(Panel(
            Markdown(result.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            expand=False
        ))
        
        if result.sources:
            console.print("\n[bold dim]Sources:[/bold dim]")
            for source in result.sources:
                console.print(f"  [dim]- {source}[/dim]")
                
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@cli.command()
def interactive():
    """Start an interactive chat session."""
    pipeline = create_pipeline()
    
    console.print(Panel(
        "Welcome to the Local RAG System Interactive Mode!\n"
        "Type 'exit' or 'quit' to leave.",
        title="[bold blue]Interactive RAG[/bold blue]",
        border_style="blue"
    ))
    
    while True:
        try:
            question = console.input("\n[bold yellow]Query:[/bold yellow] ")
            
            if question.lower() in ["exit", "quit"]:
                break
                
            if not question.strip():
                continue
                
            with console.status("[bold green]Thinking..."):
                result = pipeline.query(question)
            
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Markdown(result.answer))
            
            if result.sources:
                console.print("\n[bold dim]Sources:[/bold dim]")
                for source in result.sources:
                    console.print(f"  [dim]- {source}[/dim]")
            
            console.print("[hr]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

    console.print("\n[bold blue]Goodbye![/bold blue]")


@cli.command()
def status():
    """Check the status of the RAG system."""
    pipeline = create_pipeline()
    stats = pipeline.get_stats()
    
    console.print("[bold blue]RAG System Status[/bold blue]")
    
    # Vector store stats
    vs_stats = stats["vector_store"]
    vs_table = Table(title="Vector Database (Pinecone)", show_header=False)
    for k, v in vs_stats.items():
        vs_table.add_row(k.replace("_", " ").title(), str(v))
    console.print(vs_table)
    
    # Model stats
    m_table = Table(title="Models", show_header=True, header_style="bold magenta")
    m_table.add_column("Type")
    m_table.add_column("Model Name")
    m_table.add_column("Config")
    
    emb = stats["embedding_model"]
    m_table.add_row("Embeddings", emb["model_name"], f"Dim: {emb['dimension']}, Device: {emb['device']}")
    
    llm = stats["llm_model"]
    m_table.add_row("LLM", llm["model_name"], f"Max Len: {llm['max_length']}, Temp: {llm['temperature']}")
    
    console.print(m_table)


@cli.command()
@click.confirmation_option(prompt="Are you sure you want to delete all vectors?")
def reset():
    """Delete all indexed vectors."""
    pipeline = create_pipeline()
    try:
        pipeline.reset()
        console.print("[bold green]Success![/bold green] Vector store has been cleared.")
    except Exception as e:
        console.print(f"[bold red]Error during reset:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
