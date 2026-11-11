import json
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.syntax import Syntax

from toolkit.config import setup_logger
from toolkit.summarizer import TextSummarizer
from toolkit.classifier import TextClassifier
from toolkit.extractor import InfoExtractor
from toolkit.schemas import ToolkitResponse, ProcessingMetadata
from toolkit.rag import SimpleRAGPipeline

app = typer.Typer(
    help="AI Text Processing Toolkit - CLI for NLP Summarization, Classification, and Extraction",
    no_args_is_help=True
)
console = Console()
logger = setup_logger("cli")


def _read_input(text: Optional[str], file: Optional[Path]) -> str:
    """Helper to get text from either raw string or a file path."""
    if text:
        return text
    if file and file.is_file():
        return file.read_text(encoding="utf-8")
    raise ValueError("You must provide either direct text (--text) or a valid file path (--file).")


def _emit_output(response: ToolkitResponse, output_path: Optional[Path] = None):
    """Outputs structured JSON to console and optionally to a file."""
    data_dict = response.model_dump()
    json_str = json.dumps(data_dict, indent=2)

    console.print("\n[bold green]Result:[/bold green]")
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    console.print(syntax)

    if output_path:
        output_path.write_text(json_str, encoding="utf-8")
        console.print(f"\n[cyan]Output saved to: {output_path.resolve()}[/cyan]")


@app.command("summarize")
def summarize(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Raw input text to summarize"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to text file"),
    max_length: int = typer.Option(130, "--max-len", help="Maximum summary token length"),
    min_length: int = typer.Option(30, "--min-len", help="Minimum summary token length"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write JSON output")
):
    """Generate abstractive summary of input text with token & latency tracking."""
    try:
        content = _read_input(text, file)
        logger.info(f"Summarizing text (length: {len(content)} chars)")
        summarizer = TextSummarizer()
        res = summarizer.summarize(content, max_length=max_length, min_length=min_length)
        _emit_output(res, output)
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("classify")
def classify(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to classify"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to text file"),
    labels: List[str] = typer.Option(..., "--label", "-l", help="Candidate classification labels"),
    multi_label: bool = typer.Option(False, "--multi-label", help="Allow multiple labels with sigmoid probabilities"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write JSON output")
):
    """Run Zero-Shot classification on text against candidate labels."""
    try:
        content = _read_input(text, file)
        logger.info(f"Classifying text against labels: {labels}")
        classifier = TextClassifier()
        res = classifier.classify(content, candidate_labels=labels, multi_label=multi_label)
        _emit_output(res, output)
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command("extract")
def extract(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to extract entities from"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Path to text file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write JSON output")
):
    """Extract named entities (PER, ORG, LOC, MISC) and group them cleanly."""
    try:
        content = _read_input(text, file)
        logger.info(f"Extracting entities from text (length: {len(content)} chars)")
        extractor = InfoExtractor()
        res = extractor.extract_entities(content)
        _emit_output(res, output)
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("ask")
def ask(
    query: str = typer.Option(..., "--query", "-q", help="The question to ask"),
    context_file: Path = typer.Option(..., "--context", "-c", help="Path to text or knowledge file"),
    top_k: int = typer.Option(2, "--top-k", "-k", help="Number of context chunks to retrieve"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to write JSON output")
):
    """Run Retrieval-Augmented Generation (RAG) over a local text document."""
    try:
        if not context_file.is_file():
            raise FileNotFoundError(f"Context file '{context_file}' does not exist.")

        content = context_file.read_text(encoding="utf-8")
        logger.info(f"Indexing context file: {context_file} ({len(content)} chars)")

        rag_pipeline = SimpleRAGPipeline()
        rag_pipeline.index_documents([content])

        logger.info(f"Running RAG query: '{query}'")
        res = rag_pipeline.query(query, top_k=top_k)
        _emit_output(res, output)
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)        


if __name__ == "__main__":
    app()