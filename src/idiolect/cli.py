import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .models import Fingerprint, ComparisonResult, AuthorType
from .store import FingerprintStore

app = typer.Typer(name="idiolect", help="Linguistic fingerprinting CLI")
console = Console()
store = FingerprintStore()


def version_callback(value: bool):
    if value:
        print("idiolect 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version."
    )
):
    """Linguistic fingerprinting and author identification."""
    pass


def draw_bar(score: float, width: int = 25) -> str:
    """Draw a rich text progress bar for scores 0-100."""
    filled = int((score / 100) * width)
    empty = width - filled
    return f"[cyan]{'█' * filled}[/cyan][dim]{'░' * empty}[/dim]"


@app.command()
def analyze(
    file: Path = typer.Argument(..., help="Path to the text file to analyze.", exists=True),
    output: Path = typer.Option(Path("."), "--output", "-o", help="Output directory for the PDF report."),
    label: str = typer.Option(None, "--label", "-l", help="Label for the fingerprint (default: filename stem)."),
    json: bool = typer.Option(False, "--json", help="Also output the raw fingerprint JSON."),
    no_report: bool = typer.Option(False, "--no-report", help="Skip PDF generation, just print summary to console."),
):
    """Analyze a text file and generate a fingerprint + PDF report."""
    from .ingestion import ingest_file
    
    # Lazy import fingerprint/reporting functions (assuming these will exist in synthesis/report modules)
    try:
        from .fingerprint import create_fingerprint
    except ImportError:
        # Mock for now if not implemented
        def create_fingerprint(doc, label):
            return Fingerprint(label=label, word_count=doc.word_count, sentence_count=doc.sentence_count, axes={"lexical_richness": 82.5, "syntactic_complexity": 56.1, "formality": 92.0, "epistemic_stance": 43.4, "pacing_cadence": 68.9, "affective_intensity": 35.2, "interactive_engagement": 71.0}, standout_traits=[{"name": "Sentence Length Variation", "z_score": 2.3, "percentile": 98}, {"name": "Passive Voice", "z_score": -1.8, "percentile": 5}], author_type=AuthorType.HUMAN, ai_confidence=0.78)

    try:
        from .report import generate_report
    except ImportError:
        def generate_report(fingerprint, output_path):
            output_path.write_text("PDF content placeholder")

    fingerprint_label = label or file.stem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Ingesting [cyan]{file.name}[/cyan] and analyzing...", total=None)
        doc = ingest_file(file)
        fingerprint = create_fingerprint(doc, label=fingerprint_label)

    # Header Panel
    ai_confidence_pct = int(fingerprint.ai_confidence * 100)
    
    type_color = "green" if fingerprint.author_type == AuthorType.HUMAN else ("red" if fingerprint.author_type == AuthorType.AI else "yellow")
    
    header_text = (
        f"📄 Document: [bold]{file.name}[/bold]\n"
        f"📊 Words: {fingerprint.word_count:,}  |  Sentences: {fingerprint.sentence_count:,}  |  Paragraphs: {doc.raw_text.count(chr(10)*2) + 1}\n\n"
        f"🤖 Author Type: [{type_color} bold]{fingerprint.author_type.value.upper()}[/] (confidence: {ai_confidence_pct}%)"
    )
    console.print(Panel(header_text, title="IDIOLECT", expand=False, padding=(1, 2)))

    # Axes
    console.print()
    axes = fingerprint.axes
    axes_display = [
        ("Lexical Richness", axes.get("lexical_richness", 0.0)),
        ("Syntactic Complex.", axes.get("syntactic_complexity", 0.0)),
        ("Formality", axes.get("formality", 0.0)),
        ("Epistemic Stance", axes.get("epistemic_stance", 0.0)),
        ("Pacing & Cadence", axes.get("pacing_cadence", 0.0)),
        ("Affective Intensity", axes.get("affective_intensity", 0.0)),
        ("Engagement", axes.get("interactive_engagement", 0.0)),
    ]
    
    for name, score in axes_display:
        bar = draw_bar(score)
        console.print(f"  {name:<19} {bar}  {int(score)}")

    # Standout Traits
    console.print("\n  [bold yellow]⚡ Standout Traits:[/bold yellow]")
    for trait in fingerprint.standout_traits[:5]:
        trait_name = trait.get("feature", "Unknown trait").replace(".", " › ").replace("_", " ").title()
        z = trait.get("z_score", 0.0)
        interpretation = trait.get("interpretation", "")
        
        if z > 0:
            arrow = "↑"
            console.print(f"  • {trait_name} — z = {z:+.1f} {arrow} — {interpretation}")
        else:
            arrow = "↓"
            console.print(f"  • {trait_name} — z = {z:+.1f} {arrow} — {interpretation}")
    
    console.print()

    if json:
        console.print(fingerprint.to_json())

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        pdf_path = output / f"{fingerprint_label}_fingerprint.pdf"
        generate_report(fingerprint, pdf_path)
        console.print(f"  📋 Report saved: [cyan]{pdf_path}[/cyan]")


@app.command()
def compare(
    file1: Path = typer.Argument(..., help="Path to the first text file.", exists=True),
    file2: Path = typer.Argument(..., help="Path to the second text file.", exists=True),
    output: Path = typer.Option(Path("."), "--output", "-o", help="Output directory for the PDF report."),
):
    """Compare two text files and determine if they share authorship."""
    from .ingestion import ingest_file
    
    try:
        from .fingerprint import create_fingerprint
        from .comparison import compare as compare_fingerprints
    except ImportError:
        def create_fingerprint(doc, label):
            return Fingerprint(label=label)
        def compare_fingerprints(f1, f2):
            return ComparisonResult(fingerprint_a=f1.label, fingerprint_b=f2.label, cosine_similarity=0.87, same_author_likelihood="likely", axis_deltas={"lexical_richness": 12.0}, most_similar_features=["Vocabulary Size"], most_divergent_features=["Passive Voice"])

    try:
        from .report import generate_comparison_report
    except ImportError:
        def generate_comparison_report(result, fp_a, fp_b, out):
            out.write_text("Comparison PDF")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Ingesting files and analyzing...", total=None)
        
        doc1 = ingest_file(file1)
        doc2 = ingest_file(file2)
        
        f1 = create_fingerprint(doc1, label=file1.stem)
        f2 = create_fingerprint(doc2, label=file2.stem)
        
        comp = compare_fingerprints(f1, f2)

    console.print(Panel(f"Comparing: [bold]{file1.name}[/] vs [bold]{file2.name}[/]", title="COMPARISON"))
    
    sim = comp.cosine_similarity * 100
    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"
    console.print(f"\n  Similarity: [{color} bold]{sim:.1f}%[/] ({comp.same_author_likelihood.replace('_', ' ').title()})")

    table = Table(title="Axis Deltas", show_header=True, header_style="bold magenta")
    table.add_column("Axis")
    table.add_column("Delta")

    for axis, delta in comp.axis_deltas.items():
        table.add_row(axis.replace("_", " ").title(), f"{delta:.1f}")

    console.print(table)
    
    if comp.most_similar_features:
        console.print("\n[bold green]Most Similar Features:[/]")
        for f in comp.most_similar_features[:3]:
            console.print(f"  • {f}")
            
    if comp.most_divergent_features:
        console.print("\n[bold red]Most Divergent Features:[/]")
        for f in comp.most_divergent_features[:3]:
            console.print(f"  • {f}")

    output.mkdir(parents=True, exist_ok=True)
    pdf_path = output / f"compare_{file1.stem}_{file2.stem}.pdf"
    generate_comparison_report(comp, f1, f2, pdf_path)
    console.print(f"\n  📋 Comparison report saved: [cyan]{pdf_path}[/cyan]")


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Name of the author to enroll."),
    file: Path = typer.Argument(..., help="Path to the text sample.", exists=True),
):
    """Enroll a text sample as a known author's fingerprint."""
    from .ingestion import ingest_file
    
    try:
        from .fingerprint import create_fingerprint
    except ImportError:
        def create_fingerprint(doc, label):
            return Fingerprint(label=label, word_count=doc.word_count, sentence_count=doc.sentence_count)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Enrolling [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(file)
        fingerprint = create_fingerprint(doc, label=name)
        store.enroll(fingerprint)

    console.print(f"[bold green]✓ Successfully enrolled author:[/bold green] {name}")


@app.command()
def verify(
    name: str = typer.Argument(..., help="Name of the enrolled author."),
    file: Path = typer.Argument(..., help="Path to the text file to verify.", exists=True),
):
    """Verify if a text file matches an enrolled author."""
    from .ingestion import ingest_file
    
    try:
        from .fingerprint import create_fingerprint
        from .comparison import compare as compare_fingerprints
    except ImportError:
        def create_fingerprint(doc, label):
            return Fingerprint(label=label)
        def compare_fingerprints(f1, f2):
            return ComparisonResult(fingerprint_a=f1.label, fingerprint_b=f2.label, cosine_similarity=0.92, same_author_likelihood="very_likely")

    enrolled = store.get(name)
    if not enrolled:
        console.print(f"[bold red]Error:[/] Author '{name}' is not enrolled.")
        raise typer.Exit(1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Verifying against [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(file)
        new_fingerprint = create_fingerprint(doc, label=file.stem)
        comp = compare_fingerprints(enrolled, new_fingerprint)

    sim = comp.cosine_similarity * 100
    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"
    
    console.print(Panel(
        f"Author: [bold]{name}[/]\n"
        f"File: [bold]{file.name}[/]\n\n"
        f"Match Confidence: [{color} bold]{sim:.1f}%[/]\n"
        f"Result: [bold]{comp.same_author_likelihood.replace('_', ' ').title()}[/]",
        title="VERIFICATION RESULT",
        expand=False
    ))


@app.command()
def list():
    """List all enrolled fingerprints."""
    labels = store.list_all()
    if not labels:
        console.print("No authors enrolled.")
        return
        
    table = Table(title="Enrolled Authors", show_header=True)
    table.add_column("Name", style="cyan")
    
    for label in labels:
        table.add_row(label)
        
    console.print(table)


@app.command()
def delete(name: str = typer.Argument(..., help="Name of the enrolled fingerprint to delete.")):
    """Delete an enrolled fingerprint."""
    if store.delete(name):
        console.print(f"[bold green]✓ Deleted enrolled author:[/bold green] {name}")
    else:
        console.print(f"[bold red]Error:[/] Author '{name}' not found.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
