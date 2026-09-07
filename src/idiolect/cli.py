from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .comparison import compare as compare_fingerprints
from .fingerprint import create_fingerprint
from .ingestion import ingest_file
from .models import AuthorType
from .report import generate_comparison_report, generate_report
from .store import FingerprintStore

app = typer.Typer(name="idiolect", help="Linguistic fingerprinting CLI")
console = Console()


def get_store(db_path: Path | None = None) -> FingerprintStore:
    """Lazily instantiate FingerprintStore to avoid filesystem mutations on import."""
    return FingerprintStore(db_path=db_path) if db_path else FingerprintStore()


def version_callback(value: bool):
    if value:
        print("idiolect 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version."
    ),
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
    file: Path = typer.Argument(
        ...,
        help="Path to the text file to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("."), "--output", "-o", help="Output directory for the PDF report."
    ),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Label for the fingerprint (default: filename stem)."
    ),
    json: bool = typer.Option(False, "--json", help="Also output the raw fingerprint JSON."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip PDF generation, just print summary to console."
    ),
):
    """Analyze a text file and generate a fingerprint + PDF report."""
    fingerprint_label = label or file.stem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(
            description=f"Ingesting [cyan]{file.name}[/cyan] and analyzing...", total=None
        )
        doc = ingest_file(file)
        fingerprint = create_fingerprint(doc, label=fingerprint_label)

    # Header Panel
    ai_confidence_pct = int(fingerprint.ai_confidence * 100)
    type_color = (
        "green"
        if fingerprint.author_type == AuthorType.HUMAN
        else ("red" if fingerprint.author_type == AuthorType.AI else "yellow")
    )

    paragraphs = len([p for p in doc.cleaned_text.split("\n\n") if p.strip()]) or 1

    header_text = (
        f"📄 Document: [bold]{file.name}[/bold]\n"
        f"📊 Words: {fingerprint.word_count:,}  |  "
        f"Sentences: {fingerprint.sentence_count:,}  |  "
        f"Paragraphs: {paragraphs}\n\n"
        f"🤖 Author Type: [{type_color} bold]{fingerprint.author_type.value.upper()}[/] "
        f"(confidence: {ai_confidence_pct}%)"
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
        trait_name = (
            trait.get("feature", "Unknown trait").replace(".", " › ").replace("_", " ").title()
        )
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
    file1: Path = typer.Argument(
        ...,
        help="Path to the first text file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    file2: Path = typer.Argument(
        ...,
        help="Path to the second text file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("."), "--output", "-o", help="Output directory for the PDF report."
    ),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip PDF report generation, only print summary."
    ),
):
    """Compare two text files and determine if they share authorship."""
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description="Ingesting files and analyzing...", total=None)

        doc1 = ingest_file(file1)
        doc2 = ingest_file(file2)

        f1 = create_fingerprint(doc1, label=file1.stem)
        f2 = create_fingerprint(doc2, label=file2.stem)

        comp = compare_fingerprints(f1, f2)

    console.print(
        Panel(f"Comparing: [bold]{file1.name}[/] vs [bold]{file2.name}[/]", title="COMPARISON")
    )

    sim = comp.cosine_similarity * 100
    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"
    console.print(
        f"\n  Similarity: [{color} bold]{sim:.1f}%[/] "
        f"({comp.same_author_likelihood.replace('_', ' ').title()})"
    )

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

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        pdf_path = output / f"compare_{file1.stem}_{file2.stem}.pdf"
        generate_comparison_report(comp, f1, f2, pdf_path)
        console.print(f"\n  📋 Comparison report saved: [cyan]{pdf_path}[/cyan]")


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Name of the author to enroll."),
    file: Path = typer.Argument(
        ...,
        help="Path to the text sample.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Enroll a text sample as a known author's fingerprint."""
    store = get_store()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description=f"Enrolling [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(file)
        fingerprint = create_fingerprint(doc, label=name)
        store.enroll(fingerprint)

    console.print(f"[bold green]✓ Successfully enrolled author:[/bold green] {name}")


@app.command()
def verify(
    name: str = typer.Argument(..., help="Name of the enrolled author."),
    file: Path = typer.Argument(
        ...,
        help="Path to the text file to verify.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Verify if a text file matches an enrolled author."""
    store = get_store()
    enrolled = store.get(name)
    if not enrolled:
        console.print(f"[bold red]Error:[/] Author '{name}' is not enrolled.")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description=f"Verifying against [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(file)
        new_fingerprint = create_fingerprint(doc, label=file.stem)
        comp = compare_fingerprints(enrolled, new_fingerprint)

    sim = comp.cosine_similarity * 100
    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"

    console.print(
        Panel(
            f"Author: [bold]{name}[/]\n"
            f"File: [bold]{file.name}[/]\n\n"
            f"Match Confidence: [{color} bold]{sim:.1f}%[/]\n"
            f"Result: [bold]{comp.same_author_likelihood.replace('_', ' ').title()}[/]",
            title="VERIFICATION RESULT",
            expand=False,
        )
    )


@app.command(name="list")
def list_fingerprints():
    """List all enrolled fingerprints."""
    store = get_store()
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
    store = get_store()
    if store.delete(name):
        console.print(f"[bold green]✓ Deleted enrolled author:[/bold green] {name}")
    else:
        console.print(f"[bold red]Error:[/] Author '{name}' not found.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
