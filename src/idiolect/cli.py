import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .comparison import compare as compare_fingerprints
from .explainability import (
    MIN_RELIABLE_WORDS,
    SHORT_DOC_WARNING,
    compute_length_damping,
    explain_aligning_traits,
    format_aligning_traits_summary,
)
from .fingerprint import create_fingerprint
from .ingestion import find_text_files, ingest_file
from .models import AuthorType
from .profiling import calculate_sample_weights, classify_stability, compute_profile_consistency
from .report import generate_comparison_report, generate_report
from .store import FingerprintStore

app = typer.Typer(name="idiolect", help="Linguistic fingerprinting CLI")
console = Console()


def print_csv_rows(headers: list[str], rows: list[list[Any]]) -> None:
    """Print clean CSV formatted data to stdout."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    print(output.getvalue().strip())


def print_json_data(data: Any) -> None:
    """Print formatted JSON data to stdout."""
    print(json.dumps(data, indent=2))


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
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to a text file or directory of text files to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("artifacts"),
        "--output",
        "-o",
        help="Output directory for the PDF report(s) (default: 'artifacts').",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", "-l", help="Label for the fingerprint (single file mode only)."
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip PDF generation, just print summary to console."
    ),
):
    """Analyze a text file or directory of files and generate fingerprint + PDF reports."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

        batch_results = []
        skipped_files: list[tuple[Path, str]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=(out_format != "table"),
        ) as progress:
            task = progress.add_task(f"Analyzing {len(files)} documents...", total=len(files))
            for f in files:
                progress.update(task, description=f"Analyzing [cyan]{f.name}[/cyan]...")
                try:
                    doc = ingest_file(f)
                    fp = create_fingerprint(doc, label=f.stem)

                    if not no_report:
                        output.mkdir(parents=True, exist_ok=True)
                        pdf_path = output / f"{f.stem}_fingerprint.pdf"
                        generate_report(fp, pdf_path)

                    batch_results.append((f, doc, fp))
                except Exception as err:
                    skipped_files.append((f, str(err)))
                finally:
                    progress.advance(task)

        if not batch_results:
            console.print(f"[bold red]Error:[/] Could not analyze any documents in: {path}")
            if skipped_files:
                for f, err in skipped_files:
                    console.print(f"  • [yellow]{f.name}:[/] {err}")
            raise typer.Exit(1)

        if out_format == "json":
            batch_data = [
                {
                    "document": f.name,
                    "path": str(f),
                    "word_count": fp.word_count,
                    "short_document": fp.word_count < MIN_RELIABLE_WORDS,
                    "length_warning": (
                        SHORT_DOC_WARNING if fp.word_count < MIN_RELIABLE_WORDS else None
                    ),
                    "sentence_count": fp.sentence_count,
                    "author_type": fp.author_type.value,
                    "ai_confidence": round(fp.ai_confidence, 4),
                    "reading_ease": round(
                        fp.features.get("readability.flesch_reading_ease", 60.0), 2
                    ),
                    "lexical_richness": round(fp.axes.get("lexical_richness", 0.0), 2),
                    "syntactic_complexity": round(fp.axes.get("syntactic_complexity", 0.0), 2),
                    "axes": {k: round(v, 2) for k, v in fp.axes.items()},
                }
                for f, _doc, fp in batch_results
            ]
            print_json_data(batch_data)
            return

        if out_format == "csv":
            csv_headers = [
                "document",
                "words",
                "short_doc",
                "sentences",
                "author_type",
                "ai_confidence",
                "reading_ease",
                "lexical_richness",
                "syntactic_complexity",
            ]
            csv_rows = [
                [
                    f.name,
                    fp.word_count,
                    "yes" if fp.word_count < MIN_RELIABLE_WORDS else "no",
                    fp.sentence_count,
                    fp.author_type.value,
                    f"{fp.ai_confidence * 100:.1f}%",
                    f"{fp.features.get('readability.flesch_reading_ease', 60.0):.1f}",
                    f"{fp.axes.get('lexical_richness', 0.0):.1f}",
                    f"{fp.axes.get('syntactic_complexity', 0.0):.1f}",
                ]
                for f, _doc, fp in batch_results
            ]
            print_csv_rows(csv_headers, csv_rows)
            return

        table = Table(
            title=f"Batch Linguistic Analysis ({len(files)} Documents)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Document", style="cyan", width=24)
        table.add_column("Words", justify="right", width=9)
        table.add_column("Author Type", justify="center", width=14)
        table.add_column("AI Conf.", justify="right", width=10)
        table.add_column("Reading Ease", justify="right", width=13)
        table.add_column("Lex. Richness", justify="right", width=14)
        table.add_column("Synt. Complex.", justify="right", width=14)

        human_count = 0
        ai_count = 0
        uncertain_count = 0
        total_words = 0
        short_count = 0

        for f, doc, fp in batch_results:
            total_words += fp.word_count
            is_s = fp.word_count < MIN_RELIABLE_WORDS
            if is_s:
                short_count += 1
                words_str = f"[yellow]{fp.word_count:,} ⚠️[/yellow]"
            else:
                words_str = f"{fp.word_count:,}"

            if fp.author_type == AuthorType.HUMAN:
                human_count += 1
                type_badge = "[bold green]HUMAN[/bold green]"
            elif fp.author_type == AuthorType.AI:
                ai_count += 1
                type_badge = "[bold red]AI[/bold red]"
            else:
                uncertain_count += 1
                type_badge = "[bold yellow]UNCERTAIN[/bold yellow]"

            ai_conf_str = f"{fp.ai_confidence * 100:.1f}%"
            re = fp.features.get("readability.flesch_reading_ease", 60.0)
            re_str = f"{re:.1f}"
            lr = fp.axes.get("lexical_richness", 0.0)
            sc = fp.axes.get("syntactic_complexity", 0.0)

            table.add_row(
                f.name,
                words_str,
                type_badge,
                ai_conf_str,
                re_str,
                f"{lr:.1f}",
                f"{sc:.1f}",
            )

        console.print(table)

        short_note = (
            f"\n⚠️  [yellow]{short_count} short document(s) (<250 words) detected; "
            f"stylometric metrics exhibit higher variance.[/yellow]"
            if short_count > 0
            else ""
        )
        skip_note = (
            f"\n⚠️  [yellow]Skipped {len(skipped_files)} unreadable/corrupted file(s)[/yellow]"
            if skipped_files
            else ""
        )
        summary_text = (
            f"📂 Processed Directory: [bold]{path}[/bold]\n"
            f"📄 Documents Analyzed: {len(batch_results)}  |  Total Words: {total_words:,}\n\n"
            f"Classification Breakdown: [green bold]{human_count} Human[/], "
            f"[red bold]{ai_count} AI[/], [yellow bold]{uncertain_count} Uncertain[/]"
            f"{short_note}"
            f"{skip_note}"
        )
        if not no_report:
            summary_text += f"\n📋 PDF Reports Saved: [cyan]{output}[/cyan]"

        console.print(
            Panel(summary_text, title="BATCH ANALYSIS COMPLETE", expand=False, padding=(1, 2))
        )
        return

    # Single-file mode
    fingerprint_label = label or path.stem

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=(out_format != "table"),
    ) as progress:
        progress.add_task(
            description=f"Ingesting [cyan]{path.name}[/cyan] and analyzing...", total=None
        )
        doc = ingest_file(path)
        fingerprint = create_fingerprint(doc, label=fingerprint_label)

    paragraphs = len([p for p in doc.cleaned_text.split("\n\n") if p.strip()]) or 1

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        pdf_path = output / f"{fingerprint_label}_fingerprint.pdf"
        generate_report(fingerprint, pdf_path)

    is_short = fingerprint.word_count < MIN_RELIABLE_WORDS

    if out_format == "json":
        fp_dict = fingerprint.to_dict()
        fp_dict["short_document"] = is_short
        fp_dict["length_warning"] = SHORT_DOC_WARNING if is_short else None
        print_json_data(fp_dict)
        return

    if out_format == "csv":
        csv_headers = [
            "document",
            "words",
            "short_doc",
            "sentences",
            "paragraphs",
            "author_type",
            "ai_confidence",
            "lexical_richness",
            "syntactic_complexity",
            "formality",
            "epistemic_stance",
            "pacing_cadence",
            "affective_intensity",
            "interactive_engagement",
        ]
        csv_rows = [
            [
                path.name,
                fingerprint.word_count,
                "yes" if is_short else "no",
                fingerprint.sentence_count,
                paragraphs,
                fingerprint.author_type.value,
                f"{fingerprint.ai_confidence * 100:.1f}%",
                f"{fingerprint.axes.get('lexical_richness', 0.0):.1f}",
                f"{fingerprint.axes.get('syntactic_complexity', 0.0):.1f}",
                f"{fingerprint.axes.get('formality', 0.0):.1f}",
                f"{fingerprint.axes.get('epistemic_stance', 0.0):.1f}",
                f"{fingerprint.axes.get('pacing_cadence', 0.0):.1f}",
                f"{fingerprint.axes.get('affective_intensity', 0.0):.1f}",
                f"{fingerprint.axes.get('interactive_engagement', 0.0):.1f}",
            ]
        ]
        print_csv_rows(csv_headers, csv_rows)
        return

    if is_short:
        console.print(
            Panel(
                f"⚠️  [bold yellow]Short Document Notice "
                f"({fingerprint.word_count} words < 250 words)[/bold yellow]\n"
                f"[dim]{SHORT_DOC_WARNING}[/dim]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    # Header Panel
    ai_confidence_pct = int(fingerprint.ai_confidence * 100)
    type_color = (
        "green"
        if fingerprint.author_type == AuthorType.HUMAN
        else ("red" if fingerprint.author_type == AuthorType.AI else "yellow")
    )

    header_text = (
        f"📄 Document: [bold]{path.name}[/bold]\n"
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

    if not no_report:
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
        Path("artifacts"),
        "--output",
        "-o",
        help="Output directory for the PDF report (default: 'artifacts').",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip PDF report generation, only print summary."
    ),
):
    """Compare two text files and determine if they share authorship."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=(out_format != "table"),
    ) as progress:
        progress.add_task(description="Ingesting files and analyzing...", total=None)

        doc1 = ingest_file(file1)
        doc2 = ingest_file(file2)

        f1 = create_fingerprint(doc1, label=file1.stem)
        f2 = create_fingerprint(doc2, label=file2.stem)

        comp = compare_fingerprints(f1, f2)

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        pdf_path = output / f"compare_{file1.stem}_{file2.stem}.pdf"
        generate_comparison_report(comp, f1, f2, pdf_path)

    f1_short = f1.word_count < MIN_RELIABLE_WORDS
    f2_short = f2.word_count < MIN_RELIABLE_WORDS
    any_short = f1_short or f2_short

    if out_format == "json":
        data = {
            "file1": file1.name,
            "file2": file2.name,
            "file1_words": f1.word_count,
            "file2_words": f2.word_count,
            "short_document": any_short,
            "length_warning": SHORT_DOC_WARNING if any_short else None,
            "similarity": round(comp.cosine_similarity * 100, 2),
            "burrows_delta": round(comp.manhattan_delta, 4),
            "verdict": comp.same_author_likelihood.replace("_", " ").title(),
            "axis_deltas": {k: round(v, 2) for k, v in comp.axis_deltas.items()},
            "most_similar_features": comp.most_similar_features,
            "most_divergent_features": comp.most_divergent_features,
        }
        print_json_data(data)
        return

    if out_format == "csv":
        headers = ["axis", "delta"]
        rows = [
            [axis.replace("_", " ").title(), f"{delta:.2f}"]
            for axis, delta in comp.axis_deltas.items()
        ]
        print_csv_rows(headers, rows)
        return

    if any_short:
        short_names = []
        if f1_short:
            short_names.append(f"{file1.name} ({f1.word_count} words)")
        if f2_short:
            short_names.append(f"{file2.name} ({f2.word_count} words)")
        names_str = ", ".join(short_names)
        console.print(
            Panel(
                f"⚠️  [bold yellow]Short Document Notice "
                f"({names_str} < 250 words)[/bold yellow]\n"
                f"[dim]{SHORT_DOC_WARNING}[/dim]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

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
        console.print(f"\n  📋 Comparison report saved: [cyan]{pdf_path}[/cyan]")


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Name of the author to enroll."),
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to a text sample file or directory of text samples.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        "-r",
        help="Clear prior samples and start a fresh baseline profile.",
    ),
    decay: float = typer.Option(
        0.90,
        "--decay",
        help="Exponential decay factor for weighted rolling average (0.1 to 1.0).",
    ),
):
    """Enroll sample(s) into an author's multi-sample profile via weighted rolling average."""
    store = get_store()

    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

        samples = []
        skipped_files: list[tuple[Path, str]] = []
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
        ) as progress:
            task = progress.add_task(
                f"Ingesting {len(files)} samples for [cyan]{name}[/cyan]...", total=len(files)
            )
            for f in files:
                progress.update(task, description=f"Analyzing [cyan]{f.name}[/cyan]...")
                try:
                    doc = ingest_file(f)
                    fp = create_fingerprint(doc, label=f.stem)
                    samples.append((f.name, fp))
                except Exception as err:
                    skipped_files.append((f, str(err)))
                finally:
                    progress.advance(task)

            if not samples:
                console.print(f"[bold red]Error:[/] No samples could be ingested from: {path}")
                if skipped_files:
                    for f, err in skipped_files:
                        console.print(f"  • [yellow]{f.name}:[/] {err}")
                raise typer.Exit(1)

            composite_fp = store.enroll_samples(
                author_label=name,
                samples=samples,
                replace=replace,
                recency_decay=decay,
            )

        added_words = sum(fp.word_count for _, fp in samples)
        skip_line = (
            f"\n  ⚠️  Skipped [yellow]{len(skipped_files)}[/yellow] unreadable file(s)"
            if skipped_files
            else ""
        )
        console.print(
            f"[bold green]✓ Successfully enrolled author profile:[/bold green] "
            f"[bold cyan]{name}[/bold cyan]\n"
            f"  📂 Ingested {len(samples)} samples ({added_words:,} words added)\n"
            f"  📈 Rolling Baseline: [bold]{composite_fp.sample_count} samples[/bold]  |  "
            f"[bold]{composite_fp.word_count:,} total words[/bold] (decay={decay:.2f})"
            f"{skip_line}"
        )
        return

    # Single-file mode
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(
            description=f"Enrolling sample from [cyan]{path.name}[/cyan]...", total=None
        )
        doc = ingest_file(path)
        fp = create_fingerprint(doc, label=path.stem)
        composite_fp = store.enroll_sample(
            author_label=name,
            fingerprint=fp,
            sample_label=path.name,
            replace=replace,
            recency_decay=decay,
        )

    if composite_fp.sample_count > 1 and not replace:
        console.print(
            f"[bold green]✓ Added sample to author profile:[/bold green] "
            f"[bold cyan]{name}[/bold cyan]\n"
            f"  📄 New Sample: [cyan]{path.name}[/cyan] ({fp.word_count:,} words)\n"
            f"  📈 Updated Rolling Baseline: [bold]{composite_fp.sample_count} samples[/bold]  |  "
            f"[bold]{composite_fp.word_count:,} total words[/bold] (decay={decay:.2f})"
        )
    else:
        console.print(
            f"[bold green]✓ Successfully enrolled author:[/bold green] "
            f"[bold cyan]{name}[/bold cyan]\n"
            f"  📄 Sample #1: [cyan]{path.name}[/cyan] ({fp.word_count:,} words)"
        )


@app.command()
def verify(
    name: str = typer.Argument(..., help="Name of the enrolled author."),
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to a text file or directory of text files to verify.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
):
    """Verify if a text file or directory of files matches an enrolled author."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    store = get_store()
    enrolled = store.get(name)
    if not enrolled:
        console.print(f"[bold red]Error:[/] Author '{name}' is not enrolled.")
        raise typer.Exit(1)

    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

        batch_results = []
        skipped_files: list[tuple[Path, str]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=(out_format != "table"),
        ) as progress:
            task = progress.add_task(
                f"Verifying {len(files)} documents against {name}...", total=len(files)
            )
            for f in files:
                progress.update(task, description=f"Verifying [cyan]{f.name}[/cyan]...")
                try:
                    doc = ingest_file(f)
                    new_fingerprint = create_fingerprint(doc, label=f.stem)
                    comp = compare_fingerprints(enrolled, new_fingerprint)
                    batch_results.append((f, doc, new_fingerprint, comp))
                except Exception as err:
                    skipped_files.append((f, str(err)))
                finally:
                    progress.advance(task)

        if not batch_results:
            console.print(f"[bold red]Error:[/] Could not verify any documents in: {path}")
            if skipped_files:
                for f, err in skipped_files:
                    console.print(f"  • [yellow]{f.name}:[/] {err}")
            raise typer.Exit(1)

        if out_format == "json":
            batch_data = [
                {
                    "document": f.name,
                    "path": str(f),
                    "author": name,
                    "words": fp.word_count,
                    "short_document": fp.word_count < MIN_RELIABLE_WORDS,
                    "length_warning": (
                        SHORT_DOC_WARNING if fp.word_count < MIN_RELIABLE_WORDS else None
                    ),
                    "confidence": round(
                        comp.cosine_similarity * 100 * compute_length_damping(fp.word_count), 2
                    ),
                    "raw_confidence": round(comp.cosine_similarity * 100, 2),
                    "burrows_delta": round(comp.manhattan_delta, 4),
                    "verdict": comp.same_author_likelihood.replace("_", " ").title(),
                }
                for f, _doc, fp, comp in batch_results
            ]
            print_json_data(batch_data)
            return

        if out_format == "csv":
            csv_headers = [
                "document",
                "author",
                "words",
                "short_doc",
                "confidence",
                "burrows_delta",
                "verdict",
            ]
            csv_rows = [
                [
                    f.name,
                    name,
                    fp.word_count,
                    "yes" if fp.word_count < MIN_RELIABLE_WORDS else "no",
                    f"{comp.cosine_similarity * 100 * compute_length_damping(fp.word_count):.1f}%",
                    f"{comp.manhattan_delta:.3f}",
                    comp.same_author_likelihood.replace("_", " ").title(),
                ]
                for f, _doc, fp, comp in batch_results
            ]
            print_csv_rows(csv_headers, csv_rows)
            return

        table = Table(
            title=f"Batch Author Verification Against '{name}' ({len(files)} Documents)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Document", style="cyan", width=26)
        table.add_column("Words", justify="right", width=9)
        table.add_column("Match Confidence", justify="left", width=22)
        table.add_column("Burrows Δ", justify="right", width=12)
        table.add_column("Verdict", justify="left", width=18)

        matched_count = 0
        short_count = 0
        for f, doc, fp, comp in batch_results:
            is_s = fp.word_count < MIN_RELIABLE_WORDS
            if is_s:
                short_count += 1
                words_str = f"[yellow]{fp.word_count:,} ⚠️[/yellow]"
            else:
                words_str = f"{fp.word_count:,}"

            sim = comp.cosine_similarity * 100 * compute_length_damping(fp.word_count)
            bar = draw_bar(sim, width=12)
            sim_str = f"{sim:5.1f}% {bar}"
            delta_str = f"{comp.manhattan_delta:.3f}"
            verdict = comp.same_author_likelihood.replace("_", " ").title()
            if sim >= 80:
                verdict_styled = f"[bold green]{verdict}[/bold green]"
                matched_count += 1
            elif sim >= 60:
                verdict_styled = f"[bold yellow]{verdict}[/bold yellow]"
            else:
                verdict_styled = f"[bold red]{verdict}[/bold red]"

            table.add_row(f.name, words_str, sim_str, delta_str, verdict_styled)

        console.print(table)
        sample_info = (
            f" ({enrolled.sample_count} samples rolling baseline)"
            if enrolled.sample_count > 1
            else ""
        )
        short_note = (
            f"\n⚠️  [yellow]{short_count} short document(s) (<250 words) detected; "
            f"match confidence was proportionally damped.[/yellow]"
            if short_count > 0
            else ""
        )
        skip_note = (
            f"\n⚠️  [yellow]Skipped {len(skipped_files)} unreadable/corrupted file(s)[/yellow]"
            if skipped_files
            else ""
        )
        console.print(
            Panel(
                f"👤 Enrolled Author: [bold]{name}[/bold]{sample_info}\n"
                f"📂 Directory: [bold]{path}[/bold] ({len(batch_results)} documents)\n"
                f"🎯 Strong Matches: [bold green]{matched_count}/{len(batch_results)}[/bold green]"
                f"{short_note}"
                f"{skip_note}",
                title="BATCH VERIFICATION COMPLETE",
                expand=False,
                padding=(1, 2),
            )
        )
        return

    # Single-file mode
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=(out_format != "table"),
    ) as progress:
        progress.add_task(description=f"Verifying against [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(path)
        new_fingerprint = create_fingerprint(doc, label=path.stem)
        comp = compare_fingerprints(enrolled, new_fingerprint)

    is_short = new_fingerprint.word_count < MIN_RELIABLE_WORDS
    damping = compute_length_damping(new_fingerprint.word_count)
    raw_sim = comp.cosine_similarity * 100
    sim = raw_sim * damping

    if out_format == "json":
        data = {
            "document": path.name,
            "path": str(path),
            "author": name,
            "words": new_fingerprint.word_count,
            "short_document": is_short,
            "length_warning": SHORT_DOC_WARNING if is_short else None,
            "confidence": round(sim, 2),
            "raw_confidence": round(raw_sim, 2) if is_short else round(sim, 2),
            "burrows_delta": round(comp.manhattan_delta, 4),
            "verdict": comp.same_author_likelihood.replace("_", " ").title(),
        }
        print_json_data(data)
        return

    if out_format == "csv":
        csv_headers = [
            "document",
            "author",
            "words",
            "short_doc",
            "confidence",
            "burrows_delta",
            "verdict",
        ]
        csv_rows = [
            [
                path.name,
                name,
                new_fingerprint.word_count,
                "yes" if is_short else "no",
                f"{sim:.1f}%",
                f"{comp.manhattan_delta:.3f}",
                comp.same_author_likelihood.replace("_", " ").title(),
            ]
        ]
        print_csv_rows(csv_headers, csv_rows)
        return

    if is_short:
        console.print(
            Panel(
                f"⚠️  [bold yellow]Short Document Notice "
                f"({new_fingerprint.word_count} words < 250 words)[/bold yellow]\n"
                f"[dim]Stylometric metrics have higher variance on brief texts. "
                f"Match confidence has been proportionally damped.[/dim]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"
    sample_info = (
        f" ({enrolled.sample_count} samples rolling baseline)" if enrolled.sample_count > 1 else ""
    )
    if is_short:
        conf_line = (
            f"Match Confidence: [{color} bold]{sim:.1f}%[/] "
            f"[dim](damped from {raw_sim:.1f}% due to length)[/dim]"
        )
    else:
        conf_line = f"Match Confidence: [{color} bold]{sim:.1f}%[/]"

    console.print(
        Panel(
            f"Author: [bold]{name}[/]{sample_info}\n"
            f"File: [bold]{path.name}[/] ({new_fingerprint.word_count:,} words)\n\n"
            f"{conf_line}\n"
            f"Result: [bold]{comp.same_author_likelihood.replace('_', ' ').title()}[/]",
            title="VERIFICATION RESULT",
            expand=False,
        )
    )


@app.command()
def identify(
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to an essay text file or directory of submissions to identify.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Maximum number of candidate matches to display."
    ),
    output: Path = typer.Option(
        Path("artifacts"),
        "--output",
        "-o",
        help="Output directory for the PDF comparison report(s) (default: 'artifacts').",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip generating PDF comparison report(s)."
    ),
):
    """Identify which enrolled student/author wrote an essay based on stylometric similarity."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    store = get_store()
    enrolled_candidates = store.get_all()

    if not enrolled_candidates:
        console.print("[bold red]Error:[/] No students or authors enrolled in the database.")
        console.print(
            "Enroll candidates first using: [cyan]idiolect enroll <name> <sample.txt>[/cyan]"
        )
        raise typer.Exit(1)

    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

        batch_identifications = []
        skipped_files: list[tuple[Path, str]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=(out_format != "table"),
        ) as progress:
            task = progress.add_task(
                f"Identifying authors for {len(files)} submissions...", total=len(files)
            )
            for f in files:
                progress.update(task, description=f"Evaluating [cyan]{f.name}[/cyan]...")
                try:
                    doc = ingest_file(f)
                    essay_fp = create_fingerprint(doc, label=f.stem)

                    candidate_scores = []
                    for cand_fp in enrolled_candidates:
                        comp = compare_fingerprints(cand_fp, essay_fp)
                        candidate_scores.append((cand_fp, comp))

                    candidate_scores.sort(key=lambda x: x[1].cosine_similarity, reverse=True)
                    best_cand_fp, best_comp = candidate_scores[0]
                    raw_sim_pct = best_comp.cosine_similarity * 100

                    # Length check & proportional damping
                    is_short = essay_fp.word_count < MIN_RELIABLE_WORDS
                    damping = compute_length_damping(essay_fp.word_count)
                    damped_sim_pct = raw_sim_pct * damping

                    margin = None
                    if len(candidate_scores) > 1:
                        runner_up_fp, runner_up_comp = candidate_scores[1]
                        runner_up_pct = runner_up_comp.cosine_similarity * 100 * damping
                        margin = damped_sim_pct - runner_up_pct

                    # Explainability: top 3 aligning traits
                    aligning_traits = explain_aligning_traits(best_cand_fp, essay_fp, top_n=3)
                    traits_summary = format_aligning_traits_summary(aligning_traits)

                    if not no_report:
                        output.mkdir(parents=True, exist_ok=True)
                        safe_name = best_cand_fp.label.replace(" ", "_")
                        pdf_path = output / f"identify_{f.stem}_{safe_name}.pdf"
                        generate_comparison_report(best_comp, best_cand_fp, essay_fp, pdf_path)

                    batch_identifications.append(
                        (
                            f,
                            essay_fp,
                            best_cand_fp,
                            best_comp,
                            margin,
                            is_short,
                            raw_sim_pct,
                            damped_sim_pct,
                            aligning_traits,
                            traits_summary,
                        )
                    )
                except Exception as err:
                    skipped_files.append((f, str(err)))
                finally:
                    progress.advance(task)

        if not batch_identifications:
            console.print(f"[bold red]Error:[/] Could not identify any submissions in: {path}")
            if skipped_files:
                for f, err in skipped_files:
                    console.print(f"  • [yellow]{f.name}:[/] {err}")
            raise typer.Exit(1)

        if out_format == "json":
            batch_data = [
                {
                    "submission": f.name,
                    "path": str(f),
                    "word_count": essay_fp.word_count,
                    "short_document": is_short,
                    "length_warning": SHORT_DOC_WARNING if is_short else None,
                    "top_match": best_cand_fp.label,
                    "sample_count": best_cand_fp.sample_count,
                    "confidence": round(damped_sim_pct, 2),
                    "raw_confidence": (
                        round(raw_sim_pct, 2) if is_short else round(damped_sim_pct, 2)
                    ),
                    "burrows_delta": round(best_comp.manhattan_delta, 4),
                    "verdict": best_comp.same_author_likelihood.replace("_", " ").title(),
                    "lead_margin": round(margin, 2) if margin is not None else None,
                    "top_aligning_traits": [t["description"] for t in aligning_traits],
                }
                for (
                    f,
                    essay_fp,
                    best_cand_fp,
                    best_comp,
                    margin,
                    is_short,
                    raw_sim_pct,
                    damped_sim_pct,
                    aligning_traits,
                    _summary,
                ) in batch_identifications
            ]
            print_json_data(batch_data)
            return

        if out_format == "csv":
            csv_headers = [
                "submission",
                "words",
                "short_doc",
                "top_match",
                "confidence",
                "burrows_delta",
                "verdict",
                "lead_margin",
                "top_aligning_traits",
            ]
            csv_rows = [
                [
                    f.name,
                    essay_fp.word_count,
                    "yes" if is_short else "no",
                    best_cand_fp.label,
                    f"{damped_sim_pct:.1f}%",
                    f"{best_comp.manhattan_delta:.3f}",
                    best_comp.same_author_likelihood.replace("_", " ").title(),
                    f"+{margin:.1f}%" if margin is not None else "",
                    traits_summary,
                ]
                for (
                    f,
                    essay_fp,
                    best_cand_fp,
                    best_comp,
                    margin,
                    is_short,
                    _raw,
                    damped_sim_pct,
                    _traits,
                    traits_summary,
                ) in batch_identifications
            ]
            print_csv_rows(csv_headers, csv_rows)
            return

        # Batch Table
        table = Table(
            title=f"Batch Author Identification ({len(files)} Submissions)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Submission", style="cyan", width=20)
        table.add_column("Words", justify="right", width=9)
        table.add_column("Top Match", style="bold", width=18)
        table.add_column("Confidence", justify="left", width=18)
        table.add_column("Burrows Δ", justify="right", width=10)
        table.add_column("Verdict", justify="left", width=15)
        table.add_column("Lead Margin", justify="right", width=11)
        table.add_column("Top Aligning Traits", style="dim", width=36)

        author_counts: dict[str, int] = {}
        short_count = 0
        for (
            f,
            essay_fp,
            best_cand_fp,
            best_comp,
            margin,
            is_short,
            _raw_sim_pct,
            damped_sim_pct,
            _aligning_traits,
            traits_summary,
        ) in batch_identifications:
            if is_short:
                short_count += 1
                words_display = f"[yellow]{essay_fp.word_count:,} ⚠️[/yellow]"
            else:
                words_display = f"{essay_fp.word_count:,}"

            bar = draw_bar(damped_sim_pct, width=10)
            conf_str = f"{damped_sim_pct:5.1f}% {bar}"
            delta_str = f"{best_comp.manhattan_delta:.3f}"
            verdict = best_comp.same_author_likelihood.replace("_", " ").title()

            cand_lbl = (
                f"{best_cand_fp.label} ({best_cand_fp.sample_count}s)"
                if best_cand_fp.sample_count > 1
                else best_cand_fp.label
            )
            if damped_sim_pct >= 80:
                top_match_str = f"[bold green]{cand_lbl}[/bold green]"
            elif damped_sim_pct >= 60:
                top_match_str = f"[bold yellow]{cand_lbl}[/bold yellow]"
            else:
                top_match_str = f"[bold red]{cand_lbl}[/bold red]"

            margin_str = f"+{margin:.1f}%" if margin is not None else "—"
            author_counts[best_cand_fp.label] = author_counts.get(best_cand_fp.label, 0) + 1

            table.add_row(
                f.name,
                words_display,
                top_match_str,
                conf_str,
                delta_str,
                verdict,
                margin_str,
                traits_summary,
            )

        console.print(table)

        summary_breakdown = ", ".join(
            f"[bold cyan]{author}[/bold cyan] ({count})"
            for author, count in sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        )
        short_note = (
            f"\n⚠️  [yellow]{short_count} short submission(s) (<250 words) detected; "
            f"attribution confidence was proportionally damped.[/yellow]"
            if short_count > 0
            else ""
        )
        skip_note = (
            f"\n⚠️  [yellow]Skipped {len(skipped_files)} unreadable file(s)[/yellow]"
            if skipped_files
            else ""
        )
        summary_text = (
            f"📂 Processed Directory: [bold]{path}[/bold]\n"
            f"📄 Submissions Evaluated: {len(batch_identifications)} "
            f"against {len(enrolled_candidates)} candidates\n\n"
            f"Identified Distribution: {summary_breakdown}"
            f"{short_note}"
            f"{skip_note}"
        )
        if not no_report:
            summary_text += f"\n📋 PDF Reports Saved: [cyan]{output}[/cyan]"

        console.print(
            Panel(
                summary_text,
                title="BATCH IDENTIFICATION COMPLETE",
                expand=False,
                padding=(1, 2),
            )
        )
        return

    # Single-file mode
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        disable=(out_format != "table"),
    ) as progress:
        progress.add_task(
            description=(
                f"Evaluating [cyan]{path.name}[/cyan] against "
                f"{len(enrolled_candidates)} candidate profiles..."
            ),
            total=None,
        )
        doc = ingest_file(path)
        essay_fp = create_fingerprint(doc, label=path.stem)

        candidate_scores = []
        for cand_fp in enrolled_candidates:
            comp = compare_fingerprints(cand_fp, essay_fp)
            candidate_scores.append((cand_fp, comp))

    # Rank by cosine similarity descending
    candidate_scores.sort(key=lambda x: x[1].cosine_similarity, reverse=True)

    best_candidate_fp, best_comp = candidate_scores[0]
    raw_best_sim_pct = best_comp.cosine_similarity * 100

    # Length check & proportional damping
    is_short = essay_fp.word_count < MIN_RELIABLE_WORDS
    damping = compute_length_damping(essay_fp.word_count)
    damped_best_sim_pct = raw_best_sim_pct * damping

    margin = None
    if len(candidate_scores) > 1:
        runner_up_fp, runner_up_comp = candidate_scores[1]
        runner_up_pct = runner_up_comp.cosine_similarity * 100 * damping
        margin = damped_best_sim_pct - runner_up_pct

    # Explainability: top 4 aligning traits
    aligning_traits = explain_aligning_traits(best_candidate_fp, essay_fp, top_n=4)
    traits_summary = format_aligning_traits_summary(aligning_traits)

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        safe_name = best_candidate_fp.label.replace(" ", "_")
        pdf_path = output / f"identify_{path.stem}_{safe_name}.pdf"
        generate_comparison_report(best_comp, best_candidate_fp, essay_fp, pdf_path)

    if out_format == "json":
        data = {
            "essay": path.name,
            "path": str(path),
            "word_count": essay_fp.word_count,
            "short_document": is_short,
            "length_warning": SHORT_DOC_WARNING if is_short else None,
            "top_match": best_candidate_fp.label,
            "confidence": round(damped_best_sim_pct, 2),
            "raw_confidence": (
                round(raw_best_sim_pct, 2) if is_short else round(damped_best_sim_pct, 2)
            ),
            "burrows_delta": round(best_comp.manhattan_delta, 4),
            "verdict": best_comp.same_author_likelihood.replace("_", " ").title(),
            "lead_margin": round(margin, 2) if margin is not None else None,
            "aligning_traits": aligning_traits,
            "candidates": [
                {
                    "rank": rank,
                    "candidate": cand_fp.label,
                    "samples": cand_fp.sample_count,
                    "confidence": round(comp.cosine_similarity * 100 * damping, 2),
                    "raw_confidence": round(comp.cosine_similarity * 100, 2),
                    "burrows_delta": round(comp.manhattan_delta, 4),
                    "verdict": comp.same_author_likelihood.replace("_", " ").title(),
                }
                for rank, (cand_fp, comp) in enumerate(candidate_scores[:top_k], start=1)
            ],
        }
        print_json_data(data)
        return

    if out_format == "csv":
        csv_headers = [
            "rank",
            "candidate",
            "samples",
            "confidence",
            "burrows_delta",
            "verdict",
            "top_aligning_traits",
        ]
        csv_rows = [
            [
                rank,
                cand_fp.label,
                cand_fp.sample_count,
                f"{comp.cosine_similarity * 100 * damping:.1f}%",
                f"{comp.manhattan_delta:.3f}",
                comp.same_author_likelihood.replace("_", " ").title(),
                traits_summary if rank == 1 else "",
            ]
            for rank, (cand_fp, comp) in enumerate(candidate_scores[:top_k], start=1)
        ]
        print_csv_rows(csv_headers, csv_rows)
        return

    # Warning alert if document is short
    if is_short:
        console.print(
            Panel(
                f"⚠️  [bold yellow]Short Document Notice "
                f"({essay_fp.word_count} words < 250 words)[/bold yellow]\n"
                f"[dim]Stylometric metrics (such as MATTR, MTLD, and parse depth variance) have "
                f"higher sampling noise on brief texts. "
                f"Attribution confidence has been proportionally damped.[/dim]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    color = (
        "green" if damped_best_sim_pct >= 80 else ("yellow" if damped_best_sim_pct >= 60 else "red")
    )
    margin_text = ""
    if margin is not None:
        margin_text = (
            f"\nMargin: [bold]+{margin:.1f}%[/bold] lead over 2nd place "
            f"([cyan]{runner_up_fp.label}[/cyan] at {runner_up_pct:.1f}%)"
        )

    sample_badge = (
        f" ({best_candidate_fp.sample_count} samples rolling baseline)"
        if best_candidate_fp.sample_count > 1
        else ""
    )
    if is_short:
        conf_line = (
            f"Match Confidence: [{color} bold]{damped_best_sim_pct:.1f}%[/] "
            f"[dim](damped from {raw_best_sim_pct:.1f}% due to length: "
            f"{essay_fp.word_count} words)[/dim] "
            f"({best_comp.same_author_likelihood.replace('_', ' ').title()})"
        )
    else:
        conf_line = (
            f"Match Confidence: [{color} bold]{damped_best_sim_pct:.1f}%[/] "
            f"({best_comp.same_author_likelihood.replace('_', ' ').title()})"
        )

    header_text = (
        f"📄 Essay: [bold]{path.name}[/bold] ({essay_fp.word_count:,} words)\n"
        f"👥 Enrolled Candidates Evaluated: {len(enrolled_candidates)}\n\n"
        f"🏆 Top Match: [{color} bold]{best_candidate_fp.label}[/]{sample_badge}\n"
        f"{conf_line}"
        f"{margin_text}"
    )
    console.print(Panel(header_text, title="AUTHOR IDENTIFICATION", expand=False, padding=(1, 2)))

    # Candidate table
    table = Table(
        title=f"Candidate Ranking (Top {min(top_k, len(candidate_scores))})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Student / Author", style="cyan")
    table.add_column("Similarity", justify="left")
    table.add_column("Burrows Δ", justify="right")
    table.add_column("Verdict", justify="left")

    for rank, (cand_fp, comp) in enumerate(candidate_scores[:top_k], start=1):
        sim = comp.cosine_similarity * 100 * damping
        bar = draw_bar(sim, width=15)
        sim_col = f"{sim:5.1f}%  {bar}"
        delta_str = f"{comp.manhattan_delta:.3f}"
        verdict = comp.same_author_likelihood.replace("_", " ").title()

        cand_str = (
            f"{cand_fp.label} [dim]({cand_fp.sample_count}s)[/dim]"
            if cand_fp.sample_count > 1
            else cand_fp.label
        )
        rank_badge = f"[bold yellow]#{rank}[/]" if rank == 1 else f"#{rank}"
        table.add_row(rank_badge, cand_str, sim_col, delta_str, verdict)

    console.print(table)

    # Attribution Explainability Card
    console.print(
        "\n  [bold cyan]🔍 Top Aligning Linguistic Traits (Idiolect Drivers):[/bold cyan]"
    )
    for t in aligning_traits:
        console.print(
            f"  • [bold]{t['description'].capitalize()}[/bold] "
            f"[dim]({t['name']} — Δz = {t['z_delta']:.2f})[/dim]"
        )
    console.print()

    if not no_report:
        console.print(f"  📋 Comparison report with top candidate saved: [cyan]{pdf_path}[/cyan]")


@app.command(name="list")
def list_fingerprints(
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
):
    """List all enrolled author profiles, sample counts, and baseline consistency."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    store = get_store()
    profiles = store.list_profiles()
    if not profiles:
        if out_format == "json":
            print_json_data([])
        elif out_format == "csv":
            csv_headers = ["author", "samples", "words", "author_type", "consistency", "top_trait"]
            print_csv_rows(csv_headers, [])
        else:
            console.print("No authors enrolled.")
        return

    if out_format == "json":
        data = []
        for prof in profiles:
            fp = prof.composite_fingerprint
            cons = (
                round(compute_profile_consistency(fp.axis_stability), 2)
                if prof.sample_count > 1 and fp.axis_stability
                else None
            )
            top_t = fp.standout_traits[0].get("feature", "") if fp.standout_traits else None
            data.append(
                {
                    "author": prof.label,
                    "samples": prof.sample_count,
                    "words": prof.total_word_count,
                    "author_type": fp.author_type.value,
                    "consistency": cons,
                    "top_trait": top_t,
                }
            )
        print_json_data(data)
        return

    if out_format == "csv":
        csv_headers = ["author", "samples", "words", "author_type", "consistency", "top_trait"]
        csv_rows = []
        for prof in profiles:
            fp = prof.composite_fingerprint
            cons_str = (
                f"{compute_profile_consistency(fp.axis_stability):.1f}%"
                if prof.sample_count > 1 and fp.axis_stability
                else ""
            )
            top_t_str = (
                fp.standout_traits[0].get("feature", "").replace(".", " › ")
                if fp.standout_traits
                else ""
            )
            csv_rows.append(
                [
                    prof.label,
                    prof.sample_count,
                    prof.total_word_count,
                    fp.author_type.value,
                    cons_str,
                    top_t_str,
                ]
            )
        print_csv_rows(csv_headers, csv_rows)
        return

    table = Table(
        title=f"Enrolled Author Profiles ({len(profiles)} Registered)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Author", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("Type", justify="center")
    table.add_column("Consistency", justify="right")
    table.add_column("Top Trait", justify="left")

    for prof in profiles:
        fp = prof.composite_fingerprint
        sample_str = str(prof.sample_count)
        words_str = f"{prof.total_word_count:,}"

        if fp.author_type == AuthorType.HUMAN:
            type_str = "[bold green]HUMAN[/bold green]"
        elif fp.author_type == AuthorType.AI:
            type_str = "[bold red]AI[/bold red]"
        else:
            type_str = "[bold yellow]UNCERTAIN[/bold yellow]"

        if prof.sample_count > 1 and fp.axis_stability:
            consistency = compute_profile_consistency(fp.axis_stability)
            c_color = "green" if consistency >= 85 else ("yellow" if consistency >= 70 else "red")
            cons_str = f"[{c_color}]{consistency:.1f}%[/]"
        else:
            cons_str = "—"

        if fp.standout_traits:
            top_trait = fp.standout_traits[0]
            t_name = top_trait.get("feature", "").replace(".", " › ").replace("_", " ").title()
            z = top_trait.get("z_score", 0.0)
            arrow = "↑" if z > 0 else "↓"
            trait_str = f"{t_name[:20]} {arrow}"
        else:
            trait_str = "[dim]Average[/dim]"

        table.add_row(prof.label, sample_str, words_str, type_str, cons_str, trait_str)

    console.print(table)


@app.command()
def profile(
    name: str = typer.Argument(..., help="Name of the enrolled author to inspect."),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output directory to generate a PDF profile report.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: 'table' (default), 'json', or 'csv'.",
    ),
    json: bool = typer.Option(False, "--json", help="Alias for --format json."),
):
    """Inspect an author's multi-sample profile, consistency, and rolling baseline."""
    out_format = "json" if json else format.lower()
    if out_format not in ("table", "json", "csv"):
        console.print(
            f"[bold red]Error:[/] Invalid format '{out_format}'. Must be 'table', 'json', or 'csv'."
        )
        raise typer.Exit(1)

    store = get_store()
    prof = store.get_profile(name)
    if not prof:
        console.print(f"[bold red]Error:[/] Author '{name}' is not enrolled.")
        raise typer.Exit(1)

    fp = prof.composite_fingerprint
    samples = prof.samples
    weights = calculate_sample_weights([s.word_count for s in samples]) if samples else []

    if output:
        output.mkdir(parents=True, exist_ok=True)
        safe_name = prof.label.replace(" ", "_")
        pdf_path = output / f"profile_{safe_name}.pdf"
        generate_report(fp, pdf_path)

    if out_format == "json":
        consistency = (
            round(compute_profile_consistency(fp.axis_stability), 2)
            if prof.sample_count > 1 and fp.axis_stability
            else None
        )
        data = {
            "author": prof.label,
            "samples_count": prof.sample_count,
            "total_words": prof.total_word_count,
            "author_type": fp.author_type.value,
            "ai_confidence": round(fp.ai_confidence, 4),
            "consistency": consistency,
            "samples": [
                {
                    "index": i,
                    "label": s.sample_label,
                    "words": s.word_count,
                    "enrolled_at": s.enrolled_at,
                    "rolling_weight": round(w, 4),
                }
                for i, (s, w) in enumerate(zip(samples, weights), start=1)
            ],
            "axes": {k: round(v, 2) for k, v in fp.axes.items()},
            "axis_stability": (
                {k: round(v, 2) for k, v in fp.axis_stability.items()} if fp.axis_stability else {}
            ),
            "standout_traits": fp.standout_traits,
        }
        print_json_data(data)
        return

    if out_format == "csv":
        csv_headers = ["sample_index", "sample_label", "words", "enrolled_at", "rolling_weight"]
        csv_rows = [
            [
                i,
                s.sample_label,
                s.word_count,
                s.enrolled_at,
                f"{w * 100:.1f}%",
            ]
            for i, (s, w) in enumerate(zip(samples, weights), start=1)
        ]
        print_csv_rows(csv_headers, csv_rows)
        return

    # Header Panel
    consistency_text = ""
    if prof.sample_count > 1 and fp.axis_stability:
        consistency = compute_profile_consistency(fp.axis_stability)
        c_color = "green" if consistency >= 85 else ("yellow" if consistency >= 70 else "red")
        consistency_text = f"  |  Consistency: [{c_color} bold]{consistency:.1f}%[/]"

    type_color = (
        "green"
        if fp.author_type == AuthorType.HUMAN
        else ("red" if fp.author_type == AuthorType.AI else "yellow")
    )

    header = (
        f"👤 Author: [bold]{prof.label}[/bold]\n"
        f"📚 Enrolled Samples: [bold]{prof.sample_count}[/bold]  |  "
        f"Total Words: [bold]{prof.total_word_count:,}[/bold]{consistency_text}\n\n"
        f"🤖 Classification: [{type_color} bold]{fp.author_type.value.upper()}[/] "
        f"(confidence: {fp.ai_confidence * 100:.1f}%)"
    )
    console.print(Panel(header, title="IDIOLECT AUTHOR PROFILE", expand=False, padding=(1, 2)))

    # Samples Table
    if samples:
        weights = calculate_sample_weights([s.word_count for s in samples])
        table_s = Table(
            title="Enrolled Writing Samples (Weighted Rolling Window)",
            show_header=True,
            header_style="bold magenta",
        )
        table_s.add_column("#", justify="right", style="dim", width=4)
        table_s.add_column("Sample Label", style="cyan", width=28)
        table_s.add_column("Words", justify="right", width=9)
        table_s.add_column("Enrolled Date", justify="center", width=20)
        table_s.add_column("Rolling Weight", justify="right", width=16)

        for i, (sample, w) in enumerate(zip(samples, weights), start=1):
            date_str = (
                sample.enrolled_at[:10] if len(sample.enrolled_at) >= 10 else sample.enrolled_at
            )
            w_pct = f"{w * 100:.1f}%"
            bar = draw_bar(w * 100, width=8)
            table_s.add_row(
                str(i), sample.sample_label, f"{sample.word_count:,}", date_str, f"{w_pct} {bar}"
            )

        console.print(table_s)

    # Stylometric Axes with Stability
    console.print(
        "\n  [bold cyan]📊 Stylometric Radar Axes (Rolling Composite Baseline):[/bold cyan]"
    )
    axes_display = [
        (
            "Lexical Richness",
            fp.axes.get("lexical_richness", 0.0),
            fp.axis_stability.get("lexical_richness", 0.0),
        ),
        (
            "Syntactic Complex.",
            fp.axes.get("syntactic_complexity", 0.0),
            fp.axis_stability.get("syntactic_complexity", 0.0),
        ),
        (
            "Formality",
            fp.axes.get("formality", 0.0),
            fp.axis_stability.get("formality", 0.0),
        ),
        (
            "Epistemic Stance",
            fp.axes.get("epistemic_stance", 0.0),
            fp.axis_stability.get("epistemic_stance", 0.0),
        ),
        (
            "Pacing & Cadence",
            fp.axes.get("pacing_cadence", 0.0),
            fp.axis_stability.get("pacing_cadence", 0.0),
        ),
        (
            "Affective Intensity",
            fp.axes.get("affective_intensity", 0.0),
            fp.axis_stability.get("affective_intensity", 0.0),
        ),
        (
            "Engagement",
            fp.axes.get("interactive_engagement", 0.0),
            fp.axis_stability.get("interactive_engagement", 0.0),
        ),
    ]

    for name, score, std in axes_display:
        bar = draw_bar(score, width=20)
        if prof.sample_count > 1:
            stability_str = f"±{std:4.1f} ({classify_stability(std)})"
            console.print(f"  {name:<19} {bar}  {int(score):3d}  [dim]{stability_str}[/dim]")
        else:
            console.print(f"  {name:<19} {bar}  {int(score):3d}")

    # Standout Traits
    if fp.standout_traits:
        console.print("\n  [bold yellow]⚡ Standout Traits:[/bold yellow]")
        for trait in fp.standout_traits[:5]:
            t_name = trait.get("feature", "").replace(".", " › ").replace("_", " ").title()
            z = trait.get("z_score", 0.0)
            interp = trait.get("interpretation", "")
            arrow = "↑" if z > 0 else "↓"
            console.print(f"  • {t_name} — z = {z:+.1f} {arrow} — {interp}")

    console.print()

    if output:
        console.print(f"  📋 Profile report saved: [cyan]{pdf_path}[/cyan]\n")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Name of the enrolled author to delete."),
    sample_id: Optional[int] = typer.Option(
        None,
        "--sample-id",
        "-s",
        help="Specific sample ID to remove rather than deleting entire profile.",
    ),
):
    """Delete an enrolled author profile or a specific sample from a profile."""
    store = get_store()
    if sample_id is not None:
        updated_fp = store.delete_sample(name, sample_id)
        if updated_fp is not None:
            console.print(
                f"[bold green]✓ Removed sample #{sample_id} from profile:[/bold green] "
                f"[bold cyan]{name}[/bold cyan]\n"
                f"  📈 Baseline: [bold]{updated_fp.sample_count} samples[/bold] | "
                f"[bold]{updated_fp.word_count:,} words[/bold]"
            )
        else:
            if not store.get(name):
                console.print(
                    f"[bold green]✓ Removed last sample; deleted profile:[/bold green] {name}"
                )
            else:
                console.print(
                    f"[bold red]Error:[/] Sample ID {sample_id} not found for author '{name}'."
                )
                raise typer.Exit(1)
        return

    if store.delete(name):
        console.print(f"[bold green]✓ Deleted enrolled author:[/bold green] {name}")
    else:
        console.print(f"[bold red]Error:[/] Author '{name}' not found.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
