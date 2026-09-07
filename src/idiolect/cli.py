from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .comparison import compare as compare_fingerprints
from .fingerprint import create_fingerprint
from .ingestion import find_text_files, ingest_file, ingest_path
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
    json: bool = typer.Option(False, "--json", help="Also output raw fingerprint JSON."),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip PDF generation, just print summary to console."
    ),
):
    """Analyze a text file or directory of files and generate fingerprint + PDF reports."""
    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

        batch_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Analyzing {len(files)} documents...", total=len(files))
            for f in files:
                progress.update(task, description=f"Analyzing [cyan]{f.name}[/cyan]...")
                doc = ingest_file(f)
                fp = create_fingerprint(doc, label=f.stem)

                if not no_report:
                    output.mkdir(parents=True, exist_ok=True)
                    pdf_path = output / f"{f.stem}_fingerprint.pdf"
                    generate_report(fp, pdf_path)

                batch_results.append((f, doc, fp))
                progress.advance(task)

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

        for f, doc, fp in batch_results:
            total_words += fp.word_count
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
                f"{fp.word_count:,}",
                type_badge,
                ai_conf_str,
                re_str,
                f"{lr:.1f}",
                f"{sc:.1f}",
            )

        console.print(table)

        summary_text = (
            f"📂 Processed Directory: [bold]{path}[/bold]\n"
            f"📄 Documents Analyzed: {len(files)}  |  Total Words: {total_words:,}\n\n"
            f"Classification Breakdown: [green bold]{human_count} Human[/], "
            f"[red bold]{ai_count} AI[/], [yellow bold]{uncertain_count} Uncertain[/]"
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
    ) as progress:
        progress.add_task(
            description=f"Ingesting [cyan]{path.name}[/cyan] and analyzing...", total=None
        )
        doc = ingest_file(path)
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
        Path("artifacts"),
        "--output",
        "-o",
        help="Output directory for the PDF report (default: 'artifacts').",
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
    path: Path = typer.Argument(
        ...,
        metavar="PATH",
        help="Path to a text sample file or directory of text samples.",
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
    ),
):
    """Enroll a text sample or directory of samples as a known author's fingerprint."""
    store = get_store()

    if path.is_dir():
        files = find_text_files(path)
        if not files:
            console.print(f"[bold red]Error:[/] No supported text files found in directory: {path}")
            raise typer.Exit(1)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description=f"Enrolling [cyan]{name}[/cyan]...", total=None)
        doc = ingest_path(path)
        fingerprint = create_fingerprint(doc, label=name)
        store.enroll(fingerprint)

    if path.is_dir():
        console.print(
            f"[bold green]✓ Successfully enrolled author:[/bold green] {name} "
            f"([dim]{len(files)} documents combined, {fingerprint.word_count:,} words[/dim])"
        )
    else:
        console.print(
            f"[bold green]✓ Successfully enrolled author:[/bold green] {name} "
            f"([dim]{fingerprint.word_count:,} words[/dim])"
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
):
    """Verify if a text file or directory of files matches an enrolled author."""
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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Verifying {len(files)} documents against {name}...", total=len(files)
            )
            for f in files:
                progress.update(task, description=f"Verifying [cyan]{f.name}[/cyan]...")
                doc = ingest_file(f)
                new_fingerprint = create_fingerprint(doc, label=f.stem)
                comp = compare_fingerprints(enrolled, new_fingerprint)
                batch_results.append((f, doc, new_fingerprint, comp))
                progress.advance(task)

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
        for f, doc, fp, comp in batch_results:
            sim = comp.cosine_similarity * 100
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

            table.add_row(f.name, f"{fp.word_count:,}", sim_str, delta_str, verdict_styled)

        console.print(table)
        console.print(
            Panel(
                f"👤 Enrolled Author: [bold]{name}[/bold]\n"
                f"📂 Directory: [bold]{path}[/bold] ({len(files)} documents)\n"
                f"🎯 Strong Matches: [bold green]{matched_count}/{len(files)}[/bold green]",
                title="BATCH VERIFICATION COMPLETE",
                expand=False,
                padding=(1, 2),
            )
        )
        return

    # Single-file mode
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
    ) as progress:
        progress.add_task(description=f"Verifying against [cyan]{name}[/cyan]...", total=None)
        doc = ingest_file(path)
        new_fingerprint = create_fingerprint(doc, label=path.stem)
        comp = compare_fingerprints(enrolled, new_fingerprint)

    sim = comp.cosine_similarity * 100
    color = "green" if sim > 80 else "yellow" if sim > 60 else "red"

    console.print(
        Panel(
            f"Author: [bold]{name}[/]\n"
            f"File: [bold]{path.name}[/]\n\n"
            f"Match Confidence: [{color} bold]{sim:.1f}%[/]\n"
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
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip generating PDF comparison report(s)."
    ),
):
    """Identify which enrolled student/author wrote an essay based on stylometric similarity."""
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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Identifying authors for {len(files)} submissions...", total=len(files)
            )
            for f in files:
                progress.update(task, description=f"Evaluating [cyan]{f.name}[/cyan]...")
                doc = ingest_file(f)
                essay_fp = create_fingerprint(doc, label=f.stem)

                candidate_scores = []
                for cand_fp in enrolled_candidates:
                    comp = compare_fingerprints(cand_fp, essay_fp)
                    candidate_scores.append((cand_fp, comp))

                candidate_scores.sort(key=lambda x: x[1].cosine_similarity, reverse=True)
                best_cand_fp, best_comp = candidate_scores[0]
                best_sim_pct = best_comp.cosine_similarity * 100

                margin = None
                if len(candidate_scores) > 1:
                    runner_up_fp, runner_up_comp = candidate_scores[1]
                    margin = best_sim_pct - (runner_up_comp.cosine_similarity * 100)

                if not no_report:
                    output.mkdir(parents=True, exist_ok=True)
                    safe_name = best_cand_fp.label.replace(" ", "_")
                    pdf_path = output / f"identify_{f.stem}_{safe_name}.pdf"
                    generate_comparison_report(best_comp, best_cand_fp, essay_fp, pdf_path)

                batch_identifications.append((f, essay_fp, best_cand_fp, best_comp, margin))
                progress.advance(task)

        # Batch Table
        table = Table(
            title=f"Batch Author Identification ({len(files)} Submissions)",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Submission", style="cyan", width=22)
        table.add_column("Words", justify="right", width=8)
        table.add_column("Top Match", style="bold", width=18)
        table.add_column("Confidence", justify="left", width=18)
        table.add_column("Burrows Δ", justify="right", width=10)
        table.add_column("Verdict", justify="left", width=16)
        table.add_column("Lead Margin", justify="right", width=12)

        author_counts: dict[str, int] = {}
        for f, essay_fp, best_cand_fp, best_comp, margin in batch_identifications:
            sim = best_comp.cosine_similarity * 100
            bar = draw_bar(sim, width=10)
            conf_str = f"{sim:5.1f}% {bar}"
            delta_str = f"{best_comp.manhattan_delta:.3f}"
            verdict = best_comp.same_author_likelihood.replace("_", " ").title()

            if sim >= 80:
                top_match_str = f"[bold green]{best_cand_fp.label}[/bold green]"
            elif sim >= 60:
                top_match_str = f"[bold yellow]{best_cand_fp.label}[/bold yellow]"
            else:
                top_match_str = f"[bold red]{best_cand_fp.label}[/bold red]"

            margin_str = f"+{margin:.1f}%" if margin is not None else "—"
            author_counts[best_cand_fp.label] = author_counts.get(best_cand_fp.label, 0) + 1

            table.add_row(
                f.name,
                f"{essay_fp.word_count:,}",
                top_match_str,
                conf_str,
                delta_str,
                verdict,
                margin_str,
            )

        console.print(table)

        summary_breakdown = ", ".join(
            f"[bold cyan]{author}[/bold cyan] ({count})"
            for author, count in sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
        )
        summary_text = (
            f"📂 Processed Directory: [bold]{path}[/bold]\n"
            f"📄 Submissions Evaluated: {len(files)} "
            f"against {len(enrolled_candidates)} candidates\n\n"
            f"Identified Distribution: {summary_breakdown}"
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
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True
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
    best_sim_pct = best_comp.cosine_similarity * 100
    color = "green" if best_sim_pct >= 80 else ("yellow" if best_sim_pct >= 60 else "red")

    margin_text = ""
    if len(candidate_scores) > 1:
        runner_up_fp, runner_up_comp = candidate_scores[1]
        runner_up_pct = runner_up_comp.cosine_similarity * 100
        margin = best_sim_pct - runner_up_pct
        margin_text = (
            f"\nMargin: [bold]+{margin:.1f}%[/bold] lead over 2nd place "
            f"([cyan]{runner_up_fp.label}[/cyan] at {runner_up_pct:.1f}%)"
        )

    header_text = (
        f"📄 Essay: [bold]{path.name}[/bold] ({essay_fp.word_count:,} words)\n"
        f"👥 Enrolled Candidates Evaluated: {len(enrolled_candidates)}\n\n"
        f"🏆 Top Match: [{color} bold]{best_candidate_fp.label}[/]\n"
        f"Match Confidence: [{color} bold]{best_sim_pct:.1f}%[/] "
        f"({best_comp.same_author_likelihood.replace('_', ' ').title()})"
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
        sim = comp.cosine_similarity * 100
        bar = draw_bar(sim, width=15)
        sim_col = f"{sim:5.1f}%  {bar}"
        delta_str = f"{comp.manhattan_delta:.3f}"
        verdict = comp.same_author_likelihood.replace("_", " ").title()

        rank_badge = f"[bold yellow]#{rank}[/]" if rank == 1 else f"#{rank}"
        table.add_row(rank_badge, cand_fp.label, sim_col, delta_str, verdict)

    console.print(table)

    if not no_report:
        output.mkdir(parents=True, exist_ok=True)
        safe_name = best_candidate_fp.label.replace(" ", "_")
        pdf_path = output / f"identify_{path.stem}_{safe_name}.pdf"
        generate_comparison_report(best_comp, best_candidate_fp, essay_fp, pdf_path)
        console.print(f"\n  📋 Comparison report with top candidate saved: [cyan]{pdf_path}[/cyan]")


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
