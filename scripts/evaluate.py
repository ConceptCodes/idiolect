"""Comprehensive evaluation runner for idiolect.

Benchmarks:
1. AI Detection Benchmark (HC3 dataset: Human vs. ChatGPT)
   - Evaluates Accuracy, Precision, Recall, F1, and False Accusation Rate (FPR).
2. Multi-Author Literary Attribution (Arthur Conan Doyle vs. Jane Austen vs. Mark Twain)
   - Evaluates multi-class closed-set authorship attribution across distinct authorial voices.
3. Historical Authorship Attribution (The Federalist Papers: Hamilton vs. Madison)
   - Evaluates attribution on complete full-length political essays.
"""

import time
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from idiolect.ingestion import ingest_file
from idiolect.fingerprint import create_fingerprint
from idiolect.comparison import compare
from idiolect.models import AuthorType, Fingerprint

console = Console()
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


def make_composite_profile(files: list[Path], label: str) -> Fingerprint:
    """Build a stable composite author profile by aggregating training texts."""
    fps = [create_fingerprint(ingest_file(f), label=f.name) for f in files]
    avg_features: dict[str, float] = {}
    for k in fps[0].features:
        avg_features[k] = float(np.mean([fp.features.get(k, 0.0) for fp in fps]))
    avg_axes: dict[str, float] = {}
    for k in fps[0].axes:
        avg_axes[k] = float(np.mean([fp.axes.get(k, 0.0) for fp in fps]))
    return Fingerprint(label=label, axes=avg_axes, features=avg_features)


def evaluate_ai_detection(max_samples: int = 25):
    """Run AI content detection evaluation on the HC3 dataset."""
    ai_dir = EVAL_DIR / "ai_human"
    if not ai_dir.exists():
        console.print("[bold red]Dataset not found![/] Run: python scripts/download_eval_data.py ai")
        return

    human_files = sorted(ai_dir.glob("human_*.txt"))[:max_samples]
    ai_files = sorted(ai_dir.glob("ai_*.txt"))[:max_samples]

    console.print(Panel(
        f"Evaluating AI Detection on [bold cyan]{len(human_files)} Human[/] and [bold cyan]{len(ai_files)} AI[/] texts from HC3...",
        title="1. AI DETECTION BENCHMARK (HC3 Human vs. ChatGPT)"
    ))

    y_true = []  # 0 for human, 1 for ai
    y_pred = []  # 0 for human, 1 for ai, 0.5 for uncertain
    confidences = []

    start = time.time()
    
    # Evaluate Human samples
    for f in human_files:
        doc = ingest_file(f)
        fp = create_fingerprint(doc, label=f.name)
        y_true.append(0)
        confidences.append(fp.ai_confidence)
        if fp.author_type == AuthorType.HUMAN:
            y_pred.append(0)
        elif fp.author_type == AuthorType.AI:
            y_pred.append(1)
        else:
            y_pred.append(0.5)

    # Evaluate AI samples
    for f in ai_files:
        doc = ingest_file(f)
        fp = create_fingerprint(doc, label=f.name)
        y_true.append(1)
        confidences.append(fp.ai_confidence)
        if fp.author_type == AuthorType.AI:
            y_pred.append(1)
        elif fp.author_type == AuthorType.HUMAN:
            y_pred.append(0)
        else:
            y_pred.append(0.5)

    duration = time.time() - start

    # Compute Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    uncertain = np.sum(y_pred == 0.5)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    table = Table(title="AI Detection Performance Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold green")
    table.add_column("Description")

    table.add_row("Accuracy", f"{accuracy*100:.1f}%", "Overall correct classifications")
    table.add_row("Precision", f"{precision*100:.1f}%", "When flagged as AI, probability it is AI")
    table.add_row("Recall", f"{recall*100:.1f}%", "Proportion of AI texts successfully caught")
    table.add_row("F1 Score", f"{f1*100:.1f}%", "Harmonic balance of precision and recall")
    table.add_row("False Accusation Rate (FPR)", f"{fpr*100:.1f}%", "Human papers wrongly flagged as AI")
    table.add_row("Uncertain Rate", f"{(uncertain/total)*100:.1f}%", "Cases flagged for human teacher review")
    table.add_row("Throughput", f"{total/duration:.1f} docs/sec", f"Processed {total} docs in {duration:.1f}s")

    console.print(table)


def evaluate_literary_authorship(train_samples: int = 5, test_samples: int = 3):
    """Run Multi-Author Attribution on Conan Doyle, Jane Austen, Mark Twain."""
    auth_dir = EVAL_DIR / "authors"
    if not auth_dir.exists():
        console.print("[bold red]Literary dataset not found![/] Run: python scripts/download_eval_data.py authors")
        return

    doyle_files = sorted(auth_dir.glob("doyle_*.txt"))
    austen_files = sorted(auth_dir.glob("austen_*.txt"))
    twain_files = sorted(auth_dir.glob("twain_*.txt"))

    console.print(Panel(
        f"Evaluating 3-Way Authorship Attribution on Literary Works:\n"
        f"• Arthur Conan Doyle: {train_samples} train / {test_samples} test stories\n"
        f"• Jane Austen: {train_samples} train / {test_samples} test chapters\n"
        f"• Mark Twain: {train_samples} train / {test_samples} test chapters",
        title="2. MULTI-AUTHOR LITERARY ATTRIBUTION BENCHMARK"
    ))

    # Enroll composite baseline profiles
    p_doyle = make_composite_profile(doyle_files[:train_samples], "Conan Doyle")
    p_austen = make_composite_profile(austen_files[:train_samples], "Jane Austen")
    p_twain = make_composite_profile(twain_files[:train_samples], "Mark Twain")

    profiles = {
        "Conan Doyle": p_doyle,
        "Jane Austen": p_austen,
        "Mark Twain": p_twain
    }

    test_suite = [
        (doyle_files[train_samples:train_samples+test_samples], "Conan Doyle"),
        (austen_files[train_samples:train_samples+test_samples], "Jane Austen"),
        (twain_files[train_samples:train_samples+test_samples], "Mark Twain")
    ]

    table = Table(title="Holdout Text Predictions", show_header=True)
    table.add_column("Document", style="cyan")
    table.add_column("True Author")
    table.add_column("Predicted Author", style="bold")
    table.add_column("Doyle Delta")
    table.add_column("Austen Delta")
    table.add_column("Twain Delta")
    table.add_column("Result")

    correct = 0
    total = 0

    for files, true_author in test_suite:
        for f in files:
            fp = create_fingerprint(ingest_file(f), label=f.name)
            deltas = {name: compare(fp, prof).manhattan_delta for name, prof in profiles.items()}
            pred = min(deltas, key=deltas.get)
            match = (pred == true_author)
            if match:
                correct += 1
            total += 1
            
            table.add_row(
                f.name,
                true_author,
                f"[{ 'green' if match else 'red' }]{pred}[/]",
                f"{deltas['Conan Doyle']:.2f}",
                f"{deltas['Jane Austen']:.2f}",
                f"{deltas['Mark Twain']:.2f}",
                "[green]✓[/]" if match else "[red]✗[/]"
            )

    console.print(table)
    console.print(f"\n[bold green]Literary Attribution Accuracy:[/] [bold]{correct}/{total} ({(correct/total)*100:.1f}%)[/]\n")


def evaluate_federalist_attribution(holdout_per_author: int = 5):
    """Run Authorship Attribution on full Federalist Papers."""
    fed_dir = EVAL_DIR / "federalist"
    if not fed_dir.exists():
        console.print("[bold red]Federalist dataset not found![/] Run: python scripts/download_eval_data.py federalist")
        return

    hamilton_files = sorted(fed_dir.glob("hamilton_*.txt"))
    madison_files = sorted(fed_dir.glob("madison_*.txt"))

    console.print(Panel(
        f"Evaluating Historical Authorship Attribution on The Federalist Papers:\n"
        f"Hamilton ({len(hamilton_files)} full essays) vs. Madison ({len(madison_files)} full essays)\n"
        f"Composite baseline profiles generated from training corpora.",
        title="3. HISTORICAL ESSAY ATTRIBUTION (The Federalist Papers)"
    ))

    h_train = hamilton_files[:-holdout_per_author]
    m_train = madison_files[:-holdout_per_author]
    h_test = hamilton_files[-holdout_per_author:]
    m_test = madison_files[-holdout_per_author:]

    # Enroll composite baseline profiles
    fp_h = make_composite_profile(h_train, "Alexander Hamilton")
    fp_m = make_composite_profile(m_train, "James Madison")

    table = Table(title="Federalist Holdout Predictions", show_header=True)
    table.add_column("Essay", style="cyan")
    table.add_column("True Author")
    table.add_column("Predicted Author", style="bold")
    table.add_column("Hamilton Delta")
    table.add_column("Madison Delta")
    table.add_column("Result")

    correct = 0
    total = len(h_test) + len(m_test)

    for f in h_test:
        fp = create_fingerprint(ingest_file(f), label=f.name)
        dh = compare(fp, fp_h).manhattan_delta
        dm = compare(fp, fp_m).manhattan_delta
        pred = "Hamilton" if dh < dm else "Madison"
        match = (pred == "Hamilton")
        if match:
            correct += 1
        table.add_row(f.name, "Hamilton", f"[{ 'green' if match else 'red' }]{pred}[/]", f"{dh:.3f}", f"{dm:.3f}", "[green]✓[/]" if match else "[red]✗[/]")

    for f in m_test:
        fp = create_fingerprint(ingest_file(f), label=f.name)
        dh = compare(fp, fp_h).manhattan_delta
        dm = compare(fp, fp_m).manhattan_delta
        pred = "Madison" if dm < dh else "Hamilton"
        match = (pred == "Madison")
        if match:
            correct += 1
        table.add_row(f.name, "Madison", f"[{ 'green' if match else 'red' }]{pred}[/]", f"{dh:.3f}", f"{dm:.3f}", "[green]✓[/]" if match else "[red]✗[/]")

    console.print(table)
    console.print(f"\n[bold green]Federalist Attribution Accuracy:[/] [bold]{correct}/{total} ({(correct/total)*100:.1f}%)[/]\n")


if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    if task in ("all", "ai"):
        evaluate_ai_detection(max_samples=25)
    if task in ("all", "authors"):
        evaluate_literary_authorship(train_samples=5, test_samples=3)
    if task in ("all", "federalist"):
        evaluate_federalist_attribution(holdout_per_author=5)
