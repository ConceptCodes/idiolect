import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .models import FINGERPRINT_AXES, ComparisonResult, Fingerprint

# Color scheme
COLOR_NAVY = (26, 26, 46)  # #1a1a2e
COLOR_GREEN = (12, 206, 107)  # #0cce6b
COLOR_AMBER = (255, 164, 0)  # #ffa400
COLOR_RED = (255, 78, 66)  # #ff4e42
COLOR_TEXT = (51, 51, 51)  # #333333
COLOR_BG_ALT = (245, 245, 245)  # #f5f5f5


def _clean_str(text: Any) -> str:
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "—": "--",
        "–": "-",
        "•": "*",
        "↑": "(+)",
        "↓": "(-)",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    def cell(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = _clean_str(kwargs["text"])
        elif len(args) >= 3:
            args = list(args)
            args[2] = _clean_str(args[2])
            args = tuple(args)
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        if "text" in kwargs:
            kwargs["text"] = _clean_str(kwargs["text"])
        elif len(args) >= 3:
            args = list(args)
            args[2] = _clean_str(args[2])
            args = tuple(args)
        return super().multi_cell(*args, **kwargs)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(*COLOR_TEXT)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _get_score_color(score: float) -> tuple[int, int, int]:
    if score >= 75:
        return COLOR_GREEN
    elif score >= 50:
        return COLOR_AMBER
    else:
        return COLOR_RED


def _get_score_color_hex(score: float) -> str:
    if score >= 75:
        return "#0cce6b"
    elif score >= 50:
        return "#ffa400"
    else:
        return "#ff4e42"


def _render_score_dial(score: float, label: str, filepath: str):
    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={"projection": "polar"})

    # We want a dial from -225 deg to 45 deg = 270 deg sweep
    # Start at 225 deg (bottom left), end at -45 deg (bottom right).
    # Matplotlib polar goes counterclockwise from right.
    # Let's map 0-100 to angle.
    theta_start = np.radians(225)
    theta_end = np.radians(-45)

    color_hex = _get_score_color_hex(score)

    # Background arc
    ax.bar(0, 1, bottom=0, width=2 * np.pi, color="none")  # invisible center
    theta_range = np.linspace(theta_end, theta_start, 100)
    ax.plot(theta_range, [1] * 100, color="#e0e0e0", linewidth=10)

    # Filled arc
    fill_amount = score / 100.0
    theta_fill_end = theta_start - fill_amount * (theta_start - theta_end)
    theta_filled = np.linspace(theta_fill_end, theta_start, int(100 * fill_amount) + 1)
    if len(theta_filled) > 0:
        ax.plot(
            theta_filled,
            [1] * len(theta_filled),
            color=color_hex,
            linewidth=10,
            solid_capstyle="round",
        )

    ax.set_axis_off()
    ax.text(
        0,
        0,
        f"{int(score)}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=color_hex,
    )

    plt.tight_layout()
    plt.savefig(filepath, transparent=True, dpi=300)
    plt.close(fig)


def _render_radar_chart(fp: Fingerprint, filepath: str):
    labels = [k.replace("_", " ").title() for k in FINGERPRINT_AXES]
    values = [fp.axes.get(k, 0.0) for k in FINGERPRINT_AXES]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, color="#1a1a2e", linewidth=2)
    ax.fill(angles, values, color="#1a1a2e", alpha=0.25)

    # Fix axis to go in the right order and start at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(filepath, transparent=True, dpi=300)
    plt.close(fig)


def _render_comparison_radar(fp_a: Fingerprint, fp_b: Fingerprint, filepath: str):
    labels = [k.replace("_", " ").title() for k in FINGERPRINT_AXES]
    values_a = [fp_a.axes.get(k, 0.0) for k in FINGERPRINT_AXES]
    values_b = [fp_b.axes.get(k, 0.0) for k in FINGERPRINT_AXES]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # FP A
    ax.plot(angles, values_a, color="#0cce6b", linewidth=2, label=fp_a.label)
    ax.fill(angles, values_a, color="#0cce6b", alpha=0.2)

    # FP B
    ax.plot(angles, values_b, color="#ffa400", linewidth=2, label=fp_b.label)
    ax.fill(angles, values_b, color="#ffa400", alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 100)

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(filepath, transparent=True, dpi=300)
    plt.close(fig)


def _humanize_feature(name: str) -> str:
    if "." in name:
        name = name.split(".", 1)[1]
    return name.replace("_", " ").title()


def generate_report(fingerprint: Fingerprint, output_path: Path) -> Path:
    """Generate a PDF fingerprint report.

    Args:
        fingerprint: The fingerprint to report on.
        output_path: Where to save the PDF.

    Returns:
        The path to the generated PDF file.
    """
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # --- PAGE 1: Summary ---
        pdf.add_page()

        # Header
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(
            0, 12, "IDIOLECT — Linguistic Fingerprint Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )

        pdf.set_font("helvetica", "", 14)
        pdf.set_text_color(*COLOR_TEXT)
        pdf.cell(0, 8, f"Label: {fingerprint.label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(
            0,
            8,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        # Stats
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 6, "Document Statistics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 11)
        pdf.cell(0, 6, f"Words: {fingerprint.word_count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(
            0, 6, f"Sentences: {fingerprint.sentence_count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        if fingerprint.source_path:
            pdf.cell(
                0, 6, f"Source: {fingerprint.source_path}", new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )

        pdf.ln(10)

        # Dials calculation
        lr = fingerprint.axes.get("lexical_richness", 0)
        sc = fingerprint.axes.get("syntactic_complexity", 0)
        pc = fingerprint.axes.get("pacing_cadence", 0)

        vd_score = (lr + sc) / 2
        cf_score = pc  # Using pacing_cadence for clarity & flow
        ls_score = lr
        # Contemporaneity: derived from rare word usage, Zipf frequency, and formality
        avg_zipf = fingerprint.features.get("lexical.avg_zipf_frequency", 4.5)
        c_score = max(0.0, min(100.0, (avg_zipf - 3.0) / 3.0 * 100))

        dials = [
            (vd_score, "Voice Distinctiveness"),
            (cf_score, "Clarity & Flow"),
            (ls_score, "Lexical Sophistication"),
            (c_score, "Contemporaneity"),
        ]

        dial_paths = []
        for i, (score, label) in enumerate(dials):
            p = tmp_path / f"dial_{i}.png"
            _render_score_dial(score, label, str(p))
            dial_paths.append(p)

        # Place Dials in PDF (4 across)
        y_dials = pdf.get_y()
        col_w = pdf.epw / 4
        for i, (p, (_, label)) in enumerate(zip(dial_paths, dials)):
            x = pdf.l_margin + i * col_w
            pdf.image(str(p), x=x + (col_w - 40) / 2, y=y_dials, w=40)

            # Label
            pdf.set_xy(x, y_dials + 42)
            pdf.set_font("helvetica", "B", 9)
            pdf.multi_cell(col_w, 4, label, align="C")

        pdf.set_y(y_dials + 55)

        # AI Detection Box
        pdf.ln(10)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 8, "AI Detection Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_fill_color(*COLOR_BG_ALT)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(60, 10, f"Author: {fingerprint.author_type.value.upper()}", fill=True)
        pdf.cell(
            60,
            10,
            f"Confidence: {fingerprint.ai_confidence * 100:.1f}%",
            fill=True,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        pdf.set_font("helvetica", "", 10)
        expl = "AI indicators observed in text analysis."
        if fingerprint.ai_indicators:
            top_indicators = sorted(
                fingerprint.ai_indicators.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            signals_str = ", ".join(f"{_humanize_feature(k)} ({v:.2f})" for k, v in top_indicators)
            expl = f"Key signals: {signals_str}"
        pdf.multi_cell(0, 8, expl, fill=True)

        # --- PAGE 2: Radar Chart + Standout Traits ---
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Linguistic Fingerprint Profile", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        radar_path = tmp_path / "radar.png"
        _render_radar_chart(fingerprint, str(radar_path))
        pdf.image(str(radar_path), x=(pdf.w - 100) / 2, y=pdf.get_y(), w=100)
        pdf.set_y(pdf.get_y() + 105)

        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Standout Traits", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 11)

        if fingerprint.standout_traits:
            for trait in fingerprint.standout_traits[:8]:
                name = _humanize_feature(trait.get("feature", "Unknown"))
                z = trait.get("z_score", 0)
                arrow = "↑" if z > 0 else "↓"
                interpretation = trait.get("interpretation")
                text = f"• {name} (z = {z:+.2f} {arrow})"
                if interpretation:
                    text += f" — {interpretation}"
                pdf.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.cell(
                0,
                8,
                "No significant standout traits detected.",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

        # --- PAGE 3: Detailed Metrics Table ---
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Detailed Feature Metrics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Group by stratum
        strata = {}
        for k, v in fingerprint.features.items():
            parts = k.split(".", 1)
            if len(parts) == 2:
                s, f = parts
                strata.setdefault(s.title(), {})[f] = v

        for stratum, feats in strata.items():
            pdf.ln(4)
            pdf.set_font("helvetica", "B", 12)
            pdf.set_fill_color(*COLOR_NAVY)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 8, f" {stratum} Features", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_text_color(*COLOR_TEXT)
            pdf.set_font("helvetica", "", 10)

            fill = False
            for f_name, f_val in sorted(feats.items()):
                pdf.set_fill_color(*COLOR_BG_ALT)

                name_human = _humanize_feature(f_name)
                # Metric | Value
                pdf.cell(100, 6, name_human, fill=fill)
                pdf.cell(0, 6, f"{f_val:.4f}", fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                fill = not fill

    pdf.output(str(output_path))
    return output_path


def generate_comparison_report(
    result: ComparisonResult, fp_a: Fingerprint, fp_b: Fingerprint, output_path: Path
) -> Path:
    """Generate a PDF comparison report between two fingerprints.

    Args:
        result: The comparison result.
        fp_a: First fingerprint.
        fp_b: Second fingerprint.
        output_path: Where to save the PDF.

    Returns:
        The path to the generated PDF.
    """
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        pdf.add_page()

        # Header
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(0, 12, "IDIOLECT — Comparison Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 14)
        pdf.set_text_color(*COLOR_TEXT)
        pdf.cell(0, 8, f"Document A: {fp_a.label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 8, f"Document B: {fp_b.label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Radar
        radar_path = tmp_path / "comp_radar.png"
        _render_comparison_radar(fp_a, fp_b, str(radar_path))
        pdf.image(str(radar_path), x=(pdf.w - 120) / 2, y=pdf.get_y() + 5, w=120)
        pdf.set_y(pdf.get_y() + 130)

        # Metrics
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "Similarity Assessment", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 11)
        pdf.cell(
            0,
            6,
            f"Cosine Similarity: {result.cosine_similarity:.4f}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.cell(
            0, 6, f"Cosine Delta: {result.cosine_delta:.4f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        pdf.cell(
            0,
            6,
            f"Manhattan Delta (Burrows): {result.manhattan_delta:.4f}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.set_fill_color(*COLOR_NAVY)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(
            0,
            10,
            f" Same-Author Likelihood: {result.same_author_likelihood.replace('_', ' ').upper()}",
            fill=True,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        pdf.set_text_color(*COLOR_TEXT)

        # Similar/Divergent
        pdf.ln(10)

        x_start = pdf.l_margin
        col_w = pdf.epw / 2 - 5

        # Left column (Similar)
        y_start = pdf.get_y()
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(col_w, 8, "Most Similar Features", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        for f in result.most_similar_features[:8]:
            pdf.cell(col_w, 6, f"• {_humanize_feature(f)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Right column (Divergent)
        pdf.set_xy(x_start + col_w + 10, y_start)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(col_w, 8, "Most Divergent Features", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_xy(x_start + col_w + 10, y_start + 8)
        pdf.set_font("helvetica", "", 10)
        for f in result.most_divergent_features[:8]:
            # we need to adjust X for each new line in the right column
            pdf.set_x(x_start + col_w + 10)
            pdf.cell(col_w, 6, f"• {_humanize_feature(f)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(output_path))
    return output_path
