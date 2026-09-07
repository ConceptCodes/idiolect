import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from matplotlib.patches import Wedge

from .models import FINGERPRINT_AXES, AuthorType, ComparisonResult, Fingerprint

# Modern Forensic & Executive Palette
COLOR_NAVY = (15, 23, 42)  # Slate 900 #0F172A
COLOR_INDIGO = (79, 70, 229)  # Indigo 600 #4F46E5
COLOR_CYAN = (6, 182, 212)  # Cyan 500 #06B6D4
COLOR_EMERALD = (16, 185, 129)  # Emerald 500 #10B981
COLOR_AMBER = (245, 158, 11)  # Amber 500 #F59E0B
COLOR_CORAL = (239, 68, 68)  # Coral/Red 500 #EF4444
COLOR_TEXT_DARK = (30, 41, 59)  # Slate 800 #1E293B
COLOR_TEXT_MUTED = (100, 116, 139)  # Slate 500 #64748B
COLOR_BG_CARD = (248, 250, 252)  # Slate 50 #F8FAFC
COLOR_BG_ALT = (241, 245, 249)  # Slate 100 #F1F5F9
COLOR_BORDER = (226, 232, 240)  # Slate 200 #E2E8F0
COLOR_WHITE = (255, 255, 255)


def _clean_str(text: Any) -> str:
    """Sanitize strings for Latin-1 core font rendering in FPDF."""
    if not isinstance(text, str):
        text = str(text)
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
        "Δ": "Delta",
        "σ": "sigma",
        "›": ">",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class ReportPDF(FPDF):
    """Custom FPDF document with elegant running header and footer."""

    def __init__(self, doc_title: str = "STYLESHEET & FORENSIC LINGUISTIC REPORT"):
        super().__init__()
        self.doc_title = doc_title
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(True, margin=18)

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

    def header(self):
        self.set_xy(15, 8)
        self.set_font("helvetica", "B", 7.5)
        self.set_text_color(*COLOR_INDIGO)
        self.cell(105, 5, f"IDIOLECT  |  {self.doc_title}", align="L")
        self.set_font("helvetica", "", 7.5)
        self.set_text_color(*COLOR_TEXT_MUTED)
        self.cell(75, 5, "CONFIDENTIAL FORENSIC PROFILE", align="R")
        self.set_draw_color(*COLOR_BORDER)
        self.set_line_width(0.25)
        self.line(15, 14, 195, 14)
        self.set_xy(15, 20)

    def footer(self):
        self.set_xy(15, -14)
        self.set_draw_color(*COLOR_BORDER)
        self.set_line_width(0.25)
        self.line(15, 283, 195, 283)
        self.set_font("helvetica", "", 7.5)
        self.set_text_color(*COLOR_TEXT_MUTED)
        self.cell(
            120,
            5,
            "Idiolect Linguistic Profiling System  *  Forensic Authorship Analysis",
            align="L",
        )
        self.cell(60, 5, f"Page {self.page_no()}", align="R")


def _draw_card(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    bg_color=COLOR_BG_CARD,
    border_color=COLOR_BORDER,
    radius=2.5,
    line_width=0.3,
):
    """Draw a soft rounded card container with crisp border."""
    pdf.set_fill_color(*bg_color)
    pdf.set_draw_color(*border_color)
    pdf.set_line_width(line_width)
    pdf.rect(x, y, w, h, style="DF", round_corners=True, corner_radius=radius)


def _draw_badge(
    pdf: FPDF,
    x: float,
    y: float,
    text: str,
    bg_color: tuple,
    text_color=COLOR_WHITE,
    height=5.5,
    font_size=7.5,
) -> float:
    """Draw a rounded badge pill with centered text."""
    pdf.set_font("helvetica", "B", font_size)
    clean = _clean_str(text)
    text_w = pdf.get_string_width(clean)
    badge_w = text_w + 6
    pdf.set_fill_color(*bg_color)
    pdf.set_draw_color(*bg_color)
    pdf.rect(x, y, badge_w, height, style="DF", round_corners=True, corner_radius=height / 2)
    pdf.set_text_color(*text_color)
    pdf.set_xy(x, y + 0.5)
    pdf.cell(badge_w, height - 1, clean, align="C")
    return badge_w


def _draw_progress_bar(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    pct: float,
    fill_color: tuple,
    bg_color=COLOR_BORDER,
):
    """Draw a modern rounded horizontal meter bar."""
    pdf.set_fill_color(*bg_color)
    pdf.set_draw_color(*bg_color)
    pdf.rect(x, y, w, h, style="F", round_corners=True, corner_radius=h / 2)
    fill_w = max(h, w * (max(0.0, min(100.0, pct)) / 100.0))
    pdf.set_fill_color(*fill_color)
    pdf.set_draw_color(*fill_color)
    pdf.rect(x, y, fill_w, h, style="F", round_corners=True, corner_radius=h / 2)


def _render_score_dial(score: float, label: str, filepath: str):
    """Render a radial donut gauge with colored indicator arc."""
    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    r_out = 1.0
    r_in = 0.76

    # Background track (from 225 deg down to -45 deg = 270 deg sweep)
    w_bg = Wedge((0, 0), r_out, -45, 225, width=r_out - r_in, facecolor="#E2E8F0", edgecolor="none")
    ax.add_patch(w_bg)

    # Score arc
    score_clamped = max(0.0, min(100.0, score))
    end_deg = 225 - 270 * (score_clamped / 100.0)

    if score_clamped >= 75:
        color = "#10B981"  # Emerald
    elif score_clamped >= 50:
        color = "#06B6D4"  # Cyan
    elif score_clamped >= 35:
        color = "#F59E0B"  # Amber
    else:
        color = "#EF4444"  # Coral

    if score_clamped > 0:
        w_val = Wedge(
            (0, 0), r_out, end_deg, 225, width=r_out - r_in, facecolor=color, edgecolor="none"
        )
        ax.add_patch(w_val)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    ax.text(
        0,
        0.05,
        f"{int(round(score_clamped))}",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        0,
        -0.34,
        "/ 100",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#94A3B8",
    )

    plt.tight_layout(pad=0.1)
    plt.savefig(filepath, transparent=True)
    plt.close(fig)


def _render_radar_chart(fp: Fingerprint, filepath: str):
    """Render a 7-axis stylometric radar chart."""
    axis_names = [
        "Lexical\nRichness",
        "Syntactic\nComplexity",
        "Formality",
        "Epistemic\nStance",
        "Pacing &\nCadence",
        "Affective\nIntensity",
        "Interactive\nEngagement",
    ]
    values = [fp.axes.get(k, 0.0) for k in FINGERPRINT_AXES]

    num_vars = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), dpi=300)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#94A3B8", size=7)
    plt.ylim(0, 105)

    ax.spines["polar"].set_color("#CBD5E1")
    ax.spines["polar"].set_linewidth(1.0)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)

    ax.plot(
        angles,
        values,
        color="#4F46E5",
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor="#4F46E5",
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax.fill(angles, values, color="#6366F1", alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, size=7.5, fontweight="bold", color="#1E293B")

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(12)

    plt.tight_layout()
    plt.savefig(filepath, transparent=True)
    plt.close(fig)


def _render_comparison_radar(fp_a: Fingerprint, fp_b: Fingerprint, filepath: str):
    """Render a dual-overlay 7-axis radar chart."""
    axis_names = [
        "Lexical\nRichness",
        "Syntactic\nComplexity",
        "Formality",
        "Epistemic\nStance",
        "Pacing &\nCadence",
        "Affective\nIntensity",
        "Interactive\nEngagement",
    ]
    values_a = [fp_a.axes.get(k, 0.0) for k in FINGERPRINT_AXES]
    values_b = [fp_b.axes.get(k, 0.0) for k in FINGERPRINT_AXES]

    num_vars = len(axis_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True), dpi=300)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#94A3B8", size=7)
    plt.ylim(0, 105)

    ax.spines["polar"].set_color("#CBD5E1")
    ax.spines["polar"].set_linewidth(1.0)
    ax.grid(color="#E2E8F0", linestyle="--", linewidth=0.8)

    # Document A (Indigo)
    ax.plot(
        angles,
        values_a,
        color="#4F46E5",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        markerfacecolor="#4F46E5",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=f"A: {fp_a.label[:18]}",
    )
    ax.fill(angles, values_a, color="#6366F1", alpha=0.18)

    # Document B (Coral)
    ax.plot(
        angles,
        values_b,
        color="#F43F5E",
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        markerfacecolor="#F43F5E",
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=f"B: {fp_b.label[:18]}",
    )
    ax.fill(angles, values_b, color="#FB7185", alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_names, size=7.5, fontweight="bold", color="#1E293B")

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(filepath, transparent=True)
    plt.close(fig)


def _humanize_feature(name: str) -> str:
    """Format dot-separated feature identifier into title-cased display label."""
    if "." in name:
        name = name.split(".", 1)[1]
    return name.replace("_", " ").title()


def generate_report(fingerprint: Fingerprint, output_path: Path | None = None) -> Path:
    """Generate a high-polish forensic PDF fingerprint report.

    Args:
        fingerprint: The linguistic fingerprint profile.
        output_path: Path where the PDF will be saved.
            Defaults to 'artifacts/<label>_fingerprint.pdf'.

    Returns:
        The path to the generated PDF.
    """
    if output_path is None:
        output_path = Path("artifacts") / f"{fingerprint.label}_fingerprint.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF(doc_title="STYLESHEET & FORENSIC LINGUISTIC REPORT")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # -------------------------------------------------------------
        # PAGE 1: Executive Overview & Authorship Profile
        # -------------------------------------------------------------
        pdf.add_page()

        # 1. Hero Title Banner Card
        y_hero = 20
        _draw_card(pdf, 15, y_hero, 180, 28, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
        # Left accent stripe
        pdf.set_fill_color(*COLOR_INDIGO)
        pdf.rect(15, y_hero, 3.5, 28, style="F", round_corners=True, corner_radius=2.5)

        pdf.set_xy(23, y_hero + 3)
        pdf.set_font("helvetica", "B", 15)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(100, 6, "IDIOLECT FORENSIC REPORT")

        pdf.set_xy(23, y_hero + 10)
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(100, 5, "Stylometric Fingerprint & Authorship Integrity Profile")

        # Metadata badges on right side of Hero
        pdf.set_xy(120, y_hero + 4)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        lbl_text = (
            f"Profile: {fingerprint.label[:16]} ({fingerprint.sample_count} samples)"
            if fingerprint.sample_count > 1
            else f"Label: {fingerprint.label[:22]}"
        )
        pdf.cell(70, 5, lbl_text, align="R")

        pdf.set_xy(120, y_hero + 10)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(70, 4, f"Generated: {datetime.now().strftime('%b %d, %Y  %H:%M')}", align="R")

        if fingerprint.source_path:
            pdf.set_xy(23, y_hero + 18)
            pdf.set_font("helvetica", "I", 7.5)
            pdf.set_text_color(*COLOR_TEXT_MUTED)
            src = str(fingerprint.source_path)
            if len(src) > 75:
                src = "..." + src[-72:]
            pdf.cell(165, 4, f"Source File: {src}")

        # 2. KPI Metric Cards (4 cards across page)
        y_kpi = 52
        kpi_w = (180 - 3 * 3.5) / 4  # ~42.3 mm
        kpi_h = 22

        w_per_s = (
            (fingerprint.word_count / fingerprint.sentence_count)
            if fingerprint.sentence_count
            else 0
        )
        re = fingerprint.features.get("readability.flesch_reading_ease", 60.0)
        ttr = fingerprint.features.get(
            "lexical.ttr", fingerprint.features.get("lexical.mattr", 0.0)
        )

        kpis = [
            (f"{fingerprint.word_count:,}", "TOTAL WORDS", "Lexical tokens"),
            (f"{fingerprint.sentence_count:,}", "SENTENCES", f"Avg {w_per_s:.1f} words/sent"),
            (f"{re:.1f}", "READING EASE", "Flesch index"),
            (f"{ttr:.3f}", "LEXICAL DIVERSITY", "Type-token metric"),
        ]

        for i, (val, title, sub) in enumerate(kpis):
            kx = 15 + i * (kpi_w + 3.5)
            _draw_card(
                pdf, kx, y_kpi, kpi_w, kpi_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER
            )
            pdf.set_xy(kx + 2, y_kpi + 2.5)
            pdf.set_font("helvetica", "B", 13)
            pdf.set_text_color(*COLOR_NAVY)
            pdf.cell(kpi_w - 4, 6, val, align="C")

            pdf.set_xy(kx + 2, y_kpi + 9.5)
            pdf.set_font("helvetica", "B", 6.5)
            pdf.set_text_color(*COLOR_INDIGO)
            pdf.cell(kpi_w - 4, 4, title, align="C")

            pdf.set_xy(kx + 2, y_kpi + 14.5)
            pdf.set_font("helvetica", "", 6.5)
            pdf.set_text_color(*COLOR_TEXT_MUTED)
            pdf.cell(kpi_w - 4, 4, sub, align="C")

        # 3. Key Stylometric Dimensions (Dials)
        y_sec1 = 78
        pdf.set_xy(15, y_sec1)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(100, 5, "KEY STYLOMETRIC DIMENSIONS")

        y_dial_card = 84
        dial_card_h = 64
        _draw_card(
            pdf,
            15,
            y_dial_card,
            180,
            dial_card_h,
            bg_color=COLOR_WHITE,
            border_color=COLOR_BORDER,
        )

        lr = fingerprint.axes.get("lexical_richness", 0)
        sc = fingerprint.axes.get("syntactic_complexity", 0)
        pc = fingerprint.axes.get("pacing_cadence", 0)
        vd_score = (lr + sc) / 2
        cf_score = pc
        ls_score = lr
        avg_zipf = fingerprint.features.get("lexical.avg_zipf_frequency", 4.5)
        c_score = max(0.0, min(100.0, (avg_zipf - 3.0) / 3.0 * 100))

        dials = [
            (vd_score, "Voice Distinctiveness", "Vocabulary & syntax"),
            (cf_score, "Clarity & Cadence", "Pacing & sentence rhythm"),
            (ls_score, "Lexical Sophistication", "Rare words & breadth"),
            (c_score, "Contemporaneity", "Modern frequency alignment"),
        ]

        dial_w = 180 / 4
        for i, (score, d_title, d_sub) in enumerate(dials):
            p = tmp_path / f"dial_{i}.png"
            _render_score_dial(score, d_title, str(p))

            dx = 15 + i * dial_w
            pdf.image(str(p), x=dx + (dial_w - 32) / 2, y=y_dial_card + 3, w=32)

            pdf.set_xy(dx, y_dial_card + 40)
            pdf.set_font("helvetica", "B", 8)
            pdf.set_text_color(*COLOR_NAVY)
            pdf.cell(dial_w, 4.5, d_title, align="C")

            pdf.set_xy(dx, y_dial_card + 46)
            pdf.set_font("helvetica", "", 6.8)
            pdf.set_text_color(*COLOR_TEXT_MUTED)
            pdf.cell(dial_w, 4, d_sub, align="C")

        # 4. Authorship & AI Integrity Analysis Card
        y_sec2 = 153
        pdf.set_xy(15, y_sec2)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(100, 5, "AUTHORSHIP & AI DETECTION ANALYSIS")

        y_ai_card = 159
        ai_card_h = 75
        _draw_card(
            pdf, 15, y_ai_card, 180, ai_card_h, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER
        )

        is_human = fingerprint.author_type == AuthorType.HUMAN
        is_ai = fingerprint.author_type == AuthorType.AI

        if is_human:
            verdict_color = COLOR_EMERALD
            verdict_label = "HUMAN AUTHORSHIP VERIFIED"
            summary_text = (
                "Stylometric variance and cadence strongly align with natural human prose patterns."
            )
        elif is_ai:
            verdict_color = COLOR_CORAL
            verdict_label = "AI-GENERATED CONTENT DETECTED"
            summary_text = (
                "High syntactic uniformity and flattened emotional tone strongly indicate LLM"
                " generation."
            )
        else:
            verdict_color = COLOR_AMBER
            verdict_label = "UNCERTAIN / HYBRID SIGNALS"
            summary_text = (
                "Mixed indicators detected; text exhibits both human variance and synthetic"
                " regularities."
            )

        # Left accent stripe
        pdf.set_fill_color(*verdict_color)
        pdf.rect(15, y_ai_card, 3.5, ai_card_h, style="F", round_corners=True, corner_radius=2.5)

        # Verdict Badge
        _draw_badge(pdf, 24, y_ai_card + 6, verdict_label, verdict_color, height=6.5, font_size=8)

        # Confidence percentage on right
        pdf.set_xy(110, y_ai_card + 6)
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*COLOR_NAVY)
        conf_pct = fingerprint.ai_confidence * 100
        pdf.cell(80, 6, f"Confidence: {conf_pct:.1f}%", align="R")

        # Progress bar
        _draw_progress_bar(pdf, 24, y_ai_card + 16, 166, 4, conf_pct, verdict_color)

        # Divider line inside card
        pdf.set_draw_color(*COLOR_BORDER)
        pdf.set_line_width(0.2)
        pdf.line(24, y_ai_card + 25, 190, y_ai_card + 25)

        # Forensic Signals Section
        pdf.set_xy(24, y_ai_card + 28)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        pdf.cell(100, 4.5, "Observed Forensic Signals & Model Indicators:")

        indicators = list(fingerprint.ai_indicators.items()) if fingerprint.ai_indicators else []
        indicators.sort(key=lambda x: abs(x[1]), reverse=True)

        y_sig = y_ai_card + 35
        for k, v in indicators[:3]:
            human_name = _humanize_feature(k)
            if "Uniformity" in human_name:
                desc = f"Sentence length regularity: {v:.2f}"
            elif "Flatness" in human_name:
                desc = f"Emotional variation index: {v:.2f}"
            elif "Contraction" in human_name:
                desc = f"Contraction suppression score: {v:.2f}"
            else:
                desc = f"Relative indicator weight: {v:.2f}"

            # Bullet dot
            pdf.set_fill_color(*verdict_color)
            pdf.circle(26, y_sig + 2, 0.8, style="F")

            pdf.set_xy(30, y_sig)
            pdf.set_font("helvetica", "B", 7.5)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.cell(60, 4.5, human_name)

            pdf.set_font("helvetica", "", 7.5)
            pdf.set_text_color(*COLOR_TEXT_MUTED)
            pdf.cell(100, 4.5, f"--  {desc}")
            y_sig += 6.5

        # Summary footer text inside card
        pdf.set_xy(24, y_ai_card + 58)
        pdf.set_font("helvetica", "I", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.multi_cell(166, 4, summary_text)

        # -------------------------------------------------------------
        # PAGE 2: 7-Axis Spectrum & Standout Traits
        # -------------------------------------------------------------
        pdf.add_page()

        pdf.set_xy(15, 20)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(180, 6, "7-AXIS LINGUISTIC RADAR PROFILE", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(
            180,
            4.5,
            "Standardized multi-dimensional stylometric fingerprint mapped from 0 to 100.",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        # Radar chart container card
        y_radar_card = 33
        _draw_card(
            pdf,
            15,
            y_radar_card,
            180,
            115,
            bg_color=COLOR_WHITE,
            border_color=COLOR_BORDER,
        )

        radar_path = tmp_path / "radar.png"
        _render_radar_chart(fingerprint, str(radar_path))
        pdf.image(str(radar_path), x=15 + (180 - 110) / 2, y=y_radar_card + 2.5, w=110)

        # Standout Idiolectal Traits section
        y_traits_sec = 153
        pdf.set_xy(15, y_traits_sec)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(180, 6, "STANDOUT IDIOLECTAL TRAITS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(
            180,
            4.5,
            "Forensic features showing significant statistical divergence (|z| >= 1.0) from corpus"
            " norms.",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        y_trait = 165
        traits = fingerprint.standout_traits[:5]
        if traits:
            for trait in traits:
                _draw_card(
                    pdf,
                    15,
                    y_trait,
                    180,
                    18,
                    bg_color=COLOR_BG_CARD,
                    border_color=COLOR_BORDER,
                )

                z = trait.get("z_score", 0.0)
                is_high = z > 0
                badge_bg = COLOR_INDIGO if is_high else COLOR_AMBER
                badge_label = f"+{z:.1f} sigma HIGH" if is_high else f"{z:.1f} sigma LOW"

                _draw_badge(pdf, 20, y_trait + 4.5, badge_label, badge_bg, height=5, font_size=7)

                name = _humanize_feature(trait.get("feature", "Unknown"))
                pdf.set_xy(55, y_trait + 3)
                pdf.set_font("helvetica", "B", 8.5)
                pdf.set_text_color(*COLOR_NAVY)
                pdf.cell(80, 5, name)

                interp = trait.get(
                    "interpretation", "Diverges notably from standard corpus baselines."
                )
                pdf.set_xy(55, y_trait + 8.5)
                pdf.set_font("helvetica", "", 7.5)
                pdf.set_text_color(*COLOR_TEXT_MUTED)
                pdf.cell(130, 4.5, interp[:95])

                y_trait += 21
        else:
            _draw_card(pdf, 15, y_trait, 180, 25, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
            pdf.set_xy(20, y_trait + 8)
            pdf.set_font("helvetica", "I", 9)
            pdf.set_text_color(*COLOR_TEXT_MUTED)
            pdf.cell(
                170,
                8,
                "No extreme outlier traits (|z| >= 1.0) detected; metrics fall within standard"
                " baseline distribution.",
                align="C",
            )

        # -------------------------------------------------------------
        # PAGE 3: Detailed Feature Metrics Table
        # -------------------------------------------------------------
        pdf.add_page()

        pdf.set_xy(15, 20)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(180, 6, "COMPREHENSIVE FEATURE METRICS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(
            180,
            4.5,
            "Detailed quantitative feature extraction values grouped by linguistic analytical"
            " stratum.",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        strata: dict[str, dict[str, float]] = {}
        for k, v in fingerprint.features.items():
            parts = k.split(".", 1)
            if len(parts) == 2:
                s, f = parts
                strata.setdefault(s.title(), {})[f] = v

        for stratum, feats in sorted(strata.items()):
            pdf.ln(3)
            # Stratum header bar
            pdf.set_font("helvetica", "B", 8.5)
            pdf.set_fill_color(*COLOR_NAVY)
            pdf.set_text_color(*COLOR_WHITE)
            pdf.set_draw_color(*COLOR_NAVY)
            pdf.cell(
                180,
                6.5,
                f"  {stratum.upper()} STRATUM FEATURES ({len(feats)} Metrics)",
                fill=True,
                border=1,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

            # Table subheader
            pdf.set_font("helvetica", "B", 7.5)
            pdf.set_fill_color(*COLOR_BG_ALT)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.set_draw_color(*COLOR_BORDER)
            pdf.cell(100, 5.5, "  Metric / Feature Name", fill=True, border=1)
            pdf.cell(40, 5.5, "  Value", fill=True, border=1)
            pdf.cell(
                40,
                5.5,
                "  Stratum Category",
                fill=True,
                border=1,
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )

            pdf.set_font("helvetica", "", 7.5)
            pdf.set_text_color(*COLOR_TEXT_DARK)

            fill = False
            for f_name, f_val in sorted(feats.items()):
                pdf.set_fill_color(*(COLOR_BG_CARD if fill else COLOR_WHITE))
                name_human = _humanize_feature(f_name)

                if abs(f_val) >= 100:
                    val_str = f"{f_val:,.1f}"
                elif abs(f_val) >= 1:
                    val_str = f"{f_val:.3f}"
                else:
                    val_str = f"{f_val:.4f}"

                pdf.cell(100, 5.2, f"  {name_human}", fill=True, border="LRB")
                pdf.cell(40, 5.2, f"  {val_str}", fill=True, border="LRB")
                pdf.cell(
                    40,
                    5.2,
                    f"  {stratum}",
                    fill=True,
                    border="LRB",
                    new_x=XPos.LMARGIN,
                    new_y=YPos.NEXT,
                )
                fill = not fill

    pdf.output(str(output_path))
    return output_path


def generate_comparison_report(
    result: ComparisonResult,
    fp_a: Fingerprint,
    fp_b: Fingerprint,
    output_path: Path | None = None,
) -> Path:
    """Generate a dual forensic PDF comparison report.

    Args:
        result: The comparison result.
        fp_a: First fingerprint profile.
        fp_b: Second fingerprint profile.
        output_path: Path where the PDF will be saved.
            Defaults to 'artifacts/compare_<label_a>_<label_b>.pdf'.

    Returns:
        The path to the generated PDF.
    """
    if output_path is None:
        output_path = Path("artifacts") / f"compare_{fp_a.label}_{fp_b.label}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF(doc_title="DUAL FORENSIC COMPARISON REPORT")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # -------------------------------------------------------------
        # PAGE 1: Comparison Executive Summary
        # -------------------------------------------------------------
        pdf.add_page()

        # 1. Hero Card
        y_hero = 20
        _draw_card(pdf, 15, y_hero, 180, 26, bg_color=COLOR_BG_CARD, border_color=COLOR_BORDER)
        pdf.set_fill_color(*COLOR_INDIGO)
        pdf.rect(15, y_hero, 3.5, 26, style="F", round_corners=True, corner_radius=2.5)

        pdf.set_xy(23, y_hero + 3)
        pdf.set_font("helvetica", "B", 15)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(100, 6, "IDIOLECT COMPARISON REPORT")

        pdf.set_xy(23, y_hero + 10)
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(100, 5, "Dual Stylometric Cross-Verification & Authorship Attribution")

        pdf.set_xy(110, y_hero + 5)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        pdf.cell(80, 5, f"Comparison ID: {fp_a.label[:12]}_vs_{fp_b.label[:12]}", align="R")

        # 2. Side-by-side Document Profile Cards
        y_docs = 49
        doc_w = (180 - 4) / 2
        doc_h = 24

        # Doc A (Indigo Accent)
        _draw_card(pdf, 15, y_docs, doc_w, doc_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
        pdf.set_fill_color(*COLOR_INDIGO)
        pdf.rect(15, y_docs, 3, doc_h, style="F", round_corners=True, corner_radius=2)
        pdf.set_xy(21, y_docs + 3)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*COLOR_NAVY)
        doc_a_label = (
            f"Doc A: {fp_a.label[:16]} ({fp_a.sample_count}s)"
            if fp_a.sample_count > 1
            else f"Doc A: {fp_a.label[:22]}"
        )
        pdf.cell(doc_w - 10, 5, doc_a_label)

        pdf.set_xy(21, y_docs + 9)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        counts_a = f"Words: {fp_a.word_count:,}  |  Sentences: {fp_a.sentence_count:,}"
        pdf.cell(doc_w - 10, 4, counts_a)

        badge_a_color = COLOR_EMERALD if fp_a.author_type.value == "human" else COLOR_CORAL
        _draw_badge(
            pdf,
            21,
            y_docs + 15,
            f"Type: {fp_a.author_type.value.upper()}",
            badge_a_color,
            height=5,
            font_size=6.8,
        )

        # Doc B (Coral Accent)
        bx = 15 + doc_w + 4
        _draw_card(pdf, bx, y_docs, doc_w, doc_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
        pdf.set_fill_color(*COLOR_CORAL)
        pdf.rect(bx, y_docs, 3, doc_h, style="F", round_corners=True, corner_radius=2)
        pdf.set_xy(bx + 6, y_docs + 3)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*COLOR_NAVY)
        doc_b_label = (
            f"Doc B: {fp_b.label[:16]} ({fp_b.sample_count}s)"
            if fp_b.sample_count > 1
            else f"Doc B: {fp_b.label[:22]}"
        )
        pdf.cell(doc_w - 10, 5, doc_b_label)

        pdf.set_xy(bx + 6, y_docs + 9)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        counts_b = f"Words: {fp_b.word_count:,}  |  Sentences: {fp_b.sentence_count:,}"
        pdf.cell(doc_w - 10, 4, counts_b)

        badge_b_color = COLOR_EMERALD if fp_b.author_type.value == "human" else COLOR_CORAL
        _draw_badge(
            pdf,
            bx + 6,
            y_docs + 15,
            f"Type: {fp_b.author_type.value.upper()}",
            badge_b_color,
            height=5,
            font_size=6.8,
        )

        # 3. Verdict Card
        y_verdict = 76
        v_h = 38
        _draw_card(
            pdf,
            15,
            y_verdict,
            180,
            v_h,
            bg_color=COLOR_BG_CARD,
            border_color=COLOR_BORDER,
        )

        sim_pct = result.cosine_similarity * 100
        if sim_pct >= 80:
            v_color = COLOR_EMERALD
            v_label = "SAME AUTHOR PROBABLE"
        elif sim_pct >= 60:
            v_color = COLOR_AMBER
            v_label = "INCONCLUSIVE / MIXED SIMILARITY"
        else:
            v_color = COLOR_CORAL
            v_label = "DISTINCT AUTHORS INDICATED"

        pdf.set_fill_color(*v_color)
        pdf.rect(15, y_verdict, 3.5, v_h, style="F", round_corners=True, corner_radius=2.5)

        _draw_badge(pdf, 24, y_verdict + 5, v_label, v_color, height=6.5, font_size=8)

        pdf.set_xy(110, y_verdict + 5)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(80, 6, f"Similarity: {sim_pct:.1f}%", align="R")

        _draw_progress_bar(pdf, 24, y_verdict + 15, 166, 4, sim_pct, v_color)

        # Metrics row
        pdf.set_xy(24, y_verdict + 24)
        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        pdf.cell(55, 5, f"Cosine Sim: {result.cosine_similarity:.4f}")
        pdf.cell(55, 5, f"Cosine Delta: {result.cosine_delta:.4f}")
        pdf.cell(55, 5, f"Burrows Delta: {result.manhattan_delta:.4f}")

        # 4. Dual Radar Chart Card
        y_r_sec = 117
        pdf.set_xy(15, y_r_sec)
        pdf.set_font("helvetica", "B", 10)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(100, 5, "7-AXIS DUAL STYLOMETRIC SPECTRUM")

        y_r_card = 123
        r_card_h = 135
        _draw_card(
            pdf, 15, y_r_card, 180, r_card_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER
        )

        radar_path = tmp_path / "comp_radar.png"
        _render_comparison_radar(fp_a, fp_b, str(radar_path))
        pdf.image(str(radar_path), x=15 + (180 - 130) / 2, y=y_r_card + 3, w=130)

        # -------------------------------------------------------------
        # PAGE 2: Axis Comparison & Feature Convergence/Divergence
        # -------------------------------------------------------------
        pdf.add_page()

        pdf.set_xy(15, 20)
        pdf.set_font("helvetica", "B", 13)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(180, 6, "AXIS-BY-AXIS DELTA BREAKDOWN", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 8)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(
            180,
            4.5,
            "Comparative variance across the seven core stylometric dimensions.",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        # Table
        pdf.ln(3)
        pdf.set_font("helvetica", "B", 8)
        pdf.set_fill_color(*COLOR_NAVY)
        pdf.set_text_color(*COLOR_WHITE)
        pdf.set_draw_color(*COLOR_NAVY)

        pdf.cell(60, 6.5, "  Linguistic Axis", fill=True, border=1)
        pdf.cell(30, 6.5, f"  Doc A ({fp_a.label[:8]})", fill=True, border=1)
        pdf.cell(30, 6.5, f"  Doc B ({fp_b.label[:8]})", fill=True, border=1)
        pdf.cell(30, 6.5, "  Delta (|A-B|)", fill=True, border=1)
        pdf.cell(30, 6.5, "  Alignment", fill=True, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("helvetica", "", 7.8)
        pdf.set_text_color(*COLOR_TEXT_DARK)
        fill = False

        for axis in FINGERPRINT_AXES:
            val_a = fp_a.axes.get(axis, 0.0)
            val_b = fp_b.axes.get(axis, 0.0)
            delta = result.axis_deltas.get(axis, abs(val_a - val_b))

            pdf.set_fill_color(*(COLOR_BG_CARD if fill else COLOR_WHITE))
            axis_name = axis.replace("_", " ").title()

            if delta < 15:
                align_status = "High Match"
            elif delta < 30:
                align_status = "Moderate"
            else:
                align_status = "Divergent"

            pdf.cell(60, 5.5, f"  {axis_name}", fill=True, border="LRB")
            pdf.cell(30, 5.5, f"  {val_a:.1f}", fill=True, border="LRB")
            pdf.cell(30, 5.5, f"  {val_b:.1f}", fill=True, border="LRB")
            pdf.cell(30, 5.5, f"  {delta:.1f}", fill=True, border="LRB")
            pdf.cell(
                30,
                5.5,
                f"  {align_status}",
                fill=True,
                border="LRB",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            fill = not fill

        # Side-by-side Convergent vs Divergent Cards
        y_side = 88
        col_w = (180 - 6) / 2
        col_h = 95

        # Convergent Card (Left)
        _draw_card(pdf, 15, y_side, col_w, col_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
        pdf.set_fill_color(*COLOR_EMERALD)
        pdf.rect(15, y_side, col_w, 2.5, style="F", round_corners=True, corner_radius=1.5)

        pdf.set_xy(20, y_side + 6)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(col_w - 10, 5, "Most Convergent Features (Aligned)")

        pdf.set_xy(20, y_side + 11.5)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(col_w - 10, 4, "Habits shared most closely between both texts:")

        y_c = y_side + 18
        for f in result.most_similar_features[:8]:
            pdf.set_fill_color(*COLOR_EMERALD)
            pdf.circle(22, y_c + 2, 0.7, style="F")
            pdf.set_xy(25, y_c)
            pdf.set_font("helvetica", "", 7.8)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.cell(col_w - 15, 4.5, _humanize_feature(f))
            y_c += 6.5

        # Divergent Card (Right)
        rx = 15 + col_w + 6
        _draw_card(pdf, rx, y_side, col_w, col_h, bg_color=COLOR_WHITE, border_color=COLOR_BORDER)
        pdf.set_fill_color(*COLOR_CORAL)
        pdf.rect(rx, y_side, col_w, 2.5, style="F", round_corners=True, corner_radius=1.5)

        pdf.set_xy(rx + 5, y_side + 6)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.set_text_color(*COLOR_NAVY)
        pdf.cell(col_w - 10, 5, "Most Divergent Features (Contrasting)")

        pdf.set_xy(rx + 5, y_side + 11.5)
        pdf.set_font("helvetica", "", 7.5)
        pdf.set_text_color(*COLOR_TEXT_MUTED)
        pdf.cell(col_w - 10, 4, "Strongest markers of separation and style contrast:")

        y_d = y_side + 18
        for f in result.most_divergent_features[:8]:
            pdf.set_fill_color(*COLOR_CORAL)
            pdf.circle(rx + 7, y_d + 2, 0.7, style="F")
            pdf.set_xy(rx + 10, y_d)
            pdf.set_font("helvetica", "", 7.8)
            pdf.set_text_color(*COLOR_TEXT_DARK)
            pdf.cell(col_w - 15, 4.5, _humanize_feature(f))
            y_d += 6.5

    pdf.output(str(output_path))
    return output_path
