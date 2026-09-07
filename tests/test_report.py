from pathlib import Path

from idiolect.comparison import compare
from idiolect.fingerprint import create_fingerprint_from_text
from idiolect.report import generate_comparison_report, generate_report


def test_generate_report_pdf(tmp_path: Path):
    text = (
        "In the tranquil village of Oakhaven, the morning fog clung low to the cobblestones. "
        "Eleanor hurried past the shuttered bakeries, clutching an unread letter in her pocket. "
        "Could the rumors of the northern convoy be true? She could hardly bear the suspense."
    )
    fp = create_fingerprint_from_text(text, label="Eleanor Sample")
    pdf_out = tmp_path / "report.pdf"

    result_path = generate_report(fp, pdf_out)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000
    with open(result_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"


def test_generate_comparison_report_pdf(tmp_path: Path):
    text1 = "The quick brown fox jumps gracefully over the sleepy dog on a crisp autumn morning."
    text2 = "A fast auburn fox leapt effortlessly above the resting hound during late September."

    fp1 = create_fingerprint_from_text(text1, label="Doc A")
    fp2 = create_fingerprint_from_text(text2, label="Doc B")
    comp = compare(fp1, fp2)

    pdf_out = tmp_path / "comparison.pdf"
    result_path = generate_comparison_report(comp, fp1, fp2, pdf_out)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000
    with open(result_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-"
