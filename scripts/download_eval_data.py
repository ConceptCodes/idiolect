"""Download evaluation benchmark datasets for idiolect.

Supports:
1. 'ai-human': Paired Human vs. AI texts from HC3 and Student Paper datasets.
2. 'federalist': The canonical Federalist Papers (Hamilton vs. Madison, full essays).
3. 'authors': Multi-author literary benchmark (Conan Doyle vs. Jane Austen vs. Mark Twain).
"""

import json
import re
import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


def download_ai_human_benchmark(sample_size: int = 50) -> Path:
    """Download paired Human vs ChatGPT essays from HC3 dataset."""
    out_dir = EVAL_DIR / "ai_human"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching {sample_size} Human and {sample_size} AI samples from HC3...")
    url = f"https://datasets-server.huggingface.co/rows?dataset=Hello-SimpleAI%2FHC3&config=all&split=train&offset=0&limit={sample_size}"
    req = urllib.request.Request(url, headers={"User-Agent": "idiolect-eval/0.1"})
    
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        
    rows = data.get("rows", [])
    count = 0
    for item in rows:
        row = item.get("row", {})
        human_answers = row.get("human_answers", [])
        chatgpt_answers = row.get("chatgpt_answers", [])
        
        if human_answers and chatgpt_answers:
            h_text = human_answers[0].strip()
            ai_text = chatgpt_answers[0].strip()
            
            # Require minimum length for fair stylometric analysis
            if len(h_text.split()) >= 60 and len(ai_text.split()) >= 60:
                count += 1
                (out_dir / f"human_{count:03d}.txt").write_text(h_text, encoding="utf-8")
                (out_dir / f"ai_{count:03d}.txt").write_text(ai_text, encoding="utf-8")
                
    print(f"✓ Saved {count} Human texts and {count} AI texts to {out_dir}")
    return out_dir


def download_federalist_benchmark() -> Path:
    """Download and parse full Federalist Papers from Project Gutenberg (skipping TOC)."""
    out_dir = EVAL_DIR / "federalist"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading full Federalist Papers from Project Gutenberg...")
    url = "https://www.gutenberg.org/cache/epub/18/pg18.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=20) as resp:
        full_text = resp.read().decode("utf-8")
        
    # Skip the Table of Contents (first ~8,500 characters)
    body_text = full_text[8500:]
    
    roman_map = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
        "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
        "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30,
        "XXXI": 31, "XXXII": 32, "XXXIII": 33, "XXXIV": 34, "XXXV": 35, "XXXVI": 36, "XXXVII": 37, "XXXVIII": 38, "XXXIX": 39, "XL": 40,
        "XLI": 41, "XLII": 42, "XLIII": 43, "XLIV": 44, "XLV": 45, "XLVI": 46, "XLVII": 47, "XLVIII": 48, "XLIX": 49, "L": 50,
        "LI": 51, "LII": 52, "LIII": 53, "LIV": 54, "LV": 55, "LVI": 56, "LVII": 57, "LVIII": 58, "LIX": 59, "LX": 60,
        "LXI": 61, "LXII": 62, "LXIII": 63, "LXIV": 64, "LXV": 65, "LXVI": 66, "LXVII": 67, "LXVIII": 68, "LXIX": 69, "LXX": 70,
        "LXXI": 71, "LXXII": 72, "LXXIII": 73, "LXXIV": 74, "LXXV": 75, "LXXVI": 76, "LXXVII": 77, "LXXVIII": 78, "LXXIX": 79, "LXXX": 80,
        "LXXXI": 81, "LXXXII": 82, "LXXXIII": 83, "LXXXIV": 84, "LXXXV": 85
    }
    
    # Matches actual essay headers in body: THE FEDERALIST. No. X.
    pattern = re.compile(r"THE\s+FEDERALIST\.?\s*\r?\n+No\.\s+([IVXLCDM]+)\.?", re.IGNORECASE)
    matches = [m for m in pattern.finditer(body_text) if m.group(1).upper() in roman_map]
    
    print(f"Found {len(matches)} full essay bodies in text. Parsing...")
    
    hamilton_ids = {1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85}
    madison_ids = {10, 14, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48}
    jay_ids = {2, 3, 4, 5, 64}
    
    h_count, m_count, j_count = 0, 0, 0
    
    for i, match in enumerate(matches):
        essay_roman = match.group(1).upper()
        essay_num = roman_map[essay_roman]
        start_idx = match.start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(body_text)
        
        essay_content = body_text[start_idx:end_idx].strip()
        
        # Clean header down to greeting
        if "To the People of the State of New York:" in essay_content:
            greeting_idx = essay_content.find("To the People of the State of New York:")
            essay_content = essay_content[greeting_idx:]
            
        if essay_num in hamilton_ids:
            h_count += 1
            author = "hamilton"
        elif essay_num in madison_ids:
            m_count += 1
            author = "madison"
        elif essay_num in jay_ids:
            j_count += 1
            author = "jay"
        else:
            continue
            
        (out_dir / f"{author}_essay_{essay_num:02d}.txt").write_text(essay_content, encoding="utf-8")
        
    print(f"✓ Saved complete Federalist essays: {h_count} Hamilton, {m_count} Madison, {j_count} Jay to {out_dir}")
    return out_dir


def download_classic_authors_benchmark() -> Path:
    """Download multi-author literary benchmark: Conan Doyle, Jane Austen, Mark Twain."""
    out_dir = EVAL_DIR / "authors"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Arthur Conan Doyle (Sherlock Holmes)
    print("Downloading Arthur Conan Doyle (Sherlock Holmes) from Gutenberg...")
    url_doyle = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    req = urllib.request.Request(url_doyle, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        doyle_text = resp.read().decode("utf-8")
        
    doyle_titles = [
        "A SCANDAL IN BOHEMIA", "THE RED-HEADED LEAGUE", "A CASE OF IDENTITY",
        "THE BOSCOMBE VALLEY MYSTERY", "THE FIVE ORANGE PIPS", "THE MAN WITH THE TWISTED LIP",
        "THE ADVENTURE OF THE BLUE CARBUNCLE", "THE ADVENTURE OF THE SPECKLED BAND"
    ]
    
    doyle_matches = []
    for title in doyle_titles:
        m = re.search(rf"[IVXLCDM]+\.\s+{re.escape(title)}", doyle_text)
        if m:
            doyle_matches.append((title, m.start()))
            
    doyle_matches.sort(key=lambda x: x[1])
    for i, (title, start_pos) in enumerate(doyle_matches):
        end_pos = doyle_matches[i+1][1] if i + 1 < len(doyle_matches) else start_pos + 40000
        story_text = doyle_text[start_pos:end_pos].strip()
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')
        (out_dir / f"doyle_{slug}.txt").write_text(story_text, encoding="utf-8")
    print(f"✓ Saved {len(doyle_matches)} Conan Doyle stories.")

    # 2. Jane Austen (Pride and Prejudice)
    print("Downloading Jane Austen (Pride and Prejudice) from Gutenberg...")
    url_austen = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
    req = urllib.request.Request(url_austen, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        austen_text = resp.read().decode("utf-8")
        
    body_austen = austen_text[35900:]
    austen_matches = list(re.finditer(r"Chapter\s+([IVXLCDM]+)", body_austen, re.IGNORECASE))
    
    a_count = 0
    for i, m in enumerate(austen_matches[:8]):
        start_pos = m.start()
        end_pos = austen_matches[i+1].start() if i + 1 < len(austen_matches) else len(body_austen)
        chap_text = body_austen[start_pos:end_pos].strip()
        if len(chap_text.split()) >= 300:
            a_count += 1
            (out_dir / f"austen_chapter_{a_count:02d}.txt").write_text(chap_text, encoding="utf-8")
    print(f"✓ Saved {a_count} Jane Austen chapters.")

    # 3. Mark Twain (The Adventures of Tom Sawyer)
    print("Downloading Mark Twain (Tom Sawyer) from Gutenberg...")
    url_twain = "https://www.gutenberg.org/cache/epub/74/pg74.txt"
    req = urllib.request.Request(url_twain, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        twain_text = resp.read().decode("utf-8")
        
    body_twain = twain_text[twain_text.find("CHAPTER I\r\n"):]
    twain_matches = list(re.finditer(r"CHAPTER\s+([IVXLCDM]+)", body_twain, re.IGNORECASE))
    
    t_count = 0
    for i, m in enumerate(twain_matches[:8]):
        start_pos = m.start()
        end_pos = twain_matches[i+1].start() if i + 1 < len(twain_matches) else len(body_twain)
        chap_text = body_twain[start_pos:end_pos].strip()
        if len(chap_text.split()) >= 300:
            t_count += 1
            (out_dir / f"twain_chapter_{t_count:02d}.txt").write_text(chap_text, encoding="utf-8")
    print(f"✓ Saved {t_count} Mark Twain chapters.")
    print(f"✓ Multi-author benchmark ready at {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ai"):
        download_ai_human_benchmark(sample_size=60)
    if which in ("all", "federalist"):
        download_federalist_benchmark()
    if which in ("all", "authors"):
        download_classic_authors_benchmark()
