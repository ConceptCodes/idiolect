# Idiolect

Linguistic fingerprinting CLI — identify authorship and detect AI-generated content.

## Install

```bash
uv sync
python -m spacy download en_core_web_sm
```

## Usage

`idiolect` seamlessly supports both **single files** and **entire batch directories** across commands.

### Single File Analysis & Identification

```bash
# Analyze a text file (generates fingerprint + PDF report)
idiolect analyze essay.txt

# Compare two texts directly
idiolect compare essay1.txt essay2.txt

# Enroll a known author from a sample
idiolect enroll "Student Name" known_sample.txt

# Verify authorship against a specific enrolled student
idiolect verify "Student Name" submitted_paper.txt

# Identify which enrolled student wrote an anonymous essay
idiolect identify submission.txt

# List or delete enrolled authors
idiolect list
idiolect delete "Student Name"

# Inspect an author's multi-sample profile, consistency, and rolling baseline
idiolect profile "Student Name"
idiolect profile "Student Name" --output artifacts/
```

### Multi-Sample Profiling & Weighted Rolling Average

Authors write differently across prompts, genres, and time. Rather than relying on a static single sample or concatenated text, `idiolect` maintains a **multi-sample profile** for each author:

- **Volume Weighting**: Longer samples carry higher statistical weight ($\min(\text{word\_count}, 5000)$), reflecting lower sampling noise.
- **Recency Decay**: An exponential rolling decay ($\alpha^{(N-1-i)}$, default $\alpha=0.90$) places higher weight on recent writings while preserving long-term baseline traits.
- **Stylometric Consistency**: Evaluates intra-author variance across all 7 radar dimensions ($\pm \sigma$) and computes an overall consistency score ($0\text{--}100\%$).
- **Continuous Learning**: Enrolling new samples over time automatically updates the rolling baseline:
  ```bash
  # Append a new sample to Alice's profile (automatically recalculates rolling baseline)
  idiolect enroll "Alice" latest_essay.txt

  # Custom decay factor (e.g. 0.85 for faster adaptation to recent style)
  idiolect enroll "Alice" latest_essay.txt --decay 0.85

  # Reset baseline and clear previous samples
  idiolect enroll "Alice" fresh_sample.txt --replace
  ```

### Batch Folder Processing

Process entire directories containing `.docx`, `.pdf`, `.txt`, `.md`, and `.rst` documents in a single command:

```bash
# Batch analyze an entire folder of essays with summary breakdown table
idiolect analyze ./essays/

# Batch identify all submissions in a folder against enrolled candidates
idiolect identify ./submissions/

# Enroll an author using multiple sample documents from a folder
idiolect enroll "Student Name" ./student_samples/

# Batch verify all documents in a folder against an enrolled author
idiolect verify "Student Name" ./submissions/
```

### Native Document Ingestion (.docx & .pdf)

Students rarely submit plain text files; they submit Microsoft Word (.docx) or Adobe PDF (.pdf) documents. `idiolect` natively parses `.docx` and `.pdf` files without manual copy-pasting or external conversion tools:

- **Word Documents (`.docx`)**: Extracts paragraph text and structured table rows.
- **PDF Files (`.pdf`)**: Extracts text pages and preserves structural divisions.
- Works seamlessly in both single-file commands and folder batch processing (`.txt`, `.docx`, `.pdf`, `.md`, `.rst`).

### CSV / JSON Output Mode (LMS & Grading Pipeline Integration)

All core inspection, analysis, and identification commands support `--format table|json|csv` (with `--json` as an alias) for headless automation, scripting, and LMS integration (Canvas, Blackboard, Moodle):

```bash
# Export batch identification results as CSV for Canvas or Blackboard gradebooks
idiolect identify ./submissions/ --format csv > results.csv

# Output machine-readable JSON for LMS webhook / automated grading script
idiolect identify ./submissions/ --format json > results.json

# Batch analyze essay directory and export summary metrics to CSV
idiolect analyze ./essays/ --format csv > stylometric_metrics.csv

# Export enrolled author roster and consistency ratings
idiolect list --format csv > enrolled_students.csv

# Inspect author profile structure in JSON
idiolect profile "Student Name" --format json
```

When `--format csv` or `--format json` is selected, standard output is clean and free of ANSI styling, spinners, or table borders—allowing direct stdout redirection (`> output.csv`).

### Document Length Warning & Confidence Damping

Stylometric features (such as Moving-Average Type-Token Ratio, Measure of Textual Lexical Diversity, hapax legomena ratios, and syntactic parse tree depth variance) exhibit high statistical variance when evaluated on brief snippets (<250 words):

- **Automatic Short Document Detection**: Any submission or sample under 250 words triggers a prominent alert notice:
  ```
  ⚠️  Short Document Notice (142 words < 250 words)
      Stylometric metrics (such as MATTR, MTLD, and parse depth variance) have higher
      sampling noise on brief texts. Attribution confidence has been proportionally damped.
  ```
- **Proportional Confidence Damping**: Confidence is scaled via $\sqrt{W / 250}$ (with a 0.40 safety floor) to mitigate false-positive attribution risks on brief texts:
  ```
  Match Confidence: 75.4% (damped from 100.0% due to length: 142 words) (Strong Match)
  ```
- **Structured Reporting**: Reports document length state across all formats:
  - **Console**: Displays yellow `⚠️` indicator and explanatory banner.
  - **JSON**: Includes `"short_document": true`, `"length_warning": "..."`, and `"raw_confidence"`.
  - **CSV**: Includes `"short_doc": "yes"`.

### Attribution Explainability in `identify`

Attribution decisions shouldn't be opaque black boxes. When identifying an unknown submission, `idiolect` extracts the top 3–5 distinct linguistic traits that drove the match:

- **Trait Salience Scoring**: Evaluates feature alignment in standardized population z-score space, identifying traits where both texts share distinctive divergence from population norms.
- **Human-Readable Stylistic Insights**: Translates statistical markers into actionable descriptions:
  ```
  🔍 Top Aligning Linguistic Traits (Idiolect Drivers):
    • Semicolon Usage — high semicolon frequency (z-delta: 0.04)
    • Contraction Rate — low contraction rate (formal register) (z-delta: 0.07)
    • Subordinate Clauses — complex subordinate clause structure (z-delta: 0.11)
    • Vocabulary Richness (MATTR) — high moving-average lexical diversity (z-delta: 0.15)
  ```
- **Export Integration**: Trait descriptions are automatically included in batch summaries, JSON exports (`aligning_traits` / `top_aligning_traits`), and CSV tables.

## Persistence & Storage

Enrolled student profiles are permanently stored on disk using **SQLite** at:
`~/.idiolect/fingerprints.db`

The database maintains both the composite rolling fingerprint and individual historical writing samples in `author_samples`. Profiles persist across terminal sessions and reboots without needing to re-extract features.

Generated PDF reports default to the local [`artifacts/`](./artifacts/) directory.


