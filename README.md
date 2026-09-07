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

Process entire directories containing `.txt`, `.md`, and `.rst` documents in a single command:

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

## Persistence & Storage

Enrolled student profiles are permanently stored on disk using **SQLite** at:
`~/.idiolect/fingerprints.db`

The database maintains both the composite rolling fingerprint and individual historical writing samples in `author_samples`. Profiles persist across terminal sessions and reboots without needing to re-extract features.

Generated PDF reports default to the local [`artifacts/`](./artifacts/) directory.


