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
```

### Batch Folder Processing

Process entire directories containing `.txt`, `.md`, and `.rst` documents in a single command:

```bash
# Batch analyze an entire folder of essays with summary breakdown table
idiolect analyze ./essays/

# Batch identify all submissions in a folder against enrolled candidates
idiolect identify ./submissions/

# Enroll an author using multiple sample documents combined from a folder
idiolect enroll "Student Name" ./student_samples/

# Batch verify all documents in a folder against an enrolled author
idiolect verify "Student Name" ./submissions/
```

## Persistence & Storage

Enrolled student profiles are permanently stored on disk using **SQLite** at:
`~/.idiolect/fingerprints.db`

Each enrolled profile stores the author's label, complete serialized stylometric fingerprint, word count, and enrollment timestamp. Profiles persist across terminal sessions and reboots without needing to re-extract features.

Generated PDF reports default to the local [`artifacts/`](./artifacts/) directory.

