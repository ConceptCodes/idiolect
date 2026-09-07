# Idiolect

Linguistic fingerprinting CLI — identify authorship and detect AI-generated content.

## Install

```bash
uv sync
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Analyze a text file
idiolect analyze essay.txt

# Compare two texts
idiolect compare essay1.txt essay2.txt

# Enroll a known author
idiolect enroll "Student Name" known_sample.txt

# Verify authorship against a specific enrolled student
idiolect verify "Student Name" submitted_paper.txt

# Guess/identify which enrolled student wrote an anonymous essay
idiolect guess submission.txt

# List or manage enrolled authors
idiolect list
idiolect delete "Student Name"
```

## Persistence & Storage

Enrolled student profiles are permanently stored on disk using **SQLite** at:
`~/.idiolect/fingerprints.db`

Each enrolled profile stores the author's label, complete serialized stylometric fingerprint, word count, and enrollment timestamp. Profiles persist across terminal sessions and reboots without needing to re-extract features.

Generated PDF reports default to the local [`artifacts/`](./artifacts/) directory.

