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

# Verify authorship
idiolect verify "Student Name" submitted_paper.txt
```
