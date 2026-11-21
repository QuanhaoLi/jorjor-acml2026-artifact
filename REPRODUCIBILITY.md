
# Reproducibility Guide

## Environment

Recommended local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The package intentionally does not include a virtual environment or model weights.

## Local Integrity Check

```bash
python tools/check_artifact.py
python tools/smoke_static_artifacts.py
```

This verifies that required static artifacts exist, excluded file types are absent, all package folders have `README.md`, checksums can be generated, key JSON files load, and core claim-ledger text is present.

## Local Unit and Smoke Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```

These tests are meant for code sanity. They require a valid PyTorch installation and are not a full rerun of the paper experiments. If PyTorch is missing or partially installed, training/probe tests will fail while the static artifact checks above remain valid.

## Static Paper Evidence

Most paper tables and figures can be inspected directly from:

```bash
artifacts/paper_outputs/
artifacts/claim_audits/
artifacts/checkpoints_tables/
artifacts/figures/
artifacts/camera_ready_rebuttal/
```

See `supplement/TABLE_FIGURE_PROVENANCE.md` for mapping.

## Full Regeneration Boundary

Full regeneration of all long-run poker rows may require an external archive root with raw checkpoints and cached probe outputs. The relevant command shape is:

```bash
PYTHONPATH=src python scripts/reproduce_paper_tables.py \
  --archive-root /path/to/archive \
  --output-dir outputs/reproduced \
  --fresh50-commit <commit-or-archive-id>
```

The command shape is preserved from the research code. The submitted lightweight artifact does not bundle the large archive. The files under `supplement/camera_ready_rebuttal/code/` are an inspection snapshot rather than a self-contained executable source tree; exact reruns require the complete frozen source revision and external archive named in the source reports. The submitted ZIP supports static evidence inspection, checksum verification, and local source sanity tests, not independent regeneration of every long-run camera-ready experiment.
