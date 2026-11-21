
# Validation Status

## Default Artifact Checks

Command:

```bash
python tools/check_artifact.py
python tools/smoke_static_artifacts.py
```

Recorded result:

- package hygiene check passed;
- no excluded file types were present;
- no banned development strings, CJK characters, or local absolute paths were present;
- no missing required files;
- no missing folder README files;
- 6 static JSON evidence files loaded;
- 3 PNG manuscript figures found;
- `MANIFEST_CHECKSUMS.sha256` and `artifact_check_report.json` generated.

## Full Source Test Suite

Command:

```bash
PYTHONPATH=src python -m unittest discover tests
```

Recorded result in the research environment used to prepare this artifact:

- 303 tests run;
- 16 skipped;
- all non-skipped tests passed.

## Environment Note

Full tests require a valid PyTorch installation. If PyTorch is unavailable or partially installed, training and probing tests may fail even though the static artifact checks remain valid.

The 303-test suite covers the supported main package. It does not make the inspection-only camera-ready rebuttal snapshot self-contained or rerun the long-run five-seed/exact-posterior products without their external archive and complete frozen source revision.
