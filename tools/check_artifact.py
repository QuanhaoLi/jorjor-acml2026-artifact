"""Lightweight integrity checker for the ACML 2026 JorJor artifact."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "CODE_OF_ARTIFACT.md",
    "MANIFEST.md",
    "REPRODUCIBILITY.md",
    "supplement/CLAIM_LEDGER.md",
    "supplement/TABLE_FIGURE_PROVENANCE.md",
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/claim_gate_matrix.md",
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/diagnostic_report.json",
    "artifacts/claim_audits/k.6.6_current_product_audit_pack/source_manifest.json",
    "artifacts/claim_audits/k.5.1_full_poker_predictive_link/k.5.1_poker_predictive_link_report.json",
    "artifacts/claim_audits/k.5.2_composition_boundedness/k.5.2_composition_boundedness_report.json",
    "artifacts/claim_audits/k.5.0_synthetic_cbps_control/k.5.0_synthetic_cbps_report.json",
    "artifacts/figures/figure_3_composition_gate_per_seed_paper_flat.png",
    "artifacts/figures/figure_4_matched_composition_decisive_paper_flat.png",
    "artifacts/figures/figure_5_oracle_validation_ordered_vs_shortcut_paper_flat.png",
]

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
}

EXCLUDED_NAMES = {
    ".DS_Store",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
}

BANNED_TEXT_PATTERNS = [
    "Co" + "dex",
    "co" + "dex",
    "Chat" + "GPT",
    "Open" + "AI",
    "Created" + " by",
    "Created" + " at",
    "Last" + " Modified",
    "Lst" + " Modified",
    "/Us" + "ers/",
    "alex" + "_1",
    "acml june" + " 20 submit",
]

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file())


def find_missing_readmes() -> list[str]:
    missing: list[str] = []
    for directory in sorted(path for path in ROOT.rglob("*") if path.is_dir()):
        rel_parts = directory.relative_to(ROOT).parts
        if any(part in EXCLUDED_DIR_NAMES for part in rel_parts):
            continue
        if not (directory / "README.md").exists():
            missing.append(str(directory.relative_to(ROOT)))
    return missing


def find_text_hygiene_issues() -> list[str]:
    issues: list[str] = []
    ignored_files = {"MANIFEST_CHECKSUMS.sha256", "artifact_check_report.json"}
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if path.name in ignored_files or path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in BANNED_TEXT_PATTERNS:
            if pattern in text:
                issues.append(f"{rel}: contains banned pattern {pattern!r}")
                break
        else:
            if CJK_RE.search(text):
                issues.append(f"{rel}: contains CJK characters")
    return issues


def main() -> int:
    missing_required = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    excluded_files: list[str] = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            excluded_files.append(str(rel))
            continue
        if path.name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            excluded_files.append(str(rel))

    missing_readmes = find_missing_readmes()
    text_hygiene_issues = find_text_hygiene_issues()
    checksums = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if str(rel) in {"MANIFEST_CHECKSUMS.sha256", "artifact_check_report.json"}:
            continue
        checksums.append((sha256(path), str(rel)))

    checksum_path = ROOT / "MANIFEST_CHECKSUMS.sha256"
    checksum_path.write_text(
        "".join(f"{digest}  {rel}\n" for digest, rel in checksums),
        encoding="utf-8",
    )

    report = {
        "root": ".",
        "file_count": len(iter_files()),
        "required_file_count": len(REQUIRED_FILES),
        "missing_required": missing_required,
        "excluded_files": excluded_files,
        "missing_readmes": missing_readmes,
        "text_hygiene_issues": text_hygiene_issues,
        "checksum_manifest": "MANIFEST_CHECKSUMS.sha256",
    }
    (ROOT / "artifact_check_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ok = not missing_required and not excluded_files and not missing_readmes and not text_hygiene_issues
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
