from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.llm_full_sweep import build_llm_full_sweep, rebuild_llm_full_sweep_from_run_reports


def main() -> int:
    """Build the k.2.6 multi-configuration LLM latent-order sweep."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="5")
    parser.add_argument("--layer-ids", default="-1")
    parser.add_argument("--backends", default="oracle_posterior,composition,order_only,surface_noise")
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260604)
    parser.add_argument("--signal-probability", type=float, default=0.72)
    parser.add_argument("--surface-template-count", type=int, default=2)
    parser.add_argument("--model-id")
    parser.add_argument("--model-role", default="main_or_single_model")
    parser.add_argument("--locked-layer-id", type=int)
    parser.add_argument("--model-specs-json")
    parser.add_argument("--rebuild-run-report-glob")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hf-batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if args.rebuild_run_report_glob:
        payload = rebuild_llm_full_sweep_from_run_reports(
            output_dir=Path(args.output_dir),
            run_report_paths=_expand_report_glob(args.rebuild_run_report_glob),
            model_specs=_parse_model_specs(args.model_specs_json)
            or (
                {
                    "model_id": args.model_id,
                    "model_role": args.model_role,
                    "locked_layer_id": args.locked_layer_id,
                    "layer_ids": list(_parse_ints(args.layer_ids)),
                },
            ),
            lengths=_parse_ints(args.lengths),
            layer_ids=_parse_ints(args.layer_ids),
            backends=_parse_strings(args.backends),
        )
    else:
        payload = build_llm_full_sweep(
            output_dir=Path(args.output_dir),
            lengths=_parse_ints(args.lengths),
            layer_ids=_parse_ints(args.layer_ids),
            backends=_parse_strings(args.backends),
            max_pairs=args.max_pairs,
            bootstrap_count=args.bootstrap_count,
            bootstrap_seed=args.bootstrap_seed,
            signal_probability=args.signal_probability,
            surface_template_count=args.surface_template_count,
            model_id=args.model_id,
            model_role=args.model_role,
            locked_layer_id=args.locked_layer_id,
            model_specs=_parse_model_specs(args.model_specs_json),
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            hf_batch_size=args.hf_batch_size,
            trust_remote_code=args.trust_remote_code,
        )
    print(
        "wrote k.2.6 LLM full sweep "
        f"runs={payload['run_count']} hf_runs={payload['hf_run_count']} label={payload['final_sweep_label']}"
    )
    return 0


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one backend")
    return values


def _parse_model_specs(raw: str | None) -> tuple[dict, ...] | None:
    if raw is None:
        return None
    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("model specs JSON must be a list")
    specs = tuple(item for item in data if isinstance(item, dict))
    if len(specs) != len(data):
        raise ValueError("every model spec must be a JSON object")
    return specs


def _expand_report_glob(raw: str) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in sorted(glob.glob(raw)))
    if not paths:
        raise ValueError(f"no run reports matched: {raw}")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
