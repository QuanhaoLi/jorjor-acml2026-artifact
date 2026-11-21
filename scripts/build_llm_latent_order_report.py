from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jorjor.probe.llm_latent_order import DEFAULT_BACKENDS, build_llm_latent_order_report


def main() -> int:
    """Build the k.2 frozen-LM latent-order audit report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--signal-probability", type=float, default=0.72)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--bootstrap-count", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260604)
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--surface-template-count", type=int, default=2)
    parser.add_argument("--posterior-correlation-threshold", type=float, default=0.80)
    parser.add_argument("--equivalence-upper-threshold", type=float, default=1e-6)
    parser.add_argument("--model-id")
    parser.add_argument("--layer-id", type=int, default=-1)
    parser.add_argument("--token-selection", default="final")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--hf-batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    backends = tuple(backend.strip() for backend in args.backends.split(",") if backend.strip())
    reproduction_command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    payload = build_llm_latent_order_report(
        output_dir=Path(args.output_dir),
        length=args.length,
        signal_probability=args.signal_probability,
        max_pairs=args.max_pairs,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        backends=backends,
        surface_template_count=args.surface_template_count,
        posterior_correlation_threshold=args.posterior_correlation_threshold,
        equivalence_upper_threshold=args.equivalence_upper_threshold,
        model_id=args.model_id,
        layer_id=args.layer_id,
        token_selection=args.token_selection,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        hf_batch_size=args.hf_batch_size,
        trust_remote_code=args.trust_remote_code,
        reproduction_command=reproduction_command,
    )
    labels = ",".join(f"{case['backend']}={case['final_diagnostic_label']}" for case in payload["cases"])
    print(f"wrote k.2 LLM latent-order audit labels={labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
