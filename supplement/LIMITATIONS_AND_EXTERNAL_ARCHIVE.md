
# Limitations and External Archive Boundary

## Lightweight Package Boundary

This submission package is intentionally lightweight. It includes source code, configs, tests, static paper evidence packs, and checksums. It excludes raw long-run checkpoints and model weights.

## Requires External Archive

The following tasks may require an external archive root:

- Full retraining of poker AR models.
- Regeneration from raw checkpoints rather than static JSON/MD report packs.
- Rebuilding all probe caches for every seed from scratch.
- Re-running remote LLM/frozen-LM stress rows.

## Included Without External Archive

- Static reports and JSON outputs for main paper claims.
- Figure source JSON/SVG and current PNG exports.
- Source code and configs used by the research pipeline.
- Local smoke/unit tests.
- Package integrity and checksum generation.

## How To Report This In The Paper

Use precise language:

> The supplementary artifact includes the source code, configurations, paper table/figure source manifests, static diagnostic reports, and checksums. Full regeneration from raw long-run checkpoints requires the external archive root described in the artifact documentation.
