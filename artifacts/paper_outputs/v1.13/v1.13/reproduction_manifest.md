# Paper Table Reproduction Manifest

## Command

```text
python scripts/reproduce_paper_tables.py --archive-root checkpoints/remote_archive/v100s --output-dir outputs/paper/v1.13 --fresh50-commit 2dfcb42 --composition-commit 2dfcb42 --bootstrap-count 200 --bootstrap-seed 20260602
```

## Inputs

- archive_root: `checkpoints/remote_archive/v100s`
- fresh50_commit: `2dfcb42`
- bootstrap_count: `200`
- bootstrap_seed: `20260602`
- confidence_level: `0.95`

## Outputs

- `packs/limit_ar_no_range_bootstrap_pack.json` sha256=`2ff5b3aecf4c3b8155140bd81c4a6e8389142d8193fc347a1c8a0ab443c6a65b`
- `packs/limit_ar_no_range_bootstrap_pack.md` sha256=`8cea0f6ceacbe758a4a7af327a7f066027aab40868ece81c6cd2dbfb9e1b195a`
- `paper_tables/final_diagnostic_tables.json` sha256=`dc82e1ff25c6b1c9c1c7132a1043d21b9c6937a57c0f1677333f73c80dcad9c3`
- `paper_tables/final_diagnostic_tables.md` sha256=`62a23bafef2f60dfb955d77f8336fea3a97ce7f939c383ba9c54452591b463ff`
- `paper_tables/final_table_source_manifest.json` sha256=`61d2e1e839c6b38c6c8723218becf52a0b9ab136d64c466b505897fa6b1e9fc7`
- `packs/limit_ar_no_range_composition_nuisance_pack.json` sha256=`0b67fb6d1f8d4d64842f3c613c1914fd7f859360b1b4d5d2887befba053c021a`
- `packs/limit_ar_no_range_composition_nuisance_pack.md` sha256=`86b40197b7d7d49f950633abfed458bf008203ab22682cec1e73c3b66c39420d`

## Missing Optional Packs

- none
