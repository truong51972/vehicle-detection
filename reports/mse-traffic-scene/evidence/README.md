# Evidence provenance

The source of truth for report claims is a pinned repository file or a group-supplied draft with an explicit evidence status.

1. **Vehicle:** development evaluation JSON under runs/v0–v4, with config identities and the separate matched batch-16 Raw control under docs/v4/evidence/matched_raw. Never combine Study A and B as if the whole set shared one control.
2. **Phát:** dhphat/traffic-sign-detection at f3c30d8be4e31404a73e8621d8fb050a62c90e05. Its train args, results.csv, best.pt, scripts and demo images exist. The repository tree does not include the metrics.json described by the README for the distinct test 640/960 runs.
3. **Hiếu:** ec2003/VN-traffic-sign-detection at 1bcf75eebbb3b255f622ade63900cead00abd863. **The trained checkpoint, args.yaml, results.csv and saved test plots DO exist.** README reports test @832. Missing are a machine-readable test metrics summary and raw test dataset required for independent re-evaluation; do **not** describe the checkpoint as missing.
4. **Group draft:** 640/960 comparison values and a 140-box sign-test count were provided in an uploaded DOCX; they have not been independently reproduced with the same image IDs and environment.
5. **Integration:** no measured same-frame two-detector video job; no combined mAP or end-to-end FPS exists.

See manifest.yml for exact paths, pinned refs, figure source attribution and unsupported measurements.
