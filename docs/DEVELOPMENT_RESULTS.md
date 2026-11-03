# Six-class development matrix: partial issue #7 evidence

All six native development evaluations completed before the request to stop further
benchmarking. The original S10 checkpoints, six-class labels (including railway
`train`), policy, data roles and training results remain unchanged. No new training
or final-test evaluation ran.

Exact metrics, per-class AP/precision/recall, stage timing, checkpoint hashes,
input identities and every native artifact hash are recorded in
[matrix.json](development-v1/matrix.json). Full matching statistics, plots and
representative prediction/GT batches remain gitignored under `runs/development-v1/`.
The [prospective completion plan](development-v1/PLAN.md) and its original declaration
receipt are preserved; no final model/scale lock was published.

Native development mAP50:95, with Raw / Gamma-or-Skip inputs respectively:

- Pretrained: **13.63620% / 13.66326%**.
- Raw-trained S10 best: **16.11914% / 16.09336%**.
- Gamma-or-Skip-trained S10 best: **16.48291% / 16.51434%**.

These are new evaluations of the saved native checkpoints, distinct from the
original epoch metrics in [the S10 training report](ISSUE7_RESULTS.md). The Raw-input
pretrained point exactly reproduces the earlier baseline. Small differences from
saved training-epoch metrics must not be relabeled as a changed training result.
Single-seed point estimates do not establish a reliable Adaptive advantage.
Native exported-file stage times exclude offline enhancement, loader isolation and
application end-to-end latency.

All six caches were replayed with model construction, training and trainer
construction forbidden. Cache replay verified exact current checkpoint/data/settings
identities, result checksums and all saved plot/statistic hashes; it ran no inference.
Final-test pixels remain untouched.

## Remaining work

Per the user's stop request, no full paired bootstrap or isolated loader benchmark
was run. Supported real-data condition analysis, 100-draw group uncertainty,
loader costs, model/scale selection, final protocol lock, remote CI and PR review
remain open. Small fixture tests and the two-draw timing smoke are implementation
checks, never completed uncertainty or hardware evidence.

Notebook 04 now defaults to cached inference and **zero bootstrap draws**. It can
show point estimates without a loader measurement; missing optional loader costs
are visibly pending. Full resampling and new loader measurements are explicit
choices. Their helpers are fixture-tested, and no reporting cell automatically
locks model selection. The taxonomy/training work described by issue #14 is outside
this implementation.
