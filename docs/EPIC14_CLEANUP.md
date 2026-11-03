# Epic #14 — delete-first cleanup audit (issues #15, #17, #18, #19)

This change starts from the merge of #13 / PR #16. It simplifies **six-class
implementation only**. It does not make a five-class export, retrain a model,
change frozen roles or Gamma-or-Skip v2, touch the reserved final test, or revise
the archived experiment identities and records. Those are separate #20–#22 tasks.

## Callers and responsibility boundaries

| Module / caller | Decision | Before → after |
| --- | --- | --- |
| `finetuning.py`; notebook 03, training tests | **REMOVE / SIMPLIFY #15** | Delete custom `PairedTrainer`, `FreezeAudit`, state hashing, optimizer-step override, per-epoch JSON, extra resume checkpoint/receipt and complete/identity files. Keep a small paired dataset manifest and native `YOLO(...).train(freeze=11)`; `resume=True` uses native `last.pt`. |
| `datasets.verify_role`; notebooks 02/03/04, evaluation, loader benchmark | **SIMPLIFY #17** | Deep label coordinate/box checks happen at export/audit time. Normal training/cache preflights inspect frozen metadata, roles, paths, ID inventory, encoding policy and verified export identity without reading every label file again. |
| `offline.verify_alignment`; exports and phase loaders | **KEEP / HARDEN #17** | Export-time deep SHA/decoded-geometry checks remain. Cheap calls verify metadata/export stamps, inventories, existence, label digests recorded in manifests and owned-file safety. Reject unexpected label/image symlinks (historical raw_original source links are intentional). |
| `adaptive.export_pair`; notebook 02 | **KEEP #17** | One paired deep verification after both exports is the authoritative same-ID/label-hash/shape/split/JPEG95 check. Existing frozen export bytes are not rewritten. |
| `protocol.py`, `subsets.py`, `data.py`; notebooks 01/02 and fixture tests | **KEEP #17** | Group/duplicate/role assignment, native BDD conversion, negative-frame and coordinate handling stay scientifically necessary. Do not rewrite search/sampling algorithms or legacy `choose_splits` for LOC reduction. |
| `evaluation.py`; notebook 04 and development tests | **SIMPLIFY #18** | New checkpoint discovery needs native `args.yaml`, `best.pt`, the paired frozen role manifest and YAML binding, **not** custom training receipts. The legacy reader still verifies historical `pair.json/identity.json/complete.json` and hashed evidence. Explicit `evaluate_development(..., checkpoint, ...)` accepts a native best checkpoint directly. |
| `evaluation_stats.class_ap`, `development_report.py`; notebook 04, statistical tests | **MERGE #18** | Native `ap_per_class` aggregation is implemented in one place; a detailed option returns AP50 and AP50:95. No new IoU/matching/AP implementation or invented absent-class AP. |
| `baseline.py`, `loader_benchmark.py`; notebook 04, fixtures | **KEEP #18** | Historical COCO-80→six-class semantics and baseline provenance stay intact. The separate CPU-only loader measurement reports a different stage from native model/validator timing. New measurement is opt-in. |
| `stress.py`, `stress_eval.py`, `robustness_spec.py`; archived notebooks and stress tests | **KEEP / NO-CHANGE #19** | No active notebook imports synthetic research, calibration evaluator or future holdout robustness logic. Existing public research imports and re-export shims serve historical callers; removal would break replay/#8 without reducing active execution. |
| `notebooks/archive/`; historical evidence | **KEEP #19** | Archived notebook bytes and hashes remain unchanged. |

## I/O boundary and guarantees

**Previously** the training preparation path validated per-label geometry; every
train start rehashed and deep-decoded whole exports and rechecked label content
after already verified export publication. **Now** one deep pass remains at
creation/audit of each export. Subsequent calls validate metadata/content
*identities* through frozen manifests and check IDs, roles, expected paths,
file presence and symlinks without a second all-image hash/decode pass.
There is no measured speedup claim, and cheap preflight **does not detect
in-place content tampering without a manifest change**. Re-run explicit deep
verification whenever external tools may have modified export files; fresh
development inference still performs it before native validation.

Raw and Adaptive share exactly the same six-class GT IDs, label digests,
source dimensions and role, and both JPEG95 encodings. Native training uses
independent instantiations of the **same pinned release pretrained checkpoint**,
fixed native freeze=11 and equal declared settings/budget. Native `results.csv`
is the history; `best.pt` and `last.pt` are the canonical weights. Native
resume does not promise bitwise continuation and no longer makes a separate
atomic durable resume copy. Optimizer-step, buffer and nonzero-gradient audits
were intentionally dropped: these were implementation-specific checks and are
not equivalent native guarantees. The small run manifest is **not** a claim of
bitwise training reproducibility.

Historical six-class evaluation caches and development-analysis summaries are read-only: a legacy
`development-native-v1` cache can be reused with its recorded identity and
artifact hashes when role/protocol/policy/checkpoint/data/mapping/settings
still match. Fresh outputs use `development-native-v2` and `development-analysis-v2` identities and separate roots.
Change of class list or settings requires a fresh evaluation identity. Neither
path performs upstream notebook execution or accesses final-test image pixels.

## Test boundaries

Relevant fixtures cover frozen train/development/final roles, same paired IDs,
wrong-role rejection before pixel reads, deep-vs-light export corruption,
YAML/native args/run integrity, YOLO26n freeze boundary, independent pretrained
initialization, native result reuse/resume, a tiny CPU native training fixture,
historical completion receipts, legacy cache replay, native AP and fixed-supported
group resampling. Archived stress/robustness tests continue to guard #8.

Full S10 retraining, final-test release, five-class migration and new
benchmarking are **not** part of this PR. Historical results are not rewritten.
