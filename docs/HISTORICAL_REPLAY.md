# Historical replay and issue 41 migration

The pre-refactor baseline is commit [51687b273e7e87b5315d84f5b67202d4cc746992](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992)
(local tag `issue41-baseline`). It was preserved before removing archived notebooks.
All ten archived notebook hashes were checked against the old archive manifest before deletion.
No committed research JSON, CSV, image, checkpoint or measured result was deleted or rewritten.

For exact historical execution, use a separate clone checked out at that commit,
install its locked study environment and restore the original local sources/runs/weights.
The old [archive guide](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/README.md) records earlier source revisions,
including `99d9af2`, `2df2241`, `7380eed`, `98d7a1f` and training implementation `a853add`.
The old [archive manifest](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/manifest.json) records original paths and SHA256.
Absolute historical paths must be verified explicitly when relocating artifacts.

Historical evidence records intentionally retain original filenames and hashes. In particular,
`docs/round2-preprocessing-v2/evidence_manifest.json` points to the original PR #37 notebook;
its bytes are the screening archive below, not the later notebook at the baseline's top level.
The hash-bound `ISSUE30_IMPLEMENTATION.md` and `ISSUE30_RESULTS.md` stay byte-for-byte intact;
their execution/path instructions apply to the baseline. Current instructions live in
[the root README](../README.md). An old relative notebook path in an immutable evidence record
is a historical identifier, not a current entrypoint.

Preserved notebook bytes in Git:

- [notebooks/01_data_init.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/round1/01_data_init.ipynb) — SHA256 `2984ee6b870dd1aa47d821e86a125f6a1a5b9fcba66d5e67aece2b2ee0cfed7c`; original source `99d9af2be1b1dcb1a296f7d85867207609bc8162`.
- [notebooks/01_data_preparation.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/01_data_preparation.ipynb) — SHA256 `01af079715d574784a797e036e310da671b7e0fc6c11a0d080ca42f47df4a835`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/02_training_benchmark.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/02_training_benchmark.ipynb) — SHA256 `07997ff433961d4dc2cd1e93a01aa3b27ff5e0bd533e2aafbe2805aa604d4d66`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/03_preprocessing_experiments.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/round2-screening/03_preprocessing_experiments.ipynb) — SHA256 `74332e912de4e1e638ddebf3e7878ce3803fbe5c9215d085ca12d42f1986a429`; original source `6e9c7af31fffca5c75f787aa1cbfeb1ac99e1566`.
- [notebooks/phase2/01_data_protocol.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/01_data_protocol.ipynb) — SHA256 `5d4ed8206a12a158abf19d49278a6b097804283098df7d319fe944c92a990c0e`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/phase2/02_adaptive_preprocessing.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/02_adaptive_preprocessing.ipynb) — SHA256 `a6733c7e3961e8be55b002fef883959b5d2b3b0f2d171137f3ad8befb0b48827`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/phase2/02b_synthetic_robustness.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/02b_synthetic_robustness.ipynb) — SHA256 `0a7719742db0e4ad43b77d393e3520521ecfa2ddebafbd1a63a000a435eabff4`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/phase2/03_finetuning_benchmark.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/03_finetuning_benchmark.ipynb) — SHA256 `f242d9f92061c0fc9e0816b77b42e4b6c7df8e3051e3d9f36951f7774e7e76a1`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/phase2/evidence/02_adaptive_preprocessing_v1.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02_adaptive_preprocessing_v1.ipynb) — SHA256 `69f0b8c18f61b6707671cc0ee48ae535e230bc5ce38019ad9e318a318e58dbfd`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.
- [notebooks/phase2/evidence/02b_synthetic_robustness_v1.ipynb](https://github.com/truong51972/vehicle-detection/blob/51687b273e7e87b5315d84f5b67202d4cc746992/notebooks/archive/phase2/evidence/02b_synthetic_robustness_v1.ipynb) — SHA256 `c2f4c3a4ffe912069df75f8181a1e74a12e5b10b93c7e6ffcba88ef5bb32ed43`; original source `2df2241f6bf4e09b92499122ad039a9e59f0ad86`.

The archive-dependent smoke fixture was replaced by direct synthetic BDD fixtures.
The old paired training, development-report and route-report entrypoints/tests were superseded
by versioned native training/evaluation and shared-data tests. Their source and exact replay
remain available at the baseline. Historical research helpers remain because retained
calibration/search evidence and its tests still use them; the active training flow does not
consume their manifest chains.

The new `configs/splits.csv.gz` is a direct projection of the preserved protocol/subsets:
79,593 IDs across four roles, source/duplicate-union groups, source-relative paths, dimensions,
source image and annotation-label hashes, and the existing S10 membership bit. It copies no
boxes or label files and makes no new allocation. The CSV hash and original protocol/subset
IDs live once in `configs/data.yaml`; source GT is read from unchanged original annotations
using the existing annotation filter and six-to-five mapping. Final annotations remain metadata;
no final images are opened, decoded, hashed or exported.

New exports use a native `data.yaml` with a short binding to experiment config, shared splits,
operator, OpenCV version and aggregate exported-file hashes. There are no separate protocol,
subset, conversion, export, verification, pair, training or evaluation-identity receipts in
`runs/vN/`. A native `train.yaml` selects on the saved v0 Raw development images, preserving
the historical training comparison, while `data/data.yaml` evaluates the matching policy.
Native `args.yaml`, `results.csv`, checkpoints and one `eval/metrics.json` hold the run results.
Preparation publishes atomically; an interrupted `data.tmp` must be inspected/removed explicitly.
No background export recovery, custom trainer, dynamic augmentation, new search or GPU work
was introduced. Notebook defaults read historical evidence; all new work is explicit.

## Incoming #33 workflow

While preparing this PR, #33's independent paired preparation/training workflow
landed on main at [c11ce10d764fcbba6fabf01b5c5702dc06754edc](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc). This commit is a second preserved
replay reference, containing the original P4/P5 notebooks and helpers. Their hashes
were checked against the committed #33 validation receipt before retiring those
superseded entrypoints and tests. The versioned preparation/training and fresh-kernel
tests replace them; no compatibility receipt chain is added.

The [#33 implementation](ISSUE33_IMPLEMENTATION.md), [verified-reuse results](ISSUE33_RESULTS.md)
and all `docs/round2-training-v1/` bytes remain unchanged. Their source/notebook/test
paths, including the root README hash in `validation.json`, refer to that upstream
commit. They authenticate reuse of the same #21 Raw/Gamma-or-Skip checkpoints, not
additional training. The #35 handoff and its checkpoint identities remain intact.

Historical #33 execution sources:

- [notebooks/04_train_data_preparation.ipynb](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc/notebooks/04_train_data_preparation.ipynb) — SHA256 `66a42b5098bfd27b5c3fe03912c6f74a868556a393bc8b0924aae0626f7ad467`.
- [notebooks/05_model_training.ipynb](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc/notebooks/05_model_training.ipynb) — SHA256 `e0cf190a694843e75883eaa1fd7f55f026692372e6ab75a8b1ce92eabed58038`.
- [src/vehicle_detection/training_preparation.py](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc/src/vehicle_detection/training_preparation.py) — SHA256 `47b3ea1964f13d2cac223d69b109fcfdb0ba1c5cd4c644c35fdc49215ca8953a`.
- [src/vehicle_detection/training_report.py](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc/src/vehicle_detection/training_report.py) — SHA256 `a6f9817087be894ced686932e7bfcb985fc6d10f9190cda7c5e8646ea421c6c8`.
- [tests/test_training_phases.py](https://github.com/truong51972/vehicle-detection/blob/c11ce10d764fcbba6fabf01b5c5702dc06754edc/tests/test_training_phases.py) — SHA256 `a4e6ccd1aee03e2e90e52772a9e169057f625178a63a7d44cd8d638c9c0b087b`.
