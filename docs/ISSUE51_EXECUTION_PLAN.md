# Issue #51 — authorized full S20 execution

The user explicitly requested end-to-end execution after implementation commit
`0dba036`: address review findings, complete both full training arms and the fixed
matrix, analyze the results, and publish a PR. This supersedes the earlier
implementation-only authorization **for this declared study**. It does not
authorize policy promotion or confirmation/final-test access. Issue #47 stays open.

The [finalized protocol](ISSUE47_FULL_TRAINING_PLAN.md), native workflow, defaults,
thresholds and population identities are unchanged. The original implementation
and [software-validation receipt](issue51/validation.json) remain historical
evidence rather than being rewritten to imply they already ran the study.

## Review resolution and fixed acquisition

- P1: native CPU tests at actual training sizes 640/960 verified fixed selection
  args, loader and maximum tensor edge 640 at every epoch and terminal best
  validation, using native rectangular `pad=0`. Raw native checkpoint epoch and
  metrics are captured before stripping; fitness selection and ties are checked.
  The inference matrix retains native rectangular `pad=.5` and saves actual shapes.
- P2 uncertainty: prospectively fixed **1,000** original-source-group paired
  draws each for all/dark/bright; no draw-count changes after outcomes, missing
  class imputation/redraw, or independent-confirmation claims.
- P2 support: report GT/images/distinct source groups, per-class intervals and
  fixed-confidence TP/FP/FN, particularly dark-bus and motorcycle/bicycle. Sparse
  or contradictory evidence cannot become a clean success.
- Use all **9,781 S20 training images**, **20 epochs per arm**, identical original
  GT/negatives/Raw JPEG95 exports, independent pinned YOLO26n initialization,
  seed 2026, batch 16, FP32 AdamW, freeze 11 and the frozen disabled augmentations.
- Use the same **13,975 development images** for fixed-640 native checkpoint
  selection and all four matrix cells, in declared order 640/640, 960/960,
  640/960, 960/640. D-B is primary. No historical trained model is substituted.
- Native recovery/batch reduction, silent resume, rescue arms, additional epochs,
  evaluation sizes, retuning and data expansion remain forbidden. Preserve any
  failed/partial attempt. The current request authorizes fixing implementation
  issues; any necessary repeat must have a prospective addendum and fresh owned
  outputs, never delete or overwrite evidence or select the better attempt.

## Execution and receipts

The pinned checkpoint SHA256 is
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Preflight found Torch 2.15.0+cu132, CUDA 13.2, an RTX 5060 Ti (16 GiB), about
757 GiB free disk and 18 GiB available host RAM. CUDA arithmetic passed without
reading study pixels. The frozen-input guard passes and the declared output roots
do not exist before preparation. These are prerequisites, not training outcomes.

Run the unchanged committed
[research notebook](../notebooks/research/issue47/03_full_resolution_training.ipynb)
in a fresh kernel using `nbclient`, injecting `ALLOW_PREPARATION=True`,
`ALLOW_TRAINING=True`, `ALLOW_INFERENCE=True` into the configuration **in memory**.
Keep its cleared-output, read-only defaults in Git. No upstream notebook runs.
Save the executed copy and kernel log under ignored
`runs/issue47-s20-resolution-20e/` after acquisition; use temporary external log
paths during execution so the historical inventory does not include a changing
execution log outside the owned study root.

Native wall/VRAM receipts are persisted after each arm and timing receipts after
each cell. Preserve histories, settings, selection pass/epoch receipts, best/last
hashes, exact membership/export/GT/config/code identities and matching arrays.
Preparation and later stages verify historical tracked hashes and local file
size/mtime inventories. Source hashing/decoding is restricted to the allowed
existing Raw exports and new owned copies; reserved confirmation and final pixels,
headers and source hashes remain sealed. Annotation metadata is allowed.

The ~3 GPU-hour estimate remains a planning ceiling. The per-arm 90/150-minute
abort limits and 14 GiB allocated VRAM limit remain unchanged. The CPU bootstrap
has no added inference and its wall time/RAM will be reported separately.

Publish compact outcomes, convergence, class/cohort errors, uncertainty, costs,
integrity checks and execution limitations in a results document and PR. Never
reinterpret the PR #50 failures, infer generalization from reused development,
insert a v3 policy row, or access sealed data to resolve an unfavorable outcome.
