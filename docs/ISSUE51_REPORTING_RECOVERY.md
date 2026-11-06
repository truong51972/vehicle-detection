# Issue #51 — CPU-only reporting continuation

Both 20-epoch S20 training arms and all four declared development evaluations
completed at frozen acquisition revision `30fde1b`. Every cell passed native AP
agreement, original GT/ID checks and frozen-checkpoint verification. The notebook
parent exited during subsequent CPU bootstrap; its kernel then shut down. No
completed bootstrap/decision was published. Preserve the partial notebook, log,
resource samples and all acquisition files; see the
[interruption receipt](issue47-s20-resolution-20e/analysis_interruption.json).

The user's end-to-end request and instruction to fix encountered implementation
issues authorize finishing this stage. **No training, resume, model construction,
pixel access or image inference is repeated.** All 391,240 training exposures
and 642,850 validation image passes remain the original declared acquisition.

Continue from the four verified saved native matching arrays, fixed development
metadata, frozen cell metrics/timings and unchanged selected checkpoints. Use
the existing `paired_interval` implementation with seed 2026 and **1,000 draws
per all/dark/bright cohort**, passing original source groups exactly as declared.
Execute independent cohorts in three CPU processes, each writing an immutable
receipt immediately on completion. Parallel execution changes neither the
per-cohort RNG seed/order nor AP calculation. Keep CUDA hidden from these workers.

Bind completed cohort receipts to matching-array hashes, metadata hashes, helper
code and parameters before reuse. A partial cohort may be recomputed with the
same inputs; a completed cohort is reused only after identity/hash verification.
Do not reduce/increase draws, redraw missing classes, retune criteria, or select
among repeated intervals. Preserve CPU work already spent in the interrupted
attempt and report continuation costs separately from measured GPU cell costs.

Commit and test the small reporting continuation before running it. The original
training/inference implementation, protocol and implementation receipt remain
unchanged. Reuse the original decision and plotting helpers for final reporting;
publish the same native result schema. Verify every pre-acquisition tracked hash
and historical local size/mtime record; newly added continuation/report files
do not replace historical files. The interruption cannot promote a policy or
authorize sealed confirmation/final pixels. Issue #47 remains open.
