# Issue #47 — explicitly requested repeat after an acquisition bug

The first attempt at `e9a51cc` completed both five-epoch training runs and the
640-trained/640-inference evaluation, then failed because the implementation
accessed `model.validator`. Native `YOLO.val` does not expose that attribute.
The correct check uses the validator supplied to `on_val_end`, as in the existing
method-discovery diagnostic. No detector-benefit decision was reached.

The user explicitly requested **“train it again”** after disclosure of the
failure and lost in-memory cost measurements. This authorizes a fresh matched
pair, superseding the original plan's no-retry rule **for this one repeat only**.
Commit this addendum, repaired implementation and notebook before execution.

All scientific settings, exact populations, support policies, uncertainty,
regression/cost gates and stop rules in [the original plan](ISSUE47_TRAINING_PLAN.md)
remain fixed. Preserve the same 2,048 training IDs (hash
`0c895a25bdb405d289676bedd73d83bf7a483d3709a94a4cdb81e35e8206bdfc`),
1,027 consumed evaluation IDs, seed 2026, independent pinned initializations,
five epochs, batch 16, 640/960 training and the fixed 2×2 evaluation matrix.
Do not change recipe, thresholds or membership in response to the first output.

Use entirely new `runs/issue47-resolution-training-repeat/study` and
`docs/issue47-resolution-training-repeat` roots. Keep the original partial
`runs/issue47-resolution-training` and `docs/issue47-resolution-training`
artifacts, failed executed notebook and log unchanged. Save training wall/peak
VRAM receipts immediately after each arm and evaluation timing receipts after
each cell. Existing/incomplete repeat attempts still block execution.

The repeat budget is 20,480 training image exposures, four explicit evaluations
(4,108 images), and at most four native terminal-validation passes (4,108).
The first attempt already spent 20,480 training exposures, one explicit evaluation
(1,027) and four terminal-validation passes (4,108). Thus both attempts together
use 40,960 training exposures and at most 13,351 validation image passes. Record
both attempts, and report the fresh repeat rather than choosing the better run.
This repeated seed/population is **not independent replication or confirmation**.
No new exploration, confirmation, development or final pixels are authorized.

After this repeat, stop regardless of outcome. Any further run requires its own
justification; there is no automatic retry, batch reduction or resume. The broader
adaptive-preprocessing issue stays open; neither attempt is a full S20 arm.
