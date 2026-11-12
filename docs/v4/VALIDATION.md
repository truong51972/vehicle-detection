## Publication validation — 2026-11-12

The isolated v4 branch is based on main commit
`af9e4931d73e49a302f94627cf16fcfcc200c752`. The original active workspace and its
background v5 experiment were not changed.

- Base environment: **53 passed, 5 skipped** (`pytest -q`). Optional native
  dependencies are absent from this environment. V4 fresh-kernel preview/export
  tests execute on synthetic images, including opt-out and reuse.
- Native study environment: **25 passed** for comparison, evaluation and Canny
  contracts excluding the duplicate fresh-kernel Canny case. This includes the
  v4 factorial bootstrap identity/GT checks.
- Native notebook suite: the four fresh-kernel/structure/lifecycle tests passed,
  including tiny CPU train/eval and off-diagonal evaluation. Its first run found
  a pre-existing portability assertion against main's archived Ultralytics
  `args.yaml`; after narrowing the exception to native args archives, the failed
  documentation/portability test was rerun and **passed**. Portable source/config
  checks remain active. No BDD images or new scientific AP were used in these tests.
- `ruff check .` and `ruff format --check .`: passed. Archived `runs/` evidence is
  excluded so formatting does not rewrite executed notebook receipts.
- Windows Jupyter/ZMQ emitted its selector-thread warning; executed tests passed.
- V0–v3 notebook/config files and the frozen split table have no diff from main.
- V4 metrics and checkpoints match the completion marker and source SHA256 values;
  native evidence is marked `-text` to preserve byte identities across platforms.
- No v5 runtime modules or raw/processed BDD exports are introduced. The one
  historical `v5_raw` alias in evidence is the matched Raw control, not v5 code.

No GitHub CI result is claimed here; these are local checks. The preserved notes
below describe the original implementation stage, not the current execution state.

# v4 implementation checks — 2026-11-10

Local implementation in the downloaded `main` snapshot. No Git commit or remote
code publication has been performed.

- `python -X utf8 -m pytest -q`: **53 passed, 4 skipped**. PyTorch/Ultralytics-dependent
  tests were skipped because those optional packages are not installed. One Windows
  Jupyter/ZMQ event-loop warning was emitted; all executed tests passed.
- `ruff check .`: passed.
- `ruff format --check .`: passed (49 files).
- Every notebook validates against nbformat.
- Fresh kernels executed v4 preview and preparation on tiny synthetic sources,
  including opt-out, export and reuse. Guards rejected any training/inference
  engine import in those phases. This is software verification, not BDD evidence.
- Operator checks cover locality, uint8/geometry, deterministic output, source
  immutability, bounded offsets, saturation, tiny images, routes and invalid recipes.
- Synthetic v4 export preserves exact Raw label bytes, negative examples and
  Raw-identical Skip JPEGs; final-source reads are guarded.
- Original ZIP comparison confirms v0–v3 notebook/config/README bytes are unchanged.
  Shared helper/contract tests and the comparison notebook were extended for v4.
- Windows test fixtures now store POSIX split paths, copy fixtures without requiring
  symlink privileges and accept a GitHub source ZIP without a `.git` directory.

The local environment is `.venv` in the workspace, with an editable install of this
`main` snapshot and the base/dev/notebook packages. Installation was not resolved
from `uv.lock`; the pinned full `study` environment and native training/evaluation
checks remain to be run before real GPU work. On Windows Python 3.13, use UTF-8 mode
(`python -X utf8` / `PYTHONUTF8=1`) because existing project readers assume UTF-8.

## Research and publication status

No real BDD image preview, pilot, export, training or evaluation was executed.
The v4 policy and proposed scientific gates are unvalidated hypotheses.

Four issue bodies are ready under [issues](issues). GitHub plugin search reported
the connector installed and enabled, but issue search/create returned `Unknown tool`.
No remote issue was created. GitHub CLI and `GH_TOKEN`/`GITHUB_TOKEN` are absent locally.
Reconnect the GitHub integration before publishing the prepared issues; link child
issues to the parent after creation. No remote issue number is fabricated.
