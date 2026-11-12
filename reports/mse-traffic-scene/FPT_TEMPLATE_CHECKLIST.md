# FPT MSE + Computer Vision report audit checklist

**Scaffold phase**: structure created only. The boxes below are intentionally **not marked complete** by the presence of a placeholder QMD. Track [issue #64](https://github.com/truong51972/vehicle-detection/issues/64).

## P0 — Experimental truth

- [ ] **F01** Actual vehicle + traffic-sign same-frame demonstration, or explicitly title/scope as two independent studies with proposed integration.
- [ ] **F02** Hiếu/Phát traffic-sign reproducibility: both checkpoints/configs, test IDs/GT, 640/960 settings, test artifacts; reconcile conflicting Phát AP50:95 reports.

## P1 — Computer Vision and experiment protocol

- [ ] **F03** Keep batch-32 v0–v3 and batch-16 v0/v1/v4 as separate studies; avoid unqualified rankings.
- [ ] **F04** Verify sign-test duplicates/augmentation leakage, GT support and rare class uncertainty.
- [ ] **F05** Vehicle class imbalance, small-object recall, confidence/error analysis, lighting/occlusion cohorts and GT-versus-prediction examples.
- [ ] **F06** Analyze Canny v4 nearly always-on policy (Apply ~98.39%) and measured accuracy/cost; v3 dark-scene signal versus overall result.
- [ ] **F07** Measure/report CV preprocessing, video I/O, inference and full pipeline latency independently; no invented combined FPS.
- [ ] **F08** Clarify source/train/calibration/development/holdout roles and the vehicle `test` alias versus sign true test.

## P1 — FPT MSE template

- [ ] **F09** Genuine Chapter 2 **Literature Review**, with research gap and cited related work, not a generic knowledge tutorial.
- [ ] **F10** Chapter 4 **Simulation Results** must separate measured evidence from proposals.
- [ ] **F11** Compare two cover pages with official template; use exactly four verified team members, confirm lecturer/course/title.
- [ ] **F12** Abstract has title above it, ≤150 words (master's) and double spacing; remove internal evidence-snapshot notes.
- [ ] **F13** Centered two-line chapter headings, upper-right page numbering, correct front-matter transitions when PDF render is permitted.
- [ ] **F14** Times-family 12pt, 1.5 line spacing, no first-line indent, 12pt paragraph spacing; visual verification after render.
- [ ] **F15** Five main chapters in required order; automatic TOC, LoT, LoF, citations, references and appendices.

## P2 — Editorial quality

- [ ] **F16** Remove repeated limitations/checklists and avoid 53-table overload; keep detailed settings in appendices.
- [ ] **F17** Improve weak/dark montage, readable figures, CV-centric paired deltas, small-object error and clear captions.
- [ ] **F18** Fix glued words, stretched justified paragraphs, clipped tables/captions and Vietnamese glyph issues.
- [ ] **F19** BibTeX + source hashes; claim-by-claim evidence map; no invented statistics or missing-artifact substitution.

## Intended Computer Vision emphasis

- [ ] Explain pixel transformations, grayscale, illumination, histograms and clipping rather than optimizer tuning.
- [ ] Explain Canny gradients/edges and risks of amplifying background texture.
- [ ] Compare the same original scene with matching GT boxes and transformed variants.
- [ ] Discuss image/video geometry, resized box coordinate mapping and actual end-to-end processing cost.
- [ ] Keep YOLO26n training details limited to reproducibility controls.

## PR boundary

- [x] Create isolated `reports/mse-traffic-scene/` source scaffold.
- [x] Populate four verified member names/IDs.
- [x] Create five CV-oriented chapter outlines and evidence policy.
- [x] Prepare bibliography, appendix layout, LaTeX-style draft and future cover file.
- [ ] **NOT REQUESTED NOW:** render PDF, inspect its typography, finalize contents, conduct experiments or close F01–F19.
