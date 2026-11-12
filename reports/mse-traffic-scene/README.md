# FPT MSE Computer Vision report — Quarto scaffold

**Status: structure only. No PDF rendering, no completed report, no verified two-model integration.**

Tracks [#64](https://github.com/truong51972/vehicle-detection/issues/64). This is a new combined-report project. **Do not replace** the existing vehicle-only [docs/report.qmd](../../docs/report.qmd) or historical DOCX/PDF exports.

## Scope and emphasis

- **Course:** Computer Vision (per project-team clarification). The provided FPT template literally says *Image and Video Processing*; confirm the official course name with the lecturer before submission.
- **Core scientific subject:** image formation and quality, gamma, local illumination adjustment, CLAHE, Canny/edges, color/contrast, object size, bounding-box geometry, evaluation under darkness/occlusion, image/video pipeline and computer-vision failure cases.
- **YOLO26n:** a fixed object-detection instrument/benchmark. Explain detection outputs and the reproducibility controls needed to compare CV image-processing policies, **not** a tutorial on neural network architectures or training optimization.
- **Two branches:** five vehicle classes (BDD100K) and six traffic-sign classes (separate source and dataset). Never average mAP across the datasets.
- **Evidence boundary:** no reported joint-demo FPS, joint mAP, distance-in-metres or driving-control accuracy unless the corresponding experiments have genuinely run.

## Project organization

| File / directory | Purpose |
| --- | --- |
| \`_quarto.yml\` | Quarto book chapter order and prospective PDF configuration |
| \`index.qmd\`, \`frontmatter/\` | Preface, abstract and contributor metadata |
| \`chapters/\` | Exactly five FPT-oriented report chapters |
| \`appendices/\` | CV operator details, experiment protocol and evidence mapping |
| \`references.bib\`, \`references.qmd\` | Source-based citations, generated bibliography |
| \`styles/fpt-mse.tex\` | Initial LaTeX typography; **not yet visually validated** |
| \`styles/title-pages.tex\` | Prepared two-cover layout with four members; **not wired into render** until duplication with Quarto's default title page is resolved |
| \`evidence/\` | Provenance register, unresolved data/checkpoint requirements |
| \`figures/\` | Future source-backed diagrams and plots |
| \`FPT_TEMPLATE_CHECKLIST.md\` | All findings (F01–F19), grouped for implementation |
| \`scripts/check_scaffold.py\` | No-dependency, static structure-only validation |

## Members (provided by the team)

1. Đỗ Hữu Phát — 26MS23351
2. Trần Trung Hiếu — 26MS23355
3. Đinh Quốc Bảo — 26MS23350
4. Trường Trần Quốc — 26MS23361

These four names and IDs are authoritative for this scaffold. The original FPT template shows six example member slots; do not invent two additional members. Lecturer identity and official course wording are **to be confirmed**.

## Editing order

1. Confirm course title, lecturer, report title and permission to use a Vietnamese-language report.
2. Add actual related research to Chapter 2; separate prior literature from explanatory background.
3. Fill Chapter 3 from pinned preprocessing/evaluation code and data manifests.
4. Fill Chapter 4 from *existing verified* outputs. Keep batch-32 v0–v3 and matched batch-16 v0/v1/v4 distinct.
5. Obtain missing sign-model evidence and a genuinely measured two-branch video demo if the final title claims integrated perception.
6. Finish discussion, limitations, references and figures. Only **then** validate PDF fidelity to the 12-page FPT sample.

## Validation

The only validation intended for this PR is a static source check:

\`\`\`bash
python reports/mse-traffic-scene/scripts/check_scaffold.py
\`\`\`

**No PDF was rendered for this PR, by request.** When the team authorizes that next phase, prospective command:

\`\`\`bash
quarto render reports/mse-traffic-scene --to pdf
\`\`\`

PDF generation will require Quarto, a suitable TeX installation and licensed/installed Vietnamese-capable fonts. The current configuration (including \`Times New Roman\` and header layout) is **provisional**; do not claim visual or compilation parity until a real build, inspection and comparison are performed.

## Source policy

- No automatic train/inference/download during report generation; \`execute.enabled=false\` in this initial scaffold.
- No guessed numerical results or screenshots presented as new measurements.
- Keep cross-references and citations in native Quarto syntax; figures must identify source, dataset, split, run and whether GT or prediction.
- \`docs/report.qmd\`, experimental notebooks, \`runs/\`, and raw checkpoints are outside this PR's scope.
