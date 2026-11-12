# Report evidence registry

This folder records **which sources support each report claim**; it intentionally contains no inferred results.

- Vehicle evidence: archived reports under `reports/BDD100K_*/`, source notebooks, `docs/v4/`, and run manifests. Check comparability before combining.
- Traffic sign: external repository, source report and group-confirmed table; missing Hiếu run artifacts and image/GT for independent rerun.
- Joint integration: no measured two-model experiment is assumed.
- References to archives in an earlier DOCX are **not** proof that each artifact is tracked in this Git repository.

Future work: add a versioned manifest with file hash, checkpoint, dataset/split, experiment ID, capture method and reproducibility status for every table/figure. Do not commit private datasets or large model weights solely to make report compilation succeed.
