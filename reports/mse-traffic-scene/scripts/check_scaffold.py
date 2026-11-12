"""Static checks for the Quarto source scaffold. Does NOT invoke Quarto or render PDF."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "_quarto.yml",
    "index.qmd",
    "frontmatter/abstract.qmd",
    "frontmatter/contributors.yml",
    "chapters/01-introduction.qmd",
    "chapters/02-literature-review.qmd",
    "chapters/03-methodology.qmd",
    "chapters/04-results.qmd",
    "chapters/05-conclusions.qmd",
    "references.qmd",
    "references.bib",
    "appendices/A-cv-operators.qmd",
    "appendices/B-experiment-protocol.qmd",
    "appendices/C-evidence-register.qmd",
    "styles/fpt-mse.tex",
    "styles/title-pages.tex",
    "evidence/manifest.yml",
    "FPT_TEMPLATE_CHECKLIST.md",
]
EXPECTED_MEMBERS = {
    "Đỗ Hữu Phát": "26MS23351",
    "Trần Trung Hiếu": "26MS23355",
    "Đinh Quốc Bảo": "26MS23350",
    "Trường Trần Quốc": "26MS23361",
}


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"Missing report files: {missing}")

    config = (ROOT / "_quarto.yml").read_text(encoding="utf-8")
    contributors = (ROOT / "frontmatter/contributors.yml").read_text(encoding="utf-8")
    cover = (ROOT / "styles/title-pages.tex").read_text(encoding="utf-8")
    for name, sid in EXPECTED_MEMBERS.items():
        for path, source in [("_quarto.yml", config), ("contributors.yml", contributors)]:
            if name not in source:
                raise SystemExit(f"Missing {name} in {path}")
        if sid not in contributors or sid not in cover:
            raise SystemExit(f"Missing ID {sid} in contributor metadata or cover")

    chapter_paths = [f"chapters/0{i}-{suffix}.qmd" for i, suffix in enumerate(
        ("introduction", "literature-review", "methodology", "results", "conclusions"), 1
    )]
    positions = [config.find("    - " + path) for path in chapter_paths]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise SystemExit("FPT five-chapter ordering is missing or incorrect")
    if "type: book" not in config or "pdf:" not in config:
        raise SystemExit("Expected Quarto book and prospective PDF configuration")
    if "enabled: false" not in config:
        raise SystemExit("Scaffold must keep executable code disabled")
    if "Computer Vision" not in (ROOT / "README.md").read_text(encoding="utf-8"):
        raise SystemExit("CV course emphasis missing from README")
    for path in chapter_paths:
        content = (ROOT / path).read_text(encoding="utf-8")
        if not content.startswith("# "):
            raise SystemExit(f"Missing primary heading: {path}")
    print(f"OK — {len(REQUIRED)} required files, 5 chapters, 4 members; no render performed")


if __name__ == "__main__":
    main()
