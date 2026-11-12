"""Static, no-render validation for the single-file CV academic report."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "report.qmd"
MEMBERS = {
    "Đỗ Hữu Phát": "26MS23351",
    "Trần Trung Hiếu": "26MS23355",
    "Đinh Quốc Bảo": "26MS23350",
    "Trường Trần Quốc": "26MS23361",
}
REQUIRED_HEADINGS = (
    "# Tóm tắt {.unnumbered}",
    "# Giới thiệu (Introduction)",
    "# Tổng quan nghiên cứu (Literature Review)",
    "# Phương pháp (Methodology — Dataset + Algorithms)",
    "# Kết quả thực nghiệm (Simulation Results)",
    "# Kết luận (Conclusions)",
    "# Tài liệu tham khảo {.unnumbered}",
    "# Chi tiết các toán tử xử lý ảnh",
    "# Giao thức đánh giá và đối chứng",
    "# Xuất xứ dữ liệu, hình và mã nguồn",
)
RESOURCES = (
    "references.bib",
    "evidence/manifest.yml",
    "FPT_TEMPLATE_CHECKLIST.md",
    "README.md",
    "figures/README.md",
)


def main() -> None:
    qmds = sorted(ROOT.rglob("*.qmd"))
    if qmds != [MAIN]:
        raise SystemExit(f"Expected exactly one report.qmd, found: {qmds}")
    for relative_path in RESOURCES:
        if not (ROOT / relative_path).is_file():
            raise SystemExit(f"Missing companion resource: {relative_path}")

    text = MAIN.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise SystemExit("Missing Quarto YAML front matter")
    header = text.split("\n---\n", 1)[0]
    if "  pdf:" not in header or "documentclass: report" not in header:
        raise SystemExit("Expected standalone Quarto PDF configuration")
    if "enabled: false" not in header or "type: book" in header:
        raise SystemExit("Unexpected code-execution/Quarto Book configuration")
    for name, sid in MEMBERS.items():
        if name not in header or sid not in header:
            raise SystemExit(f"Missing contributor: {name} / {sid}")

    positions = [text.find(h) for h in REQUIRED_HEADINGS]
    if any(p < 0 for p in positions) or positions != sorted(positions):
        raise SystemExit("Report headings missing or not in FPT order")
    if text.count("\n\\appendix\n") != 1:
        raise SystemExit("Exactly one LaTeX appendix switch is required")

    abstract = text.split("# Tóm tắt {.unnumbered}", 1)[1].split("**Từ khóa:**", 1)[0]
    abstract_words = len(abstract.split())
    if abstract_words > 150:
        raise SystemExit(f"Abstract has {abstract_words} whitespace-separated words (>150)")

    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([A-Za-z0-9_-]+),", bib))
    citations = set(re.findall(r"(?<![A-Za-z0-9])@([A-Za-z][A-Za-z0-9_-]*)", text))
    missing_refs = sorted(
        c for c in citations if not c.startswith(("fig-", "tbl-")) and c not in keys
    )
    if missing_refs:
        raise SystemExit(f"Unresolved bibliography keys: {missing_refs}")

    fig_labels = set(re.findall(r"\{#(fig-[A-Za-z0-9_-]+)\b", text))
    used_figs = set(re.findall(r"@((?:fig-)[A-Za-z0-9_-]+)", text))
    if missing := sorted(used_figs - fig_labels):
        raise SystemExit(f"Unresolved internal figure references: {missing}")

    paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    missing_images = [p for p in paths if not (ROOT / p).is_file()]
    if missing_images:
        raise SystemExit(f"Missing local figures: {missing_images}")

    print(
        "OK: 1 QMD, 4 members, 5 chapters, 3 appendices, "
        f"{abstract_words}-word abstract, {len(keys)} bibliography entries, "
        f"{len(paths)} source-backed images; no PDF render."
    )


if __name__ == "__main__":
    main()
