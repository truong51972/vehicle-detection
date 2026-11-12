"""Validate the single report.qmd source scaffold without rendering PDF."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "report.qmd"
COMPANIONS = ("references.bib", "evidence/manifest.yml", "FPT_TEMPLATE_CHECKLIST.md", "README.md")
MEMBERS = {
    "Đỗ Hữu Phát": "26MS23351",
    "Trần Trung Hiếu": "26MS23355",
    "Đinh Quốc Bảo": "26MS23350",
    "Trường Trần Quốc": "26MS23361",
}
HEADINGS = (
    "# Tóm tắt {.unnumbered}",
    "# Giới thiệu (Introduction)",
    "# Tổng quan nghiên cứu (Literature Review)",
    "# Phương pháp (Methodology — Dataset + Algorithms)",
    "# Kết quả thực nghiệm (Simulation Results)",
    "# Kết luận (Conclusions)",
    "# Tài liệu tham khảo {.unnumbered}",
    "# Tham số và công thức các toán tử xử lý ảnh",
    "# Giao thức đánh giá, ngưỡng và chi phí",
    "# Nguồn, artifact và giới hạn tái lập",
)


def main() -> None:
    qmds = sorted(ROOT.rglob("*.qmd"))
    if qmds != [MAIN]:
        raise SystemExit(f"Expected exactly one report.qmd, found: {qmds}")
    for relative_path in COMPANIONS:
        if not (ROOT / relative_path).is_file():
            raise SystemExit(f"Missing source companion: {relative_path}")
    source = MAIN.read_text(encoding="utf-8")
    if not source.startswith("---\n") or "\n---\n" not in source[4:]:
        raise SystemExit("Missing YAML front matter")
    yaml_header = source.split("\n---\n", 1)[0]
    if "  pdf:" not in yaml_header or "documentclass: report" not in yaml_header:
        raise SystemExit("Expected standalone Quarto PDF report metadata")
    if "type: book" in source or "    - index.qmd" in source:
        raise SystemExit("Unexpected Quarto Book configuration")
    if "enabled: false" not in yaml_header:
        raise SystemExit("QMD code execution must be disabled in scaffold")
    for name, student_id in MEMBERS.items():
        if name not in yaml_header or student_id not in yaml_header:
            raise SystemExit(f"Missing member or ID: {name} / {student_id}")
    positions = [source.find(heading) for heading in HEADINGS]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise SystemExit("Required sections missing or out of FPT order")
    if source.count("\n\\appendix\n") != 1:
        raise SystemExit("Expected exactly one PDF appendix switch")
    print("OK — one report.qmd, four members, five chapters, references, three appendices. No render performed.")


if __name__ == "__main__":
    main()
