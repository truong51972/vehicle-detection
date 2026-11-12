# Combined FPT MSE Computer Vision report

**Chỉ có một Quarto source chính:** [`report.qmd`](report.qmd). Toàn bộ Abstract, năm chương và phụ lục nằm trong file này, không dùng Quarto Book hay các QMD con.

## Trạng thái

- **Đã viết bản thảo nội dung** có căn cứ cho 5 chương, gồm Literature Review thực chất, Methodology tập trung Computer Vision, bảng kết quả Vehicle và hai repo biển báo, phần giới hạn, kết luận và phụ lục truy vết.
- **Chưa render PDF** theo yêu cầu. Typography, bìa hai trang, Abstract spacing, số trang và trích dẫn cần một lượt kiểm tra PDF riêng.
- Chưa chạy train/eval/inference mới hay video ghép hai detector; không tự phát minh số liệu.

## Nguồn thực nghiệm

**Vehicle:** [truong51972/vehicle-detection](https://github.com/truong51972/vehicle-detection), các run v0–v4; tách **Study A (v0–v3 batch 32)** và **Study B (Raw matched/v4 batch 16)**. Development evaluation không phải sealed final holdout.

**Traffic sign — Phát:** [dhphat/traffic-sign-detection](https://github.com/dhphat/traffic-sign-detection/tree/f3c30d8be4e31404a73e8621d8fb050a62c90e05) có checkpoint, cấu hình, history, code eval và demo. Script kiểm tra 640/960, nhưng JSON test riêng chưa được commit tại phiên bản đã đọc.

**Traffic sign — Hiếu:** [ec2003/VN-traffic-sign-detection](https://github.com/ec2003/VN-traffic-sign-detection/tree/1bcf75eebbb3b255f622ade63900cead00abd863) có **best.pt**, args.yaml, results.csv, notebook, hình test. README ghi **test @832**: AP50 97,27%, AP50:95 78,36%, **khác** bảng 640/960 trong draft nhóm.

Nguồn ảnh: `figures/README.md` ghi rõ hình tự vẽ và hình từ repository Hiếu (Apache-2.0, có lưu ý điều khoản dataset). Hình vehicle được **tham chiếu** tương đối tới `../../docs/images` của repo, không sao chép.

## Trọng tâm môn học

Computer Vision: ánh sáng, tương phản, Gamma, local-shadow, CLAHE, Canny, shape/edge, histogram, pHash, lựa chọn augmentation và hình học bbox. YOLO26n chỉ là công cụ đo detection; mô tả training được giữ đủ để kiểm soát phương pháp so sánh. Chưa tuyên bố nhận thức cảnh tích hợp cho đến khi có đo video chung.

Thành viên đã xác nhận: Đỗ Hữu Phát (26MS23351), Trần Trung Hiếu (26MS23355), Đinh Quốc Bảo (26MS23350), Trường Trần Quốc (26MS23361).

**Cần xác nhận** tên giảng viên, tên học phần chính thức (template cung cấp ghi *Image and Video Processing* còn yêu cầu nhóm dùng *Computer Vision*), và format tiếng Việt trước khi nộp.

## Kiểm tra file (không dựng PDF)

```bash
python reports/mse-traffic-scene/scripts/check_scaffold.py
```

## Xuất PDF về sau (chưa thực hiện)

```bash
quarto render reports/mse-traffic-scene/report.qmd --to pdf
```

Quarto/Pandoc, LaTeX và font Times có hỗ trợ tiếng Việt là các prerequisite. `report.qmd` chứa YAML format PDF tạm thời; một build tương lai cần đo chính xác khoảng cách lề, bìa, TOC, List of Tables/Figures, chapter headings, citation formatting và thứ tự số trang theo FPT template.

## Khi cập nhật kết quả mới

Chỉ thay số liệu sau khi có `metrics.json` / `results.csv` cùng manifest hash và run/commit. Mỗi hình ghi đối tượng, test/development, preprocessing, model và nguồn. Không tự hợp nhất mAP xe/biển báo, không lấy số lần benchmark riêng để suy joint FPS.
