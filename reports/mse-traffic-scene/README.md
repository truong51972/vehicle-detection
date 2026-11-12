# FPT MSE Computer Vision — báo cáo Quarto

**Chỉ có một file báo cáo: [`report.qmd`](report.qmd).** Không còn Quarto Book hay các file `.qmd` tách theo chương.

Trạng thái: **mới là khung**, chưa chuyển đầy đủ nội dung Word và **chưa render PDF**.

## Tổ chức

Trong `report.qmd` theo đúng thứ tự: metadata 4 thành viên/MSSV, Abstract, Chapter 1 Introduction, Chapter 2 Literature Review, Chapter 3 Methodology (Dataset + Computer Vision Algorithms), Chapter 4 Simulation Results, Chapter 5 Conclusions, References, các phụ lục A–C.

Toàn bộ nội dung học thuật và cấu hình Quarto/PDF được lưu trong một file `report.qmd`. Các file phụ trợ không phải báo cáo: `references.bib` (trích dẫn), `evidence/manifest.yml` (nguồn kết quả), `FPT_TEMPLATE_CHECKLIST.md` (19 findings), `figures/` (ảnh tương lai), `scripts/check_scaffold.py` (kiểm tra cấu trúc).

Giữ nguyên `docs/report.qmd` của project (nghiên cứu Vehicle Detection, xuất HTML) và các báo cáo/experiment hiện hữu.

## Trọng tâm học thuật

Môn **Computer Vision**: xử lý ảnh theo độ sáng, histogram, Gamma, CLAHE, Canny/biên, ROI, ảnh nhỏ, hình học bounding box, độ che khuất, tiền xử lý video, so sánh lỗi trên cùng frame. YOLO26n là công cụ phát hiện để đánh giá các biến đổi ảnh, không phải trọng tâm tối ưu Deep Learning.

Template FPT có chữ *Image and Video Processing*; cần xác nhận tên học phần chính thức, tên giảng viên và định dạng bìa trước khi nộp. Giữ **đúng bốn thành viên** do nhóm cung cấp, không thêm thành viên theo ví dụ sáu người trong template.

## Kiểm tra khung (không render)

```bash
python reports/mse-traffic-scene/scripts/check_scaffold.py
```

## Xuất PDF về sau — hiện chưa chạy

```bash
quarto render reports/mse-traffic-scene/report.qmd --to pdf
```

Cần Quarto, TeX và font Times tương thích tiếng Việt. Cấu hình format PDF hiện mới là dự kiến; hai trang bìa theo FPT, Abstract double spacing, bố cục tiêu đề chương, số trang và khả năng render đều chưa được kiểm chứng trực quan.

## Quy tắc nguồn

- Chỉ lấy bảng/đồ thị từ evidence có định danh run, dataset, split và checkpoint.
- Giữ riêng Study A v0–v3 batch 32 và Study B v0/v1/v4 batch 16.
- Kết quả biển báo Phát/Hiếu được nhóm cung cấp nhưng còn thiếu artifact độc lập của Hiếu.
- Chưa được báo cáo FPS/mAP hệ thống ghép khi chưa chạy hai detector thực tế.
- Không mở sealed final holdout hoặc tự chạy train/inference khi dựng báo cáo.
