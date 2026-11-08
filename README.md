# CO5177 — Phân tích và trực quan dữ liệu

Landing page cho Bài tập lớn môn **Nền tảng lập trình cho phân tích và trực quan dữ liệu** — HCMUT, học kỳ 261 (2026–2027).

## Điểm nổi bật

- Thiết kế responsive, lấy cảm hứng từ nhịp thị giác của Sequence nhưng có bố cục và nhận diện riêng.
- Trình bày project pipeline: mô tả dữ liệu, chuẩn bị, mô hình hóa, đánh giá.
- Tab tương tác cho Tabular, Text và Image.
- Ba trang EDA riêng, kèm notebook và báo cáo tổng hợp.
- Không dùng framework hay bước build; tương thích GitHub Pages.
- Có semantic HTML, keyboard navigation, focus state và chế độ giảm chuyển động.

## Chạy local

Có thể mở trực tiếp file index.html, hoặc chạy một static server:

    npx serve .

Sau đó truy cập địa chỉ được in trong terminal.

## Cấu trúc

    .
    ├── index.html       # Nội dung và cấu trúc trang
    ├── styles.css       # Design system, responsive layout, animation
    ├── script.js        # Menu, tab dữ liệu, reveal, parallax
    ├── assignment-vne-v4.pdf  # Đề bài và thang điểm của môn học
    ├── tabular-eda.html, text-eda.html, image-eda.html
    ├── notebooks/       # Notebook Colab và mã Python tương ứng
    ├── CO5177 Report Latex/report.pdf  # Báo cáo tổng hợp
    └── README.md

## Triển khai GitHub Pages

Trong repository GitHub, vào **Settings → Pages**, chọn:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/(root)**

Trang sẽ được xuất bản sau khi workflow Pages hoàn tất.

## Tác giả

[Khanh Hoang Duy Dinh](https://github.com/khanhhoangduydinh)

## Tabular assignment

- Website report: `tabular-eda.html`
- Notebook: `notebooks/tabular_koi_eda.ipynb`
- Editable percent-format source: `notebooks/tabular_koi_eda.py`
- Dataset snapshot: `data/koi_cumulative.csv`
- Generated figures: `reports/figures/`
- Consolidated PDF report: `CO5177 Report Latex/report.pdf`

Open the notebook in Colab:

<https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb>

Notebook là bản mở trong Colab; mỗi cell code có hướng dẫn tiếng Việt ngay phía trên.
Khi sửa notebook, cần cập nhật cả file `.py` tương ứng trước khi commit. Đường dẫn
Colab mở bản đã push lên nhánh `main`, không tự đồng bộ các thay đổi chưa push.

## Text assignment

- Website report: `text-eda.html`
- Notebook: `notebooks/text_victsd_eda.ipynb`
- Editable percent-format source: `notebooks/text_victsd_eda.py`
- Dataset: [ViCTSD](https://github.com/tarudesu/ViCTSD)
- Generated figures and summary: `reports/`
- Consolidated PDF report: `CO5177 Report Latex/report.pdf`

Open the notebook in Colab:

<https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/text_victsd_eda.ipynb>

## Image assignment

- Website report: `image-eda.html`
- Notebook: `notebooks/image_intel_eda.ipynb`
- Editable percent-format source: `notebooks/image_intel_eda.py`
- Dataset: [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- Generated figures, summary and prediction CSV: `reports/`
- Consolidated PDF report: `CO5177 Report Latex/report.pdf`

Open the notebook in Colab:

<https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/image_intel_eda.ipynb>

Notebook ảnh tải dữ liệu Kaggle và trọng số MobileNetV3 ở lần chạy đầu; nên bật GPU
trong Colab và chạy lần lượt từ đầu. Video thuyết trình cho ba bài hiện chưa có.

Các tệp `image_intel_eda.ipynb` ở thư mục gốc, `notebooks/image_galaxy10_eda.*`
và `Report Latex/` là những bản khác còn lưu trong repository. Landing page hiện
dẫn đến ba notebook trong `notebooks/` và báo cáo `CO5177 Report Latex/report.pdf`
được liệt kê ở trên.
