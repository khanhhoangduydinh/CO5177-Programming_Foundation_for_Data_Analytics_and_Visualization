# DataFlow

Landing page cho Bài tập lớn môn **Nền tảng lập trình cho phân tích và trực quan dữ liệu** — HCMUT, học kỳ 261 (2026–2027).

## Điểm nổi bật

- Thiết kế responsive, lấy cảm hứng từ nhịp thị giác của Sequence nhưng có bố cục và nhận diện riêng.
- Trình bày project pipeline: mô tả dữ liệu, chuẩn bị, mô hình hóa, đánh giá.
- Tab tương tác cho Tabular, Text và Image.
- Dashboard rubric dựa trên brief môn học.
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
    ├── Assigment.pdf    # Brief gốc của môn học
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
- Executed notebook: `notebooks/tabular_koi_eda.ipynb`
- Editable percent-format source: `notebooks/tabular_koi_eda.py`
- Dataset snapshot: `data/koi_cumulative.csv`
- Generated figures: `reports/figures/`
- PDF report: `reports/tabular-koi-eda-report.pdf`
- PDF generator: `reports/generate_tabular_report.py`

Open the notebook in Colab:

<https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb>

The notebook is the source of truth. Edit the `.py` or `.ipynb`, run all cells,
then commit and push; the GitHub-backed Colab link always opens the latest
version from `main`.
