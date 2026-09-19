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
