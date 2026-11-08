![CO5177 — From data to insight: tabular, text and image](assets/readme-banner.svg)

<p align="center">
  <a href="https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/"><img alt="Website CO5177" src="https://img.shields.io/badge/Website-CO5177-173f4c?style=flat-square"></a>
  <a href="https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb"><img alt="Open in Google Colab" src="https://img.shields.io/badge/Notebooks-Google%20Colab-d35b12?style=flat-square&amp;logo=googlecolab&amp;logoColor=white"></a>
  <a href="CO5177%20Report%20Latex/report.pdf"><img alt="Báo cáo tổng hợp PDF" src="https://img.shields.io/badge/Report-PDF-618e81?style=flat-square"></a>
  <img alt="Ba loại dữ liệu" src="https://img.shields.io/badge/Data-Tabular%20%7C%20Text%20%7C%20Image-efae00?style=flat-square">
</p>

# CO5177 — Phân tích và trực quan dữ liệu

Repository bài tập lớn môn **Programming Foundation for Data Analytics and Visualization (CO5177)**, Trường Đại học Bách khoa — ĐHQG TP.HCM, học kỳ HK261 (2026–2027). Bài làm sử dụng Python để khám phá dữ liệu, trực quan hóa, tiền xử lý, xây dựng mô hình và đánh giá kết quả trên ba dạng dữ liệu.

**Nhóm:** Trailblazer · **Thực hiện:** Dinh Hoang Duy Khanh · **Giảng viên:** Dr. Le Thanh Sach

## Xem nhanh

| Bài | Dữ liệu và bài toán | Trang EDA | Notebook Colab |
| --- | --- | --- | --- |
| Tabular | NASA Kepler KOI: phân loại tín hiệu thành ba nhóm | [Xem Tabular EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/tabular-eda.html) | [Mở Tabular](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/tabular_koi_eda.ipynb) |
| Text | ViCTSD: phân loại bình luận tiếng Việt theo hai nhãn | [Xem Text EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/text-eda.html) | [Mở Text](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/text_victsd_eda.ipynb) |
| Image | Intel Image Classification: phân loại sáu loại cảnh | [Xem Image EDA](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/image-eda.html) | [Mở Image](https://colab.research.google.com/github/khanhhoangduydinh/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/blob/main/notebooks/image_intel_eda.ipynb) |

- [Trang chính](https://khanhhoangduydinh.github.io/CO5177-Programming_Foundation_for_Data_Analytics_and_Visualization/) · [Báo cáo tổng hợp PDF](CO5177%20Report%20Latex/report.pdf) · [Đề bài và thang điểm](assignment-vne-v4.pdf)
- Video thuyết trình cho các bài **chưa có**.

## Nội dung từng bài

### 1. Tabular — Kepler Objects of Interest

- **Dữ liệu:** 9.564 dòng, 30 cột trong bản snapshot [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html). Nhãn `koi_disposition` gồm `CONFIRMED`, `CANDIDATE` và `FALSE POSITIVE`.
- **Thực hiện:** kiểm tra dữ liệu thiếu và ngoại lệ, trực quan các đặc trưng, tránh đưa thông tin liên quan trực tiếp đến nhãn vào đầu vào mô hình; so sánh mô hình phân loại.
- **Kết quả đã lưu:** Balanced Random Forest đạt accuracy **0,758** và macro F1 **0,727** trên tập test. Số liệu nằm trong [tabular_koi_summary.json](reports/tabular_koi_summary.json).
- **Mã nguồn:** [notebook](notebooks/tabular_koi_eda.ipynb) · [Python](notebooks/tabular_koi_eda.py) · [dữ liệu](data/koi_cumulative.csv).

### 2. Text — ViCTSD

- **Dữ liệu:** [ViCTSD](https://github.com/tarudesu/ViCTSD) gồm 10.000 bình luận tiếng Việt; bộ dữ liệu ban đầu chia 7.000 train, 2.000 validation và 1.000 test. Hai nhãn được phân tích riêng là `Toxicity` và `Constructiveness`.
- **Thực hiện:** thống kê và trực quan dữ liệu văn bản, kiểm tra bình luận trùng giữa các tập, chuyển văn bản thành đặc trưng TF-IDF và so sánh các mô hình phân loại.
- **Kết quả đã lưu:** trên tập test đã lọc, bài toán Toxicity đạt accuracy **0,842**, macro F1 **0,679**; Constructiveness đạt accuracy **0,807**, macro F1 **0,800**. Chi tiết: [text_victsd_summary.json](reports/text_victsd_summary.json).
- **Giới hạn:** sau khi loại các bình luận trùng với train, vẫn còn một bình luận chuẩn hóa xuất hiện ở cả validation và test. Các kết quả đã lưu chưa loại trường hợp này; cần loại và chạy lại trước khi coi hai tập hoàn toàn độc lập.
- **Mã nguồn:** [notebook](notebooks/text_victsd_eda.ipynb) · [Python](notebooks/text_victsd_eda.py) · [thông tin dữ liệu](data/README.md).

### 3. Image — Intel Image Classification

- **Dữ liệu:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) có 17.034 ảnh có nhãn, mỗi ảnh 150 × 150 px, thuộc sáu lớp: buildings, forest, glacier, mountain, sea và street.
- **Thực hiện:** kiểm tra phân bố lớp và ảnh mẫu, tăng cường dữ liệu cho tập train, dùng MobileNetV3 pretrained để trích xuất đặc trưng và so sánh hai cách phân loại. Validation được dùng để chọn mô hình; tập test riêng dùng cho đánh giá cuối.
- **Kết quả đã lưu:** mô hình Logistic Regression trên đặc trưng pretrained đạt accuracy **0,914** và macro F1 **0,916** trên 3.000 ảnh test. Chi tiết: [image_intel_summary.json](reports/image_intel_summary.json).
- **Lưu ý:** với phương án huấn luyện head, các tham số backbone được cố định nhưng thống kê chạy của Batch Normalization vẫn có thể thay đổi ở chế độ train.
- **Mã nguồn:** [notebook](notebooks/image_intel_eda.ipynb) · [Python](notebooks/image_intel_eda.py) · [dự đoán test](reports/image_intel_test_predictions.csv).

Các điểm số trên là kết quả đã được lưu của các lần chạy trước, không phải cam kết rằng mọi lần chạy sẽ cho kết quả giống hệt.

## Chạy notebook trên Google Colab

1. Mở notebook tương ứng bằng liên kết trong bảng **Xem nhanh**.
2. Đọc hướng dẫn tiếng Việt trước mỗi cell và chọn **Runtime → Run all**. Lần chạy đầu cần Internet để tải các gói hoặc dữ liệu cần thiết.
3. Với bài Image, nên chọn GPU trong **Runtime → Change runtime type**. Notebook tải ảnh Intel và trọng số pretrained nếu chưa có; thời gian chạy và dung lượng phụ thuộc môi trường Colab.
4. Khi cần lưu thay đổi, chọn **File → Save a copy in Drive** hoặc lưu bản sao về GitHub. Liên kết Colab ở trên mở notebook từ nhánh `main` trên GitHub; thay đổi chưa được lưu/push không tự cập nhật lên repository.

Mỗi bài có một notebook `.ipynb` và file Python dạng percent `.py` tương ứng. Đây là hai file riêng: nếu chỉnh một file, hãy kiểm tra và cập nhật file còn lại khi cần; chúng không tự đồng bộ.

## Chạy website trên máy

Cần Python 3 đã cài trên máy. Mở terminal tại thư mục repository và chạy:

```bash
python -m http.server 8000
```

Mở `http://localhost:8000/` trong trình duyệt. Website là HTML/CSS/JavaScript tĩnh, không cần bước build. Dùng server thay vì mở trực tiếp file để các trang có thể tải dữ liệu JSON đúng cách.

Để cài các thư viện Python cho notebook trên máy, có thể dùng `pip install -r requirements-eda.txt` trong môi trường ảo. Với PyTorch/torchvision, hãy chọn bản phù hợp với hệ điều hành và CPU/GPU nếu môi trường không cài được cấu hình mặc định.

## Cấu trúc repository

```text
├── index.html, styles.css, script.js   # Trang chính và tương tác
├── tabular-eda.html, text-eda.html, image-eda.html
├── eda.css, eda.js                    # Giao diện dùng chung cho các trang EDA
├── assets/                            # Logo, banner README và ảnh mẫu Intel
├── notebooks/                         # Notebook Colab và file Python
├── data/                              # Snapshot KOI và các tập ViCTSD
├── reports/                           # Biểu đồ, dữ liệu kết quả và báo cáo cũ
├── CO5177 Report Latex/               # Mã LaTeX và PDF báo cáo tổng hợp
├── tests/                             # Kiểm tra tính toàn vẹn của một số bài
├── assignment-vne-v4.pdf              # Đề bài
└── requirements-eda.txt               # Các thư viện Python chính
```

Thư mục `Report Latex/`, notebook Galaxy10 và `image_intel_eda.ipynb` ở thư mục gốc là các bản lưu khác; trang hiện tại dùng ba notebook trong `notebooks/` và báo cáo tổng hợp ở `CO5177 Report Latex/report.pdf`.

## Nguồn dữ liệu

- [NASA Exoplanet Archive — Kepler Objects of Interest](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)
- [UIT-ViCTSD — Vietnamese Constructive and Toxic Speech Detection](https://github.com/tarudesu/ViCTSD)
- [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Tác giả

[Dinh Hoang Duy Khanh](https://github.com/khanhhoangduydinh) · Nhóm Trailblazer · HCMUT
