# Tự kiểm tra bài tập lớn CO5177

Đối chiếu với `assignment-vne-v4.pdf`, phiên bản 4, mục 3–7. Đây là **đánh giá tham khảo**, không phải điểm do giảng viên chấm. Trọng số dưới đây lấy từ đề; điểm ước lượng trong từng nhóm do người kiểm tra tự đặt vì đề không quy định cách chia chi tiết bên trong mỗi nhóm.

## Điểm tham khảo

| Thành phần trong đề | Tối đa | Ước lượng | Căn cứ và giới hạn |
|---|---:|---:|---|
| Chất lượng notebook | 50 | 42 | Có ba notebook cho ba loại dữ liệu. Tabular, Text và Image đã chạy hết trong môi trường cục bộ sau khi sửa lỗi Image. Chưa xác minh Run all trên Google Colab và chưa đối chiếu các metric của lần chạy kiểm tra mới. |
| Báo cáo, giải thích, Markdown | 20 | 17 | Có báo cáo tổng hợp, các trang EDA và hướng dẫn tiếng Việt cho từng cell code. Cần đối chiếu metric của lần chạy cuối với báo cáo. |
| Video demo và trình bày | 15 | 0 | Hiện chưa có video Tabular, Text hoặc Image. Không thể đánh giá khả năng trình bày khi chưa có video. |
| Sáng tạo, mở rộng và kết luận | 10 | 7 | Có hình so sánh mô hình/đặc trưng, xử lý dữ liệu và ví dụ dự đoán sai; các metric đã lưu có trong báo cáo. Điểm tối đa phụ thuộc việc kiểm tra đầy đủ từng hình, kết luận và lần chạy cuối của từng bài. |
| Hình thức nộp | 5 | 3 | Có GitHub Pages, các trang riêng, notebook và PDF tổng hợp. Đề yêu cầu PDF và video cho từng bài; các trang hiện dùng chung PDF theo yêu cầu của chủ bài, còn video chưa có. Chưa xác minh đăng ký GroupLink/GroupRegistration hoặc nộp LMS. |
| **Tổng ước lượng** | **100** | **69** | **6,9/10**, mang tính thận trọng tại thời điểm kiểm tra; không phải điểm chính thức. |

Nếu các kết quả được đối chiếu và Run all thành công trên Colab, điểm phần notebook có thể được đánh giá cao hơn. Ngược lại, lỗi phát sinh trên Colab hoặc yêu cầu chưa đáp ứng có thể làm điểm thực tế thấp hơn.

## Kiểm tra yêu cầu chính

| Mục | Bằng chứng | Trạng thái |
|---|---|---|
| Quy mô bài cá nhân | Đề yêu cầu Tabular và một loại tự chọn; repository có Tabular, Text và Image. | Đủ số loại nếu đăng ký cá nhân. Nếu đăng ký nhóm, cần xác minh danh sách thành viên và yêu cầu của nhóm. |
| Dữ liệu Tabular | Kepler KOI: 9.564 dòng, 30 cột; có giá trị thiếu, cột số, cột phân loại và mất cân bằng. | Đáp ứng các ngưỡng quy mô ghi trong đề. |
| Dữ liệu Text | ViCTSD: 10.000 bình luận tiếng Việt. | Đáp ứng ngưỡng 2.000 văn bản. |
| Dữ liệu Image | Intel Image Classification: 17.034 ảnh có nhãn, sáu lớp, ảnh nguồn 150 × 150; hai hướng dùng MobileNetV3 pretrained. | Đáp ứng ngưỡng 5.000 ảnh, ba lớp và khuyến nghị độ phân giải. |
| Nội dung phân tích | Có EDA, chia dữ liệu, tiền xử lý, so sánh mô hình, metric và confusion matrix trong các notebook/trang EDA. | Có minh chứng trong repository; cần xác nhận lại bằng Run all trên Colab. |
| Thành viên và môn học | Landing page và từng trang bài con hiển thị Trailblazer, Dinh Hoang Duy Khanh và MSSV 2670306. | Cần đối chiếu tên nhóm và thành viên với sheet đăng ký thực tế. |
| Notebook | `notebooks/tabular_koi_eda.ipynb`, `notebooks/text_victsd_eda.ipynb`, `notebooks/image_intel_eda.ipynb`. | Có hướng dẫn tiếng Việt trước mỗi cell code. Cả ba chạy hết ở máy kiểm tra; không đồng nghĩa đã chạy trên Colab. |
| Báo cáo | `CO5177 Report Latex/report.pdf`; `Trailblazer-report.pdf` có trong repository. | Ba trang hiện dẫn đến báo cáo tổng hợp; xác nhận với giảng viên nếu cần ba PDF tách riêng. Chưa xác minh nộp LMS. |
| Video | Chưa có ba link video YouTube. | Chưa đáp ứng; cần video Public hoặc Unlisted cho từng bài. |

## Các con số đã công bố

- **Tabular:** Random Forest đạt test accuracy 0,758 và macro F1 0,727 trong `reports/tabular_koi_summary.json` và báo cáo tổng hợp.
- **Text:** mô hình toxicity đạt test accuracy 0,842 và macro F1 0,679 trên 985 bình luận sau khi lọc trùng với train, theo `reports/text_victsd_summary.json`.
- **Image:** mô hình Logistic Regression trên đặc trưng MobileNetV3 đạt test accuracy 0,914 và macro F1 0,916 theo `reports/image_intel_summary.json`.

Các số trên là kết quả đã lưu của lần chạy trước. Không thay chúng bằng kết quả kiểm tra mới nếu chưa cập nhật đồng bộ notebook, hình, website và PDF.

## Giới hạn kỹ thuật cần biết

1. **Text:** sau khi loại trùng với train, dữ liệu local được kiểm tra vẫn có một dòng test có bình luận chuẩn hóa trùng validation. Các metric công bố vẫn bao gồm dòng này, nên chưa thể gọi test hoàn toàn độc lập với validation. Nếu sửa bộ lọc, cần chạy lại và cập nhật tất cả bảng, hình, website và báo cáo.
2. **Image:** code đóng băng các tham số `features`, nhưng thống kê chạy của BatchNorm có thể vẫn thay đổi khi `head_model.train()`. Vì vậy diễn giải chính xác là đóng băng *tham số backbone*, không bảo đảm mọi trạng thái backbone bất biến.
3. **Image notebook:** bản notebook cũ gặp lỗi với cấu trúc thư mục Kaggle lồng nhau và có thể đặt classifier mới trên CPU khi dùng GPU. Hai lỗi đã được sửa và notebook chạy hết trong môi trường cục bộ; chưa xác minh trên Colab.
4. **Notebook và Python source:** Tabular và Text có nội dung cell tương ứng. Bản Python Image có kiểm tra file mẫu local và dùng `print`, trong khi notebook kiểm tra thư mục và dùng `display`; cần duy trì các khác biệt này có chủ đích khi đồng bộ.
5. **Tái lập:** phân tích cục bộ không chứng minh Colab luôn chạy được. Kết quả phụ thuộc phiên bản thư viện, Internet, quyền tải dataset, tài nguyên GPU và dữ liệu nguồn.

## Trước khi nộp

1. Mở từng link Colab từ GitHub Pages, chọn **Runtime → Run all**, xác nhận không có lỗi và kiểm tra metric/hình so với report.
2. Quyết định có loại dòng trùng validation/test của bài Text hay không; nếu loại, chạy lại và cập nhật toàn bộ kết quả liên quan.
3. Chuẩn bị ba video, đặt Public hoặc Unlisted, sau đó cập nhật link thật trên từng trang.
4. Kiểm tra tên nhóm, thành viên và landing page trên GroupRegistration/GroupLink; xác nhận yêu cầu PDF riêng cho từng bài với giảng viên.
5. Nộp `Trailblazer-report.pdf` lên LMS theo hướng dẫn của môn và kiểm tra thời hạn trên LMS.
