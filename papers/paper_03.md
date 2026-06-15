# MCMamba: A Multi-Scale Correlation-Aware Model with Mamba for Stock Price Forecasting

## 1. Thông tin chung

- **Tác giả:** Siyi Fan, Bo Yang, Weijun Xia và Feng Shen
- **Năm xuất bản:** 2026
- **Từ khóa:** Stock price forecasting, multi-scale temporal modeling, correlation matrix, Mamba architecture
- **Người tóm tắt:** Huy

## 2. Vấn đề nghiên cứu

Các mô hình Transformer hiện tại chủ yếu sử dụng cửa sổ thời gian cố định. Điều này hạn chế khả năng nắm bắt các biến động thời gian ở nhiều tỷ lệ, từ biến động ngắn hạn đến xu hướng dài hạn.

Các cơ chế attention cũng thường tập trung vào tương quan tạm thời của từng mẫu dữ liệu, vì vậy có thể bỏ qua cấu trúc quan hệ toàn cục tương đối ổn định giữa các cổ phiếu trên thị trường.

## 3. Mục đích nghiên cứu

Bài báo đề xuất mô hình **MCMamba** nhằm:

- Mô hình hóa diễn biến thời gian của cổ phiếu ở nhiều tỷ lệ.
- Học quan hệ động giữa các cổ phiếu.
- Xây dựng một ma trận tương quan toàn cục để biểu diễn các quan hệ tương đối ổn định.
- Kết hợp Mamba và attention để nâng cao hiệu quả dự báo giá cổ phiếu.

## 4. Ý tưởng và mô hình đề xuất

MCMamba kết hợp:

- **Mamba:** Xử lý chuỗi dài với độ phức tạp tuyến tính.
- **Multi-scale temporal modeling:** Học thông tin ngắn hạn, trung hạn và dài hạn.
- **Attention:** Học quan hệ động giữa các cổ phiếu.
- **Global correlation matrix:** Lưu giữ cấu trúc quan hệ toàn cục giữa các cổ phiếu.

## 5. Phương pháp nghiên cứu

### 5.1. Quy trình

Mô hình gồm bốn bước chính:

1. **Market-Information-Guided Feature Gating:** Điều chỉnh đặc trưng dưới sự dẫn dắt của thông tin thị trường.
2. **Multi-scale Intra-Stock Modeling:** Mô hình hóa diễn biến nội bộ của từng cổ phiếu ở nhiều tỷ lệ thời gian.
3. **Correlation-Matrix-Guided Inter-Stock Modeling:** Mô hình hóa tương tác giữa các cổ phiếu dựa trên ma trận tương quan.
4. **Aggregation and Prediction:** Tổng hợp đặc trưng và tạo kết quả dự báo.

### 5.2. Thuật toán

MCMamba sử dụng kiến trúc Mamba để xử lý chuỗi thời gian dài với độ phức tạp tuyến tính, kết hợp cơ chế attention và một ma trận tương quan toàn cục được học trực tiếp từ dữ liệu.

Hàm Mean Squared Error được sử dụng làm hàm mất mát:

\[
\mathcal{L} = \operatorname{MSE}(r, \hat{r})
\]

## 6. Dữ liệu thực nghiệm

Bài báo sử dụng hai bộ dữ liệu:

- **CSI 300**
- **CSI 800**

Đây là dữ liệu của thị trường chứng khoán Trung Quốc A-share trong giai đoạn từ ngày 01/01/2008 đến ngày 31/12/2022.

Đặc trưng đầu vào gồm:

- Bộ chỉ báo Alpha158.
- 63 đặc trưng về chỉ số thị trường.

## 7. Chỉ số đánh giá

Đánh giá khả năng xếp hạng:

- Information Coefficient (IC)
- RankIC
- ICIR
- RankICIR

Đánh giá danh mục đầu tư:

- Annualized Excess Return (AR)
- Information Ratio (IR)

## 8. Thiết lập thực nghiệm

- **Nền tảng:** Qlib
- **Phần cứng:** NVIDIA L20
- Thực nghiệm được lặp lại năm lần với các random seed khác nhau.
- Kết quả cuối cùng là giá trị trung bình của năm lần chạy.
- Các tham số được tối ưu trên tập validation.
- Quá trình huấn luyện sử dụng early stopping.

## 9. Kết quả chính

MCMamba đạt kết quả tốt nhất trên tất cả các chỉ số của cả hai bộ dữ liệu.

| Bộ dữ liệu | IC | RankIC |
|---|---:|---:|
| CSI 300 | 0.072 | 0.082 |
| CSI 800 | 0.056 | 0.071 |

Thời gian tính toán:

| Bộ dữ liệu | Huấn luyện | Suy luận |
|---|---:|---:|
| CSI 300 | 145.5 giây/epoch | 9.8 giây/epoch |
| CSI 800 | 198.3 giây/epoch | 24.7 giây/epoch |

> **Lưu ý:** Bản tóm tắt PDF ghi “CSI1800” tại một số vị trí, nhưng phần mô tả dữ liệu ghi CSI 800. Nội dung Markdown sử dụng tên CSI 800 để thống nhất.

## 10. So sánh với các mô hình khác

MCMamba được so sánh với 14 mô hình baseline, bao gồm:

- LSTM
- XGBoost
- Transformer
- MASTER
- SAMBA
- GHOST
- CEEMDAN và các mô hình khác

MCMamba vượt trội trên toàn bộ các chỉ số và sự khác biệt có ý nghĩa thống kê với \(p < 0.01\).

## 11. Điểm mạnh

- Nắm bắt đồng thời quan hệ động và cấu trúc quan hệ toàn cục ổn định giữa các cổ phiếu.
- Xử lý tốt chuỗi thời gian dài nhờ độ phức tạp tuyến tính của Mamba.
- Kết hợp được ưu điểm của Mamba và self-attention.
- Cân bằng tốt giữa hiệu suất dự báo và chi phí tính toán.

## 12. Hạn chế

- Mới chỉ được kiểm chứng trên thị trường chứng khoán Trung Quốc.
- Chiến lược danh mục Top-K mới chỉ được mô phỏng đơn giản.
- Chưa tính đầy đủ các yếu tố thực tế như chi phí giao dịch và tính thanh khoản.
- Chỉ đánh giá một khoảng thời gian dự báo cố định.

## 13. Tính mới

Đây là mô hình tích hợp đồng thời:

- Chiến lược lấy mẫu thời gian đa tỷ lệ vào mạng Mamba.
- Ma trận tương quan toàn cục được học trực tiếp từ dữ liệu.
- Cơ chế attention được hỗ trợ bởi thông tin tương quan toàn cục.

Thiết kế này giúp mô hình học được cả biến động nhiều thang thời gian và quan hệ ổn định giữa các cổ phiếu.

## 14. Khả năng áp dụng

Mô hình có mức độ khả thi cao đối với bài toán dự báo tài chính. Số lượng tham số tăng khoảng 223% so với MASTER, nhưng thời gian huấn luyện chỉ tăng khoảng 6.8% đến 10.3%, cho thấy sự đánh đổi hợp lý để cải thiện độ chính xác.

Tuy nhiên, MCMamba được thiết kế chuyên biệt cho dữ liệu nhiều cổ phiếu và dự báo tài chính. Vì vậy, mô hình không phù hợp bằng TimeXer nếu bài toán của nhóm là dự báo một biến mục tiêu từ bộ dữ liệu thời tiết.

## 15. Hướng mở rộng

- Tích hợp thêm thông tin tần số thời gian vào mô hình hóa đa tỷ lệ.
- Kiểm thử trên các thị trường tài chính khác.
- Thử nghiệm với nhiều khoảng thời gian dự báo.
- Thiết kế kịch bản backtest gần với điều kiện giao dịch thực tế.
- Bổ sung chi phí giao dịch và ràng buộc thanh khoản.

## 16. Ghi chú

Trong thực nghiệm của bài báo, các phương pháp lấy mẫu đơn giản như mean pooling và max pooling đem lại hiệu quả cao và ổn định hơn các kỹ thuật phức tạp như CNN hoặc attention khi mô hình hóa dữ liệu đa tỷ lệ.
