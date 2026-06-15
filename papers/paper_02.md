# TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

## 1. Thông tin chung

- **Tác giả:** Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang và Mingsheng Long
- **Năm xuất bản:** 2024
- **Từ khóa:** Time series forecasting, exogenous variables, endogenous variables, Transformer, self-attention, cross-attention, multivariate time series
- **Người tóm tắt:** Hiệp

## 2. Vấn đề nghiên cứu

Các mô hình dự báo đơn biến thường chỉ sử dụng lịch sử của biến mục tiêu nên bỏ qua thông tin từ các yếu tố bên ngoài.

Trong khi đó, các mô hình dự báo đa biến thường xem tất cả các biến có vai trò như nhau và dự báo đồng thời toàn bộ biến. Cách tiếp cận này chưa phân biệt rõ:

- **Biến nội sinh:** Biến mục tiêu cần dự báo.
- **Biến ngoại sinh:** Các biến hỗ trợ cung cấp thông tin cho việc dự báo.

Ngoài ra, dữ liệu ngoại sinh có thể bị thiếu, lệch thời gian, khác tần suất lấy mẫu hoặc có độ dài lịch sử khác nhau.

## 3. Mục đích nghiên cứu

Bài báo xây dựng mô hình **TimeXer** để:

- Sử dụng nhiều biến ngoại sinh nhằm nâng cao độ chính xác khi dự báo một hoặc nhiều biến nội sinh.
- Học đồng thời phụ thuộc thời gian bên trong biến mục tiêu và quan hệ giữa biến mục tiêu với các biến ngoại sinh.
- Xử lý dữ liệu ngoại sinh không đồng nhất về độ dài, tần suất hoặc thời điểm quan sát.

## 4. Ý tưởng và mô hình đề xuất

TimeXer biểu diễn dữ liệu bằng ba loại token:

- **Patch token:** Biểu diễn các đoạn lịch sử của biến nội sinh.
- **Global token:** Tổng hợp thông tin toàn cục của biến nội sinh.
- **Variate token:** Biểu diễn từng biến ngoại sinh.

Self-attention học phụ thuộc thời gian giữa các patch của biến nội sinh. Cross-attention sử dụng global token làm truy vấn để chọn lọc thông tin hữu ích từ các biến ngoại sinh.

## 5. Phương pháp nghiên cứu

### 5.1. Quy trình

1. Chia chuỗi lịch sử của biến nội sinh thành các đoạn không chồng lấn.
2. Biểu diễn mỗi đoạn thành một patch token.
3. Thêm một global token để tổng hợp thông tin toàn cục của biến nội sinh.
4. Biểu diễn mỗi biến ngoại sinh bằng một variate token.
5. Sử dụng self-attention để học quan hệ theo thời gian của biến nội sinh.
6. Sử dụng cross-attention để global token truy vấn và chọn lọc thông tin từ các biến ngoại sinh.
7. Đưa biểu diễn cuối cùng qua lớp chiếu tuyến tính để tạo dự báo.

### 5.2. Thuật toán

TimeXer sử dụng một Transformer encoder với:

- Patch-level endogenous embedding.
- Global endogenous token.
- Variate-level exogenous embedding.
- Self-attention cho phụ thuộc thời gian của biến mục tiêu.
- Cross-attention cho tác động từ biến ngoại sinh đến biến nội sinh.
- Hàm mất mát L2 để tối ưu sai lệch giữa dự báo và giá trị thực.

## 6. Dữ liệu thực nghiệm

Bài báo thực nghiệm trên 12 bộ dữ liệu thuộc các lĩnh vực điện năng, khí tượng và giao thông:

- Năm bộ dữ liệu EPF
- ECL
- Weather
- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Traffic

Bộ Weather gồm 21 chỉ số khí tượng, được đo 10 phút một lần tại trạm khí tượng của Viện Max Planck trong năm 2020. Một chỉ số được chọn làm biến nội sinh và 20 chỉ số còn lại được sử dụng làm biến ngoại sinh.

Phân chia dữ liệu Weather:

- **Tập huấn luyện:** 36,792 mẫu
- **Tập validation:** 5,271 mẫu
- **Tập kiểm tra:** 10,540 mẫu

## 7. Thiết lập thực nghiệm

- **Framework:** PyTorch
- **Phần cứng:** NVIDIA RTX 4090 24 GB
- **Trình tối ưu:** Adam
- **Learning rate ban đầu:** \(10^{-4}\)
- **Số epoch tối đa:** 10
- **Early stopping:** Có
- **Số TimeXer block:** \(\{1, 2, 3\}\)
- **Kích thước biểu diễn:** \(\{128, 256, 512\}\)
- **Độ dài đầu vào:** 96
- **Các khoảng dự báo:** 96, 192, 336 và 720
- **Patch length:** 16
- **Chỉ số đánh giá:** MSE và MAE

## 8. Kết quả chính

Bài báo không sử dụng chỉ số Accuracy. Trong 28 thiết lập dự báo dài hạn có biến ngoại sinh, TimeXer đạt:

- MSE tốt nhất ở 23 thiết lập.
- MAE tốt nhất ở 23 thiết lập.

Một số kết quả tiêu biểu:

| Bộ dữ liệu/thí nghiệm | MSE | MAE |
|---|---:|---:|
| Trung bình năm bộ EPF | 0.307 | 0.265 |
| Weather đa biến | 0.241 | 0.271 |
| Weather một biến mục tiêu | 0.002 | 0.031 |
| Khí tượng quy mô lớn, 3,850 trạm | 0.200 | Không báo cáo |

Trên bộ ECL với 320 biến ngoại sinh, TimeXer cần khoảng:

- 33 ms cho mỗi vòng lặp huấn luyện.
- 0.95 GB bộ nhớ.

Mô hình sử dụng ít bộ nhớ hơn iTransformer nhưng chậm hơn một số mô hình tuyến tính. Độ phức tạp theo số biến ngoại sinh là \(O(C)\), thấp hơn mức \(O(C^2)\) của iTransformer.

## 9. So sánh với các mô hình khác

Trên năm bộ EPF:

| Mô hình | MSE | MAE |
|---|---:|---:|
| TimeXer | 0.307 | 0.265 |
| iTransformer | 0.335 | 0.289 |
| PatchTST | 0.330 | 0.282 |
| TiDE | 0.412 | 0.338 |

Trên bộ khí tượng quy mô lớn, MSE của TimeXer là 0.200, tốt hơn iTransformer 0.207, PatchTST 0.208, DLinear 0.212 và RLinear 0.216.

## 10. Điểm mạnh

- Phù hợp trực tiếp với bài toán nhiều biến đầu vào và một biến mục tiêu.
- Phân biệt rõ vai trò của biến nội sinh và biến ngoại sinh.
- Học đồng thời quan hệ theo thời gian và quan hệ giữa các biến.
- Có khả năng xử lý dữ liệu thiếu, lệch thời gian, khác tần suất và khác độ dài.
- Cross-attention giúp kết quả dễ giải thích hơn và tiết kiệm bộ nhớ khi có nhiều biến ngoại sinh.

## 11. Hạn chế

- Self-attention giữa các patch vẫn có độ phức tạp bậc hai theo số patch.
- Hiệu quả dự báo phụ thuộc nhiều vào chất lượng lịch sử của biến mục tiêu.
- Có thể dự báo đúng xu hướng nhưng chưa chính xác với các đỉnh tăng hoặc giảm đột ngột.
- Chủ yếu mô hình hóa ảnh hưởng từ biến ngoại sinh đến biến mục tiêu, chưa khai thác đầy đủ quan hệ giữa các biến ngoại sinh.

## 12. Tính mới

- Kết hợp patch token và global token cho biến nội sinh với variate token cho biến ngoại sinh.
- Global token đóng vai trò cầu nối giữa self-attention theo thời gian và cross-attention giữa các biến.
- Cho phép sử dụng Transformer nguyên bản cho bài toán dự báo có biến ngoại sinh mà không cần thay đổi các thành phần nền tảng của Transformer.

## 13. Khả năng áp dụng cho bài toán của nhóm

TimeXer có tính khả thi cao đối với bài toán của nhóm vì đúng với yêu cầu:

\[
X \in \mathbb{R}^{T \times d} \rightarrow y \in \mathbb{R}^{T}
\]

Nhóm có thể chọn một đại lượng thời tiết, chẳng hạn lượng mưa hoặc nhiệt độ, làm biến mục tiêu. Các chỉ số khí tượng còn lại cùng với đặc trưng thời gian được sử dụng làm biến ngoại sinh.

Mã nguồn chính thức đã được công bố, nhưng nhóm cần điều chỉnh bộ nạp dữ liệu và cấu hình đầu ra cho phù hợp với bộ dữ liệu Weather đang sử dụng.

## 14. Hướng mở rộng

- Bổ sung `hour`, `day_of_week`, `month`, `is_weekend` và đặc trưng Fourier làm biến ngoại sinh.
- Thay full attention bằng linear attention để giảm chi phí tính toán.
- Bổ sung cơ chế học quan hệ giữa các biến ngoại sinh.
- Mở rộng sang dự báo xác suất.
- Dự báo đồng thời nhiều khoảng thời gian.

## 15. Ghi chú

Các biến ngoại sinh chỉ cung cấp thông tin hỗ trợ và không nhất thiết phải được dự báo. Khi thay dữ liệu ngoại sinh bằng giá trị 0 hoặc nhiễu ngẫu nhiên, hiệu quả chỉ giảm nhẹ. Ngược lại, khi lịch sử biến nội sinh bị mất, kết quả suy giảm nghiêm trọng.

TimeXer cũng gặp khó khăn khi dự báo các giá trị đột biến trên bộ Traffic.
