# CA-DE: Heterogeneous Component-Aware and Antagonistic Dependency-Enhanced Dual Paths for Long-Term Time Series Forecasting

## 1. Thông tin chung

- **Tác giả:** Xiaoyue Liu, Weimin Li, Fangfang Liu và Quanke Pan
- **Năm xuất bản:** 2026
- **Từ khóa:** Long-term time series forecasting, multivariate time series, feature decoupling, negative dependency modeling
- **Người tóm tắt:** Hiếu

## 2. Vấn đề nghiên cứu

Các mô hình dự báo chuỗi thời gian đa biến hiện nay thường xử lý đặc trưng thời gian và quan hệ giữa các biến trong cùng một không gian đặc trưng. Điều này gây ra sự liên kết chặt giữa các loại đặc trưng và làm giảm khả năng biểu diễn của mô hình.

Bên cạnh đó, nhiều mô hình sử dụng cùng một kiến trúc cho cả thành phần xu hướng và mùa vụ, mặc dù hai thành phần này có đặc điểm khác nhau. Các quan hệ tương quan âm giữa các biến cũng chưa được khai thác rõ ràng.

## 3. Mục đích nghiên cứu

Bài báo đề xuất mô hình **CA-DE** nhằm:

- Tách biệt việc học đặc trưng thời gian và quan hệ giữa các biến thành hai nhánh song song.
- Học riêng thành phần xu hướng và thành phần mùa vụ.
- Khai thác cả quan hệ tương quan thuận và tương quan nghịch giữa các biến.
- Kết hợp kết quả của hai nhánh tại tầng dự báo cuối cùng.

## 4. Ý tưởng và mô hình đề xuất

CA-DE gồm hai nhánh chính:

1. **Component-Aware Path (CA):** Phân rã chuỗi thời gian thành xu hướng và mùa vụ, sau đó sử dụng kiến trúc phù hợp để học từng thành phần.
2. **Dependency-Enhanced Path (DE):** Tạo thêm chuỗi đảo để biểu diễn tương quan âm và dùng attention để học quan hệ giữa các biến.

Dự báo từ hai nhánh được kết hợp bằng một cơ chế hợp nhất thích nghi.

## 5. Phương pháp nghiên cứu

### 5.1. Quy trình

1. Dữ liệu đầu vào được đưa đồng thời vào nhánh CA và nhánh DE.
2. Nhánh CA phân rã chuỗi thành xu hướng và mùa vụ rồi học riêng từng thành phần.
3. Nhánh DE tạo chuỗi đảo để biểu diễn tương quan âm.
4. Cross-attention được sử dụng để học các quan hệ thuận và nghịch giữa các biến.
5. Kết quả của hai nhánh được kết hợp bằng một cổng thích nghi để tạo dự báo cuối cùng.

### 5.2. Các thành phần chính

- **HTA:** Học xu hướng ở nhiều tỷ lệ.
- **PSC:** Học các chu kỳ nổi bật bằng Fourier và convolution.
- **ADG:** Tạo chuỗi đảo theo công thức:

  \[
  \tilde{x} = \max(x) + \min(x) - x
  \]

- **BDF:** Sử dụng cross-attention để học quan hệ thuận và nghịch giữa các biến.
- **Adaptive Fusion:** Tự động xác định trọng số của hai nhánh và hợp nhất dự báo.

## 6. Dữ liệu thực nghiệm

Bài báo sử dụng tám bộ dữ liệu chuỗi thời gian đa biến:

- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Weather
- Exchange
- Electricity
- Traffic

Số lượng biến trong các bộ dữ liệu dao động từ 7 đến 862, thuộc các lĩnh vực năng lượng, tài chính, khí tượng và giao thông.

## 7. Thiết lập thực nghiệm

- **Độ dài đầu vào:** 96
- **Các khoảng dự báo:** 96, 192, 336 và 720
- **Trình tối ưu:** Adam
- **Learning rate:** 0.002
- **Batch size:** 16; riêng Traffic sử dụng batch size 8
- **Số epoch tối đa:** 10
- **Early stopping patience:** 3
- **Phần cứng:** NVIDIA RTX 4090 24 GB
- **Chỉ số đánh giá:** MSE và MAE

## 8. Kết quả chính

CA-DE đạt hạng nhất hoặc hạng nhì trong **34/40 cấu hình đánh giá**. Mô hình thể hiện tốt nhất trên Exchange và có kết quả cạnh tranh trên các bộ ETT, Weather và Electricity.

Kết quả MSE/MAE trung bình:

| Bộ dữ liệu | MSE | MAE |
|---|---:|---:|
| ETTh1 | 0.429 | 0.436 |
| ETTh2 | 0.363 | 0.395 |
| ETTm1 | 0.382 | 0.395 |
| ETTm2 | 0.273 | 0.320 |
| Weather | 0.240 | 0.273 |
| Electricity | 0.174 | 0.267 |
| Exchange | 0.259 | 0.345 |
| Traffic | 0.490 | 0.290 |

Trên bộ Exchange, MSE và MAE trung bình lần lượt giảm khoảng **26.9%** và **13.6%** so với mô hình đứng thứ hai.

Trên ETTh2 với khoảng dự báo 96, CA-DE cần khoảng:

- 0.039 giây cho mỗi iteration.
- 0.010 GFLOPs.
- 0.070 triệu tham số.

Mô hình nhanh hơn TimeMixer và nhẹ hơn đáng kể so với PatchTST, TimesNet và FEDformer.

## 9. So sánh với các mô hình khác

CA-DE được so sánh với RLMamba, MSPatch, TimeMixer, Vcformer, iTransformer, PatchTST, DLinear, TimesNet và FEDformer.

Mô hình cũng tốt hơn TimeCMA và Time-LLM trong toàn bộ 20 cấu hình so sánh với các phương pháp dựa trên mô hình ngôn ngữ lớn. Tuy nhiên, trên bộ Traffic, CA-DE có MSE kém hơn RLMamba và iTransformer.

## 10. Điểm mạnh

- Kiến trúc nhẹ và chi phí tính toán thấp.
- Tách rõ việc học đặc trưng thời gian và quan hệ giữa các biến.
- Sử dụng cấu trúc khác nhau cho xu hướng và mùa vụ.
- Chủ động khai thác cả tương quan thuận và tương quan nghịch.
- Có khả năng áp dụng cho nhiều lĩnh vực chuỗi thời gian đa biến.

## 11. Hạn chế

- Hiệu quả giảm khi dữ liệu có quá nhiều biến và nhiều quan hệ dư thừa, điển hình là Traffic với 862 biến.
- Chưa xử lý tốt các sự kiện đột ngột và chuỗi có tính không dừng mạnh.
- Bài báo chưa cung cấp đầy đủ mã nguồn và một số chi tiết triển khai.
- Ký hiệu trọng số cổng trong công thức và hình minh họa chưa hoàn toàn nhất quán.

## 12. Tính mới

- Tách hoàn toàn đặc trưng thời gian và quan hệ giữa các biến thành hai nhánh song song, chỉ kết hợp tại tầng dự báo.
- Sử dụng kiến trúc riêng cho thành phần xu hướng và mùa vụ.
- Tạo chuỗi đảo để biến tương quan âm thành quan hệ dễ nhận biết hơn đối với attention.

## 13. Khả năng áp dụng cho bài toán của nhóm

Mô hình có thể áp dụng cho dữ liệu năng lượng, tài chính, khí tượng, giao thông và dữ liệu cảm biến đa biến. Với khoảng 0.070 triệu tham số và thời gian huấn luyện thấp, mô hình phù hợp với điều kiện tính toán hạn chế.

Đối với bộ dữ liệu có nhiều biến, nhóm nên bổ sung bước lựa chọn biến hoặc giảm chiều dữ liệu để hạn chế các quan hệ dư thừa.

## 14. Hướng mở rộng

- Bổ sung cơ chế lựa chọn biến.
- Kết hợp mô hình phát hiện bất thường.
- Cải thiện khả năng xử lý dữ liệu không dừng.
- Nghiên cứu quan hệ có độ trễ giữa các biến.
- Kiểm chứng có kiểm soát hiệu quả của cơ chế mô hình hóa tương quan âm.
- Khai thác tốt hơn các đặc trưng trong miền tần số.

## 15. Ghi chú

Tác giả lưu ý rằng CA-DE còn hạn chế trên dữ liệu có số lượng biến rất lớn và nhiều quan hệ dư thừa. Khi áp dụng thực tế, cần điều chỉnh số tầng downsampling, số chu kỳ trội và kích thước kernel tích chập phù hợp với từng bộ dữ liệu.
