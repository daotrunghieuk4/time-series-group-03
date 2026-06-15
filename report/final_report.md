# Dự báo lượng mưa từ dữ liệu thời tiết đa biến

## 1. Giới thiệu bài toán

Bài toán của nhóm là dự báo lượng mưa từ dữ liệu thời tiết đa biến theo thời gian. Dữ liệu đầu vào gồm nhiều biến khí tượng như nhiệt độ, độ ẩm, áp suất, gió, bức xạ và các biến thời gian. Đầu ra là một biến mục tiêu duy nhất: lượng mưa (`rain`).

Dạng bài toán:

**Input:** `X ∈ R^(T × d)`

**Output:** `y ∈ R^T`

Trong đó:

* `T`: số bước thời gian.
* `d`: số biến đầu vào.
* `X`: chuỗi thời gian nhiều chiều.
* `y`: chuỗi lượng mưa cần dự báo.

Trong thực nghiệm, nhóm xây dựng bài toán dự báo một bước tiếp theo:

**Input window:** `X[t-L+1:t] ∈ R^(L × d)`

**Output:** `y[t+1] ∈ R`

Với `L = 144`, tương ứng 24 giờ dữ liệu nếu tần suất lấy mẫu là 10 phút.

## 2. Mục tiêu thực nghiệm

Mục tiêu chính của thực nghiệm là xây dựng và so sánh các mô hình dự báo lượng mưa từ dữ liệu thời tiết nhiều biến.

Các mục tiêu cụ thể:

* Kiểm tra và làm sạch dữ liệu thời tiết.
* Tạo các đặc trưng thời gian và đặc trưng chu kỳ.
* Chia dữ liệu thành train, validation và test theo đúng thứ tự thời gian.
* Xây dựng ba mô hình dự báo gồm Linear Regression, XGBoost và GRU.
* Đánh giá mô hình bằng MAE, RMSE và sMAPE.
* Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự báo.

## 3. Mô tả dữ liệu

Bộ dữ liệu gồm các quan sát thời tiết được ghi nhận theo thời gian với tần suất 10 phút. Mỗi dòng dữ liệu tương ứng với một thời điểm quan sát.

Một số biến chính trong dữ liệu:

| Nhóm biến | Ví dụ                       |
| --------- | --------------------------- |
| Thời gian | `date`                      |
| Áp suất   | `p`                         |
| Nhiệt độ  | `T`, `Tpot`, `Tdew`, `Tlog` |
| Độ ẩm     | `rh`, `sh`, `H2OC`          |
| Hơi nước  | `VPmax`, `VPact`, `VPdef`   |
| Gió       | `wv`, `max. wv`, `wd`       |
| Mưa       | `rain`, `raining`           |
| Bức xạ    | `SWDR`, `PAR`, `max. PAR`   |

Biến mục tiêu là:

```text
rain
```

Đây là lượng mưa tại từng thời điểm. Do lượng mưa thường có nhiều giá trị bằng 0, bài toán này khó hơn so với các bài toán dự báo chuỗi liên tục thông thường.

## 4. Tiền xử lý dữ liệu

Dữ liệu được tiền xử lý trước khi đưa vào mô hình. Quy trình gồm các bước sau:

1. Đọc dữ liệu gốc từ `weather.csv`.
2. Chuyển cột thời gian sang kiểu `datetime`.
3. Sắp xếp dữ liệu theo thứ tự thời gian.
4. Loại bỏ các thời điểm bị trùng.
5. Resample dữ liệu về tần suất 10 phút.
6. Nội suy missing values theo thời gian.
7. Xử lý outlier bằng phương pháp IQR.
8. Tạo đặc trưng thời gian và Fourier.
9. Chuẩn hóa dữ liệu bằng `StandardScaler`.
10. Chia dữ liệu thành train, validation và test.

Tỷ lệ chia dữ liệu:

| Tập dữ liệu | Tỷ lệ |
| ----------- | ----: |
| Train       |   70% |
| Validation  |   15% |
| Test        |   15% |

Việc chia dữ liệu được thực hiện theo thứ tự thời gian, không shuffle, để tránh rò rỉ dữ liệu từ tương lai vào quá trình huấn luyện.

## 5. Tạo đặc trưng

Ngoài các biến thời tiết ban đầu, nhóm tạo thêm các đặc trưng thời gian:

* `hour`
* `day_of_week`
* `month`
* `is_weekend`

Các đặc trưng Fourier được thêm vào để mô hình học tính chu kỳ:

* `sin_day`
* `cos_day`
* `sin_week`
* `cos_week`
* `sin_year`
* `cos_year`

Các đặc trưng này giúp mô hình nhận biết quy luật theo ngày, tuần và năm.

## 6. Các mô hình sử dụng

### 6.1. Linear Regression

Linear Regression được dùng làm mô hình baseline. Mô hình này học quan hệ tuyến tính giữa các biến đầu vào và lượng mưa ở thời điểm tiếp theo.

Ưu điểm:

* Đơn giản.
* Huấn luyện nhanh.
* Dễ làm mốc so sánh.

Hạn chế:

* Không học tốt quan hệ phi tuyến.
* Không mô hình hóa tốt phụ thuộc chuỗi dài.

### 6.2. XGBoost

XGBoost là mô hình boosting dựa trên cây quyết định. Mô hình này phù hợp với dữ liệu dạng bảng và có khả năng học các quan hệ phi tuyến.

Ưu điểm:

* Mạnh với dữ liệu dạng bảng.
* Có thể học tương tác giữa các biến.
* Ít nhạy với thang đo dữ liệu hơn mạng nơ-ron.

Hạn chế:

* Không trực tiếp học chuỗi thời gian dài.
* Phụ thuộc nhiều vào đặc trưng đã tạo sẵn.

### 6.3. GRU

GRU là mô hình mạng nơ-ron hồi tiếp dùng cho dữ liệu chuỗi. Mô hình nhận vào cửa sổ 144 bước thời gian trước đó để dự báo lượng mưa ở bước tiếp theo.

Ưu điểm:

* Phù hợp với dữ liệu chuỗi thời gian.
* Có khả năng học phụ thuộc theo thời gian.
* Nhẹ hơn LSTM.

Hạn chế:

* Cần nhiều thời gian huấn luyện hơn.
* Có thể khó dự báo chính xác các đỉnh mưa đột ngột.

## 7. Chỉ số đánh giá

Nhóm sử dụng ba chỉ số chính:

| Chỉ số | Ý nghĩa                                       |
| ------ | --------------------------------------------- |
| MAE    | Sai số tuyệt đối trung bình                   |
| RMSE   | Căn bậc hai của sai số bình phương trung bình |
| sMAPE  | Sai số phần trăm tuyệt đối đối xứng           |

Do lượng mưa có nhiều giá trị bằng 0, nhóm sử dụng sMAPE thay cho MAPE tiêu chuẩn để tránh lỗi chia cho 0.

## 8. Kết quả thực nghiệm

| Mô hình           |      MAE |     RMSE | sMAPE (%) |       R2 |
| ----------------- | -------: | -------: | --------: | -------: |
| Linear Regression | 0.002693 | 0.020186 |  3.426139 | 0.410617 |
| XGBoost           | 0.002665 | 0.022433 |  3.461736 | 0.272082 |
| GRU               | 0.002393 | 0.019489 |  3.129758 | 0.450607 |

Kết quả cho thấy GRU là mô hình tốt nhất theo ba chỉ số chính:

* MAE thấp nhất.
* RMSE thấp nhất.
* sMAPE thấp nhất.

Điều này cho thấy việc sử dụng thông tin chuỗi trong cửa sổ thời gian giúp cải thiện khả năng dự báo lượng mưa.

## 9. Hình ảnh kết quả

### 9.1. So sánh giá trị thực tế và dự báo

![y\_true\_vs\_y\_pred](../figures/y_true_vs_y_pred.png)

Biểu đồ cho thấy giá trị lượng mưa thực tế và dự báo của ba mô hình trên cùng một đoạn dữ liệu test.

### 9.2. So sánh chỉ số mô hình

![model\_metrics\_comparison](../figures/model_metrics_comparison.png)

Biểu đồ so sánh MAE, RMSE và sMAPE giữa ba mô hình.

## 10. Nhận xét

GRU đạt kết quả tốt nhất vì mô hình này sử dụng toàn bộ cửa sổ thời gian thay vì chỉ dùng đặc trưng tại một thời điểm. Với dữ liệu thời tiết, lượng mưa thường phụ thuộc vào trạng thái khí tượng trong nhiều giờ trước đó, nên mô hình chuỗi như GRU có lợi thế hơn.

Linear Regression có kết quả khá ổn dù là mô hình đơn giản. Điều này cho thấy các biến thời tiết và đặc trưng thời gian có chứa thông tin hữu ích cho bài toán dự báo.

XGBoost không đạt kết quả tốt nhất trong thí nghiệm này. Nguyên nhân có thể là mô hình chưa tận dụng đầy đủ quan hệ theo thời gian dài như GRU.

## 11. Hạn chế

Một số hạn chế của thực nghiệm:

* Dữ liệu lượng mưa có nhiều giá trị bằng 0.
* Các thời điểm mưa lớn xuất hiện ít, gây mất cân bằng dữ liệu.
* Mô hình còn khó dự báo chính xác các đỉnh mưa đột ngột.
* Thực nghiệm mới chỉ dự báo T+1.
* Chưa thử các mô hình hiện đại hơn như TimeXer, TimeMixer hoặc iTransformer.

## 12. Kết luận

Nhóm đã xây dựng hoàn chỉnh bài toán dự báo lượng mưa từ chuỗi thời gian thời tiết đa biến. Dữ liệu được tiền xử lý, tạo đặc trưng, chuẩn hóa và chia tập theo đúng thứ tự thời gian.

Ba mô hình được xây dựng và đánh giá gồm:

* Linear Regression
* XGBoost
* GRU

Trong ba mô hình, GRU cho kết quả tốt nhất với:

* MAE = 0.002393
* RMSE = 0.019489
* sMAPE = 3.129758%

Kết quả cho thấy mô hình chuỗi thời gian có khả năng khai thác thông tin lịch sử tốt hơn trong bài toán dự báo lượng mưa.

## 13. Hướng phát triển

Trong tương lai, nhóm có thể mở rộng theo các hướng:

* Dự báo nhiều bước thời gian, ví dụ T+6, T+12 hoặc T+24.
* Thử nghiệm TimeXer để mô hình hóa rõ hơn biến mục tiêu và biến ngoại sinh.
* Thử nghiệm TimeMixer hoặc iTransformer.
* Bổ sung kỹ thuật xử lý mất cân bằng cho các thời điểm có mưa.
* Kết hợp mô hình phân loại mưa/không mưa với mô hình hồi quy lượng mưa.
* Đánh giá riêng trên các thời điểm có mưa thay vì toàn bộ tập test.

