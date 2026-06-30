# Time Series Group 03

## 1. Tên nhóm và thành viên

Tên repository: `time-series-group-03`

Tên nhóm: Nhóm 03

Thành viên:
- Đào Trung Hiếu
- Nguyễn Văn Hiệp
- Nguyễn Minh Huy

## 2. Chủ đề nghiên cứu

Đề tài của nhóm là dự báo chuỗi thời gian dài hạn trên bộ dữ liệu **Electricity** bằng mô hình **TimeMixer**.

Bài toán được xây dựng theo dạng:

```text
Input:  X ∈ R^(L × C)
Output: Y ∈ R^(H × C)
```

Trong đó:
- `L`: độ dài cửa sổ quan sát quá khứ.
- `H`: tầm dự báo.
- `C`: số biến/kênh trong chuỗi thời gian.
- Với thực nghiệm chính: `L = 96`, `C = 321`, `H ∈ {96, 192, 336, 720}`.

## 3. Mô tả bộ dữ liệu

Nhóm sử dụng bộ dữ liệu **Electricity**, gồm mức tiêu thụ điện của 321 khách hàng/đơn vị, được ghi nhận theo tần suất 1 giờ.

Dữ liệu được đặt trong thư mục:

```text
data/
├── raw/
│   └── electricity.csv
└── processed/
    ├── train.csv
    ├── validation.csv
    ├── test.csv
    └── split_summary.csv
```

Dữ liệu đã được chia theo thứ tự thời gian thành ba tập train/validation/test với tỷ lệ xấp xỉ 70%/10%/20%:

| Split | Số dòng | Thời gian bắt đầu | Thời gian kết thúc |
|---|---:|---|---|
| Train | 18412 | 2016-07-01 02:00:00 | 2018-08-07 05:00:00 |
| Validation | 2632 | 2018-08-07 06:00:00 | 2018-11-24 21:00:00 |
| Test | 5260 | 2018-11-24 22:00:00 | 2019-07-02 01:00:00 |

## 4. Ba bài báo đã đọc

Nhóm giữ nguyên ba bài tóm tắt paper từ bản GitHub cũ:

```text
papers/
├── paper_01.md   # CA-DE
├── paper_02.md   # TimeXer
└── paper_03.md   # MCMamba
```

Ba paper này được dùng làm nền tham khảo về dự báo chuỗi thời gian đa biến, biến ngoại sinh, mô hình đa tỷ lệ và quan hệ giữa các biến.

## 5. Phương pháp thực hiện

Quy trình thực hiện gồm bốn bước chính.

### 5.1. Khám phá dữ liệu

Notebook:

```text
notebooks/01_data_exploration.ipynb
```

Nội dung:
- Đọc dữ liệu `electricity.csv`.
- Kiểm tra số dòng, số cột.
- Kiểm tra missing values.
- Kiểm tra cột thời gian.
- Vẽ mẫu hình tiêu thụ điện theo thời gian.

### 5.2. Tiền xử lý và tạo đặc trưng

Notebook:

```text
notebooks/02_feature_engineering.ipynb
```

Các bước chính:
1. Chuyển cột `date` về dạng datetime.
2. Sắp xếp dữ liệu theo thời gian.
3. Kiểm tra missing values.
4. Tạo đặc trưng thời gian như `hour`, `day_of_week`, `month`, `is_weekend`.
5. Tạo Fourier features cho chu kỳ ngày và tuần.
6. Chia train/validation/test theo đúng thứ tự thời gian.
7. Tạo mẫu học bằng cửa sổ trượt với `seq_len = 96`.

### 5.3. Xây dựng mô hình

Notebook:

```text
notebooks/03_models.ipynb
```

Nhóm so sánh ba mô hình:

#### Mô hình 1: Seasonal Naive

Đây là baseline thống kê, dự báo bằng cách lặp lại giá trị ở cùng pha mùa vụ trong quá khứ.

#### Mô hình 2: XGBoost

Đây là baseline học máy truyền thống, sử dụng đặc trưng trễ và đặc trưng thời gian.

#### Mô hình 3: TimeMixer

Đây là mô hình chính của nhóm. TimeMixer sử dụng ý tưởng **decomposable multiscale mixing**, gồm:
- Hạ lấy mẫu đa quy mô.
- Phân rã chuỗi thành trend và seasonal.
- Past-Decomposable-Mixing.
- Future-Multipredictor-Mixing.

Code TimeMixer gốc từ bản ZIP được giữ trong:

```text
src/timemixer_core/
```

Bốn file chính theo cấu trúc bắt buộc được đặt tại:

```text
src/
├── data_loader.py
├── features.py
├── models.py
└── evaluation.py
```

### 5.4. Đánh giá mô hình

Notebook:

```text
notebooks/04_evaluation.ipynb
```

Các chỉ số đánh giá:
- MSE
- MAE
- RMSE

Kết quả được lưu trong:

```text
results/metrics.csv
```

Hình kết quả được lưu trong:

```text
figures/
├── timemixer_full_test_lookback96.png
├── seasonal_naive_full_test_lookback96.png
└── xgboost_full_test_lookback96.png
```

Thư mục `figures/` chỉ giữ ba hình được dùng trực tiếp trong báo cáo để tránh làm rối repository. Các hình metric phụ, alias và bản PDF đã được bỏ khỏi bản nộp.

## 6. Bảng kết quả mô hình

| Model | Lookback | Horizon | MSE | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| TimeMixer | 96 | 96 | 0.1526 | 0.2449 | 0.3906 |
| TimeMixer | 96 | 192 | 0.1656 | 0.2567 | 0.4069 |
| TimeMixer | 96 | 336 | 0.1854 | 0.2749 | 0.4306 |
| TimeMixer | 96 | 720 | 0.2236 | 0.3119 | 0.4729 |
| Seasonal Naive | 96 | 96 | 0.3195 | 0.3295 | 0.5652 |
| Seasonal Naive | 96 | 192 | 0.3028 | 0.3275 | 0.5503 |
| Seasonal Naive | 96 | 336 | 0.3261 | 0.3469 | 0.5711 |
| Seasonal Naive | 96 | 720 | 0.3661 | 0.3783 | 0.6050 |
| XGBoost | 96 | 96 | 0.7905 | 0.6877 | 0.8891 |
| XGBoost | 96 | 192 | 0.9910 | 0.7873 | 0.9955 |
| XGBoost | 96 | 336 | 1.1582 | 0.8619 | 1.0762 |
| XGBoost | 96 | 720 | 1.1618 | 0.8672 | 1.0779 |

Theo kết quả thực nghiệm, TimeMixer đạt sai số thấp nhất trên cả bốn tầm dự báo.

## 7. Hình kết quả

Ba hình dưới đây là bản trực quan chuẩn được đưa vào báo cáo sau khi chạy thí nghiệm.

### 7.1. TimeMixer

![TimeMixer full test](figures/timemixer_full_test_lookback96.png)

### 7.2. Seasonal Naive

![Seasonal Naive full test](figures/seasonal_naive_full_test_lookback96.png)

### 7.3. XGBoost

![XGBoost full test](figures/xgboost_full_test_lookback96.png)

## 8. Cấu trúc repository bắt buộc

```text
time-series-group-03/
├── README.md
├── papers/
│   ├── paper_01.md
│   ├── paper_02.md
│   └── paper_03.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_models.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── figures/
│   ├── timemixer_full_test_lookback96.png
│   ├── seasonal_naive_full_test_lookback96.png
│   └── xgboost_full_test_lookback96.png
├── results/
│   └── metrics.csv
├── report/
│   └── final_report.md
├── requirements.txt
└── .gitignore
```

## 9. Cách chạy code

### 9.1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 9.2. Chuẩn bị dữ liệu

Đặt file dữ liệu vào:

```text
data/raw/electricity.csv
```

Bản ZIP đầy đủ đã có sẵn file này. Nếu dùng bản nhẹ không kèm data, cần copy file `electricity.csv` vào đúng vị trí trên.

### 9.3. Chạy notebook theo thứ tự

```text
notebooks/01_data_exploration.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_models.ipynb
notebooks/04_evaluation.ipynb
```

### 9.4. Chạy TimeMixer từ code gốc

Trong Python:

```python
from src.models import run_timemixer
run_timemixer(pred_len=96)
```

Hoặc chạy trực tiếp code gốc trong `src/timemixer_core` và trỏ dữ liệu về `data/raw/electricity.csv`.

## 10. Kết luận

Nhóm đã thay thế bản GRU/weather cũ bằng bài toán dự báo phụ tải điện sử dụng TimeMixer. Kết quả cho thấy TimeMixer vượt Seasonal Naive và XGBoost trên cả bốn horizon, đặc biệt nhờ khả năng học biểu diễn đa quy mô và phân tách thông tin mùa vụ--xu hướng.
