# Processed data

Thư mục này chứa các file dữ liệu sau khi chia theo thứ tự thời gian, không shuffle.

Tỷ lệ chia dùng theo protocol của TimeMixer/Dataset_Custom:

| Split | Số dòng | Tỷ lệ | Thời gian bắt đầu | Thời gian kết thúc |
|---|---:|---:|---|---|
| Train | 18412 | 70% | 2016-07-01 02:00:00 | 2018-08-07 05:00:00 |
| Validation | 2632 | 10% | 2018-08-07 06:00:00 | 2018-11-24 21:00:00 |
| Test | 5260 | 20% | 2018-11-24 22:00:00 | 2019-07-02 01:00:00 |

Các file chính:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
data/processed/split_summary.csv
```

Lưu ý: phần chuẩn hóa dữ liệu trong TimeMixer được thực hiện bằng `StandardScaler` fit trên tập train, sau đó transform validation/test để tránh rò rỉ dữ liệu từ tương lai.
