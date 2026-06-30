# Patch notes - TimeMixer aligned version

Bản này giữ nguyên core TimeMixer theo repo gốc, chỉ sửa các file phụ để baseline không bị lệch so với protocol của `Dataset_Custom`.

## File đã sửa

### `run_xgboost.py`
- Bỏ pipeline tự thêm IQR clipping/outlier processing để dữ liệu gần với repo/paper hơn.
- Chuẩn hóa giống `Dataset_Custom`: `StandardScaler` fit trên train 70% đầu, transform toàn bộ.
- Sửa lỗi lệch vùng test: test input bắt đầu tại `len(data) - num_test - seq_len`, target đầu tiên bắt đầu tại `len(data) - num_test`.
- Mặc định train XGBoost chỉ trên train set, giống TimeMixer dùng validation để chọn model. Có thể thêm `--train_with_val` nếu muốn baseline dùng train+val.
- Thêm `--use_gpu_xgb` cho môi trường XGBoost có CUDA.
- Thêm `--random_seed` để sampling ổn định hơn.
- Lưu kết quả ra `./results/xgboost_baseline_results.csv`.

### `run_seasonal_naive.py`
- Bỏ IQR clipping/outlier processing.
- Chuẩn hóa và chia dữ liệu giống `Dataset_Custom`.
- Sửa/giữ indexing test sao cho trùng với TimeMixer: có 96 điểm lookback ngay trước test target range.
- Thêm `--test_step` và lưu kết quả ra `./results/seasonal_naive_results.csv`.

### `data_provider/data_loader.py`
- Sửa tương thích pandas mới:
  `df_stamp.drop(['date'], 1)` -> `df_stamp.drop(columns=['date'])`.
- Không thay đổi logic chia dữ liệu hoặc logic mô hình.

## Lệnh chạy TimeMixer chính

```bash
bash scripts/long_term_forecast/ECL_script/TimeMixer_unify.sh
```

## Lệnh chạy baseline gợi ý

Chạy Seasonal Naive full test:

```bash
python -u run_seasonal_naive.py --pred_lens 96 192 336 720 --test_step 1
```

Chạy XGBoost nhanh để kiểm tra:

```bash
python -u run_xgboost.py \
  --pred_lens 96 \
  --test_step 10 \
  --n_estimators 30 \
  --max_depth 4 \
  --n_channels_train 20 \
  --n_windows_train 300 \
  --n_horizons_train 15
```

Chạy XGBoost toàn bộ test cho horizon 96:

```bash
python -u run_xgboost.py \
  --pred_lens 96 \
  --test_step 1 \
  --n_estimators 30 \
  --max_depth 4 \
  --n_channels_train 20 \
  --n_windows_train 300 \
  --n_horizons_train 15
```

Bật GPU cho XGBoost nếu môi trường hỗ trợ CUDA:

```bash
python -u run_xgboost.py \
  --pred_lens 96 \
  --test_step 1 \
  --n_estimators 30 \
  --max_depth 4 \
  --n_channels_train 20 \
  --n_windows_train 300 \
  --n_horizons_train 15 \
  --use_gpu_xgb
```

## Ghi chú báo cáo

TimeMixer được giữ theo cấu hình repo/paper. XGBoost và Seasonal Naive là baseline bổ sung để tham khảo, không phải baseline chính trong paper TimeMixer.
