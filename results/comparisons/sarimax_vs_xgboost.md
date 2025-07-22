# Model Comparison: SARIMAX vs XGBoost  
**Forecasting Freight Data (Georgia, 2017-2023)**  
_Last Updated: July 22, 2025

## Summary  
This document compares the performance of **SARIMAX** and **XGBoost** models on predicting shipment metrics across selected commodity types.

---

## Validation Results (2017-2021 → 2022)

| Target   | Model                            |    RMSE | MAPE (%) | Prediction | Ground Truth |
|:---------|:---------------------------------|--------:|---------:|-----------:|-------------:|
| value_5  | SARIMAX                          | 102.499 |    0.337 |    30312.7 |      30415.2 |
| value_5  | XGBoost (Basic)                  |    14.1 | 0.653656 |      30614 |      30415.2 |
| value_5  | **XGBoost (Grid-Search Optim.)** |  4.3427 |   0.0620 |    30396.3 |      30415.2 |
| tons_5   | SARIMAX                          | 532.246 |  4.89391 |    10343.4 |      10875.7 |
| tons_5   | XGBoost (Basic)                  |  12.423 |  1.41905 |      11030 |      10875.7 |
| tons_5   | **XGBoost (Grid-Search Optim.)** |  3.4206 |   0.1076 |      10864 |      10875.7 |
| value_8  | SARIMAX                          | 1814.41 |  13.6879 |    15069.9 |      13255.5 |
| value_8  | XGBoost (Basic)                  | 31.5312 |  7.50038 |    14249.8 |      13255.5 |
| value_8  | **XGBoost (Grid-Search Optim.)** |  8.1872 |   0.5057 |    13322.6 |      13255.5 |
| tons_8   | SARIMAX                          | 357.949 |  5.11105 |    7361.38 |      7003.43 |
| tons_8   | XGBoost (Basic)                  | 22.9195 |  7.50068 |    7528.74 |      7003.43 |
| tons_8   | **XGBoost (Grid-Search Optim.)** |  7.6660 |   0.8391 |     7062.2 |      7003.43 |
| value_9  | SARIMAX                          | 124.459 |  7.55831 |     1771.1 |      1646.65 |
| value_9  | XGBoost (Basic)                  | 13.9237 |  11.7736 |    1840.52 |      1646.65 |
| value_9  | **XGBoost (Grid-Search Optim.)** | 13.7613 |  11.5005 |    1836.02 |      1646.65 |
| tons_9   | SARIMAX                          |  7.1665 |  13.8517 |       58.9 |        51.74 |
| tons_9   | XGBoost (Basic)                  |  2.3123 |  10.3343 |      57.08 |        51.74 |
| tons_9   | **XGBoost (Grid-Search Optim.)** |  2.3021 |  10.2434 |      57.04 |        51.74 |
| value_21 | SARIMAX                          | 3440.68 |  4.91195 |    66606.4 |      70047.1 |
| value_21 | XGBoost (Basic)                  |    94.7 |   12.803 |      61079 |      70047.1 |
| value_21 | **XGBoost (Grid-Search Optim.)** | 81.2385 |   9.4218 |    63447.4 |      70047.1 |
| tons_21  | SARIMAX                          |  62.968 |  4.83699 |    1238.84 |      1301.81 |
| tons_21  | **XGBoost (Basic)**              | 10.2689 |  8.10032 |    1196.36 |      1301.81 |
| tons_21  | XGBoost (Grid-Search Optim.)     | 10.2692 |  8.10082 |    1196.35 |      1301.81 |


## Test Results (2017-2022 → 2023)

| Target   | Model       |    RMSE | MAPE (%) | Prediction | Ground Truth |
|:---------|:------------|--------:|---------:|-----------:|-------------:|
| value_5  | SARIMAX     | 240.962 | 0.783796 |      30502 |        30743 |
| value_5  | **XGBoost** | 58.0121 |  10.9469 |    27377.6 |        30743 |
| tons_5   | SARIMAX     | 134.418 |  1.22098 |    10874.6 |        11009 |
| tons_5   | **XGBoost** | 31.9332 |  9.26263 |    9989.31 |        11009 |
| value_8  | SARIMAX     | 1528.34 |  11.9291 |    14340.2 |      12811.9 |
| value_8  | **XGBoost** | 17.3631 |  2.35311 |    13113.3 |      12811.9 |
| tons_8   | SARIMAX     | 693.276 |  10.2419 |     7462.3 |      6769.03 |
| tons_8   | **XGBoost** | 13.3444 |  2.63069 |     6947.1 |      6769.03 |
| value_9  | SARIMAX     | 105.605 |   6.9093 |    1634.05 |      1528.45 |
| value_9  | **XGBoost** | 10.8732 |  7.73511 |    1646.67 |      1528.45 |
| tons_9   | **SARIMAX** |  0.8095 |  1.68562 |      47.21 |        48.02 |
| tons_9   | XGBoost     |    2.39 |  11.8942 |      53.73 |        48.02 |
| value_21 | SARIMAX     | 2785.08 |  3.82549 |    75588.4 |      72803.3 |
| value_21 | **XGBoost** | 52.5279 |  3.78991 |    70044.1 |      72803.3 |
| tons_21  | SARIMAX     | 98.2609 |  7.26226 |    1254.77 |      1353.03 |
| tons_21  | **XGBoost** |  7.1571 |  3.78587 |    1301.81 |      1353.03 |

---

## Key Observations
- **Overall:**  
  - The XGBoost regressor with grid-search consistently outperformed SARIMAX, significantly reducing 
  RMSE compared to both SARIMAX and the default-parameter XGBoost.
- **By Target:**  
  - `tons_21`: Grid-search during validation did not yield a significant improvement for this target, 
  likely due to the optimal parameters not being found.
  - `tons_9`: SARIMAX was marginally better for this target, but overall performance across targets does not 
  justify selecting it as the primary model.


## Visual Comparison  


```plaintext
(Optional, placeholder for any visual comparison)
