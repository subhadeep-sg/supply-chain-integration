## Day 1 (June 8, 2025)
- Downloaded FAF5.6.1 State Database (2018-2023) from [Bureau of Transportation Statistics](https://www.bts.gov/faf)
- Reading and exploring metadata using 'bts_freight_eda.ipynb'.
- Selected data based on state (Georgia) and trade type (Domestic).
- Attempting to select data based on specific commodity types.
- **Blockers:** Need a companion dataset for some sort of yearly analysis.
- **Tomorrow:** Explore using commodity types and import Customer related data from Georgia

## Day 2 (June 9, 2025)
- Selected 4 customer oriented commodities.
- Using [Federal Reserve Economic Data](https://fred.stlouisfed.org/) for income data and [US Census Bureau](https://data.census.gov/) for population data.
- For household/median income [FRED, Real Median Household Income by State, Annual](https://fred.stlouisfed.org/series/MEHOINUSGAA672N)
- Including retail sales [FRED, Monthly State Retail Sales](https://fred.stlouisfed.org/release?rid=477&soid=19&t=ga&ob=pv&od=desc)
- US Census, State Annual Estimate of Population [2010-2019](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html) and [2020-2024](https://www.census.gov/data/datasets/time-series/demo/popest/2020s-state-total.html).
- Joined the three tables for Georgia between years.
- Restructure shipment data to inbound and outbound domestic for each commodity and select only inbound for analysis of customer-oriented commodities.
- Restructuring the metrics (ton, value) and the year in a way that allows join with income/population data.
- Fixed datetime errors before joining the tables.
- **Blockers:** 
- **Tomorrow:** Recheck all units and then regression analysis.

## Day 3 (June 10, 2025)
- Modifying median income unit from dollars to millions of dollars.
- Removing median income not adjusted for inflation.
- Obtained correlation matrix for commodity types to the 3 features.
- Missed dataset for year 2017 and projections till 2050, adding that to extend the dataset.
- Not all years available, usable only years 2007, 2012, 2017.
- **Tomorrow:** 
- Decision support system: Which commodities will grow fastest based on the given 3 features?

## Day 4 (June 11, 2025)
- Making a separate script to clean data and save into csv for easy reading.
- Added State population estimate for [2000-2010](https://www.census.gov/data/datasets/time-series/demo/popest/intercensal-2000-2010-state.html)
- Found FAF data between [2013-2018](https://www.bts.gov/faf/faf4)
**Tomorrow** Continue cleaning up data and visualize time-series data.

## Day 5 (June 13, 2025)
- Continuing to integrate more data from 2013-2018, the size is slowing things down a bit. Looking to filter columns in each dataframe before concatenating.
**Blockers:** Index issues due to differently named columns.

## Day 6 (June 14, 2025)
- While combining FAF4 and 5, I have to account for the inflation-adjusted $ value to be different. This would require me to convert one of them to the other.
- Require [GDP Implicit Price Deflator](https://fred.stlouisfed.org/series/A191RD3A086NBEA#) data for 2017 base data. Will be using 2017 base dollar value for the overall dataset (future years are estimations from 2017).
- To convert constant dollar values from one base year to another, you can use the following formula:

    ![Equation](../figures/deflation_conversion.svg)

- Obtained a cleaned csv with adjusted value for 2017 base. There is however a big gap between estimated 2016 value and base 2017 value.
**Tomorrow:** Visualizations and analysis on the cleaned data.

## Day 7 (June 17, 2025)
- Added visualizations of value and commodity groups across 2012-2023.
- Adding lagged value as a feature: value_commodity_t-1 and tons_commodity_t-1.
- Will set aside 2023 row for prediction and delete 2012 due to lack of lagged feature.
- Separate data processing after the cleaned data on "forecast_prep.py" saved in "forecast_prep.csv".
- Due to lack of 2024, 2025 data for some of the features (income and retail trade GDP), 
plan is to do a rolling validation.
- Train the model on 2012-2018 predict 2019, 2012-2019 predict 2020 etc.
- Set up training data for training first model (expected value of commodity group 5, meat/seafood).
- **Tomorrow:** Figure out how to use ARIMA.

## Day 8 (June 18, 2025)
- Understanding ARIMA model order (p, d, q) and fixing errors while model fitting and testing.
- Predictions are way off, high error rate.

## Day 9 (June 19, 2025)
- Examining the trend for value_5 there is a gap between base 2012 and base 2017.
- Considering using a scaling factor to join FAF4 and FAF5 data.
- May have misunderstood the reprocessed data, looking to see documentation for FAF4 and 5.

## Day 10 (June 20, 2025)
- The gap between the FAF4 and FAF5 estimates for the same year.

    <img src="../figures/faf4_faf5_comparisons.png" width="600">

- Prepared functions for generating trends of value by commodity and adding more years for better visual clarity.
- Writing these functions as utils into ``visualization.py``.
- Moving data cleaning files into utils.
- Changed the folder structure to make it cleaner and more readable. Added the project structure to the README.

### Visualizations
| Commodity Code | Shipment Value by Commodity                                      |
|----------------|------------------------------------------------------------------|
| 5              | <img src="../figures/faf4_5_comparison_value_5.png" width="400"> |
| 8              | <img src="../figures/faf4_5_comparison_value_8.png" width="400"> |
| 9              | <img src="../figures/faf4_5_comparison_value_9.png" width="400">    |
| 21             | <img src="../figures/faf4_5_comparison_value_21.png" width="400">   |

## Day 11 (June 21, 2025)
- I wanted to go back and check whether I should have applied the GDP Implicit Price Deflator.
- Generated a new csv minus the inflation.
- From eye-test it doesn't seem to make a big difference, there is still a clear gap between 2016-2017.

| Commodity Code | Shipment Tons by Commodity                                    |
|----------------|---------------------------------------------------------------|
| 5              | <img src="../figures/faf4_5_comparison_tons_5.png" width="400">  |
| 8              | <img src="../figures/faf4_5_comparison_tons_8.png" width="400">  |
| 9              | <img src="../figures/faf4_5_comparison_tons_9.png" width="400">  |
| 21             | <img src="../figures/faf4_5_comparison_tons_21.png" width="400"> |

- Will move on and try to apply forecasting/trend analysis over 2017-2023 data instead to avoid the gap.
- Build the integration pipeline before coming back to address this issue.

## Day 12 (June 22, 2025)
- Removing rows before 2017.
- Just learned that ARIMA is used for single feature prediction while SARIMAX is used with multiple features.
- Added script for validation, metrics and visualizing results.
- Added ``config.py`` variables for easy access across all files (commodity mapping and predictive feature names).
- model_order was set to (1, 1, 1) due to the small training size.

#### Validation of trained SARIMAX (2017-2021) on 2022.

| Commodity Code          | Forecasting Value for Commodity                         |
|-------------------------|---------------------------------------------------------|
| 5 (Meat/Seafood)        | <img src="../results/forecast_value_5.png" width="400"> |
| 8 (Alcoholic beverages) | <img src="../results/forecast_value_8.png" width="400">    |
| 9 (Tobacco prods.)      | <img src="../results/forecast_value_9.png" width="400">    |
| 21  (Pharmaceuticals)   | <img src="../results/forecast_value_21.png" width="400">   |

| Commodity Code          | Forecasting Tons for Commodity                       |
|-------------------------|------------------------------------------------------|
| 5 (Meat/Seafood)        | <img src="../results/forecast_tons_5.png" width="400">  |
| 8 (Alcoholic beverages) | <img src="../results/forecast_tons_8.png" width="400">  |
| 9 (Tobacco prods.)      | <img src="../results/forecast_tons_9.png" width="400">  |
| 21 (Pharmaceuticals)    | <img src="../results/forecast_tons_21.png" width="400"> |

**Tomorrow:** Will start with integration - set up FastAPI project structure.

## Day 13 (June 23, 2025)
- Having trouble conceptualizing what the user will input when they attempt to use my forecasting tool.
- Option between letting the user input a period of data (say 2017:2022) and then giving me 
the 3 external feature data for 2023 and expecting a forecast, OR, providing only the 3 features and expecting
output for the very next year, the user then can only query for the year upto which model is trained.
- For now will keep it simpler and do a sequential prediction to the trained data -- Single-step ahead prediction.
- Created a subdirectory for api. Defined input types in "schemas.py". Defined forecast logic in ``routes.py``
Called them in ``main.py``. Setup forecast options for user in ``config.py``.
- Added /forecast POST endpoint using FastAPI.
- Allowing users to input word based label and converting that to the target_label parameter
using the mapping dictionary from config.
- Debugging errors at the moment with combining training data and user-input data.

## Day 14 (June 24, 2025)
- Running the API using
``
uvicorn api.main:app --reload
``
- Added unit conversions so that user can input features normally and then this is converted internally in "routes.py"
- Ran into some problems with running the API because the port 8000 seemed to be in the process from yesterday, tried killing it but didn't work.
Using port 8001 right now.
- Fixed errors and got a proper forecast as output from API.
- Had a problem with duplicate columns due to redundant processes in ``apply_scaler.py`` and ``routes.py`` which is fixed.
- Testing the "test_main.http" file and got ok response for all 3 (root, options and forecast).
- Adding error handling features.
- Checked Pharmaceuticals instead of Meat/seafood and tons as well and they all work in the forecast endpoint.
- Added `requirements.txt` and updated README with brief project description and instructions for using the API endpoints.
- Adding `form.html` in a new `templates/` folder to add a visual interface.
- Modified `main.py` to add the HTML template.
- Got the basic template running in `http://127.0.0.1:8000`.
- Updated gitignore to avoid larger csv files in `freight_data/raw` and updated `requirements.txt` to include all necessary libraries.

With this I think the basic project is working.
**Next Steps:** 
- UI Improvements, Adding success/failure message on submit, descriptions for input fields.
- Add more models for improvement of forecasting logic, maybe add multiple years of predictions, allow model switching.
- Storing predictions as a log.
- Containerize with Docker, deploy on a platform (Render, Heroku etc.).
- I've just realized there is another thing I've missed about the forecast prediction. So the dataset is working with 2017
Dollar value so any prediction will be adjusted to that year, so giving the output for 2022 will involve a conversion
to display on the user's end.

## Day 15 (July 20, 2025)
- Okay so starting today, the goal is to examine the current SARIMAX model and see if I can add improvements,
and look at exploring what other models I could potentially set up for.
- Adding grid search on p, d, q parameters of SARIMAX.
```
possible_orders = [
        [1, 1, 1], [0, 0, 0],
        [0, 0, 1], [0, 1, 0],
        [0, 1, 1], [1, 0, 1],
        [1, 1, 0], [1, 0, 0],
        [2, 1, 0], [2, 1, 2],
        [1, 1, 2], [2, 1, 1]
    ]
```
- Ok the basic pipeline worked, and now I have the optimal model giving me a result in validation, next step is to
find this optimal order based model for each of the commodities and give it's predictions and storing them
reliably and then moving onto XGBoost's predictions for comparison.

## Day 16 (July 21, 2025)
- Looking to make a table to store all results for all commodity and targets.
- Learned today about using df.to_markdown as a way to easily make a copy-paste-able markdown version of the table.
- I had to install a dependency `pip install tabulate`.

### SARIMAX Validation Results (2017-2021 Train → 2022 Valid)
| Target   |   p, d, q |       RMSE |   MAPE (%) |   Prediction |   Ground Truth |
|:---------|----------:|-----------:|-----------:|-------------:|---------------:|
| value_5  |       212 |  102.499   |    0.337   |   30312.7    |     30415.2    |
| tons_5   |       212 |  532.246   |    4.89391 |   10343.4    |     10875.7    |
| value_8  |       212 | 1814.41    |   13.6879  |   15069.9    |     13255.5    |
| tons_8   |       212 |  357.949   |    5.11105 |    7361.38   |      7003.43   |
| value_9  |       212 |  124.459   |    7.55831 |    1771.1    |      1646.65   |
| tons_9   |       212 |    7.16654 |   13.8517  |      58.9043 |        51.7378 |
| value_21 |       212 | 3440.68    |    4.91195 |   66606.4    |     70047.1    |
| tons_21  |       212 |   62.9685  |    4.83699 |    1238.84   |      1301.81   |


### SARIMAX Test Results (2017-2022 Train → 2023 Test)

| Target   | (p,d,q) | RMSE   | MAPE (%) | Prediction | Ground Truth |
|:---------|:----------|------------:|----------:|-------------:|---------------:|
| value_5  | (2, 1, 2) |  240.962    |  0.783796 |   30502      |     30743      |
| tons_5   | (2, 1, 2) |  134.418    |  1.22098  |   10874.6    |     11009      |
| value_8  | (2, 1, 2) | 1528.34     | 11.9291   |   14340.2    |     12811.9    |
| tons_8   | (2, 1, 2) |  693.276    | 10.2419   |    7462.3    |      6769.03   |
| value_9  | (2, 1, 2) |  105.605    |  6.9093   |    1634.05   |      1528.45   |
| tons_9   | (2, 1, 2) |    0.809503 |  1.68562  |      47.2144 |        48.0239 |
| value_21 | (2, 1, 2) | 2785.08     |  3.82549  |   75588.4    |     72803.3    |
| tons_21  | (2, 1, 2) |   98.2609   |  7.26226  |    1254.77   |      1353.03   |

- Now working on the `xgboost_forecast.py` script to add the same predictions but with XGBoost for comparison.
- For now just checking basic XGBRegressor() working, and then will try some grid search (maybe).

### XGBoost Validation (2017-2021 Train → 2022 Valid)
| Target   |    RMSE |   MAPE (%) |   Prediction |   Ground Truth |
|:---------|--------:|-----------:|-------------:|---------------:|
| value_5  | 14.1    |   0.653656 |   30614      |     30415.2    |
| tons_5   | 12.423  |   1.41905  |   11030      |     10875.7    |
| value_8  | 31.5312 |   7.50038  |   14249.8    |     13255.5    |
| tons_8   | 22.9195 |   7.50068  |    7528.74   |      7003.43   |
| value_9  | 13.9237 |  11.7736   |    1840.52   |      1646.65   |
| tons_9   |  2.3123 |  10.3343   |      57.0845 |        51.7378 |
| value_21 | 94.7    |  12.803    |   61079      |     70047.1    |
| tons_21  | 10.2689 |   8.10032  |    1196.36   |      1301.81   |

- Now I've applied a small amount of grid-search.
- Also learned about using product from itertools to do efficient grid search parameter unpacking.
### XGBoost Grid-Search Optimized Validation (2017-2021 Train → 2022 Valid)
| Target   |     RMSE |   MAPE (%) |   Prediction |   Ground Truth |
|:---------|---------:|-----------:|-------------:|---------------:|
| value_5  |  4.3427  |  0.0620052 |   30396.3    |     30415.2    |
| tons_5   |  3.42065 |  0.107587  |   10864      |     10875.7    |
| value_8  |  8.18723 |  0.505681  |   13322.6    |     13255.5    |
| tons_8   |  7.66601 |  0.839128  |    7062.2    |      7003.43   |
| value_9  | 13.7613  | 11.5005    |    1836.02   |      1646.65   |
| tons_9   |  2.30211 | 10.2434    |      57.0375 |        51.7378 |
| value_21 | 81.2385  |  9.4218    |   63447.4    |     70047.1    |
| tons_21  | 10.2692  |  8.10082   |    1196.35   |      1301.81   |

- Now moving to testing.

### XGBoost Test Results (2017-2022 Train → 2023 Test)
| Target   |    RMSE |   MAPE (%) |   Prediction |   Ground Truth | Test Params                                                                     |
|:---------|--------:|-----------:|-------------:|---------------:|:--------------------------------------------------------------------------------|
| value_5  | 58.0121 |   10.9469  |    27377.6   |     30743      | {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.2, 'subsample': 0.8}    |
| tons_5   | 31.9332 |    9.26263 |     9989.31  |     11009      | {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.012, 'subsample': 1.0} |
| value_8  | 17.3631 |    2.35311 |    13113.3   |     12811.9    | {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.012, 'subsample': 0.8}  |
| tons_8   | 13.3444 |    2.63069 |     6947.1   |      6769.03   | {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.012, 'subsample': 0.8}  |
| value_9  | 10.8732 |    7.73511 |     1646.67  |      1528.45   | {'n_estimators': 250, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8}   |
| tons_9   |  2.39   |   11.8942  |       53.736 |        48.0239 | {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.8}    |
| value_21 | 52.5279 |    3.78991 |    70044.1   |     72803.3    | {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.2, 'subsample': 0.8}   |
| tons_21  |  7.1571 |    3.78587 |     1301.81  |      1353.03   | {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.2, 'subsample': 1.0}   |

- Goal for tomorrow is to put these results side by side for a table comparison and then add PostgreSQL
integration.

## Day 17 (July 22, 2025)
- Table comparison in `./results/comparisons/sarimax_vs_xgboost.md`. Yet to add plots but will do that after database
integration.
- Created database `supply-chain-integration` and made an owner `freight-user`.
- Installed `pip install psycopg2 sqlalchemy`.
- Set up a separate password for my freight-user and then accessed the db in `./database/setup.py`.
- Moved processed data into a schema. Choosing to not do this for the raw data because there's too many
individual csv to cover. Will only do this for the processed data and the validation/test results.
### Processed schema
|   year |   value_5 |   tons_5 |   value_8 |   tons_8 |   value_9 |   tons_9 |   value_21 |   tons_21 |   population |   median_income |   retail_trade_gdp |
|-------:|----------:|---------:|----------:|---------:|----------:|---------:|-----------:|----------:|-------------:|----------------:|-------------------:|
|   2012 |   11166.5 |  3912.97 |   5192.35 |  2778.14 |   1473.72 |  24.3644 |    38894.7 |   402.789 |  9.90143e+06 |          0.6201 |            25886.8 |
|   2013 |   10935.1 |  3957.42 |   4837.58 |  2707.45 |   1124.18 |  23.5562 |    27015.6 |   376.037 |  9.97248e+06 |          0.5983 |            27403.9 |
|   2014 |   10808.1 |  3900.16 |   4912    |  2749.1  |   1092.91 |  22.9007 |    27296.2 |   379.942 |  1.00673e+07 |          0.6221 |            28584.6 |
|   2015 |   11145.4 |  4016.58 |   4832.9  |  2704.83 |   1104.29 |  23.1388 |    27637.4 |   384.692 |  1.01784e+07 |          0.6378 |            30876.7 |
|   2016 |   11403.5 |  4101.22 |   5232.33 |  2928.38 |   1068.85 |  22.3962 |    26276.6 |   365.75  |  1.03019e+07 |          0.6665 |            32555.2 |
|   2017 |   26427.4 |  9524.72 |  11005.5  |  5969.3  |   2064.78 |  53.971  |    57498   |   943.353 |  1.04103e+07 |          0.7096 |            33869.1 |
|   2018 |   29923.3 |  10795.1 |  12348.8  |  6524.38 |   1957.05 |  61.4908 |    58204.1 |  1081.71  |  1.05111e+07 |          0.6696 |            35560.5 |
|   2019 |   30314.8 |  10911.9 |  12498.5  |  6603.47 |   1877.44 |  58.9894 |    59540.2 |  1106.54  |  1.06174e+07 |          0.6694 |            36771.7 |
|   2020 |     30614 |    11030 |  14249.7  |  7528.69 |   1909.52 |  59.9973 |    61079   |  1135.14  |  1.07329e+07 |          0.6933 |            38498.9 |
|   2021 |   30379.7 |  10885.7 |  14476.6  |  7648.59 |   1816.75 |  57.0824 |    64373.2 |  1196.36  |  1.07921e+07 |          0.6886 |            45540.4 |
|   2022 |   30415.2 |  10875.7 |  13255.5  |  7003.43 |   1646.65 |  51.7378 |    70047.1 |  1301.81  |  1.09318e+07 |          0.7042 |            48679.6 |
|   2023 |     30743 |    11009 |  12811.9  |  6769.03 |   1528.45 |  48.0239 |    72803.3 |  1353.03  |  1.10644e+07 |          0.7242 |            52847.5 |

- Moved remaining 4 csv into the db: 
`processed_data`
`sarimax_validation_results`
`sarimax_test_results`
`xgb_validation_results`
`xgb_test_results`
