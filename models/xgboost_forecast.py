import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from utils.feature_engineering import apply_scaler
from config import commodity_mapping, predictive_features, feature_units
from utils.visualization import plot_forecast_vs_actual
from dotenv import load_dotenv
import logging
import warnings
from itertools import product

load_dotenv()
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
root_dir = os.getenv('project_root_dir')


def xgboost_grid_search(train, y_train, valid, y_valid):
    param_grid = {
        'n_estimators': [50, 100, 120, 150, 200, 250],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.012, 0.025, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    best_rmse = float('inf')
    best_params = None
    results = []

    keys, values = zip(*param_grid.items())
    for v in product(*values):
        params = dict(zip(keys, v))
        model = XGBRegressor(**params)
        model.fit(train, y_train)
        prediction = model.predict(valid)

        rmse = np.sqrt(root_mean_squared_error(y_valid, prediction))
        mape = mean_absolute_percentage_error(y_valid, prediction) * 100

        results.append({'params': params, 'rmse': rmse, 'mape': mape})

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    return best_params, pd.DataFrame(results)


def xgboost_pipeline(train, valid, target, params=None, mode='valid'):
    logger.info(f'Entering xgboost pipeline for {target}..')
    y_train = train[target]
    y_valid = valid[target]
    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)
    valid = valid.drop(columns=tons_columns + value_columns)

    logger.info(f'Starting XGBRegressor train...')

    if mode == 'valid':
        best_params, search_results = xgboost_grid_search(train=train, y_train=y_train, valid=valid, y_valid=y_valid)
        logger.info(f'Best params: {best_params}')
        model = XGBRegressor(**best_params)
        model.fit(train, y_train)
    elif mode == 'test':
        model = XGBRegressor(**params)
        model.fit(train, y_train)
    else:
        raise ValueError(f'Invalid mode {mode}, enter either "valid" or "test"')

    prediction = model.predict(valid)

    rmse = np.sqrt(root_mean_squared_error(y_valid, prediction))
    mape = mean_absolute_percentage_error(y_valid, prediction) * 100

    logger.info(f'Returning scores and predictions for {target}...')
    if mode == 'valid':
        row = pd.DataFrame(data={
            'Target': [target],
            'RMSE': [rmse],
            'MAPE (%)': [mape],
            'Prediction': [prediction],
            'Ground Truth': [y_valid.iloc[0]],
            'Best Params': [best_params]
        })
        return row, best_params
    elif mode == 'test':
        row = pd.DataFrame(data={
            'Target': [target],
            'RMSE': [rmse],
            'MAPE (%)': [mape],
            'Prediction': [prediction],
            'Ground Truth': [y_valid.iloc[0]],
            'Test Params': [params]
        })
        return row, params


def main():
    st = time.time()
    shipment_df = pd.read_csv(f'{root_dir}/freight_data/processed/Georga_AIS_2012-2023_minus_inflation.csv')
    shipment_df.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)

    shipment_df = shipment_df[shipment_df.Year >= 2017]
    shipment_df['Year'] = pd.to_datetime(shipment_df['Year'], format='%Y')
    shipment_df.set_index('Year', inplace=True)
    shipment_df.index.freq = 'YS'

    x_train = shipment_df.loc['2017': '2021'].copy()
    x_valid = shipment_df.loc[['2022']].copy()

    # features = ['MEHOINUSGAA672N', 'GARETAILNQGSP', 'Population']
    x_train_scaled, fitted_scaler = apply_scaler(x_train, predictive_features, 'train', scaler=StandardScaler())
    x_valid_scaled, _ = apply_scaler(x_valid, predictive_features, mode='valid', scaler=fitted_scaler)

    value_5_row, value_5_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='value_5')
    value_8_row, value_8_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='value_8')
    value_9_row, value_9_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='value_9')
    value_21_row, value_21_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='value_21')

    tons_5_row, tons_5_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='tons_5')
    tons_8_row, tons_8_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='tons_8')
    tons_9_row, tons_9_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='tons_9')
    tons_21_row, tons_21_param = xgboost_pipeline(x_train_scaled, x_valid_scaled, target='tons_21')

    validation_results = pd.concat([value_5_row, tons_5_row, value_8_row, tons_8_row,
                                    value_9_row, tons_9_row, value_21_row, tons_21_row], axis=0)

    os.makedirs(f'{root_dir}/results/xgboost', exist_ok=True)
    validation_results.to_csv(f'{root_dir}/results/xgboost/xgb_validation.csv', index=False)
    logger.info(f'Validation results stored in ./results/xgboost/xgb_validation.csv')
    # print(validation_results.to_markdown(index=False))

    x_train = shipment_df.loc['2017': '2022'].copy()
    x_test = shipment_df.loc[['2023']].copy()
    x_train_scaled, fitted_scaler = apply_scaler(x_train, predictive_features, 'train', scaler=StandardScaler())
    x_test_scaled, _ = apply_scaler(x_test, predictive_features, mode='test', scaler=fitted_scaler)

    value_5_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='value_5', mode='test',
                                      params=value_5_param)
    value_8_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='value_8', mode='test',
                                      params=value_8_param)
    value_9_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='value_9', mode='test',
                                      params=value_9_param)
    value_21_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='value_21', mode='test',
                                       params=value_21_param)

    tons_5_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='tons_5', mode='test', params=tons_5_param)
    tons_8_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='tons_8', mode='test', params=tons_8_param)
    tons_9_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='tons_9', mode='test', params=tons_9_param)
    tons_21_row, _ = xgboost_pipeline(x_train_scaled, x_test_scaled, target='tons_21', mode='test',
                                      params=tons_21_param)

    test_results = pd.concat([value_5_row, tons_5_row, value_8_row, tons_8_row, value_9_row, tons_9_row,
                              value_21_row, tons_21_row], axis=0)

    print(test_results.to_markdown(index=False))

    test_results.to_csv(f'{root_dir}/results/xgboost/xgb_test.csv', index=False)

    logger.info(f'Runtime: {time.time() - st}')


if __name__ == '__main__':
    main()
