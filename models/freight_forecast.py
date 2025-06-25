import os
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.feature_engineering import apply_scaler
from config import commodity_mapping, predictive_features, feature_units
from utils.visualization import plot_forecast_vs_actual


# df = pd.read_csv('../freight_data/processed/forecast_prep.csv')
#
# df['Year'] = pd.to_datetime(df['Year'], format='%Y')
# df.set_index('Year', inplace=True)
# df.index.freq = 'YS'
# commodity_mapping = {'5': 'Meat/seafood', '8': 'Alcoholic beverages',
#                      '9': 'Tobacco prods.', '21': 'Pharmaceuticals'}
#
# x_train = df.loc['2013':'2018'].copy()
# x_test = df.loc[['2019']].copy()
# # x_test = pd.DataFrame(x_test, columns=df.columns)
#
# scaler = StandardScaler()
# features = ['MEHOINUSGAA672N', 'GARETAILNQGSP', 'Population',
#             'tons_5_lagged', 'tons_8_lagged', 'tons_9_lagged', 'tons_21_lagged',
#             'value_5_lagged', 'value_8_lagged', 'value_9_lagged', 'value_21_lagged']
# value_columns = ['value_5', 'value_8', 'value_9', 'value_21']
# tons_columns = ['tons_5', 'tons_8', 'tons_9', 'tons_21']
#
#
# # def apply_scaler(data_frame, features, mode):
# #     shipment_features = data_frame[features]
# #     if mode == 'train':
# #         scaled_shipments = scaler.fit_transform(shipment_features)
# #     elif mode == 'test':
# #         scaled_shipments = scaler.transform(shipment_features)
# #     scaled_shipments = pd.DataFrame(scaled_shipments, index=shipment_features.index,
# #     columns=shipment_features.columns)
# #     data_frame[features] = scaled_shipments[features]
# #     return data_frame
#
#
# x_train_scaled, fitted_scaler = apply_scaler(x_train, features, 'train', scaler=scaler)
# x_test_scaled, _ = apply_scaler(x_test, features, mode='test', scaler=fitted_scaler)
#
# y_train = x_train_scaled['value_5']
# y_test = x_test_scaled['value_5']
#
# specific_cols = ['tons_8_lagged', 'tons_9_lagged', 'tons_21_lagged',
#                  'value_8_lagged', 'value_9_lagged', 'value_21_lagged']
#
# x_train_scaled = x_train_scaled.drop(columns=tons_columns + value_columns + specific_cols)
# x_test_scaled = x_test_scaled.drop(columns=tons_columns + value_columns + specific_cols)
#
# arima_order = (0, 2, 3)
# model = ARIMA(endog=y_train, exog=x_train_scaled, order=arima_order)
# model_fit = model.fit()
#
# start_index = y_test.index[0]
# end_index = y_test.index[-1]
# predictions = model_fit.predict(start=start_index, end=end_index, exog=x_test_scaled)
# predictions.index = y_test.index
# print("Prediction on expected value", predictions.iloc[0])
# print("Actual value", y_test.iloc[0])
# print("Mean square error:", mean_squared_error(y_test, predictions))
# print("Root mean square error:", root_mean_squared_error(y_test, predictions))

def sarimax_validation(train, valid, target):
    y_train = train[target]
    y_valid = valid[target]
    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)
    valid = valid.drop(columns=tons_columns + value_columns)

    model = SARIMAX(endog=y_train, exog=train, order=(1, 1, 1))
    model_fit = model.fit(disp=0)
    start_index = y_valid.index[0]
    end_index = y_valid.index[-1]
    predictions = model_fit.predict(start=start_index, end=end_index, exog=valid)
    predictions.index = y_valid.index

    return predictions, y_train, y_valid


def metrics(target, prediction):
    print("Prediction on expected value", prediction.iloc[0])
    print("Actual value", target.iloc[0])
    # print("Mean square error:", mean_squared_error(target, prediction))
    print("Root mean square error:", root_mean_squared_error(target, prediction))
    print('Mean Absolute Percentage Error (MAPE): ', np.mean(np.abs((target - prediction) / target)) * 100)


def forecast_validation(train, valid, target, library):
    pred, y_train, y_valid = sarimax_validation(train=train, valid=valid, target=target)
    metrics(y_valid, pred)
    target = target.split('_')

    plot_forecast_vs_actual(y_train=y_train, y_valid=y_valid, predictions=pred,
                            feature_type=target[0], commodity_code=int(target[1]),
                            library=library, save_path=True)


if __name__ == '__main__':
    st = time.time()
    shipment_df = pd.read_csv('../freight_data/processed/Georga_AIS_2012-2023_minus_inflation.csv')
    shipment_df.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)

    shipment_df = shipment_df[shipment_df.Year >= 2017]
    shipment_df['Year'] = pd.to_datetime(shipment_df['Year'], format='%Y')
    shipment_df.set_index('Year', inplace=True)
    shipment_df.index.freq = 'YS'

    x_train = shipment_df.loc['2017': '2021'].copy()
    x_valid = shipment_df.loc[['2022']].copy()
    x_test = shipment_df.loc[['2023']].copy()
    # x_test = pd.DataFrame(x_test, columns=df.columns)

    # features = ['MEHOINUSGAA672N', 'GARETAILNQGSP', 'Population']
    x_train_scaled, fitted_scaler = apply_scaler(x_train, predictive_features, 'train', scaler=StandardScaler())
    x_valid_scaled, _ = apply_scaler(x_valid, predictive_features, mode='valid', scaler=fitted_scaler)

    forecast_validation(x_train_scaled, x_valid_scaled, 'value_5', library='matplotlib')
    forecast_validation(x_train_scaled, x_valid_scaled, 'tons_5', library='matplotlib')

    forecast_validation(x_train_scaled, x_valid_scaled, 'value_8', library='matplotlib')
    forecast_validation(x_train_scaled, x_valid_scaled, 'tons_8', library='matplotlib')

    forecast_validation(x_train_scaled, x_valid_scaled, 'value_9', library='matplotlib')
    forecast_validation(x_train_scaled, x_valid_scaled, 'tons_9', library='matplotlib')

    forecast_validation(x_train_scaled, x_valid_scaled, 'value_21', library='matplotlib')
    forecast_validation(x_train_scaled, x_valid_scaled, 'tons_21', library='matplotlib')

    print('Runtime:', time.time() - st)
