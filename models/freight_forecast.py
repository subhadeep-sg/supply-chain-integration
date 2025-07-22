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
from dotenv import load_dotenv
import logging
import warnings


load_dotenv()
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
root_dir = os.getenv('project_root_dir')


def sarimax_validation(train, valid, target, order=(1, 1, 1)):
    y_train = train[target]
    y_valid = valid[target]
    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)
    valid = valid.drop(columns=tons_columns + value_columns)

    logger.info(f'Starting SARIMAX train...')
    model = SARIMAX(endog=y_train, exog=train, order=order)
    model_fit = model.fit(disp=0)
    logger.debug(f'Model fit..')

    start_index = y_valid.index[0]
    end_index = y_valid.index[-1]
    predictions = model_fit.predict(start=start_index, end=end_index, exog=valid)
    predictions.index = y_valid.index

    logger.info(f'Returning SARIMAX predictions...')
    return predictions, y_train, y_valid


def metrics(gt, prediction, verbose=False):
    rmse = root_mean_squared_error(gt, prediction)
    mape = np.mean(np.abs((gt - prediction) / gt)) * 100

    if verbose:
        print("Prediction on expected value", prediction.iloc[0])
        print("Actual value", gt.iloc[0])
        # print("Mean square error:", mean_squared_error(target, prediction))
        print("Root mean square error:", root_mean_squared_error(gt, prediction))
        print('Mean Absolute Percentage Error (MAPE): ', np.mean(np.abs((gt - prediction) / gt)) * 100)

    return rmse, mape


def forecast_validation(train, valid, target, library):
    pred, y_train, y_valid = sarimax_validation(train=train, valid=valid, target=target)
    _, _ = metrics(y_valid, pred, verbose=True)
    target = target.split('_')

    plot_forecast_vs_actual(y_train=y_train, y_valid=y_valid, predictions=pred,
                            feature_type=target[0], commodity_code=int(target[1]),
                            library=library, save_path=True)


def grid_search_sarimax(train, y_train, valid, y_valid):
    logger.info(f'Starting grid search on SARIMAX..')
    possible_orders = [
        [1, 1, 1], [0, 0, 0],
        [0, 0, 1], [0, 1, 0],
        [0, 1, 1], [1, 0, 1],
        [1, 1, 0], [1, 0, 0],
        [2, 1, 0], [2, 1, 2],
        [1, 1, 2], [2, 1, 1]
    ]

    dataframe = pd.DataFrame()

    for order in possible_orders:
        extract_order = (order[0], order[1], order[2])
        model = SARIMAX(endog=y_train, exog=train, order=order)
        model = model.fit(disp=0)
        start_index = y_valid.index[0]
        end_index = y_valid.index[-1]
        prediction = model.predict(start=start_index, end=end_index, exog=valid)

        rmse, mape = metrics(gt=y_valid, prediction=prediction)
        extract_order = ''.join(str(digit) for digit in extract_order)

        row = pd.DataFrame(data={'order': [extract_order], 'rmse': [rmse], 'mape': [mape]})
        dataframe = pd.concat([dataframe, row], axis=0)

    dataframe = dataframe.sort_values(by=['rmse', 'mape'], ascending=True)
    logger.debug(f'Dataframe: {dataframe.columns}, dataframe size: {len(dataframe)}')
    logger.debug(f'Grid search validation results: \n{dataframe}')

    order_string = dataframe['order'].max()

    logger.debug(f'Best model order is {order_string}')

    extracted_order = [int(digit) for digit in order_string]
    extracted_order = (extracted_order[0], extracted_order[1], extracted_order[2])

    logger.info(f'Returning grid search results and optimal p,d, q order...')
    return dataframe, extracted_order, order_string


def sarimax_train_prediction(train, y_train, x_pred, y_true, order):
    logger.info(f'Starting SARIMAX train...')
    model = SARIMAX(endog=y_train, exog=train, order=order)
    model_fit = model.fit(disp=0)
    logger.info(f'Optimal model fit..')

    start_index = y_true.index[0]
    end_index = y_true.index[-1]
    predictions = model_fit.predict(start=start_index, end=end_index, exog=x_pred)
    predictions.index = y_true.index

    rmse, mape = metrics(gt=y_true, prediction=predictions, verbose=False)
    logger.info(f'Returning scores and prediction...')
    return rmse, mape, predictions


def sarimax_model_pipeline(train, valid, target):
    logger.info(f'Entering model pipeline for {target}..')
    y_train = train[target]
    y_valid = valid[target]
    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)
    valid = valid.drop(columns=tons_columns + value_columns)

    logger.info(f'Starting SARIMAX train...')
    results, extracted_order, order_string = grid_search_sarimax(train=train, y_train=y_train, valid=valid,
                                                                 y_valid=y_valid)

    rmse, mape, pred = sarimax_train_prediction(train=train, y_train=y_train,
                                                x_pred=valid, y_true=y_valid,
                                                order=extracted_order)

    row = pd.DataFrame(data={'target': [target], 'p, d, q': [order_string], 'rmse': [rmse], 'mape': [mape]})

    logger.info(f'Returning optimal SARIMAX results and model for {target}...')
    return row, extracted_order


def sarimax_test_pipeline(train, test, target, model_order):
    logger.info(f'Entering sarimax test pipeline for {target}..')
    y_train = train[target]
    y_test = test[target]
    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)
    test = test.drop(columns=tons_columns + value_columns)

    logger.info(f'Starting SARIMAX train...')
    # results, extracted_order, order_string = grid_search_sarimax(train=train, y_train=y_train, valid=test, y_valid=y_test)

    rmse, mape, pred = sarimax_train_prediction(train=train, y_train=y_train,
                                                x_pred=test, y_true=y_test,
                                                order=model_order)

    row = pd.DataFrame(data={
        'target': [target],
        'p, d, q': [model_order],
        'rmse': [rmse], 'mape': [mape],
        'prediction': [pred.iloc[0]],
        'ground_truth': [y_test.iloc[0]]
    })

    return row


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

    # Main pipeline
    # forecast_validation(x_train_scaled, x_valid_scaled, 'value_5', library='matplotlib')
    # forecast_validation(x_train_scaled, x_valid_scaled, 'tons_5', library='matplotlib')
    #
    # forecast_validation(x_train_scaled, x_valid_scaled, 'value_8', library='matplotlib')
    # forecast_validation(x_train_scaled, x_valid_scaled, 'tons_8', library='matplotlib')
    #
    # forecast_validation(x_train_scaled, x_valid_scaled, 'value_9', library='matplotlib')
    # forecast_validation(x_train_scaled, x_valid_scaled, 'tons_9', library='matplotlib')
    #
    # forecast_validation(x_train_scaled, x_valid_scaled, 'value_21', library='matplotlib')
    # forecast_validation(x_train_scaled, x_valid_scaled, 'tons_21', library='matplotlib')

    # Trying SARIMAX improvements

    value_5_row, value_5_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='value_5')
    value_8_row, value_8_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='value_8')
    value_9_row, value_9_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='value_9')
    value_21_row, value_21_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='value_21')

    tons_5_row, tons_5_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='tons_5')
    tons_8_row, tons_8_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='tons_8')
    tons_9_row, tons_9_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='tons_9')
    tons_21_row, tons_21_order = sarimax_model_pipeline(train=x_train_scaled, valid=x_valid_scaled, target='tons_21')

    validation_results = pd.concat([value_5_row, tons_5_row, value_8_row, tons_8_row, value_9_row, tons_9_row,
                                    value_21_row, tons_21_row], axis=0)
    logger.debug(f'Validation results: \n{validation_results}')

    os.makedirs(f'{root_dir}/results/sarimax', exist_ok=True)
    validation_results.to_csv(f'{root_dir}/results/sarimax/sarimax_validation.csv', index=False)
    logger.info(f'Validation scores saved in ./results/sarimax/sarimax_validation.csv')

    x_train = shipment_df.loc['2017': '2022'].copy()
    x_test = shipment_df.loc[['2023']].copy()
    x_train_scaled, fitted_scaler = apply_scaler(x_train, predictive_features, 'train', scaler=StandardScaler())
    x_test_scaled, _ = apply_scaler(x_test, predictive_features, mode='test', scaler=fitted_scaler)

    value_5_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='value_5', model_order=value_5_order)
    value_8_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='value_8', model_order=value_8_order)
    value_9_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='value_9', model_order=value_9_order)
    value_21_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='value_21',
                                         model_order=value_21_order)

    tons_5_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='tons_5', model_order=tons_5_order)
    tons_8_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='tons_8', model_order=tons_8_order)
    tons_9_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='tons_9', model_order=tons_9_order)
    tons_21_row = sarimax_test_pipeline(train=x_train_scaled, test=x_test, target='tons_21', model_order=tons_21_order)

    test_results = pd.concat([value_5_row, tons_5_row, value_8_row, tons_8_row, value_9_row, tons_9_row,
                              value_21_row, tons_21_row], axis=0)

    logger.debug(f'Test results: \n{test_results}')
    test_results.to_csv(f'{root_dir}/results/sarimax/sarimax_test.csv', index=False)
    logger.info(f'Test scores saved in ./results/sarimax/sarimax_test.csv')

    # print(test_results.to_markdown(index=False))

    logger.info(f'Runtime: {time.time() - st}')


if __name__ == '__main__':
    main()
