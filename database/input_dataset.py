from sqlalchemy import create_engine, text
import os
import pandas as pd
from dotenv import load_dotenv
import logging
import time

logging.basicConfig(level=logging.INFO)
load_dotenv()
root_dir = os.getenv('project_root_dir')
database_url = os.getenv("DATABASE_URL")
logger = logging.getLogger(__name__)


def create_results_schema(table_name, model_name):
    if model_name == 'sarimax':
        create_results_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                target TEXT,
                model_order TEXT,
                rmse FLOAT,
                mape FLOAT,
                prediction FLOAT,
                ground_truth FLOAT
            );
        """
    elif model_name == 'xgboost':
        create_results_query = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        target TEXT,
                        rmse FLOAT,
                        mape FLOAT,
                        prediction FLOAT,
                        ground_truth FLOAT,
                        params TEXT
                    );
                """
    else:
        raise ValueError(f'Model {model_name} does not exist.')
    engine = create_engine(database_url)
    with engine.connect() as connection:
        connection.execute(text(create_results_query))
        logging.info(f"Table {table_name} created...")


def insert_results_data(table_name, model_name, mode):
    logger.info(f"Inserting {model_name} {mode} results into table {table_name}...")

    if mode == 'valid':
        mode = 'validation'
    elif mode == 'testing':
        mode = 'test'
    elif mode not in ['validation', 'test']:
        raise ValueError(f"Invalid mode {mode}")

    if model_name == 'sarimax':
        df = pd.read_csv(f'{root_dir}/results/{model_name}/sarimax_{mode}.csv')
        df = df.rename(columns={'p, d, q': 'Model Order', 'MAPE (%)': 'MAPE'})
        df = df[['Target', 'Model Order', 'RMSE', 'MAPE', 'Prediction', 'Ground Truth']]

    elif model_name == 'xgboost':
        df = pd.read_csv(f'{root_dir}/results/{model_name}/xgb_{mode}.csv')
        df = df.rename(columns={'Best Params': 'Params', 'Test Params': 'Params', 'MAPE (%)': 'MAPE'})
        df = df[['Target', 'RMSE', 'MAPE', 'Prediction', 'Ground Truth', 'Params']]

    else:
        raise ValueError(f'Model {model_name} does not exist.')

    engine = create_engine(database_url)
    df.to_sql(table_name, engine, if_exists='append', index=False)
    logger.info(f'{mode} data inserted successfully..')


def create_processed_schema(table_name):
    logger.info(f"Creating table {table_name} ...")
    create_data_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                year INT PRIMARY KEY,
                value_5 FLOAT,
                tons_5 FLOAT,
                value_8 FLOAT,
                tons_8 FLOAT,
                value_9 FLOAT,
                tons_9 FLOAT,
                value_21 FLOAT,
                tons_21 FLOAT,
                population FLOAT,
                median_income FLOAT,
                retail_trade_gdp FLOAT
            );
            """
    engine = create_engine(database_url)
    with engine.connect() as connection:
        connection.execute(text(create_data_query))
    logging.info(f"Table {table_name} created...")


def insert_processed_data(table_name):
    logger.info(f'Starting insertion...')
    engine = create_engine(database_url)
    df = pd.read_csv(f'{root_dir}/freight_data/processed/Georga_AIS_2012-2023_minus_inflation.csv')

    df = df.rename(columns={
        'Unnamed: 0': 'year',
        'MEHOINUSGAA672N': 'median_income',
        'GARETAILNQGSP': 'retail_trade_gdp',
        'Population': 'population'
    })
    df['year'] = df['year'].astype(int)

    # Reorder
    df = df[['year', 'value_5', 'tons_5', 'value_8', 'tons_8', 'value_9', 'tons_9',
             'value_21', 'tons_21', 'population', 'median_income', 'retail_trade_gdp']]

    df.to_sql(table_name, engine, if_exists='append', index=False)
    logger.info('Data inserted successfully..')


def print_schema(table_name):
    engine = create_engine(database_url)
    df = pd.read_sql_table(table_name, engine)
    print(df.to_markdown(index=False))


def main():
    start = time.time()

    # create_processed_schema(table_name='processed_data')
    # insert_processed_data(table_name='processed_data')
    print_schema('processed_data')

    create_results_schema(table_name='sarimax_validation_results', model_name='sarimax')
    insert_results_data(table_name='sarimax_validation_results', model_name='sarimax', mode='validation')
    print_schema('sarimax_validation_results')

    create_results_schema(table_name='sarimax_test_results', model_name='sarimax')
    insert_results_data(table_name='sarimax_test_results', model_name='sarimax', mode='test')
    print_schema('sarimax_test_results')

    create_results_schema(table_name='xgb_validation_results', model_name='xgboost')
    insert_results_data(table_name='xgb_validation_results', model_name='xgboost', mode='validation')
    print_schema('xgb_validation_results')

    create_results_schema(table_name='xgb_test_results', model_name='xgboost')
    insert_results_data(table_name='xgb_test_results', model_name='xgboost', mode='test')
    print_schema('xgb_test_results')

    logger.info(f'Runtime: {time.time() - start}')


if __name__ == '__main__':
    main()
