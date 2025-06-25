import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np


def add_lagging_commodity(data_frame):
    tons_commodity = [col for col in data_frame.columns if 'tons' in col]
    value_commodity = [col for col in data_frame.columns if 'value' in col]
    for col in tons_commodity:
        data_frame[f'{col}_lagged'] = data_frame[col].shift(1)
    for col in value_commodity:
        data_frame[f'{col}_lagged'] = data_frame[col].shift(1)
    return data_frame


def apply_scaler(data_frame, features, mode, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    shipment_features = data_frame[features]
    if mode == 'train':
        scaled_shipments = scaler.fit_transform(shipment_features)
    elif mode == 'test' or mode == 'valid':
        scaled_shipments = scaler.transform(shipment_features)
    else:
        return None

    scaled_shipments = pd.DataFrame(scaled_shipments, index=shipment_features.index, columns=shipment_features.columns)
    data_frame[features] = scaled_shipments[features]

    if data_frame[features].isnull().any().any():
        print("NaNs after scaling")
    if np.isinf(data_frame[features].to_numpy()).any():
        print("Infs after scaling")

    return data_frame, scaler


if __name__ == '__main__':
    df = pd.read_csv('../freight_data/processed/Georgia_Annual_Inbound_Shipments_2012-2023.csv', index_col=0)

    updated_df = add_lagging_commodity(df)
    updated_df = updated_df.dropna()

    # scaler = StandardScaler()
    # features = ['MEHOINUSGAA672N', 'GARETAILNQGSP', 'Population',
    #             'tons_5_lagged', 'tons_8_lagged', 'tons_9_lagged', 'tons_21_lagged',
    #             'value_5_lagged', 'value_8_lagged', 'value_9_lagged', 'value_21_lagged']
    #
    # shipment_features = updated_df[features]
    # scaled_shipments = scaler.fit_transform(shipment_features)
    # scaled_shipments = pd.DataFrame(scaled_shipments, index=shipment_features.index, columns=shipment_features.columns)
    #
    # updated_df[features] = scaled_shipments[features]

    # feature_correlations = updated_df.corr()
    # print(feature_correlations)

    # feature_correlations = feature_correlations.drop(columns=['tons_5', 'tons_8', 'tons_9', 'tons_21',
    #                                                           'value_5', 'value_8', 'value_9', 'value_21'])

    print(updated_df)
    updated_df.to_csv('forecast_prep.csv', index_label='Year')
