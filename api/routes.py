from fastapi import APIRouter
from .schemas import ForecastRequest
from .model_utils import sarimax_training
import pandas as pd
import numpy as np
from utils.feature_engineering import apply_scaler
from sklearn.preprocessing import StandardScaler
from config import forecast_options, predictive_features, unit_conversion

router = APIRouter()


@router.post('/forecast')
def forecast(request: ForecastRequest):
    target_feature = forecast_options.get(request.target_label)
    if not target_feature:
        return {'Error': 'Invalid target_label. Please choose among valid options.'}

    data = pd.DataFrame([request.dict()])
    data.index = pd.date_range(start=f'{request.year}', periods=1, freq='YS')
    data.index.name = 'Year'

    data_features_dict = request.features
    user_data = pd.DataFrame([data_features_dict])
    user_data.index = pd.date_range(start=f'{request.year}', periods=1, freq='YS')
    user_data.index.name = 'Year'
    # user_data['MEHOINUSGAA672N'] = user_data['MEHOINUSGAA672N'].apply(lambda x: x * 10e-6)
    for feature, factor in unit_conversion.items():
        if feature in user_data.columns:
            user_data[feature] = user_data[feature].apply(lambda x: x * factor)

    train_df = pd.read_csv('freight_data/processed/Georga_AIS_2012-2023_minus_inflation.csv')
    train_df.rename(columns={'Unnamed: 0': 'Year'}, inplace=True)
    train_df = train_df[train_df['Year'] >= 2017]
    train_df = train_df[train_df['Year'] < request.year]
    train_df['Year'] = pd.to_datetime(train_df['Year'], format='%Y')
    train_df.set_index('Year', inplace=True)
    train_df.index.freq = 'YS'

    combined_df = pd.concat([train_df, user_data])
    combined_df = combined_df.sort_index()

    scaler = StandardScaler()
    combined_df_scaled, fitted_scaler = apply_scaler(combined_df.copy(), features=predictive_features,
                                                     mode='train', scaler=scaler)

    # combined_df_scaled = pd.concat([non_scaled_df, features_scaled], axis=1)

    train_scaled = combined_df_scaled.iloc[:-1]
    test_scaled = combined_df_scaled.iloc[-1:].copy()

    assert not np.isinf(combined_df_scaled[predictive_features].values).any(), "Infs detected after scaling!"
    assert not combined_df_scaled[predictive_features].isna().any().any(), "NaNs detected after scaling!"

    model = sarimax_training(train_scaled, target=target_feature)
    prediction = model.predict(start=test_scaled.index[0],
                               end=test_scaled.index[-1],
                               exog=test_scaled[predictive_features])

    return {
        'year': request.year,
        'forecast': round(prediction.iloc[0], 2),
        'target_label': target_feature
    }


@router.get('/options')
def get_forecast_options():
    return {'forecast_options': list(forecast_options.keys())}
