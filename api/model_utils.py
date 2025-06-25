from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


def sarimax_training(train, target):
    train = train.loc[:, ~train.columns.duplicated()]
    y_train = train[target].copy()

    value_columns = [col for col in train.columns if 'value' in col]
    tons_columns = [col for col in train.columns if 'tons' in col]
    train = train.drop(columns=tons_columns + value_columns)

    y_train = y_train.astype(float)
    train = train.astype(float)

    if train.isnull().any().any():
        print("NaNs found in exog:\n", train[train.isnull().any(axis=1)])
    if np.isinf(train.to_numpy()).any():
        print("Infs found in exog")

    train = train.dropna()
    y_train = y_train.loc[train.index]

    model = SARIMAX(endog=y_train, exog=train, order=(1, 1, 1))
    model_fit = model.fit(disp=0)

    return model_fit


def sarimax_predict(model, exog_df, index):
    predictions = model.predict(start=index[0], end=index[-1], exog=exog_df)
    predictions.index = index
    return predictions


def prepare_exog(df, target):
    values_tons = [col for col in df.columns if 'value_' in col or 'tons_' in col]
    return df.drop(columns=values_tons)
