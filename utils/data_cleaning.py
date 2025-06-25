import pandas as pd
import time
from sklearn.preprocessing import StandardScaler


def filter_domestic_state(dataframe):
    """
    Selecting trade_type 1 (Domestic) and State Destination 13 (Georgia) - Only Domestic Inbound shipments
    Dropping distance/mile columns for current scope.
    """
    foreign_columns = [col for col in dataframe.columns if 'fr_' in col or 'orig' in col]
    mile_dist_cols = [col for col in dataframe.columns if 'miles' in col or 'dist' in col]
    dataframe = dataframe.drop(columns=foreign_columns + mile_dist_cols)

    current_value_columns = {}
    for col in dataframe.columns:
        if 'curval_' in col:
            current_value_columns[col] = 'current_value' + col.split('_')[1]

    dataframe = dataframe.rename(columns=current_value_columns)

    dest_col = [col for col in dataframe.columns if 'dest' in col]
    dataframe = dataframe[(dataframe[dest_col[0]] == 13) & (dataframe.trade_type == 1)]
    dataframe = dataframe.drop(columns=['trade_type'])

    return dataframe


def separate_metric_and_year(dataframe, metric_substring):
    """
    Assumes the metric to be in the format 'metric_name_{year}'.
    """
    metric_df = dataframe[dataframe.index.str.startswith(metric_substring)]
    metric_df.columns = [metric_substring + str(col_name) for col_name in metric_df.columns]
    metric_df.index = metric_df.index.str.replace(metric_substring, '')
    return metric_df


def external_features_by_state(population_df_list, fred_data, state_name):
    state_pops = [data_frame.loc[f'.{state_name}'] for data_frame in population_df_list]
    state_pop_combined = pd.concat(state_pops)
    state_pop_combined = state_pop_combined.drop('Unnamed: 1', errors='ignore')

    fred_data['observation_date'] = pd.to_datetime(fred_data['observation_date'])
    fred_data['year'] = fred_data['observation_date'].dt.year
    fred_data = fred_data.drop(columns=['observation_date'])
    fred_data = fred_data.set_index('year')

    external_features = pd.concat([fred_data, state_pop_combined], axis=1)
    external_features = external_features.dropna()
    external_features = external_features.rename(columns={f".{state_name}": "Population"})
    return external_features


def select_commodity_groups(dataframe, commodity_groups):
    data_frame = dataframe.copy()
    data_frame = data_frame[data_frame['sctg2'].isin(commodity_groups.keys())]
    data_frame = data_frame.groupby(by=['sctg2']).agg('sum')
    return data_frame


def combine_features(shipment_data, external_features):
    transformed_df = shipment_data.copy()

    value_df = separate_metric_and_year(transformed_df, 'value_')
    tons_df = separate_metric_and_year(transformed_df, 'tons_')

    shipments_inbound = pd.concat([tons_df, value_df], axis=1)
    shipments_inbound['year'] = pd.to_datetime(shipments_inbound.index)
    shipments_inbound['year'] = shipments_inbound['year'].dt.year
    shipments_inbound = shipments_inbound.set_index('year')

    shipments_inbound = shipments_inbound.sort_index()

    combined_features = pd.concat([shipments_inbound, external_features], axis=1)
    combined_features = combined_features.dropna()
    # MEHOINUSGAA646N does not account for inflation.
    combined_features = combined_features.drop(columns=['MEHOINUSGAA646N'])
    # Converting unit from Dollars to Millions of Dollars
    combined_features['MEHOINUSGAA672N'] = combined_features['MEHOINUSGAA672N'].apply(lambda x: x * 10e-6)
    return combined_features


def apply_deflation(shipment_data, deflation_data):
    deflation_data['year'] = pd.to_datetime(deflation_data.index)
    deflation_data['year'] = deflation_data['year'].dt.year
    deflation_data = deflation_data.set_index('year')

    shipment_data = pd.concat([shipment_data, deflation_data], axis=1)
    shipment_data = shipment_data.rename(columns={'A191RD3A086NBEA': 'deflate'})
    shipment_data = shipment_data.dropna()
    shipment_data = shipment_data.sort_index()
    shipment_data['conversion_factor'] = shipment_data['deflate'].apply(
        lambda x: float(100.0 / x))
    shipment_data['conversion_factor'].values[shipment_data['conversion_factor'] < 1.0] = float(
        1.0)
    value_columns = [col for col in shipment_data.columns if 'val' in col]

    for col in value_columns:
        shipment_data[col] = shipment_data[col] * shipment_data['conversion_factor']

    shipment_data = shipment_data.drop(columns=['deflate', 'conversion_factor'])
    return shipment_data


if __name__ == '__main__':
    st = time.time()
    # FAF Data
    df2 = pd.read_csv('../freight_data/raw/FAF5.6.1_State.csv')
    df = pd.read_csv('../freight_data/raw/FAF5.6.1_Reprocessed_1997-2012_State.csv')
    df3 = pd.read_csv('../freight_data/raw/FAF5.6.1_State_2018-2023.csv')
    df4 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2013.csv')
    df5 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2014.csv')
    df6 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2015.csv')
    df7 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2016.csv')
    df8 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2017.csv')

    implicit_deflate = pd.read_csv('../freight_data/raw/A191RD3A086NBEA.csv', index_col=0)

    state_pop19 = pd.read_excel('../freight_data/raw/nst-est2019-01.xlsx', header=3, index_col=0)
    state_pop24 = pd.read_excel('../freight_data/raw/NST-EST2024-POP.xlsx', header=3, index_col=0)
    retails = pd.read_csv('../freight_data/raw/fredgraph.csv')

    df = filter_domestic_state(df)
    df2 = filter_domestic_state(df2)
    df3 = filter_domestic_state(df3)
    df4 = filter_domestic_state(df4)
    df5 = filter_domestic_state(df5)
    df6 = filter_domestic_state(df6)
    df7 = filter_domestic_state(df7)
    df8 = filter_domestic_state(df8)

    df = pd.concat([df, df2, df3, df4, df5, df6, df7, df8])

    # georgia_state_pop = state_pop19.loc['.Georgia']
    # georgia_state_pop = pd.concat([georgia_state_pop, state_pop24.loc['.Georgia']])
    # georgia_state_pop = georgia_state_pop.drop('Unnamed: 1')
    #
    # retails['observation_date'] = pd.to_datetime(retails['observation_date'])
    # retails['year'] = retails['observation_date'].dt.year
    # retails = retails.drop(columns=['observation_date'])
    # retails = retails.set_index('year')
    #
    # additional_info_df = pd.concat([retails, georgia_state_pop], axis=1)
    # additional_info_df = additional_info_df.dropna()
    # additional_info_df = additional_info_df.rename(columns={".Georgia": "Population"})

    external_features = external_features_by_state([state_pop19, state_pop24],
                                                   fred_data=retails, state_name='Georgia')

    commodity_mapping = {5: 'Meat/seafood', 8: 'Alcoholic beverages',
                         9: 'Tobacco prods.', 21: 'Pharmaceuticals'}
    # inbound_shipments = df[(df.sctg2 == 5) | (df.sctg2 == 8) | (df.sctg2 == 9) | (df.sctg2 == 21)]
    aggregated_inbound = select_commodity_groups(df, commodity_groups=commodity_mapping)

    tons_and_value_columns = [col_name for col_name in aggregated_inbound.columns if
                              'tons' in col_name or 'val' in col_name]
    annual_shipments_in = aggregated_inbound[tons_and_value_columns]
    annual_shipments_in = annual_shipments_in.transpose()

    # transformed_df = annual_shipments_in.copy()
    #
    # value_df = separate_metric_and_year(transformed_df, 'value_')
    # tons_df = separate_metric_and_year(transformed_df, 'tons_')
    #
    # annual_shipments_inbound = pd.concat([tons_df, value_df], axis=1)
    # annual_shipments_inbound['year'] = pd.to_datetime(annual_shipments_inbound.index)
    # annual_shipments_inbound['year'] = annual_shipments_inbound['year'].dt.year
    # annual_shipments_inbound = annual_shipments_inbound.set_index('year')
    #
    # annual_shipments_inbound = annual_shipments_inbound.sort_index()
    #
    # annual_shipments_inbound = pd.concat([annual_shipments_inbound, additional_info_df], axis=1)
    # annual_shipments_inbound = annual_shipments_inbound.dropna()
    # # MEHOINUSGAA646N does not account for inflation.
    # annual_shipments_inbound = annual_shipments_inbound.drop(columns=['MEHOINUSGAA646N'])
    # annual_shipments_inbound['MEHOINUSGAA672N'] = annual_shipments_inbound['MEHOINUSGAA672N'].apply(lambda x: x * 10e-6)

    annual_shipments_inbound = combine_features(annual_shipments_in, external_features)

    # Multiplying inflation factor to convert 2012 base data to 2017 base
    # implicit_deflate['year'] = pd.to_datetime(implicit_deflate.index)
    # implicit_deflate['year'] = implicit_deflate['year'].dt.year
    # implicit_deflate = implicit_deflate.set_index('year')

    # annual_shipments_inbound = pd.concat([annual_shipments_inbound, implicit_deflate], axis=1)
    # annual_shipments_inbound = annual_shipments_inbound.rename(columns={'A191RD3A086NBEA': 'deflate'})
    # annual_shipments_inbound = annual_shipments_inbound.dropna()
    # annual_shipments_inbound = annual_shipments_inbound.sort_index()
    # annual_shipments_inbound['conversion_factor'] = annual_shipments_inbound['deflate'].apply(
    #     lambda x: float(100.0 / x))
    # annual_shipments_inbound['conversion_factor'].values[annual_shipments_inbound['conversion_factor'] < 1.0] = float(
    #     1.0)
    # value_columns = [col for col in annual_shipments_inbound.columns if 'val' in col]
    #
    # for col in value_columns:
    #     annual_shipments_inbound[col] = annual_shipments_inbound[col] * annual_shipments_inbound['conversion_factor']
    #
    # annual_shipments_inbound = annual_shipments_inbound.drop(columns=['deflate', 'conversion_factor'])

    # annual_shipments_inbound = apply_deflation(annual_shipments_inbound, implicit_deflate)
    # annual_shipments_inbound.to_csv('Georgia_Annual_Inbound_Shipments_2012-2023.csv')
    # annual_shipments_inbound.to_csv('../freight_data/processed/Georgia_AIS_2012-2023_inflation_adjusted.csv')
    annual_shipments_inbound.to_csv('../freight_data/processed/Georga_AIS_2012-2023_minus_inflation.csv')

    print("Time taken:", time.time() - st)
