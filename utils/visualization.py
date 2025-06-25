import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time
import plotly.graph_objects as go
from config import commodity_mapping, feature_units


def filter_georgia_value(data_frame, year, commodity_code, feature_type):
    relevant_cols = [col for col in data_frame.columns if 'dms_dest' in col or 'sctg2' in col]
    year_specific_col = [col for col in data_frame.columns if f'{feature_type}_{year}' in col]
    data_frame = data_frame[(data_frame[relevant_cols[0]] == 13) & (data_frame['sctg2'] == commodity_code)]
    data_frame = data_frame[relevant_cols + year_specific_col]
    data_frame = data_frame.dropna()
    data_frame = data_frame.drop(columns=[relevant_cols[0], 'sctg2'])
    data_frame = data_frame.sum(axis=0)
    return data_frame


def commodity_based_value_trend(commodity_code, feature_type):
    df_2012 = pd.read_csv('../freight_data/raw/FAF5.6.1_Reprocessed_1997-2012_State.csv')
    df_2012faf4 = pd.read_csv('../freight_data/raw/FAF4.5.1_State.csv')
    df_2013 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2013.csv')
    df_2014 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2014.csv')
    df_2015 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2015.csv')
    df_2016 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2016.csv')
    df_2017faf4 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2017.csv')
    df_2017faf5 = pd.read_csv('../freight_data/raw/FAF5.6.1_State.csv')
    df_2018faf4 = pd.read_csv('../freight_data/raw/FAF4.5.1_State_2018.csv')
    df_2018faf5 = pd.read_csv('../freight_data/raw/FAF5.6.1_State_2018-2023.csv')
    df_2019 = df_2018faf5.copy()

    df_2012 = filter_georgia_value(df_2012, 2012, commodity_code, feature_type=feature_type)
    df_2012faf4 = filter_georgia_value(df_2012faf4, 2012, commodity_code, feature_type=feature_type)
    df_2013 = filter_georgia_value(df_2013, 2013, commodity_code, feature_type=feature_type)
    df_2014 = filter_georgia_value(df_2014, 2014, commodity_code, feature_type=feature_type)
    df_2015 = filter_georgia_value(df_2015, 2015, commodity_code, feature_type=feature_type)
    df_2016 = filter_georgia_value(df_2016, 2016, commodity_code, feature_type=feature_type)
    df_2017faf4 = filter_georgia_value(df_2017faf4, 2017, commodity_code, feature_type=feature_type)
    df_2017faf5 = filter_georgia_value(df_2017faf5, 2017, commodity_code, feature_type=feature_type)
    df_2018faf4 = filter_georgia_value(df_2018faf4, 2018, commodity_code, feature_type=feature_type)
    df_2018faf5 = filter_georgia_value(df_2018faf5, 2018, commodity_code, feature_type=feature_type)
    df_2019 = filter_georgia_value(df_2019, 2019, commodity_code, feature_type=feature_type)

    plot_data_dict = {
        2012: {'FAF4': df_2012faf4.iloc[0], 'FAF5': df_2012.iloc[0]},
        2013: {'FAF4': df_2013.iloc[0], 'FAF5': np.nan},
        2014: {'FAF4': df_2014.iloc[0], 'FAF5': np.nan},
        2015: {'FAF4': df_2015.iloc[0], 'FAF5': np.nan},
        2016: {'FAF4': df_2016.iloc[0], 'FAF5': np.nan},
        2017: {'FAF4': df_2017faf4.iloc[0], 'FAF5': df_2017faf5.iloc[0]},
        2018: {'FAF4': df_2018faf4.iloc[0], 'FAF5': df_2018faf5.iloc[0]},
        2019: {'FAF4': np.nan, 'FAF5': df_2019.iloc[0]},
    }
    return plot_data_dict


def plot_from_data_dict(plot_data, feature_type, commodity_code):
    plot_df = pd.DataFrame.from_dict(plot_data, orient='index')
    plot_df.index.name = 'Year'

    plot_df.index = plot_df.index.astype(int)

    plt.figure()
    plot_df.plot(ax=plt.gca(), marker='o', linewidth=2)

    plt.title(f'{feature_type}/Years: {commodity_mapping[commodity_code]}\n Georgia (FAF4 vs FAF5 Methodologies)')
    plt.xlabel('Year')
    plt.ylabel(f'Total shipment {feature_type}\n ({commodity_mapping[commodity_code]} in Georgia)')
    plt.grid(True)
    plt.legend(title='FAF Version')
    plt.xticks(plot_df.index)
    plt.tight_layout()
    plt.show()


def plot_forecast_vs_actual(y_train, y_valid, predictions, feature_type, commodity_code, library, save_path=False):
    plot_data_dict = {}
    for year, value in y_train.items():
        year_str = year.year if hasattr(year, 'year') else int(str(year))
        plot_data_dict[year_str] = {
            'Estimates': value,
            'Forecast': value,
        }

    for year, value in y_valid.items():
        year_str = year.year if hasattr(year, 'year') else int(str(year))
        plot_data_dict.setdefault(year_str, {})['Estimates'] = value

    for year, value in predictions.items():
        year_str = year.year if hasattr(year, 'year') else int(str(year))
        plot_data_dict.setdefault(year_str, {})['Forecast'] = value

    plot_df = pd.DataFrame.from_dict(plot_data_dict, orient='index')
    plot_df.index.name = 'Year'
    plot_df.index = plot_df.index.astype(int)

    if library == 'matplotlib':
        plt.figure()
        ax = plt.gca()
        ax.plot(plot_df.index, plot_df['Forecast'], marker='o', linewidth=2,
                label='Forecast', color='orange',
                # linestyle='--',
                )

        ax.plot(plot_df.index, plot_df['Estimates'], marker='o', linewidth=2,
                label='Estimates', color='blue')

        plt.title(
            f'Forecast of {feature_type} ({feature_units[feature_type]}) of {commodity_mapping[commodity_code]}\n in domestic shipments to Georgia')
        plt.xlabel('Year')
        plt.ylabel(f'Total shipment {feature_type} ({feature_units[feature_type]})\n ({commodity_mapping[commodity_code]} Georgia)')
        plt.grid(True)
        plt.legend(title='')
        plt.xticks(plot_df.index)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'../results/forecast_{feature_type}_{commodity_code}.png')
        plt.show()

    elif library == 'plotly':
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['Forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=2)  # , dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=plot_df.index,
            y=plot_df['Estimates'],
            mode='lines+markers',
            name='Estimates',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f'Forecast of {feature_type} ({feature_units[feature_type]}) of {commodity_mapping[commodity_code]}<br>in domestic shipments to Georgia',
            xaxis_title='Year',
            yaxis_title=f'Total shipment {feature_type} ({feature_units[feature_type]})<br>({commodity_mapping[commodity_code]} Georgia)',
            legend_title='',
            template='plotly_white',
            margin=dict(l=80, r=40, t=80, b=50)  # Increase left margin
        )

        if save_path:
            fig.write_image(f'../results/forecast_{feature_type}_{commodity_code}.png')


if __name__ == "__main__":
    st = time.time()
    # plot_data_trend = commodity_based_value_trend(commodity_code=5, feature_type='value')
    # plot_from_data_dict(plot_data_trend, commodity_code=5, feature_type='Value')
    #
    # plot_data_trend = commodity_based_value_trend(commodity_code=8, feature_type='value')
    # plot_from_data_dict(plot_data_trend, commodity_code=8, feature_type='Value')
    #
    # plot_data_trend = commodity_based_value_trend(commodity_code=9, feature_type='value')
    # plot_from_data_dict(plot_data_trend, commodity_code=9, feature_type='Value')
    #
    # plot_data_trend = commodity_based_value_trend(commodity_code=21, feature_type='value')
    # plot_from_data_dict(plot_data_trend, commodity_code=21, feature_type='Value')

    plot_data_trend = commodity_based_value_trend(commodity_code=5, feature_type='tons')
    plot_from_data_dict(plot_data_trend, commodity_code=5, feature_type='Tons')

    plot_data_trend = commodity_based_value_trend(commodity_code=8, feature_type='tons')
    plot_from_data_dict(plot_data_trend, commodity_code=8, feature_type='Tons')

    plot_data_trend = commodity_based_value_trend(commodity_code=9, feature_type='tons')
    plot_from_data_dict(plot_data_trend, commodity_code=9, feature_type='Tons')

    plot_data_trend = commodity_based_value_trend(commodity_code=21, feature_type='tons')
    plot_from_data_dict(plot_data_trend, commodity_code=21, feature_type='Tons')

    print('Runtime:', time.time() - st)
