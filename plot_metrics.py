import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np


def get_avg_smape_for_pollutant(results_dict, city_name, pollutant_name):
    """
    calculate the mean smape for city and pollutant
    :param results_dict: (dict) return value of train_eval_model.run()
    :param city_name: (str) 'London' or 'Beijing'
    :param pollutant_name: (str) from ['PM10', 'PM2.5', 'O3', 'CO', 'SO2', 'NO2']

    :return: average smape for city and pollutant
    """
    smape_values = []
    for key, metrics in results_dict.items():
        if city_name in key and pollutant_name in key:
            smape_values.append(metrics['smape_score'])

    return np.mean(smape_values)


def plot_smape(results_dict, filename):
    """
    creates a plot (.html) saved locally for the smape metric
    :param results_dict: (dict) return value of train_eval_model.run()
    :param filename: (str) filename to save the .html plot to

    :return: None
    """
    width = 0.01
    pollutants = ['PM10', 'PM2.5', 'O3', 'CO', 'SO2', 'NO2']
    fig = go.Figure()

    london_smape = [get_avg_smape_for_pollutant(results_dict, 'London', p) for p in pollutants]
    fig.add_trace(go.Bar(
        y=[width * 2, width * 4.5, width * 7, width * 9.5, width * 12, width * 14.5],
        x=london_smape,
        text=[str(round(v, 1)) + '%' for v in london_smape],
        width=width,
        textposition='auto',
        marker_color='#F082CF',  # marker color can be a single color value or an iterable
        orientation='h',
        name='London'
    ))

    beijing_smape = [get_avg_smape_for_pollutant(results_dict, 'Beijing', p) for p in pollutants]
    fig.add_trace(go.Bar(
        y=[width * 1, width * 3.5, width * 6, width * 8.5, width * 11, width * 13.5],
        x=beijing_smape,
        text=[str(round(v, 1)) + '%' for v in beijing_smape],
        width=width,
        textposition='auto',
        marker_color='#4AB6EC',  # marker color can be a single color value or an iterable
        orientation='h',
        name='Beijing'
    ))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[i * 10 for i in range(21)],
            ticktext=[i * 10 for i in range(21)]
        )
    )

    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[width * 1.5, width * 4, width * 6.5, width * 9, width * 11.5, width * 14],
            ticktext=pollutants,
        )
    )

    fig.update_xaxes(title='SMAPE (%)', title_font=dict(size=14))
    fig.update_yaxes(title='Pollutant', title_font=dict(size=14))
    fig.update_traces(textposition='inside', textfont_size=14)
    fig.update_layout(title_text="AVG SMAPE values")

    plot(fig, filename=filename)
