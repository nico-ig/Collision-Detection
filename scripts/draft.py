#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import tsod
import tsod.hampel
from IPython.display import display
import statsmodels.tsa.stattools as stsm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

from utils import (
    print_header, print_section, print_descriptive_stats,
    load_data, get_high_risk_threshold, set_global_options,
    load_data_full, get_high_risk_threshold
)

import warnings
warnings.filterwarnings('ignore', message='Series.__setitem__ treating keys as positions is deprecated')

def eda(file):
    print_header("Iniciando EDA")

    set_global_options()

   # Dados da tabela
    data = {
        "Feature": [
            "risk", "max_risk_scaling", "mahalanobis_distance", "c_sigma_t", "max_risk_estimate",
            "c_sigma_rdot", "miss_distance", "c_position_covariance_det", "c_sigma_n", "c_sigma_r",
            "c_obs_used", "c_sigma_ndot", "relative_position_n", "c_recommended_od_span",
            "relative_position_r", "c_sedr", "SSN", "c_crdot_t", "relative_speed",
            "c_time_lastob_end", "c_time_lastob_start", "c_cr_area_over_mass", "c_cd_area_over_mass"
        ],
        "Lag0": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "Lag1": [0.4254,0.3090,0.4245,0.4428,0.3690,0.4424,0.3186,0.3748,0.3909,0.4204,0.5237,0.4002,0.3285,0.4697,0.3463,0.5479,0.5142,0.2949,0.3201,0.3918,0.3843,0.4894,0.5201],
        "Lag2": [0.2853,0.1587,0.2868,0.3104,0.2247,0.3102,0.1862,0.2268,0.2259,0.2674,0.3105,0.2323,0.1926,0.2509,0.1910,0.3390,0.2758,0.1510,0.1558,0.1045,0.1126,0.2523,0.3126],
        "Lag3": [0.1849,0.0711,0.1867,0.2205,0.1358,0.2205,0.1204,0.1346,0.1164,0.1652,0.1668,0.1286,0.1268,0.1200,0.0932,0.1983,0.1240,0.0697,0.0556,-0.0334,-0.0204,0.1124,0.1843],
        "Lag4": [0.0844,0.0028,0.0913,0.1209,0.0428,0.1210,0.0194,0.0491,0.0240,0.0694,0.0524,0.0362,0.0231,0.0199,-0.0007,0.0867,0.0213,0.0012,-0.0181,-0.0664,-0.0579,0.0143,0.0825],
        "Lag5": [0.0063,-0.0404,0.0184,0.0425,-0.0257,0.0425,-0.0430,-0.0145,-0.0436,-0.0052,-0.0319,-0.0306,-0.0400,-0.0506,-0.0654,-0.0049,-0.0556,-0.0420,-0.0656,-0.0976,-0.0908,-0.0565,-0.0026],
        "Lag6": [-0.0578,-0.0689,-0.0387,-0.0231,-0.0713,-0.0228,-0.0724,-0.0628,-0.0909,-0.0616,-0.0919,-0.0802,-0.0707,-0.1029,-0.1046,-0.0765,-0.1108,-0.0715,-0.0961,-0.1161,-0.1150,-0.1052,-0.0656]
    }

    df_acf = pd.DataFrame(data)

    # Transpor para plot
    df_plot = df_acf.set_index("Feature").T

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    for feature in df_plot.columns:
        ax.plot(df_plot.index, df_plot[feature], marker='o', label=feature.replace('_', '-'))

    ax.set_xlabel("Lag")
    ax.set_ylabel("Auto correlação")
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=8)  # legenda abaixo
    plt.tight_layout()
    fig.savefig(f"auto-correlacao.pgf") 


    df = load_data_full(file, False)
    
    df = df.reset_index()
    biggest_series_id = df.groupby('event_id').size().idxmax()

    biggest_series = df[df['event_id'] == biggest_series_id][['risk', 'time_to_tca', 'c_sigma_rdot']]

    fig, ax = plt.subplots(figsize=(10,6))

    for col in biggest_series.columns:
        if col != "time_to_tca":
            label = col.replace('_', '-')
            ax.plot(biggest_series["time_to_tca"], biggest_series[col], marker='o', label=label)

    ax.set_xlabel("Tempo até a colisão")
    ax.set_ylabel("Valor")
    fig.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.gca().invert_xaxis()
    fig.savefig(f"exemplo-serie.pgf") 

    print_header("Concluído")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file to use on the EDA')
    args = parser.parse_args()
    
    eda(args.file)