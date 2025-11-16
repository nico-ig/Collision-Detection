#!/usr/bin/env python

import pandas as pd
import os
import argparse
import statsmodels.api as sm
from IPython.display import display
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils import print_header, print_section, set_global_options, \
    load_data, print_action, get_orders_grid, get_max_iter, get_method, \
    save_models, get_seed, get_n_clusters, get_kmeans_metric, print_descriptive_stats, \
    get_interval, save_data, get_enforce_stationarity, get_enforce_invertibility, \
    save_models, get_method, get_mle_regression

#conjunto de teste, prepopular o modelo?
#3. Selectionar o melhor modelo para cada cluster
#3. Script de predicao (ver em qual cluster o evento cai e fazer a predicao, aproveitar os dados da série nova, usar o msm scaler)
#gerar matriz de confusão

def fit_models(input_file, output_dir):
    print_header("Iniciando Treinamento")
    set_global_options()
    df = load_data(input_file)

    scaler = TimeSeriesScalerMeanVariance()
    df = normalize_df(df, scaler)
    save_models({"scaler": scaler}, os.path.join(output_dir, "scaler.pkl"))
    print_descriptive_stats(df, "Dataset normalizado")

    km = TimeSeriesKMeans(
        n_clusters=get_n_clusters(),
        metric=get_kmeans_metric(),
        random_state=get_seed(),
        max_iter=get_max_iter(),
        n_jobs=-1,
        verbose=True,
    )
    df = get_clusters(df, km)
    save_models({"km": km}, os.path.join(output_dir, "km.pkl"))

    print_action("Removendo event_id e atualizando index (cluster, time_to_tca)")
    df = df.reset_index()
    df = df.drop(columns=['event_id'])
    df = df.set_index(['cluster', 'time_to_tca'])

    print_header("Clusters")
    print_clusters_stats(df)

    input_dir = os.path.dirname(input_file)
    save_data(df, os.path.join(input_dir, "clusterized_data.csv"))

    cluster_cnt = 0
    for cluster, group in df.groupby('cluster'):
        print_header(f"Treinando modelos para cluster {cluster} ({cluster_cnt} / {get_n_clusters()})")
        cluster_model = train_cluster_models(group)
        save_models(cluster_model, os.path.join(output_dir, f"cluster_{cluster}"))
        cluster_cnt += 1

    print_header("Concluído")

def normalize_df(df, scaler):
    feature_df = df.drop(columns=['risk'])    
    scaled_data = scaler.fit_transform(feature_df)
    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1])
    feature_df = pd.DataFrame(scaled_data, columns=feature_df.columns, index=feature_df.index)
    df = pd.concat([df['risk'], feature_df], axis=1)
    return df

def get_clusters(df, km):
    print_header("Iniciando Clusterização")
    y_pred = km.fit_predict(df)
    df['cluster'] = y_pred
    return df

def print_clusters_stats(df):
    range_stats = df.groupby('cluster').agg(
        lambda x: (x.std() / (x.max() - x.min())) * 100
    )
    print_section("Amplitude do desvio padrão relativa por cluster (%)")
    display(range_stats)
    for cluster, group in df.groupby('cluster'):
        print_descriptive_stats(group, f"Cluster {cluster}")

def train_cluster_models(cluster):
    cluster = normalize_cluster_frequency(cluster)
    print_action("Removendo cluster")
    cluster = cluster.drop(columns=['cluster'])
    models = train_models(cluster)
    return models

def normalize_cluster_frequency(df):
    interval = get_interval()
    print_action("Mudando index para (time_to_tca)")
    df = df.reset_index()
    df = df.set_index('time_to_tca').sort_index()
    df = df.resample(interval, origin='start').mean().bfill()
    print_action(f"Eventos reamostrados com intervalo de {interval} horas")
    return df

def train_models(df):
    endog = df[['risk']]
    exog = df.drop(columns=['risk'])

    models = {}
    orders_grid = get_orders_grid()
    for order in orders_grid:
        print_section(f"Treinando ordem {order}")
        models[order] = fit_single_model(endog, exog, order)
        display(models[order].summary())

    return models

def fit_single_model(endog, exog, order):
    model = sm.tsa.SARIMAX(
        endog=endog, 
        exog=exog, 
        order=order,
        enforce_stationarity=get_enforce_stationarity(),
        enforce_invertibility=get_enforce_invertibility(),
        mle_regression=get_mle_regression(),
    )
    return model.fit(
        maxiter=get_max_iter(),
        method=get_method(),
        disp=False,
        low_memory=True,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit ARIMAX models')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to use on the fit')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the directory to save the models')
    args = parser.parse_args()
    
    fit_models(args.input, args.output)