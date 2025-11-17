#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import os
import argparse
import statsmodels.api as sm
from IPython.display import display
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesResampler
from utils import print_header, print_section, set_global_options, \
    load_data, print_action, get_orders_grid, get_max_iter, get_method, \
    save_models, get_seed, get_n_clusters, get_kmeans_metric, print_descriptive_stats, \
    save_data, get_enforce_stationarity, get_enforce_invertibility, \
    get_missing_values_interpolation, get_time_conflict_resolution, get_method, get_mle_regression, get_njobs

#3. Selectionar o melhor modelo para cada cluster
#3. Script de predicao (ver em qual cluster o evento cai e fazer a predicao, aproveitar os dados da série nova, usar o msm scaler)
#gerar matriz de confusão

def fit_models(input_file, output_dir):
    print_header("Iniciando Treinamento")
    set_global_options()
    df = load_data(input_file)

    max_size = df.groupby(level=0).size().max()
    resampler = TimeSeriesResampler(sz=max_size)
    df = resample_df(df, resampler)
    save_data(df, os.path.join(output_dir, "resampled_data.csv"))
    save_models({"resampler": resampler}, os.path.join(output_dir, "resampler.pkl"))
    print_action("Removendo resampler")
    del resampler

    scaler = TimeSeriesScalerMeanVariance(
        per_timeseries=False,
        per_feature=True,
    )
    df = normalize_df(df, scaler)
    save_models({"scaler": scaler}, output_dir)
    print_descriptive_stats(df, "Dataset normalizado")
    print_action("Removendo scaler")
    del scaler

    km = TimeSeriesKMeans(
        n_clusters=get_n_clusters(),
        metric=get_kmeans_metric(),
        random_state=get_seed(),
        max_iter=get_max_iter(),
        n_jobs= get_njobs(),
        n_jobs_barycenter=1,
        verbose=True,
    )

    df = get_clusters(df, km)
    df = df.sort_values(['time_to_tca'])
    print_clusters_stats(df.drop(columns=['event_id']))
    save_models({"km": km}, output_dir)
    save_data(df, os.path.join(output_dir, "clusterized_data.csv"))
    print_action("Removendo km")
    del km

    cluster_cnt = 0
    for cluster, group in df.groupby('cluster'):
        print_header(f"Treinando modelos para cluster {cluster} ({cluster_cnt} / {get_n_clusters()})")
        cluster_model = train_cluster_models(group)
        save_models(cluster_model, os.path.join(output_dir, f"cluster_{cluster}"))
        print_action("Removendo cluster")
        del cluster_model
        cluster_cnt += 1

    save_data(df, os.path.join(output_dir, "final_data.csv")) # This should be the same as clusterized_data.csv
    print_header("Concluído")

def resample_df(df, resampler):
    print_action("Reamostrando dados")
    resampled_dfs = pd.DataFrame()
    columns = df.columns
    for event_id, group in df.groupby('event_id'):
        first_time_to_tca = group.index.get_level_values('time_to_tca').min()
        last_time_to_tca = group.index.get_level_values('time_to_tca').max()
        series_3d = transform_3d(group)
        resampled_3d = resampler.fit_transform(series_3d)
        resampled_group = transform_to_2d_dataframe(resampled_3d, columns)
        resampled_group['event_id'] = event_id
        new_time_to_tca = pd.date_range(
            start=first_time_to_tca,
            end=last_time_to_tca,
            periods=resampled_3d.shape[1]
        )
        resampled_group['time_to_tca'] = new_time_to_tca
        resampled_group = resampled_group.bfill().ffill()
        resampled_dfs = pd.concat([resampled_dfs, resampled_group])
    resampled_dfs = resampled_dfs.set_index(['event_id', 'time_to_tca'])
    return resampled_dfs

def transform_to_2d_dataframe(series_3d, columns):
    n_series, n_timesteps, n_features = series_3d.shape
    series_2d = series_3d.reshape(n_series * n_timesteps, n_features)
    return pd.DataFrame(series_2d, columns=columns)

def transform_3d(df):
    n_series = 1
    n_timesteps = df.shape[0]
    n_features = df.shape[1]
    return df.values.reshape(n_series, n_timesteps, n_features)

def normalize_df(df, scaler):
    risk = df['risk']
    index = df.index
    feature_df = df.drop(columns=['risk'])
    events_series = create_events_series(feature_df)
    normalized_events_series = scaler.fit_transform(events_series)
    normalized_df = transform_to_2d_dataframe(normalized_events_series, feature_df.columns)
    normalized_df['risk'] = risk.values
    normalized_df = normalized_df.set_index(index)
    return normalized_df
    
def create_events_series(df):
    events_list = []
    for _, group in df.groupby('event_id'):
        event_series = transform_3d(group)
        events_list.append(event_series)
    events_series = np.concatenate(events_list, axis=0)
    return events_series

def get_clusters(df, km):
    print_header("Iniciando Clusterização")
    events_series = create_events_series(df)
    y_pred = km.fit_predict(events_series)
    unique_events = df.index.get_level_values('event_id').unique()
    cluster_mapping = dict(zip(unique_events, y_pred))
    df = df.reset_index()
    df['cluster'] = df['event_id'].map(cluster_mapping)
    print_action("Atualizando index (cluster, time_to_tca)")
    df = df.set_index(['cluster', 'time_to_tca'])
    return df

def print_clusters_stats(df):
    range_stats = df.groupby('cluster').agg(
        lambda x: (x.std() / (x.max() - x.min())) * 100
    )
    print_header("Clusters")
    print_section("Amplitude do desvio padrão relativa por cluster (%)")
    display(range_stats)
    for cluster, group in df.groupby('cluster'):
        print_descriptive_stats(group, f"Cluster {cluster}")

def train_cluster_models(cluster):
    cluster = remove_unecessary_columns(cluster)
    cluster, interval = normalize_cluster_frequency(cluster)
    trained_models = train_models(cluster)
    models = {
        'cluster': cluster,
        'interval': interval,
        'trained_models': trained_models
    }
    return models

def remove_unecessary_columns(df):
    print_action("Resetando index")
    df = df.reset_index()
    print_action("Removendo coluna 'cluster' do dataset")
    df = df.drop(columns=['cluster'])
    return df

def normalize_cluster_frequency(df):
    method = get_missing_values_interpolation()
    time_bucket_conflict = get_time_conflict_resolution()
    print_action("Mudando index para (time_to_tca)")
    df = df.reset_index()
    df = df.set_index('time_to_tca').sort_index()
    interval = df.index.sort_values().diff().dropna().mean()
    df = df.resample(interval, origin='start').agg(time_bucket_conflict).interpolate(method=method)
    df = df.sort_index()
    print_action(f"Eventos reamostrados com intervalo de {interval} horas")
    return df, interval

def train_models(df):
    endog = df[['risk']]
    exog = df.drop(columns=['risk'])
    display(endog.head())
    display(exog.head())
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