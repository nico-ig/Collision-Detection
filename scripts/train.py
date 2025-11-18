#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from sklearn.metrics import confusion_matrix, classification_report 
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.preprocessing import TimeSeriesResampler
from utils import (
    print_header, print_section, set_global_options,
    load_data, print_action, get_max_iter,
    save_models, get_seed, get_kmeans_metric,
    print_descriptive_stats,
    save_data, get_high_risk_threshold,
    get_njobs
) 

def fit_models(input_file, output_dir, n_clusters):
    print_header("Iniciando treinamento")
    set_global_options()
    output_dir = os.path.join(output_dir, "training")
    df = load_data(input_file)

    max_size = df.groupby(level=0).size().max()
    resampler = TimeSeriesResampler(sz=max_size)
    df = resample_df(df, resampler)
    save_data(df, os.path.join(output_dir, "resampled.csv"))
    save_models({"resampler": resampler}, output_dir)
    print_action("Removendo resampler")
    del resampler

    scaler = TimeSeriesScalerMeanVariance(
        per_timeseries=False,
        per_feature=True,
    )
    df = normalize_df(df, scaler)
    save_data(df, os.path.join(output_dir, "normalized.csv"))
    save_models({"scaler": scaler}, output_dir)
    print_descriptive_stats(df, "Dataset normalizado")
    print_action("Removendo scaler")
    del scaler

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=get_kmeans_metric(),
        random_state=get_seed(),
        max_iter=get_max_iter(),
        n_jobs=get_njobs(),
        verbose=True,
    )

    df = get_clusters(df, km)
    df = df.sort_values(['time_to_tca'])
    print_clusters_stats(df.drop(columns=['event_id']))
    save_models({"km": km}, output_dir)
    save_data(df, os.path.join(output_dir, "clusterized.csv"))
    print_action("Removendo km")
    del km

    print_header("Analisando e Mapeando Risco dos Clusters")
    
    max_risk_per_event = df.groupby('event_id')['risk'].max()
    event_to_cluster_map = df.reset_index()[['event_id', 'cluster']].drop_duplicates().set_index('event_id')
    cluster_risk_data = max_risk_per_event.to_frame(name='max_risk').join(event_to_cluster_map)
    cluster_risk_stats = cluster_risk_data.groupby('cluster')['max_risk'].agg(['mean', 'median', 'max', 'count'])
    
    print_section("Estatísticas de Risco (baseado no Risco Máx. do Evento) por Cluster:")
    
    cluster_risk_mapping = {}
    for cluster_id, stats in cluster_risk_stats.iterrows():
        if stats['median'] > get_high_risk_threshold():
            cluster_risk_mapping[cluster_id] = 'High'
        else:
            cluster_risk_mapping[cluster_id] = 'Low/Medium'
    
    true_labels = max_risk_per_event.apply(
        lambda x: 'High' if x > get_high_risk_threshold() else 'Low/Medium'
    ).rename('true_risk_label')

    df = df.reset_index()

    df['risk_label'] = df['cluster'].map(lambda x: 'High' if cluster_risk_mapping[x] == 'High' else 'Low/Medium')
    predicted_labels = df.reset_index().drop_duplicates(subset=['event_id'])[
        ['event_id', 'risk_label']
    ].set_index('event_id')['risk_label']

    print_section(f"Mapeamento de Risco (Limiar de Média > {get_high_risk_threshold()}):")
    display(cluster_risk_mapping)
    
    print_predict_stats(true_labels, predicted_labels, cluster_risk_mapping)

    save_models({"cluster_risk_mapping": cluster_risk_mapping}, output_dir)
    print_action("Mapeamento de risco do cluster salvo.")

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

def print_predict_stats(true_labels, predicted_labels, cluster_risk_mapping):
    df_metrics = true_labels.to_frame().join(predicted_labels)
    y_true = df_metrics['true_risk_label']
    y_pred = df_metrics['risk_label']
    
    y_true_unique = set(y_true.dropna())
    y_pred_unique = set(y_pred.dropna())
    cluster_risk_mapping_unique = set(cluster_risk_mapping.values())

    unique_labels = sorted(list(y_true_unique.union(y_pred_unique)))
    target_names = sorted(list(cluster_risk_mapping_unique))
    
    if len(unique_labels) != len(target_names):
        target_names = unique_labels

    target_names = sorted(target_names)
    target_names = [str(x) for x in target_names]

    y_true = y_true.values.astype(str)
    y_pred = y_pred.values.astype(str)

    print_section("Matriz de Confusão")
    cm = confusion_matrix(
        y_true, 
        y_pred, 
        labels=target_names
    )
    cm = pd.DataFrame(cm, index=[f'Real: {c}' for c in target_names], columns=[f'Previsto: {c}' for c in target_names])
    display(cm)

    print_section("Relatório de Classificação (Precision, Recall, F1-Score)")
    report = classification_report(y_true, y_pred, labels=target_names, zero_division=0)
    display(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit ARIMAX models')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to use on the fit')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the directory to save the models')
    parser.add_argument('-n', '--n_clusters', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()
    
    fit_models(args.input, args.output, args.n_clusters)
