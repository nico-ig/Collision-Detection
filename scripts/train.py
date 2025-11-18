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

    cluster_risk_mapping = classify_clusters(df)
    true_labels = create_true_labels(df)
    print_predict_stats(df, cluster_risk_mapping, true_labels)

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
    df = assign_cluster(df, y_pred)
    return df

def assign_cluster(df, clusters):
    unique_events = df.index.get_level_values('event_id').unique()
    cluster_mapping = dict(zip(unique_events, clusters))
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

def classify_clusters(df):
    print_header("Classificando Clusters")
    cluster_risk_mapping = {}
    for cluster_id, group in df.groupby('cluster'):
        print_descriptive_stats(group, f"Cluster {cluster_id}")
        last_events = df.groupby('event_id').tail(3)
        cluster_risk_mapping[cluster_id] = 'High' if is_cluster_high_risk(last_events) else 'Low/Medium'
    return cluster_risk_mapping

def is_cluster_high_risk(df):
    return df['risk'].quantile(0.1) >= get_high_risk_threshold() * 1.2

def print_predict_stats(df, cluster_risk_mapping, true_labels):        
    predicted_labels = create_predicted_labels(df, cluster_risk_mapping)
    
    df_metrics = true_labels.to_frame().join(predicted_labels)
    y_true = df_metrics['true_risk_label']
    y_pred = df_metrics['risk_label']
    
    target_names = prepare_labels_for_classification(y_true, y_pred, cluster_risk_mapping)
    y_true_array = y_true.astype(str).values
    y_pred_array = y_pred.astype(str).values
    
    print_confusion_matrix(y_true_array, y_pred_array, target_names)
    print_classification_report(y_true_array, y_pred_array, target_names)

def create_true_labels(df):
    last_risk = df.groupby('event_id')['risk'].last()
    risk_labels = last_risk.apply(
        lambda x: 'High' if x >= get_high_risk_threshold() else 'Low/Medium'
    )
    risk_labels.name = 'true_risk_label'
    return risk_labels

def create_predicted_labels(df, cluster_risk_mapping):
    event_clusters = df.reset_index().drop_duplicates('event_id').set_index('event_id')['cluster']
    return event_clusters.map(cluster_risk_mapping).rename('risk_label')

def prepare_labels_for_classification(y_true, y_pred, cluster_risk_mapping):
    unique_true = set(y_true.dropna())
    unique_pred = set(y_pred.dropna())
    unique_mapped = set(cluster_risk_mapping.values())
    
    all_unique_labels = sorted(unique_true.union(unique_pred))
    target_names = sorted(unique_mapped)
    
    target_names = ['High', 'Low/Medium']
    
    return [str(label) for label in sorted(target_names)]

def print_confusion_matrix(y_true, y_pred, target_names):
    print_section("Matriz de Confusão")
    cm = confusion_matrix(y_true, y_pred, labels=target_names)
    cm_df = pd.DataFrame(
        cm, 
        index=[f'Real: {label}' for label in target_names],
        columns=[f'Previsto: {label}' for label in target_names]
    )
    display(cm_df)

def print_classification_report(y_true, y_pred, target_names):
    print_section("Relatório de Classificação (Precision, Recall, F1-Score)")
    report = classification_report(
        y_true, y_pred, labels=target_names, zero_division=0
    )
    display(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit ARIMAX models')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to use on the fit')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the directory to save the models')
    parser.add_argument('-n', '--n_clusters', type=int, required=True, help='Number of clusters')
    args = parser.parse_args()
    
    fit_models(args.input, args.output, args.n_clusters)
