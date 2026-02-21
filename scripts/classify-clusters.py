#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from sklearn.metrics import confusion_matrix, classification_report
from utils import (
    print_header, print_section, set_global_options,
    load_data, print_action, get_max_iter,
    save_models, get_seed, get_kmeans_metric,
    print_descriptive_stats,
    save_data, get_high_risk_threshold,
    get_njobs
) 

def classify(input_file, output_dir):
    print_header("Iniciando treinamento")
    set_global_options()
    output_dir = os.path.join(output_dir, "classify")
    df = load_data(input_file)

    print_clusters_stats(df)
    cluster_risk_mapping = classify_clusters(df)
    display(cluster_risk_mapping)
    true_labels = create_true_labels(df)
    print_predict_stats(df, cluster_risk_mapping, true_labels)

    save_models({"cluster_risk_mapping": cluster_risk_mapping}, output_dir)
    print_action("Mapeamento de risco do cluster salvo.")

    print_header("Concluído")

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
        last_events = group.groupby('event_id').tail(3)
        cluster_risk_mapping[cluster_id] = 'High' if is_cluster_high_risk(last_events) else 'Low/Medium'
    return cluster_risk_mapping

def is_cluster_high_risk(df):
    cluster_value = df.groupby('event_id').last()['risk'].quantile(0.95)
    threshold = get_high_risk_threshold()
    display(f"Cluster value: {cluster_value}, Threshold: {threshold}")
    return cluster_value >= threshold

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
    unique_mapped = set(cluster_risk_mapping.values())

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
    args = parser.parse_args()
    
    classify(args.input, args.output)
