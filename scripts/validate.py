#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from utils import (
    print_header, set_global_options,
    load_data, print_action, load_model,
    save_data, get_high_risk_threshold
)
from train import (
    resample_df, normalize_df, create_events_series,
    print_predict_stats
)

def validate_classification(input_file, output_dir, models_dir):
    models_dir = os.path.join(models_dir, 'training')
    print_header("Iniciando validação de CLASSIFICAÇÃO")
    set_global_options()

    output_dir = os.path.join(output_dir, "validation")
    df = load_data(input_file)
    
    true_risks = df.groupby('event_id').tail(1)
    df = df.groupby('event_id', group_keys=False).apply(lambda x: x.iloc[:-1])  

    resampler = load_model(models_dir, 'resampler')
    df = resample_df(df, resampler)
    save_data(df, os.path.join(output_dir, "resampled.csv"))
    print_action("Removendo resampler")
    del resampler

    scaler = load_model(models_dir, 'scaler')
    df = normalize_df(df, scaler) 
    save_data(df, os.path.join(output_dir, "normalized.csv"))
    print_action("Removendo scaler")
    del scaler

    km = load_model(models_dir, 'km')
    cluster_risk_mapping = load_model(models_dir, 'cluster_risk_mapping')
    
    print_action("Mapeamento de risco carregado:")
  
    df_classified = predict_clusters_and_classify(df, km, cluster_risk_mapping)
    
    df_classified = df_classified.sort_values(['time_to_tca'])
    save_data(df_classified, os.path.join(output_dir, "classified_events.csv"))
    print_action("Removendo km")
    del km    
    
    print_header("Avaliação de Desempenho do Classificador")

    final_risk_per_event = true_risks.groupby('event_id')['risk'].max()
    true_labels = final_risk_per_event.apply(
        lambda x: 'High' if x > get_high_risk_threshold() else 'Low/Medium'
    ).rename('true_risk_label')

    predicted_labels = df_classified.reset_index().drop_duplicates(subset=['event_id'])[
        ['event_id', 'risk_label']
    ].set_index('event_id')['risk_label']
    
    print_predict_stats(true_labels, predicted_labels, cluster_risk_mapping)

    print_header("Concluído")
    
def predict_clusters_and_classify(df, km, cluster_risk_mapping):
    print_action("Prevendo clusters e classificando novos eventos")
    events_series = create_events_series(df)
    y_pred = km.predict(events_series)
    unique_events = df.index.get_level_values('event_id').unique()
    event_to_cluster_map = dict(zip(unique_events, y_pred))    
    df_reset = df.reset_index()
    df_reset['cluster'] = df_reset['event_id'].map(event_to_cluster_map)
    df_reset['risk_label'] = df_reset['cluster'].map(cluster_risk_mapping)
    return df_reset.set_index(['event_id', 'time_to_tca'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and Classify new events')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to validate')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the directory to save validation results')
    parser.add_argument('-m', '--models', type=str, required=True, help='Path to the directory where models are saved')
    args = parser.parse_args()
    
    validate_classification(args.input, args.output, args.models)