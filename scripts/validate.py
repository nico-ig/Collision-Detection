#!/usr/bin/env python3 

import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from utils import (
    print_header, set_global_options,
    load_data, print_action, load_model,
    save_data
)
from train import (
    resample_df, normalize_df, create_events_series,
    print_predict_stats, create_true_labels, assign_cluster
)

def validate_classification(input_file, output_dir, models_dir):
    models_dir = os.path.join(models_dir, 'training')
    print_header("Iniciando validação de CLASSIFICAÇÃO")
    set_global_options()

    output_dir = os.path.join(output_dir, "validation")
    df = load_data(input_file)
    
    true_labels = create_true_labels(df)
    df = remove_last_events(df)

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
    df = predict_clusters(df, km)
    print_action("Removendo km")
    del km
      
    cluster_risk_mapping = load_model(models_dir, 'cluster_risk_mapping')
    print_predict_stats(df, cluster_risk_mapping, true_labels)

    save_data(df, os.path.join(output_dir, "classified_events.csv"))
    print_header("Concluído")
    
def remove_last_events(df):
    return df.groupby('event_id', group_keys=False).apply(lambda x: x.iloc[:-1]) 

def predict_clusters(df, km):
    print_action("Prevendo clusters e classificando novos eventos")
    
    events_series = create_events_series(df)
    y_pred = km.predict(events_series)
    df = assign_cluster(df, y_pred)
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate and Classify new events')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to validate')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the directory to save validation results')
    parser.add_argument('-m', '--models', type=str, required=True, help='Path to the directory where models are saved')
    args = parser.parse_args()
    
    validate_classification(args.input, args.output, args.models)