#!/usr/bin/env python

import pandas as pd
import argparse
from pathlib import Path

from print_dataset_stats import print_dataset_stats

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset."""
    df = pd.read_csv(csv_path).sort_values('time_to_tca', ascending=False)
    return df

def get_near_final_events(events, expected_size):
    """Get events with latest CDM < 2 day to TCA."""
    latest_cdms = events.nth(0)
    ratio = (latest_cdms['time_to_tca'] < 2).mean()
    near_final_cdm = latest_cdms[latest_cdms['time_to_tca'] < 2]
    near_final_events = near_final_cdm.sample(n=int(expected_size * ratio))
    return near_final_events

def get_early_events(events, expected_size):
    second_to_last_cdms = events.nth(1)
    ratio = (second_to_last_cdms['time_to_tca'] >= 2).mean()
    early_enough_cdm = second_to_last_cdms[second_to_last_cdms['time_to_tca'] >= 2]
    early_events = early_enough_cdm.sample(n=int(expected_size * ratio))
    return early_events

def get_additional_events_ids(df, current_ids, expected_size):
    """Get additional events to reach the desired validation set size."""
    remaining_size = expected_size - len(current_ids)
    if remaining_size <= 0:
        return current_ids
    
    available_events = df[~df['event_id'].isin(current_ids)]['event_id'].unique()
    sample_size = min(remaining_size, len(available_events))
    new_events_ids = pd.Series(available_events).sample(n=sample_size)
    return pd.Index(current_ids).append(pd.Index(new_events_ids))

def print_final_stats(df, train_df, val_df):
    print_dataset_stats(df, 'Full')
    print()
    print_dataset_stats(train_df, 'Train')
    print()
    print_dataset_stats(val_df, 'Validation')

def save_dataset(df, path):
    """Save dataset to CSV file."""
    df.to_csv(path, index=False)

def split_dataset(csv_path, val_percent):
    csv_path = Path(csv_path)
    df = load_and_prepare_data(csv_path)    
        
    events = df.groupby('event_id', as_index=False)
    val_expected_size = int(len(events) * (val_percent)/100)

    val_near_final = get_near_final_events(events, val_expected_size)
    val_early = get_early_events(events, val_expected_size)

    val_ids = pd.concat([val_near_final, val_early])['event_id'].unique()
    val_ids = get_additional_events_ids(df, val_ids, val_expected_size)

    train_df = df[~df['event_id'].isin(val_ids)]
    val_df = df[df['event_id'].isin(val_ids)]

    print_final_stats(df, train_df, val_df)

    dir_path = csv_path.parent
    base_name = csv_path.stem

    save_dataset(train_df, dir_path / f'{base_name}_train.csv')  
    save_dataset(val_df, dir_path / f'{base_name}_val.csv')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train and validation sets')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file to split')
    parser.add_argument('-p', '--percent', type=int, required=True, choices=range(0, 101), help='Percentage of the dataset to be used for validation')
    args = parser.parse_args()
    
    split_dataset(args.file, args.percent)