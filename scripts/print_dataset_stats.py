#!/usr/bin/env python
import pandas as pd
import argparse

def print_dataset_stats(df, name):
    print(f"{name} set characteristics:")
    print(f"- Events with latest CDM < 2 day to TCA: {int((df['time_to_tca'] < 2).mean() * 100)}%")
    print(f"- Events with second to last CDM >= 2 days to TCA: {int((df['time_to_tca'] >= 2).mean() * 100)}%")
    print(f"- Number of events: {len(df['event_id'].unique())} from {len(df)} rows")
    print(f"- Average rows per event: {int(len(df)/df['event_id'].nunique())}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print dataset stats')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    print_dataset_stats(df, 'Full')