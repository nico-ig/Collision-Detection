#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import tsod
import tsod.hampel
from IPython.display import display
import statsmodels.tsa.stattools as stsm

from utils import print_header, print_section, print_descriptive_stats, load_data, get_high_risk_threshold, set_global_options

import warnings
warnings.filterwarnings('ignore', message='Series.__setitem__ treating keys as positions is deprecated')

def eda(file):
    print_header("Iniciando EDA")

    set_global_options()
    df = load_data(file)
    
    print_dataset_characteristics(df)
    print_data_quality(df)
    print_events_characteristics(df)
    print_risk_characteristics(df)
    print_time_characteristics(df)
    print_constant_values(df)
    print_outliers(df)
    print_acf(df)
    print_correlation(df)

    print_header("Concluído")

def print_dataset_characteristics(df):
    print_header("Características do dataset")

    print_section("Primeiras linhas")
    display(df.head())

    print_section("Tipos de dados")
    var_types = pd.DataFrame({
        'Tipo': df.dtypes.astype(str),
        'Classificação': ['Discreta' if 'int' in str(tipo) else 'Contínua' 
                        for tipo in df.dtypes]
    }, index=df.dtypes.index)
    display(var_types)

    print_descriptive_stats(df.drop(columns=['event_id']), "Dataset")

def print_data_quality(df):
    print_header("Qualidade dos dados")

    print_section("Valores nulos")
    null_counts = df.isnull().sum()
    null_cols = pd.DataFrame({
        'Qtd. Nulos': null_counts.values
    }, index=null_counts.index)
    display(null_cols)

    print_section("Valores infinitos")
    inf_df = pd.DataFrame({
        'Infinitos (+)': (df == np.inf).sum(),
        'Infinitos (-)': (df == -np.inf).sum()
    })
    display(inf_df)
    
def print_events_characteristics(df):
    print_header("Características dos eventos")

    print_section("Quantidade de eventos")
    print(f"Total de eventos: {df['event_id'].nunique()}")

    event_sizes = df.groupby('event_id').size()
    print_section("Quantidade de observações por quantidade de eventos")    
    freq_table = event_sizes.value_counts().sort_index(ascending=False)
    freq_df = pd.DataFrame({
        'Qtd. Eventos': freq_table.values,
        '%': (freq_table.values / len(event_sizes) * 100)
    }, index=freq_table.index)

    freq_df.index.name = 'Qtd. Observações'
    display(freq_df)
       
    print_descriptive_stats(pd.DataFrame(event_sizes, columns=['Qtd. Observações']), "Quantidade de observações por evento")

def print_risk_characteristics(df):
    print_header("Características do risco")
    print_risk_distribution_by_event(df)
    print_risk_distribution_by_time(df)
    print_risk_transitions(df)

def print_risk_distribution_by_event(df):
    print_section("Quantidade de eventos de alto e baixo risco")
    risk_stats = df.groupby('event_id')['risk'].agg(
        ['first', 'last', lambda x: x.iloc[-2] if len(x) > 1 else None]
    ).dropna()
    risk_stats.columns = ['first', 'last', 'penultimate']

    high_risk_events_first = (risk_stats['first'] >= get_high_risk_threshold()).sum()
    high_risk_events_penultimate = (risk_stats['penultimate'] >= get_high_risk_threshold()).sum()
    high_risk_events_last = (risk_stats['last'] >= get_high_risk_threshold()).sum()
    
    low_risk_events_first = (risk_stats['first'] < get_high_risk_threshold()).sum()
    low_risk_events_penultimate = (risk_stats['penultimate'] < get_high_risk_threshold()).sum()
    low_risk_events_last = (risk_stats['last'] < get_high_risk_threshold()).sum()

    risk_counts = pd.DataFrame({
        'Alto Risco': [
            high_risk_events_first,
            high_risk_events_penultimate,
            high_risk_events_last
        ],
        '% A.': [
            (high_risk_events_first / len(risk_stats)) * 100,
            (high_risk_events_penultimate / len(risk_stats)) * 100,
            (high_risk_events_last / len(risk_stats)) * 100
        ],
        'Baixo Risco': [
            low_risk_events_first,
            low_risk_events_penultimate,
            low_risk_events_last
        ],
        '% B.': [
            (low_risk_events_first / len(risk_stats)) * 100,
            (low_risk_events_penultimate / len(risk_stats)) * 100,
            (low_risk_events_last / len(risk_stats)) * 100
        ]
    }, index=['Primeira obs.', 'Penúltima obs.', 'Última obs.'])
    display(risk_counts)

def print_risk_distribution_by_time(df):  
    last_obs = df.groupby('event_id').nth(-1).reset_index()
    process_observations(last_obs, "Último")
    
    penultimate_obs = df.groupby('event_id').nth(-2).reset_index()
    process_observations(penultimate_obs, "Penúltimo")

def process_observations(obs_df, label):   
    bins = pd.timedelta_range(start='0 days', end='8 days', freq='1D')
    
    print_section(f"Distribuição do {label} Risco por Faixa de Tempo")
    risk_counts = obs_df.groupby(
        pd.cut(obs_df['time_to_tca'], bins=bins, right=False), observed=True
    )['risk'].agg([
        ('Alto Risco', lambda x: (x >= get_high_risk_threshold()).sum()),
        ('Baixo Risco', lambda x: (x < get_high_risk_threshold()).sum())
    ])
    display(risk_counts[['Alto Risco', 'Baixo Risco']])

def print_risk_transitions(df):
    print_section("Contagem de Oscilações por Dia")
    daily_df = prepare_daily_data(df)
    state_changes = get_state_changes(daily_df)
    transitions = count_daily_transitions(state_changes)
    oscillations = detect_oscillations(state_changes)
    first_trans = get_first_transitions(state_changes)
    last_trans = get_last_transitions(state_changes)
    daily_totals = get_daily_totals(daily_df) 
    result_df = build_result_dataframe(oscillations, transitions, first_trans, last_trans, daily_totals)
    display(result_df.T)

def prepare_daily_data(df):
    daily_df = df.copy()
    daily_df['date'] = daily_df.index.round('d')
    daily_df['risk_state'] = np.where(daily_df['risk'] >= get_high_risk_threshold(), 'high', 'low')
    return daily_df

def get_state_changes(daily_df):
    last_states = (daily_df.groupby(['event_id', 'date'])['risk_state']
                   .last()
                   .reset_index()
                   .sort_values(['event_id', 'date'], ascending=[True, False]))
    last_states['prev_state'] = last_states.groupby('event_id')['risk_state'].shift(1)
    state_changes = last_states.dropna().query('risk_state != prev_state')
    return state_changes

def count_daily_transitions(state_changes):
    transitions = state_changes.groupby('date')['prev_state'].value_counts().unstack(fill_value=0)
    daily_low_to_high = transitions.get('low', pd.Series(index=transitions.index, dtype=int))
    daily_high_to_low = transitions.get('high', pd.Series(index=transitions.index, dtype=int))
    return daily_low_to_high, daily_high_to_low

def detect_oscillations(state_changes):
    oscillation_data = (
        state_changes.groupby('event_id')
        .filter(lambda x: len(x) >= 3)
        .assign(prev_state2=lambda x: x.groupby('event_id')['risk_state'].shift(2))
        .query('risk_state == prev_state2')
    )
    low_high_low = oscillation_data.query('risk_state == "low"').groupby('date').size()
    high_low_high = oscillation_data.query('risk_state == "high"').groupby('date').size()
    return low_high_low, high_low_high

def get_first_transitions(state_changes):
    first_transitions = state_changes.groupby('event_id').first().reset_index()
    first_trans_high = first_transitions[first_transitions['risk_state'] == 'high'].groupby('date').size()
    first_trans_low = first_transitions[first_transitions['risk_state'] == 'low'].groupby('date').size()
    return first_trans_high, first_trans_low

def get_last_transitions(state_changes):
    last_transitions = state_changes.groupby('event_id').last().reset_index()
    last_trans_high = last_transitions[last_transitions['risk_state'] == 'high'].groupby('date').size()
    last_trans_low = last_transitions[last_transitions['risk_state'] == 'low'].groupby('date').size()
    return last_trans_high, last_trans_low

def get_daily_totals(daily_df):
    daily_last_states = daily_df.groupby(['event_id', 'date'])['risk_state'].last().reset_index()    
    daily_high = daily_last_states[daily_last_states['risk_state'] == 'high'].groupby('date').size()
    daily_low = daily_last_states[daily_last_states['risk_state'] == 'low'].groupby('date').size()
    return daily_high, daily_low

def build_result_dataframe(oscillations, transitions, first_trans, last_trans, daily_totals):
    low_high_low, high_low_high = oscillations
    daily_low_to_high, daily_high_to_low = transitions
    first_trans_high, first_trans_low = first_trans
    last_trans_high, last_trans_low = last_trans
    daily_high, daily_low = daily_totals
    result_df = pd.DataFrame({
        'Low → High → Low': low_high_low,
        'High → Low → High': high_low_high,
        'Low → High': daily_low_to_high,
        'High → Low': daily_high_to_low,
        'Última transição High': last_trans_high,
        'Última transição Low': last_trans_low,
        'Primeira transição High': first_trans_high,
        'Primeira transição Low': first_trans_low,
        'Total High': daily_high,
        'Total Low': daily_low
    }).fillna(0).astype(int)
    return result_df

def print_time_characteristics(df):
    print_header("Características do tempo")
    print_time_diff(df)
    time_stats_df = get_time_stats(df)
    print_time_stats(time_stats_df)
    print_time_distribution(time_stats_df)

def print_time_diff(df):
    diffs = []
    for _, group in df.groupby('event_id'):
        if len(group) > 1:
            event_diffs = group.index.to_series().diff().dropna()
            diffs.extend(event_diffs)
    
    diffs_series = pd.Series(diffs)
    print_datetime_descriptive_stats(diffs_series, "Diferenças de tempo média entre observações")

def get_time_stats(df):
    time_stats = []
    for event_id, group in df.groupby('event_id'):
        times = group.index.sort_values(ascending=False)
        time_stats.append({
            'event_id': event_id,
            'first': times[0],
            'last': times[-1] if len(times) > 1 else None,
            'penultimate': times[-2] if len(times) > 1 else None
        })
    return pd.DataFrame(time_stats).dropna()
        
def print_time_stats(time_stats_df):
    print_datetime_descriptive_stats(time_stats_df['first'], "Primeiro time_to_tca por evento")
    print_datetime_descriptive_stats(time_stats_df['penultimate'], "Penúltimo time_to_tca por evento")
    print_datetime_descriptive_stats(time_stats_df['last'], "Último time_to_tca por evento")
    
    last_first_diff = time_stats_df['last'] - time_stats_df['first']
    print_datetime_descriptive_stats(last_first_diff, "Diferença entre último e primeiro time_to_tca por evento")
    
    last_penultimate_diff = time_stats_df['last'] - time_stats_df['penultimate']
    print_datetime_descriptive_stats(last_penultimate_diff, "Diferença entre último e penúltimo time_to_tca por evento")
    
    penultimate_first_diff = time_stats_df['penultimate'] - time_stats_df['first']
    print_datetime_descriptive_stats(penultimate_first_diff, "Diferença entre penúltimo e primeiro time_to_tca por evento")
    
def print_time_distribution(time_stats_df):
    print_header("Distribuição de Eventos por Faixa de Tempo")
    bins = pd.timedelta_range(start='0 days', end='8 days', freq='1D')
    result = pd.DataFrame({
        'Primeira observação': pd.cut(time_stats_df['first'], bins=bins).value_counts(),
        'Penúltima observação': pd.cut(time_stats_df['penultimate'], bins=bins).value_counts(),
        'Última observação': pd.cut(time_stats_df['last'], bins=bins).value_counts(),
    }).astype(int)    
    display(result)

def print_datetime_descriptive_stats(values, section_name):
    print_section(f"Estatísticas descritivas ({section_name})")
    stats = {
        'mean': values.mean().round('h'),
        'std': values.std().round('h'),
        'min': values.min().round('h'),
        '25%': values.quantile(0.25).round('h'),
        '50%': values.quantile(0.5).round('h'),
        '75%': values.quantile(0.75).round('h'),
        'max': values.max().round('h'),
    }
    display(pd.DataFrame(stats, index=['Valor']).T)

def print_constant_values(df):
    print_header("Valores Constantes")
    print_section("Valores constantes por evento")
    results = {col: 0 for col in df.columns if col != 'event_id'}
    detector = tsod.ConstantValueDetector(window_size=3, threshold=3)
    for _, group in df.groupby('event_id'):
        for col in group.columns:
            if col == 'event_id':
                continue
            detector.fit(group[col])
            res = detector.detect(group[col])
            results[col] += np.array(res).sum()

    result_df = pd.DataFrame(results, index=['Qtd. Constantes']).T
    display(result_df)

def print_outliers(df):
    print_header("Outliers")
    print_section("Outliers por evento")
    results = {col: 0 for col in df.columns if col != 'event_id'}
    detector = tsod.hampel.HampelDetector(window_size=3, threshold=3)
    for _, group in df.groupby('event_id'):
        for col in group.columns:
            if col == 'event_id':
                continue
            detector.fit(group[col])
            res = detector.detect(group[col])
            results[col] += np.array(res).sum()
    result_df = pd.DataFrame(results, index=['Qtd. Outliers']).T
    display(result_df)

def print_acf(df):
    print_header("ACF por Lag")
    n_lags = 6
    results = {col: [] for col in df.columns if col != 'event_id'}    
    for col in df.columns:
        if col == 'event_id':
            continue
        for _, group in df.groupby('event_id'):
            acf_vals = stsm.acf(group[col], nlags=n_lags)
            results[col].append(acf_vals)
        cols = [f'Lag {i}' for i in range(n_lags + 1)]
        print_descriptive_stats(pd.DataFrame(results[col], columns=cols), f"ACF {col}")

def print_correlation(df):
    print_header("Análise de Correlação com Risco")
    print_daily_risk_correlations(df)
    print_last_k_risk_correlations(df, 2)
    
def print_daily_risk_correlations(df):
    filtered_df = df.copy()
    filtered_df = filtered_df.drop(columns=['event_id'])
    filtered_df.index = filtered_df.index.round('d')
    
    risk_correlations = pd.DataFrame()
    for day, group in filtered_df.groupby(filtered_df.index):
        corr_matrix = group.corr()
        risk_corr = corr_matrix.loc[['risk']].T
        risk_corr = risk_corr.rename(columns={'risk': f"{day}"})
        risk_correlations = pd.concat([risk_correlations, risk_corr], axis=1)    
    risk_correlations = risk_correlations.drop('risk')
    print_section("Correlação diária com risco")
    display(risk_correlations)
    
def print_last_k_risk_correlations(df, k):
    result_dfs = []
    for k in range(1, k + 1):
        kth_entries = df.groupby('event_id').nth(-k).reset_index()
        kth_entries = kth_entries.drop(columns=['event_id', 'time_to_tca'])        
        corr_matrix = kth_entries.corr()        
        risk_corr = corr_matrix.loc[['risk']].drop('risk', axis=1)
        risk_corr = risk_corr.rename(index={'risk': f'-{k}'})
        result_dfs.append(risk_corr)
    
    combined_df = pd.concat(result_dfs, axis=0) 
    print_section(f"Correlação com risco para os últimos {k} eventos")   
    display(combined_df.T)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file to use on the EDA')
    args = parser.parse_args()
    
    eda(args.file)