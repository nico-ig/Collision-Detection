
import pandas as pd
import joblib
import os
from IPython.display import display

LINE_LENGTH = 110

def print_header(title):
    print("=" * LINE_LENGTH)
    print(f"{title:^{LINE_LENGTH}}")
    print("=" * LINE_LENGTH)

def print_section(action):
    action_line_length = LINE_LENGTH // 2
    print("-" * action_line_length)
    print(f"{action:^{action_line_length}}")
    print("-" * action_line_length)

def print_action(action):
    print(f"----> {action}")

def print_descriptive_stats(values, section_name):
    print_section(f"Estatísticas descritivas ({section_name})")
    stats = values.describe().T.sort_index()[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    display(stats)

def set_global_options():
    print_action("Configurando opções globais")
    pd.set_option('display.precision', 4)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None) 

def load_data(file):
    print_action("Carregando dados")
    df = pd.read_csv(file)
    # events = df['event_id'].unique()[:100]
    # df = df[df['event_id'].isin(events)]
    print_action(f"Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas")

    print_action("Removendo colunas")
    df = df[['event_id', 'time_to_tca', 'risk', 'c_time_lastob_end', 'c_time_lastob_start', 'miss_distance', 'relative_position_n', 'c_cd_area_over_mass', 'c_cr_area_over_mass', 'c_sedr', 'c_obs_used', 'c_sigma_t']]
    print_action(f"Dataset final com {len(df)} linhas e {len(df.columns)} colunas")
    
    max_time_to_tca = df['time_to_tca'].max()
    df['time_to_tca'] = df['time_to_tca'].apply(lambda x: pd.Timestamp(max_time_to_tca - x, unit='d'))

    df = sort_data(df)

    print_action("Atualizando index (event_id, time_to_tca)")
    df = df.set_index(['event_id', 'time_to_tca'])

    return df

def sort_data(df):
    print_action("Ordenando por evento e time_to_tca")
    return df.sort_values(['event_id', 'time_to_tca'])

def save_data(df, path):
    print_action(f"Salvando dados em {path}")
    df = df.reset_index()
    df['time_to_tca'] = (df['time_to_tca'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / (60 * 60 * 24)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def get_high_risk_threshold():
    return -6

def get_orders_grid():
    return [(p,d,q) for p in (0,1) for d in (0,1) for q in (0,1)]

def save_models(models, path):
    print_action(f"Salvando modelos em {path}")
    for index, model in models.items():
        base_path = os.path.join(path, str(index))
        os.makedirs(base_path, exist_ok=True)
        file_path = os.path.join(base_path, f"{index}.pkl")
        joblib.dump(model, file_path, compress=3)
    print_action(f"Modelos salvos em {path}")

def get_max_iter():
    return 200

def get_method():
    return "powell"

def get_seed():
    return 157

def get_n_clusters():
    return 3

def get_kmeans_metric():
    return "dtw"

def get_interval():
    return "4h"

def get_enforce_stationarity():
    return False

def get_enforce_invertibility():
    return False

def get_method():
    return "powell"

def get_mle_regression():
    return False
