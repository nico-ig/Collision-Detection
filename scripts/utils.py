
import pandas as pd
import joblib
import os
import math
from IPython.display import display
import multiprocessing

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
    joblib.parallel_backend('loky')

def load_data(file, convert_time_to_tca=True, add_columns=[]):
    print_action("Carregando dados")
    df = pd.read_csv(file)
    # event_ids = df['event_id'].unique()[10:20]
    # df = df[df['event_id'].isin(event_ids)]
    print_action(f"Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas")

    print_action("Removendo colunas")
    features = [
        'event_id', 
        'time_to_tca', 
        'risk', 
        'c_time_lastob_end', 
        'c_time_lastob_start', 
        'c_cd_area_over_mass',
        'c_sedr', 
        'c_obs_used', 
        'c_sigma_t',
        'cluster'
    ]
    features.extend(add_columns)
    features = [col for col in features if col in df.columns]
    df = df[features]
    print_action(f"Dataset final com {len(df)} linhas e {len(df.columns)} colunas")
    
    if convert_time_to_tca:
        df['time_to_tca'] = df['time_to_tca'].apply(lambda x: pd.Timestamp(math.ceil(-x*24), unit='h'))

    print_action("Ordenando por evento e time_to_tca")
    df = df.sort_values(['event_id', 'time_to_tca'])

    print_action("Atualizando index (event_id, time_to_tca)")
    df = df.set_index(['event_id', 'time_to_tca'])

    return df

def save_data(df, path, convert_time_to_tca=True):
    print_action(f"Salvando dados em {path}")
    index = df.index
    df = df.reset_index()
    if convert_time_to_tca:
        df['time_to_tca'] = (df['time_to_tca'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / (60 * 60 * 24)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    df.set_index(index, inplace=True)

def save_models(models, path):
    print_action(f"Salvando modelos em {path}")
    for index, model in models.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_path = os.path.join(path, f"{index}.pkl")
        print_action(f"Salvando modelo {index} em {file_path}")
        joblib.dump(model, file_path, compress=3)

def load_model(path, model_name):
    model_name = os.path.splitext(os.path.basename(model_name))[0]
    file_path = os.path.join(path, f"{model_name}.pkl")
    print_action(f"Carregando modelo de: {file_path}")
    model = joblib.load(file_path)
    return model

def load_clusters(path):
    print_action(f"Carregando clusters de: {path}")
    clusters = {}
    for cluster in os.listdir(path):
        cluster_idx = int(cluster.split('_')[1])
        cluster_path = os.path.join(path, cluster)
        print_action(f"Carregando cluster {cluster_idx} de {cluster_path}")
        models = load_model(cluster_path, 'trained_models')
        interval = load_model(cluster_path, 'interval')
        clusters[cluster_idx] = {'models': models, 'interval': interval}
    return clusters

def get_high_risk_threshold():
    return -6

def get_njobs():
    return max(1, multiprocessing.cpu_count() - 1)  

def get_max_iter():
    return 100

def get_seed():
    return 157

def get_kmeans_metric():
    return "dtw"

