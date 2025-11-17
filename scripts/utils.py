
import pandas as pd
import joblib
import os
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
    joblib.parallel_backend('loky', n_jobs=get_njobs(), inner_max_n_jobs=1)

def load_data(file, convert_time_to_tca=True, add_columns=[]):
    print_action("Carregando dados")
    df = pd.read_csv(file)
    print_action(f"Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas")

    print_action("Removendo colunas")
    features = [
        'event_id', 
        'time_to_tca', 
        'risk', 
        'c_time_lastob_end', 
        'c_time_lastob_start', 
        'miss_distance', 
        'relative_position_n', 
        'c_cd_area_over_mass', 
        'c_cr_area_over_mass', 
        'c_sedr', 
        'c_obs_used', 
        'c_sigma_t'
    ]
    features.extend(add_columns)
    df = df[features]
    print_action(f"Dataset final com {len(df)} linhas e {len(df.columns)} colunas")
    
    if convert_time_to_tca:
        max_time_to_tca = df['time_to_tca'].max()
        df['time_to_tca'] = df['time_to_tca'].apply(lambda x: pd.Timestamp(max_time_to_tca - x, unit='d'))

    df = sort_data(df)

    print_action("Atualizando index (event_id, time_to_tca)")
    df = df.set_index(['event_id', 'time_to_tca'])

    return df

def sort_data(df):
    print_action("Ordenando por evento e time_to_tca")
    return df.sort_values(['event_id', 'time_to_tca'])

def save_data(df, path, convert_time_to_tca=True):
    print_action(f"Salvando dados em {path}")
    df = df.reset_index()
    if convert_time_to_tca:
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

def get_njobs():
    return max(1, multiprocessing.cpu_count() - 1)  

def get_max_iter():
    return 100

def get_seed():
    return 157

def get_n_clusters():
    return 3

def get_kmeans_metric():
    return "softdtw"

def get_enforce_stationarity():
    return False

def get_enforce_invertibility():
    return False

def get_method():
    return "powell"

def get_mle_regression():
    return False

def get_missing_values_interpolation():
    return "akima"

def get_time_conflict_resolution():
    return {
        'risk': 'max', 
        'c_time_lastob_end': 'max', 
        'c_time_lastob_start': 'min', 
        'miss_distance': 'min', 
        'relative_position_n': 'mean', 
        'c_cd_area_over_mass': 'max', 
        'c_cr_area_over_mass': 'max', 
        'c_sedr': 'max', 
        'c_obs_used': 'sum', 
        'c_sigma_t': 'max'
    }

def load_models(base_path):
    print_action(f"Carregando modelos de: {os.path.abspath(base_path)}")
    models = {}
    km_model = None
    scaler = None
    
    km_path = os.path.join(base_path, 'km', 'km.pkl')
    if os.path.exists(km_path):
        km_model = joblib.load(km_path)
        print_action("Modelo KMeans carregado com sucesso")
    
    scaler_path = os.path.join(base_path, 'scaler', 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print_action("Scaler carregado com sucesso")
    
    for cluster_dir in (d for d in os.listdir(base_path) if d.startswith('cluster_')):
        cluster_num = int(cluster_dir.split('_')[1])
        cluster_path = os.path.join(base_path, cluster_dir)
        models[cluster_num] = {}
                
        for coord_dir in (d for d in os.listdir(cluster_path) if d.startswith('(')):
            model_path = os.path.join(cluster_path, coord_dir, f"{coord_dir}.pkl")
            if os.path.exists(model_path):
                coord = eval(coord_dir)
                models[cluster_num][coord] = joblib.load(model_path)
                print_action(f"Cluster {cluster_num}, coordenadas {coord}: OK")
    
    total = sum(len(m) for m in models.values())
    print_action(f"Total de {total} modelo(s) de previsão carregado(s)")
    return models, km_model, scaler