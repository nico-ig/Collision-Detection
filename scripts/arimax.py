#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.metrics as skl

from utils import print_header, print_section, set_global_options, load_data, print_action, get_orders_grid, get_max_iter, get_batch_size, get_min_observations, get_method, save_models


# ---------- config ----------
FEATURES = ['mahalanobis_distance', 'miss_distance', 't_j2k_inc', 't_j2k_sma']
ORDERS_GRID = [(p,d,q) for p in (0,1) for d in (0,1) for q in (0,1)]
HIGH_RISK_THRESH = -6
BETA_FOR_F = 2.0
# ----------------------------

# ---------- hyperparameters ----------
METHOD = "powell"
MAX_ITER = 100
TREND = 't'
ENFORCE_STATIONARITY = False
ENFORCE_INVERTIBILITY = False
SAME_LOSS_TOLERANCE = 1e-3
MIN_EVENT_SIZE = 6
MAX_EVENTS = 5
ROUND_DECIMALS = 4
# ----------------------------

def main(input_file):
    """Executa pipeline principal"""
    print("Carregando e pré-processando dados...")
    # df_train = pd.read_csv(input_file)
    # df_train = preprocess_data(df_train)
    df_train = load_data(input_file)
    df_train['y'] = df_train['risk'].round(ROUND_DECIMALS)

    # print("Carregando e pré-processando dados de validação...")
    # df_val = pd.read_csv('dataset/data_val.csv')
    # df_val = preprocess_data(df_val)

    # print("Carregando e pré-processando dados de teste...")
    # df_test = pd.read_csv('dataset/test_data.csv')
    # df_test = preprocess_data(df_test)

    print(f"Número de eventos de treino: {df_train['event_id'].nunique()}")
    # print(f"Número de eventos de validação: {df_val['event_id'].nunique()}")
    # print(f"Número de eventos de teste: {df_test['event_id'].nunique()}")

    orders_weight = find_orders_weights(df_train)

    # print("Gerando previsões...")
    # preds = predict(df_test, orders_weight)

    # print("Previsões com pesos:")
    # print(pd.DataFrame(preds['weigthed']['preds']))

    # print("Previsões simples:")
    # print(preds_simple)

    # print("Previsões com melhor ordem:")
    # print(preds_best_order)

def preprocess_data(df):
    df = df.sort_values(['time_to_tca'], ascending=False).reset_index(drop=True)
    unique_events = df['event_id'].unique()
    unique_events = [eid for eid in unique_events if len(df[df['event_id'] == eid]) >= MIN_EVENT_SIZE]
    df = df[df['event_id'].isin(unique_events)]
    
    if MAX_EVENTS is not None:
        unique_events = unique_events[:MAX_EVENTS]
        df = df[df['event_id'].isin(unique_events)]
    
    df = df[FEATURES + ['risk', 'event_id', 'time_to_tca']]
    df['y'] = df['risk'].round(ROUND_DECIMALS)
    return df

def find_orders_weights(df_train):
    """Encontra os pesos de cada ordem ARIMAX"""
    print("Encontrando pesos das ordens...")

    orders_weight = {}
    val_loss = []
    for _, group in df_train.groupby('event_id'):
        group = group.copy().reset_index(drop=True)
        best_orders = find_best_orders_for_event(group)
        orders_weight = update_orders_weigth(orders_weight, best_orders)
        #pred = validate_orders(df_val, orders_weight)
        #print("Perda validação:")
        #val_loss.append(pred['weigthed']['loss'])
        #print(val_loss)

    print("Pesos encontrados:", orders_weight)
    return orders_weight

def find_best_orders_for_event(group):
    """Encontra as melhores ordem ARIMAX para um evento específico"""
    print(f"Encontrando melhores orderns para event_id {group.iloc[0]['event_id']}")

    X_train = group[FEATURES].iloc[:-1]
    y_train = group['y'].iloc[:-1]

    X_next = group[FEATURES].iloc[-1]
    y_true = group['y'].iloc[-1]

    best_orders = []
    best_loss = np.inf

    for order in ORDERS_GRID:
        model = fit_model(y_train, X_train, order)
        y_pred = predict_final_risk(model, X_next)
        print(f"y_true: {y_true} - y_pred: {y_pred}")
        loss = calculate_loss([y_true], [y_pred])
        loss_diff = loss - best_loss
        if abs(loss_diff) <= SAME_LOSS_TOLERANCE:
            best_orders.append(order)
        elif loss_diff < 0:
            best_orders = [order]
            best_loss = loss

    return best_orders

def fit_model(y_train, X_train, order):
    """Ajusta modelo ARIMAX com os parâmetros fornecidos"""
    model = sm.tsa.SARIMAX(
        endog=y_train, 
        exog=X_train, 
        order=order,
        trend=TREND,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY
    )
    return model.fit(method=METHOD, maxiter=MAX_ITER, disp=False)

def predict_final_risk(model, X_val):
    pred = model.forecast(steps=1, exog=X_val)
    return float(pred.iloc[0])

def calculate_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_high_risk = y_true >= HIGH_RISK_THRESH
    pred_high_risk = y_pred >= HIGH_RISK_THRESH

    if (len(y_true[true_high_risk]) == 0 or len(y_pred[true_high_risk]) == 0):
        return 0

    mse = skl.mean_squared_error(y_true[true_high_risk], y_pred[true_high_risk])
    fbeta = skl.fbeta_score(true_high_risk.astype(int), pred_high_risk.astype(int), beta=BETA_FOR_F, zero_division=0.0)

    if fbeta == 0:
        return float('inf')

    return mse / fbeta

def update_orders_weigth(orders, best_orders):
    for order in best_orders:
        if order not in orders:
            orders[order] = 0
        orders[order] += 1
    return orders

def validate_orders(df_val, orders_weight):
    return predict(df_val, orders_weight)

def predict(df, orders_weights):
    """Predição com a média ponderada das predições de cada ordem"""
    weighted_preds = []

    for eid, group in df.groupby('event_id'):
        group = group.copy().reset_index(drop=True)
        X_train = group[FEATURES].iloc[:-1]
        y_train = group['y'].iloc[:-1]
        
        X_next = group[FEATURES].iloc[-1]
        y_true = group['y'].iloc[-1]
  
        weighted_pred = predict_weighted(X_next, X_train, y_train, orders_weights)
        
        weighted_preds.append({
            'event_id': eid,
            'y_true': y_true,
            'y_pred': weighted_pred,
            'true_positive': y_true >= HIGH_RISK_THRESH and weighted_pred >= HIGH_RISK_THRESH,
            'true_negative': y_true < HIGH_RISK_THRESH and weighted_pred < HIGH_RISK_THRESH,
            'false_positive': y_true < HIGH_RISK_THRESH and weighted_pred >= HIGH_RISK_THRESH,
            'false_negative': y_true >= HIGH_RISK_THRESH and weighted_pred < HIGH_RISK_THRESH
        })
    
    y_true = [pred['y_true'] for pred in weighted_preds]
    y_pred = [pred['y_pred'] for pred in weighted_preds]

    preds = {
        'weigthed': {
            'loss': calculate_loss(y_true, y_pred),
            'preds': weighted_preds
        }
    }

    return preds

def predict_weighted(X_next, X_test, y_test, orders_weights):
    group_pred = 0
    for order, weight in orders_weights.items():
        model = fit_model(y_test, X_test, order)
        y_pred = predict_final_risk(model, X_next)
        group_pred += weight * y_pred
            
    group_pred = group_pred / sum(orders_weights.values())
    return group_pred

# def predict_simple(eid, y_val, X_val, X_test, y_test):
#     orders_weights = {}
#     for order in ORDERS_GRID:
#         orders_weights[order] = 1
#     return predict_weighted(eid, y_val, X_val, X_test, y_test, orders_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit ARIMAX models')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the CSV file to use on the fit')
    args = parser.parse_args()

    main(args.input)


