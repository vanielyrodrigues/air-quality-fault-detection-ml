"""Atributos por janela e construção dos classificadores."""
import os
import json
from math import gcd
from functools import reduce
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False
FALHAS_ALVO = ['oscilacao', 'lacuna', 'queda']

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce')

def _maior_sequencia_true(mask_bool) -> int:
    maior = 0
    atual = 0
    for v in mask_bool:
        if bool(v):
            atual += 1
            maior = max(maior, atual)
        else:
            atual = 0
    return maior

def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return 0.0
    x = x[ok]
    y = y[ok]
    x_mean = x.mean()
    y_mean = y.mean()
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0
    num = np.sum((x - x_mean) * (y - y_mean))
    return float(num / den)

def extrair_atributos_janela(sub: pd.DataFrame, coluna: str) -> dict:
    serie = _to_numeric_safe(sub[coluna]).astype(float)
    vals = serie.to_numpy(dtype=float)
    feats = {'n': len(vals), 'nan_count': int(np.isnan(vals).sum()), 'nan_prop': float(np.isnan(vals).mean()) if len(vals) else 0.0}
    if len(vals) == 0 or np.isnan(vals).all():
        feats.update({'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'amplitude': 0.0, 'last': 0.0, 'first': 0.0, 'last_first_diff': 0.0, 'slope': 0.0, 'abs_diff_mean': 0.0, 'abs_diff_max': 0.0, 'prop_zero': 0.0, 'prop_quase_zero': 0.0, 'prop_repetido': 0.0, 'mudanca_sinal': 0.0, 'energia_variacao': 0.0, 'max_seq_zero': 0.0, 'max_seq_constante': 0.0, 'fim_is_nan': 1.0, 'fim_is_zero': 0.0})
        for i in range(len(vals)):
            feats[f'lag_{i + 1}'] = 0.0
        return feats
    s_fill = pd.Series(vals).interpolate(limit_direction='both').ffill().bfill()
    vals_fill = s_fill.to_numpy(dtype=float)
    diffs = np.diff(vals_fill)
    abs_diffs = np.abs(diffs)
    zero_mask = np.isclose(vals_fill, 0.0, atol=1e-09)
    quase_zero_mask = np.abs(vals_fill) <= 0.1
    repetido_mask = np.zeros(len(vals_fill), dtype=float)
    if len(vals_fill) > 1:
        repetido_mask[1:] = np.isclose(vals_fill[1:], vals_fill[:-1], atol=1e-09).astype(float)
    sinal = np.sign(diffs)
    mudou_sinal = 0.0
    if len(sinal) > 1:
        mudou_sinal = float(np.sum(sinal[1:] * sinal[:-1] < 0))
    const_mask = np.zeros(len(vals_fill), dtype=bool)
    if len(vals_fill) > 1:
        const_mask[1:] = np.isclose(vals_fill[1:], vals_fill[:-1], atol=1e-09)
    feats.update({'mean': float(np.mean(vals_fill)), 'std': float(np.std(vals_fill)), 'min': float(np.min(vals_fill)), 'max': float(np.max(vals_fill)), 'median': float(np.median(vals_fill)), 'amplitude': float(np.max(vals_fill) - np.min(vals_fill)), 'last': float(vals_fill[-1]), 'first': float(vals_fill[0]), 'last_first_diff': float(vals_fill[-1] - vals_fill[0]), 'slope': _slope(vals_fill), 'abs_diff_mean': float(abs_diffs.mean()) if len(abs_diffs) else 0.0, 'abs_diff_max': float(abs_diffs.max()) if len(abs_diffs) else 0.0, 'prop_zero': float(zero_mask.mean()), 'prop_quase_zero': float(quase_zero_mask.mean()), 'prop_repetido': float(repetido_mask.mean()), 'mudanca_sinal': mudou_sinal, 'energia_variacao': float(abs_diffs.sum()) if len(abs_diffs) else 0.0, 'max_seq_zero': float(_maior_sequencia_true(zero_mask)), 'max_seq_constante': float(_maior_sequencia_true(const_mask)), 'fim_is_nan': float(np.isnan(vals[-1])), 'fim_is_zero': float(np.isclose(vals_fill[-1], 0.0, atol=1e-09))})
    for i, v in enumerate(vals_fill):
        feats[f'lag_{i + 1}'] = float(v)
    return feats

def inicios_janelas(n, janela, passo):
    if janela <= 0 or passo <= 0 or passo > janela:
        raise ValueError('Exige 0 < passo <= janela')
    if n < janela:
        return []
    inicios = list(range(0, n - janela + 1, passo))
    if inicios[-1] != n - janela:
        inicios.append(n - janela)
    return inicios

def montar_dataset_janelas(df: pd.DataFrame, coluna: str, falha_alvo: str, janela: int, passo: int, rotulo_modo: str='fim') -> pd.DataFrame:
    df = df.copy().sort_values('Datetime').reset_index(drop=True)
    rows = []
    if len(df) < janela:
        return pd.DataFrame()
    for ini in inicios_janelas(len(df), janela, passo):
        fim = ini + janela
        sub = df.iloc[ini:fim].copy()
        labels = sub['label'].astype(str).tolist()
        if rotulo_modo == 'fim':
            y = int(labels[-1] == falha_alvo)
        elif rotulo_modo == 'qualquer':
            y = int(falha_alvo in labels)
        elif rotulo_modo == 'maioria':
            y = int(sum((l == falha_alvo for l in labels)) > len(labels) / 2.0)
        else:
            raise ValueError("rotulo_modo inválido = 'fim', 'qualquer' ou 'maioria'.")
        feats = extrair_atributos_janela(sub, coluna)
        feats['y'] = y
        feats['falha_alvo'] = falha_alvo
        feats['janela'] = int(janela)
        feats['passo'] = int(passo)
        feats['Datetime_ini'] = sub['Datetime'].iloc[0]
        feats['Datetime_fim'] = sub['Datetime'].iloc[-1]
        rows.append(feats)
    ds = pd.DataFrame(rows)
    if ds.empty:
        return ds
    return ds.sort_values('Datetime_fim').reset_index(drop=True)

def contar_eventos_binarios(vetor) -> int:
    normal = True
    cont = 0
    for valor in vetor:
        if int(valor) == 1:
            if normal:
                cont += 1
                normal = False
        else:
            normal = True
    return int(cont)

def _criar_modelo(nome_modelo: str, ratio: float):
    nome_modelo = nome_modelo.lower()
    if nome_modelo == 'xgboost':
        if not _HAS_XGB:
            return None
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=int(os.environ.get('PROTOCOLO_N_JOBS', '-1')), eval_metric='logloss', scale_pos_weight=max(1.0, float(ratio)))
    if nome_modelo == 'random_forest':
        return RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=int(os.environ.get('PROTOCOLO_N_JOBS', '-1')), class_weight='balanced_subsample')
    if nome_modelo == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0001, learning_rate='adaptive', max_iter=600, random_state=42)
    if nome_modelo == 'catboost':
        if not _HAS_CAT:
            return None
        return CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, loss_function='Logloss', eval_metric='F1', random_seed=42, thread_count=int(os.environ.get('PROTOCOLO_N_JOBS', '-1')), verbose=False)
    raise ValueError(f'Modelo não suportado: {nome_modelo}')

