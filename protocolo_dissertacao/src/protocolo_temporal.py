"""Divisão cronológica e preparação independente das partições."""
import argparse, hashlib, json, importlib.util
from pathlib import Path
import pandas as pd
from ajustes import reamostrar_e_imputar
from simulacao_falhas import preparar_base

def separar(dados, inicio, fim, coluna):
    dados = dados.copy()
    dados['Datetime'] = pd.to_datetime(dados['Datetime'])
    dados = dados.sort_values('Datetime')
    dados = dados[(dados.Datetime >= inicio) & (dados.Datetime <= fim)]
    if dados.empty:
        raise ValueError('Sem dados no período.')
    grade = pd.date_range(dados.Datetime.min().floor('10min'), dados.Datetime.max().floor('10min'), freq='10min')
    c1, c2 = (grade[int(len(grade) * 0.6)], grade[int(len(grade) * 0.8)])
    segmentos = [dados[dados.Datetime < c1], dados[(dados.Datetime >= c1) & (dados.Datetime < c2)], dados[dados.Datetime >= c2]]
    bases, resumo = ({}, [])
    for nome, seg in zip(['treino', 'validacao', 'teste'], segmentos):
        if seg.empty:
            raise ValueError(f'{nome}: vazio')
        base = reamostrar_e_imputar(seg, '10min')
        ausentes = int((~base[f'{coluna}_observado']).sum())
        base = preparar_base(base, coluna)
        base['label'], base['evento_id'] = ('normal', '')
        if len(base) < 100 or base[coluna].isna().any():
            raise ValueError(f'{nome}: insuficiente ou NaN')
        bases[nome] = base
        resumo.append(dict(particao=nome, inicio=str(base.Datetime.min()), fim=str(base.Datetime.max()), amostras=len(base), sem_medicao_antes_imputacao=ausentes))
    assert bases['treino'].Datetime.max() < bases['validacao'].Datetime.min()
    assert bases['validacao'].Datetime.max() < bases['teste'].Datetime.min()
    return (bases, resumo)
