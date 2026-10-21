"""Configuração e aplicação das injeções de falhas."""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ajustes import carregar_dados, filtrar_periodo, reamostrar_e_imputar
from simulacao_falhas import preparar_base, injetar_intervalo_por_tempo, balancear_falhas, resumo_eventos_injetados, LABEL_OSC, LABEL_LACUNA, LABEL_QUEDA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(BASE_DIR, 'dados')
FREQ_PADRAO = '10min'
PASTA_RESULTADOS = os.path.join(BASE_DIR, 'resultados')
PASTA_RELATORIOS = os.path.join(BASE_DIR, 'relatorios')
COLUNA_ALVO = 'MP2,5_1'
COLUNAS_ANALISADAS = ['MP2,5_1', 'MP10_1', 'Temperatura', 'Umidade']
NOMES_VARIAVEIS = {'MP2,5_1': 'MP25_1', 'MP10_1': 'MP10_1', 'Temperatura': 'Temperatura', 'Umidade': 'Umidade'}
PASTA_RESULTADOS_SEM_ERRO = os.path.join(PASTA_RESULTADOS, 'sem_erro')
PASTA_RESULTADOS_COM_ERRO = os.path.join(PASTA_RESULTADOS, 'com_erro')
PASTA_RELATORIOS_SEM_ERRO = os.path.join(PASTA_RELATORIOS, 'sem_erro')
PASTA_RELATORIOS_COM_ERRO = os.path.join(PASTA_RELATORIOS, 'com_erro')

def config_falhas_padrao():
    return {'oscilacao': {'duracao_pts': (36, 72), 'n_eventos': 25, 'amp': (2.0, 4.0)}, 'lacuna': {'duracao_pts': (24, 48), 'n_eventos': 25}, 'queda': {'duracao_pts': (18, 36), 'n_eventos': 25, 'delta': (-18.0, -6.0)}}

def criar_base_com_falhas(df_original, coluna, pasta_resultados, nome_base, incluir_falhas_finais=True, seed=42, config_custom=None):
    config_balanco = config_custom if config_custom is not None else config_falhas_padrao()
    df_com, log_eventos = balancear_falhas(df_original.copy(), coluna, config=config_balanco, return_log=True, seed=seed)
    if incluir_falhas_finais and (not df_com.empty):
        fim_serie = df_com['Datetime'].max()
        amp_final = config_balanco.get('oscilacao', {}).get('amp', (1.5, 3.0))
        delta_final = config_balanco.get('queda', {}).get('delta', (-10.0, -4.0))
        df_com = injetar_intervalo_por_tempo(df_com, coluna, fim_serie - pd.Timedelta(minutes=180), fim_serie - pd.Timedelta(minutes=150), modo=LABEL_OSC, amp=amp_final, evento_id='oscilacao_final')
        df_com = injetar_intervalo_por_tempo(df_com, coluna, fim_serie - pd.Timedelta(minutes=90), fim_serie - pd.Timedelta(minutes=30), modo=LABEL_LACUNA, evento_id='lacuna_final')
        df_com = injetar_intervalo_por_tempo(df_com, coluna, fim_serie - pd.Timedelta(minutes=25), fim_serie - pd.Timedelta(minutes=10), modo=LABEL_QUEDA, delta=delta_final, evento_id='queda_final')
        extras = pd.DataFrame([{'evento_id': 'oscilacao_final', 'falha': LABEL_OSC}, {'evento_id': 'lacuna_final', 'falha': LABEL_LACUNA}, {'evento_id': 'queda_final', 'falha': LABEL_QUEDA}])
        log_eventos = pd.concat([log_eventos, extras], ignore_index=True)
    path_log = os.path.join(pasta_resultados, f'{nome_base}_eventos_injetados.csv')
    log_eventos.to_csv(path_log, index=False)
    print(f'✅ Log de eventos injetados salvo em: {os.path.abspath(path_log)}')
    return (df_com, log_eventos)

def config_base_variavel(coluna: str) -> dict:
    configs = {'MP2,5_1': {'oscilacao': {'duracao_pts': (36, 72), 'amp': (2.0, 4.0)}, 'lacuna': {'duracao_pts': (24, 48)}, 'queda': {'duracao_pts': (18, 36), 'delta': (-18.0, -6.0)}}, 'MP10_1': {'oscilacao': {'duracao_pts': (36, 72), 'amp': (4.0, 8.0)}, 'lacuna': {'duracao_pts': (24, 48)}, 'queda': {'duracao_pts': (18, 36), 'delta': (-35.0, -10.0)}}, 'Temperatura': {'oscilacao': {'duracao_pts': (36, 72), 'amp': (0.4, 1.2)}, 'lacuna': {'duracao_pts': (24, 48)}, 'queda': {'duracao_pts': (18, 36), 'delta': (-3.0, -1.0)}}, 'Umidade': {'oscilacao': {'duracao_pts': (36, 72), 'amp': (3.0, 8.0)}, 'lacuna': {'duracao_pts': (24, 48)}, 'queda': {'duracao_pts': (18, 36), 'delta': (-15.0, -5.0)}}}
    if coluna not in configs:
        raise ValueError(f'Variável sem configuração de falhas: {coluna}')
    return configs[coluna]

def config_falhas_aleatoria(seed, coluna: str):
    rng = np.random.default_rng(seed)
    base = config_base_variavel(coluna)
    return {'oscilacao': {'duracao_pts': base['oscilacao']['duracao_pts'], 'n_eventos': int(rng.integers(20, 31)), 'amp': base['oscilacao']['amp']}, 'lacuna': {'duracao_pts': base['lacuna']['duracao_pts'], 'n_eventos': int(rng.integers(20, 31))}, 'queda': {'duracao_pts': base['queda']['duracao_pts'], 'n_eventos': int(rng.integers(20, 31)), 'delta': base['queda']['delta']}}
