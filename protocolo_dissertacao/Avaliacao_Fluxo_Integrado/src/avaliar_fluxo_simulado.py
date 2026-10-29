"""Avalia retrospectivamente a união de intervalos no teste simulado."""
import os
for chave in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[chave] = '1'
os.environ['PROTOCOLO_N_JOBS'] = '1'
import argparse
import hashlib
import json
import pickle
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ajustes import reamostrar_e_imputar
from regras_deterministicas import detectar_lacunas, detectar_stuck_e_zero
from executar_integrado import COLUNAS, atributos, predizer
import ml_pipeline as ml

ALVOS = ('oscilacao', 'queda', 'lacuna')

def carregar_ou_ajustar(args, var, config, rodada, treino):
    modelo, alvo = config.modelo, config.falha
    nome = f'{modelo}_{alvo}_{rodada}.joblib'
    caminhos = [Path(args.artefatos_arquivados)/var/'modelos'/nome,
                Path(args.artefatos_integrados)/var/'modelos'/nome]
    ilegivel = []
    for arquivo in caminhos:
        if arquivo.exists():
            try:
                return joblib.load(arquivo), 'arquivado'
            except (EOFError, OSError, pickle.UnpicklingError, ValueError):
                ilegivel.append(str(arquivo))
    inicio, X = atributos(treino, COLUNAS[var], int(config.janela), int(config.avanco))
    y = np.array([int(treino.iloc[s:s+int(config.janela)].label.eq(alvo).any()) for s in inicio])
    if len(np.unique(y)) != 2:
        raise ValueError(f'Treino sem ambas as classes: {var}/{modelo}/{alvo}/{rodada}')
    obj = ml._criar_modelo(modelo, 1.5*(y == 0).sum()/max(1, y.sum()))
    if obj is None: raise RuntimeError(f'Biblioteca indisponível: {modelo}')
    if modelo == 'catboost': obj.set_params(allow_writing_files=False)
    with warnings.catch_warnings(record=True) as aviso:
        warnings.simplefilter('always'); obj.fit(X, y)
    caminho = Path(args.saida)/'modelos_reajustados'/var/nome
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, caminho, compress=3)
    return obj, 'reajustado_fonte_ilegivel' if ilegivel else 'reajustado_no_treino'

def matriz(y, pred):
    return dict(tp=int((y & pred).sum()), fp=int((~y & pred).sum()),
                fn=int((y & ~pred).sum()), tn=int((~y & ~pred).sum()))

def calcula(dados, var, rodada, args, configs):
    entrada = Path(args.bases)/var/'bases'/f'rodada_{rodada}'
    dados = pd.read_csv(entrada/'teste.csv', parse_dates=['Datetime'])
    treino = pd.read_csv(entrada/'treino.csv', parse_dates=['Datetime'])
    eventos = pd.read_csv(entrada/'log_teste.csv', parse_dates=['inicio','fim'])
    coluna = COLUNAS[var]
    bruta = dados[['Datetime', coluna]].copy()
    base = reamostrar_e_imputar(bruta, '10min')
    if len(base) != len(dados) or not np.array_equal(base.Datetime, dados.Datetime):
        raise ValueError('Grade do teste simulado mudou na preparação.')
    n = len(base)
    marca = {k: np.zeros(n, dtype=bool) for k in ('oscilacao','queda','lacuna','stuck','stuck_at_zero')}
    for evento in detectar_lacunas(bruta, coluna) + detectar_stuck_e_zero(base, coluna):
        marca[evento['falha_detectada']][evento['inicio_idx']:evento['fim_idx']+1] = True
    historico = []
    for c in configs[configs.variavel.eq(var)].itertuples(index=False):
        starts, X = atributos(base, coluna, int(c.janela), int(c.avanco))
        obj, origem = carregar_ou_ajustar(args, var, c, rodada, treino)
        previsto = predizer(obj, c.modelo, X)
        for s, p in zip(starts, previsto):
            if p: marca[c.falha][s:s+int(c.janela)] = True
        historico.append(dict(variavel=var, modelo=c.modelo, falha=c.falha, rodada=rodada,
                              janela=int(c.janela), avanco=int(c.avanco), origem=origem))
    linhas, eventos_saida = [], []
    for categoria in ALVOS:
        verdadeiro = dados.label.eq(categoria).to_numpy()
        positivo = marca[categoria]
        z = matriz(verdadeiro, positivo)
        precisao = z['tp']/(z['tp']+z['fp']) if z['tp']+z['fp'] else 0.
        recall = z['tp']/(z['tp']+z['fn']) if z['tp']+z['fn'] else 0.
        f1 = 2*z['tp']/(2*z['tp']+z['fp']+z['fn']) if 2*z['tp']+z['fp']+z['fn'] else 0.
        ecat = eventos[eventos.falha.eq(categoria)]
        cobertos = 0
        for e in ecat.itertuples():
            mask = dados.Datetime.between(e.inicio, e.fim).to_numpy()
            if not mask.any(): raise ValueError('Evento sem posições na grade.')
            acerto = bool((mask & positivo).any())
            cobertos += acerto
            eventos_saida.append(dict(variavel=var, rodada=rodada, categoria=categoria,
                                      evento=e.evento_id, inicio=e.inicio, fim=e.fim,
                                      posicoes=int(mask.sum()), coberto=acerto,
                                      pontos_sinalizados=int((mask & positivo).sum())))
        linhas.append(dict(variavel=var, rodada=rodada, categoria=categoria,
            eventos_injetados=len(ecat), eventos_cobertos=cobertos,
            eventos_nao_cobertos=len(ecat)-cobertos, **z, precisao=precisao,
            recall=recall, f1=f1, proporcao_pontos_sinalizados=float(positivo.mean())))
    # Para stuck e stuck-at-zero, estes testes não contêm rótulo injetado correspondente.
    # A sinalização é descritiva e não uma contagem de falsos positivos físicos.
    extra = [dict(variavel=var, rodada=rodada, categoria=c,
                  pontos_sinalizados=int(marca[c].sum())) for c in ('stuck','stuck_at_zero')]
    uniao = np.logical_or.reduce([marca[c] for c in marca])
    verdadeiro = dados.label.isin(ALVOS).to_numpy()
    z = matriz(verdadeiro, uniao)
    qualquer = dict(variavel=var, rodada=rodada, **z,
                    fracao_positiva=float(uniao.mean()),
                    definicao='qualquer categoria sinalizada versus qualquer evento injetado de oscilacao, queda ou lacuna')
    return linhas, eventos_saida, extra, historico, qualquer

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bases', required=True); p.add_argument('--configuracoes', required=True)
    p.add_argument('--artefatos-arquivados', required=True); p.add_argument('--artefatos-integrados', required=True)
    p.add_argument('--saida', required=True); p.add_argument('--rodadas', nargs='+', type=int, default=[1,2,3,4,5])
    p.add_argument('--variaveis', nargs='+', choices=list(COLUNAS), default=list(COLUNAS))
    args = p.parse_args(); saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=False)
    configs = pd.read_csv(args.configuracoes)
    registros, detalhes, outros, fontes, qualquer = [], [], [], [], []
    for rodada in args.rodadas:
        for var in args.variaveis:
            a,b,c,d,e = calcula(None, var, rodada, args, configs)
            registros.extend(a); detalhes.extend(b); outros.extend(c); fontes.extend(d); qualquer.append(e)
            print(f'Avaliado: {var}, rodada {rodada}', flush=True)
    df = pd.DataFrame(registros); df.to_csv(saida/'metricas_por_rodada.csv', index=False)
    pd.DataFrame(detalhes).to_csv(saida/'eventos_injetados.csv', index=False)
    pd.DataFrame(outros).to_csv(saida/'sinalizacoes_persistencia.csv', index=False)
    pd.DataFrame(fontes).to_csv(saida/'origem_modelos.csv', index=False)
    pd.DataFrame(qualquer).to_csv(saida/'qualquer_categoria.csv', index=False)
    resumo = df.groupby(['variavel','categoria'], sort=False).agg(
        rodadas=('rodada','count'), eventos=('eventos_injetados','sum'), cobertos=('eventos_cobertos','sum'),
        nao_cobertos=('eventos_nao_cobertos','sum'), f1_medio=('f1','mean'), f1_dp=('f1','std'),
        precisao_media=('precisao','mean'), recall_medio=('recall','mean'),
        prop_sinalizada_media=('proporcao_pontos_sinalizados','mean')).reset_index()
    resumo.to_csv(saida/'sintese_por_categoria.csv', index=False)
    (saida/'LEIA_PRIMEIRO.txt').write_text(
        'Avaliação retrospectiva no mesmo teste já examinado. Os rótulos indicam injeções, não falhas físicas.\n'
        'Oscilação, Queda e Lacuna são avaliadas por posição e cobertura de evento.\n'
        'Positivo em qualquer janela estende a marca a todas as posições da janela; isso não indica horário do alerta.\n'
        'Stuck e stuck-at-zero não foram injetados no conjunto simultâneo e sua saída é apenas descritiva.\n'
        'A preparação integrada interpola ausências antes do classificador, inclusive lacunas simuladas: '
        'isso difere da entrada dos classificadores nas tabelas originais.\n'
        'A estratégia de união foi construída após observação dos testes e não constitui teste independente.\n'
        'As métricas são por posição, ao contrário das métricas por janela do capítulo original.\n', encoding='utf-8')

if __name__ == '__main__': main()
