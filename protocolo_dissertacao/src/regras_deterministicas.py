"""Avaliação das regras de ausência e persistência das leituras."""
import os
import numpy as np
import pandas as pd
from ajustes import carregar_dados, filtrar_periodo, reamostrar_e_imputar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(BASE_DIR, 'dados')
PASTA_SAIDA = os.path.join(BASE_DIR, 'resultados_regras_deterministicas_v2')
FREQ = '10min'
COLUNAS_ANALISADAS = ['MP2,5_1', 'MP10_1', 'Temperatura', 'Umidade']
MIN_PONTOS_LACUNA = 6
MIN_PONTOS_STUCK = 24
MIN_PONTOS_STUCK_ZERO = 24
N_EVENTOS = 5
DURACAO_LACUNA = (6, 12)
DURACAO_STUCK = (24, 36)
DURACAO_STUCK_ZERO = (24, 36)
SEED_BASE = 20260823
TOL = 1e-09
MARGEM_ENTRE_EVENTOS = 12
MARGEM_PREEXISTENTE = 12

def _to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def _intervalos_contiguos(mask, datetimes, min_pontos=1):
    mask = np.asarray(mask, dtype=bool)
    datetimes = pd.to_datetime(datetimes).reset_index(drop=True)
    eventos = []
    inicio = None
    for i, flag in enumerate(mask):
        if flag and inicio is None:
            inicio = i
        terminou = inicio is not None and (not flag or i == len(mask) - 1)
        if terminou:
            fim = i - 1 if not flag else i
            tamanho = fim - inicio + 1
            if tamanho >= min_pontos:
                eventos.append({'inicio_idx': int(inicio), 'fim_idx': int(fim), 'inicio': datetimes.iloc[inicio], 'fim': datetimes.iloc[fim], 'duracao_pts': int(tamanho)})
            inicio = None
    return eventos

def _sobrepoe(a_ini, a_fim, b_ini, b_fim):
    return not (a_fim < b_ini or b_fim < a_ini)

def _eventos_para_mask(n, eventos, margem=0):
    mask = np.zeros(n, dtype=bool)
    if eventos is None:
        return mask
    if isinstance(eventos, pd.DataFrame):
        registros = eventos.to_dict('records')
    else:
        registros = list(eventos)
    for e in registros:
        if 'inicio_idx' not in e or 'fim_idx' not in e:
            continue
        i0 = max(0, int(e['inicio_idx']) - margem)
        i1 = min(n - 1, int(e['fim_idx']) + margem)
        mask[i0:i1 + 1] = True
    return mask

def detectar_lacunas(df, coluna, min_pontos=MIN_PONTOS_LACUNA):
    valores = _to_numeric(df[coluna])
    mask = valores.isna().to_numpy()
    eventos = _intervalos_contiguos(mask, df['Datetime'], min_pontos=min_pontos)
    for e in eventos:
        e['falha_detectada'] = 'lacuna'
        e['valor_constante'] = np.nan
    return eventos

def detectar_stuck_e_zero(df, coluna, min_stuck=MIN_PONTOS_STUCK, min_zero=MIN_PONTOS_STUCK_ZERO, tol=TOL):
    valores = _to_numeric(df[coluna]).to_numpy(dtype=float)
    datas = pd.to_datetime(df['Datetime']).reset_index(drop=True)
    eventos = []
    n = len(valores)
    i = 0
    while i < n:
        vi = valores[i]
        if np.isnan(vi):
            i += 1
            continue
        if np.isclose(vi, 0.0, atol=tol):
            j = i + 1
            while j < n:
                vj = valores[j]
                if np.isnan(vj):
                    break
                if not np.isclose(vj, 0.0, atol=tol):
                    break
                j += 1
            tamanho = j - i
            if tamanho >= min_zero:
                eventos.append({'falha_detectada': 'stuck_at_zero', 'inicio_idx': int(i), 'fim_idx': int(j - 1), 'inicio': datas.iloc[i], 'fim': datas.iloc[j - 1], 'duracao_pts': int(tamanho), 'valor_constante': 0.0})
            i = j
            continue
        j = i + 1
        while j < n:
            vj = valores[j]
            if np.isnan(vj):
                break
            if np.isclose(vj, vi, atol=tol):
                j += 1
            else:
                break
        tamanho = j - i
        if tamanho >= min_stuck:
            eventos.append({'falha_detectada': 'stuck', 'inicio_idx': int(i), 'fim_idx': int(j - 1), 'inicio': datas.iloc[i], 'fim': datas.iloc[j - 1], 'duracao_pts': int(tamanho), 'valor_constante': float(vi)})
        i = j
    return eventos

def detectar_falha(df, coluna, falha):
    if falha == 'lacuna':
        return detectar_lacunas(df, coluna, min_pontos=MIN_PONTOS_LACUNA)
    todos = detectar_stuck_e_zero(df, coluna, min_stuck=MIN_PONTOS_STUCK, min_zero=MIN_PONTOS_STUCK_ZERO)
    return [e for e in todos if e['falha_detectada'] == falha]

def preparar_periodo(dados, coluna, inicio, fim):
    df = filtrar_periodo(dados, inicio, fim)
    df = reamostrar_e_imputar(df, FREQ)
    if coluna not in df.columns:
        raise ValueError(f'Coluna não encontrada: {coluna}')
    cols = ['Datetime', coluna]
    col_observado = f'{coluna}_observado'
    if col_observado in df.columns:
        cols.append(col_observado)
    df = df[cols].copy()
    df[coluna] = _to_numeric(df[coluna])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df

def _intervalo_livre(ocupado, proibido, inicio, fim, margem):
    a = max(0, inicio - margem)
    b = min(len(ocupado), fim + margem)
    if ocupado[a:b].any():
        return False
    if proibido[a:b].any():
        return False
    return True

def _sortear_eventos(n_total, n_eventos, duracao_range, rng, proibido=None, margem=MARGEM_ENTRE_EVENTOS):
    ocupado = np.zeros(n_total, dtype=bool)
    if proibido is None:
        proibido = np.zeros(n_total, dtype=bool)
    else:
        proibido = np.asarray(proibido, dtype=bool)
    eventos = []
    dur_min, dur_max = duracao_range
    borda = max(30, dur_max + margem)
    if n_total <= 2 * borda + dur_max:
        raise ValueError('Período muito curto para inserir os eventos com segurança.')
    tentativas = 0
    max_tentativas = 20000
    while len(eventos) < n_eventos and tentativas < max_tentativas:
        tentativas += 1
        dur = int(rng.integers(dur_min, dur_max + 1))
        inicio = int(rng.integers(borda, n_total - borda - dur))
        fim = inicio + dur
        if _intervalo_livre(ocupado=ocupado, proibido=proibido, inicio=inicio, fim=fim, margem=margem):
            ocupado[inicio:fim] = True
            eventos.append((inicio, fim))
    if len(eventos) < n_eventos:
        raise RuntimeError(f'Não foi possível sortear {n_eventos} eventos sem sobreposição e longe de padrões preexistentes.')
    return sorted(eventos)

def injetar_eventos(df, coluna, falha, n_eventos, duracao_range, seed, eventos_preexistentes=None):
    rng = np.random.default_rng(seed)
    out = df.copy().sort_values('Datetime').reset_index(drop=True)
    out[coluna] = _to_numeric(out[coluna])
    proibido = _eventos_para_mask(n=len(out), eventos=eventos_preexistentes, margem=MARGEM_PREEXISTENTE)
    intervalos = _sortear_eventos(n_total=len(out), n_eventos=n_eventos, duracao_range=duracao_range, rng=rng, proibido=proibido)
    log = []
    for ordem, (i0, i1) in enumerate(intervalos, start=1):
        valor_injetado = np.nan
        if falha == 'lacuna':
            out.loc[i0:i1 - 1, coluna] = np.nan
        elif falha == 'stuck':
            valor = out.loc[i0, coluna]
            if pd.isna(valor) or np.isclose(float(valor), 0.0, atol=TOL):
                a = max(0, i0 - 20)
                b = min(len(out) - 1, i0 + 20)
                candidatos = out.loc[a:b, coluna]
                candidatos = _to_numeric(candidatos).dropna()
                candidatos = candidatos[~np.isclose(candidatos.astype(float), 0.0, atol=TOL)]
                if candidatos.empty:
                    raise RuntimeError(f'Sem valor não-zero válido para injetar stuck em {coluna}.')
                valor = float(candidatos.iloc[0])
            valor_injetado = float(valor)
            out.loc[i0:i1 - 1, coluna] = valor_injetado
        elif falha == 'stuck_at_zero':
            valor_injetado = 0.0
            out.loc[i0:i1 - 1, coluna] = 0.0
        else:
            raise ValueError(f'Falha desconhecida: {falha}')
        log.append({'evento_id': f'{falha}_{ordem:02d}', 'falha': falha, 'inicio_idx': int(i0), 'fim_idx': int(i1 - 1), 'inicio': out.loc[i0, 'Datetime'], 'fim': out.loc[i1 - 1, 'Datetime'], 'duracao_pts': int(i1 - i0), 'valor_injetado': valor_injetado})
    return (out, pd.DataFrame(log))

def _achar_sobreposicao(evento, candidatos):
    for j, cand in candidatos.iterrows():
        if _sobrepoe(int(evento['inicio_idx']), int(evento['fim_idx']), int(cand['inicio_idx']), int(cand['fim_idx'])):
            return j
    return None

def avaliar_eventos(log_injetados, eventos_detectados_pos, eventos_preexistentes):
    inj = log_injetados.copy().reset_index(drop=True)
    det_pos = pd.DataFrame(eventos_detectados_pos).copy()
    pre = pd.DataFrame(eventos_preexistentes).copy()
    colunas_evento = ['falha_detectada', 'inicio_idx', 'fim_idx', 'inicio', 'fim', 'duracao_pts', 'valor_constante']
    if det_pos.empty:
        det_pos = pd.DataFrame(columns=colunas_evento)
    if pre.empty:
        pre = pd.DataFrame(columns=colunas_evento)
    inj['detectado'] = False
    inj['indice_deteccao_pos'] = pd.NA
    deteccoes_pos_usadas_por_injecao = set()
    for i, row in inj.iterrows():
        candidatos = det_pos[det_pos['falha_detectada'].astype(str) == str(row['falha'])]
        idx_match = _achar_sobreposicao(row, candidatos)
        if idx_match is not None:
            inj.loc[i, 'detectado'] = True
            inj.loc[i, 'indice_deteccao_pos'] = int(idx_match)
            deteccoes_pos_usadas_por_injecao.add(idx_match)
    adicionais = det_pos.loc[~det_pos.index.isin(deteccoes_pos_usadas_por_injecao)].copy()
    adicionais['classificacao'] = 'nova_nao_explicada'
    adicionais['indice_preexistente'] = pd.NA
    for idx, row in adicionais.iterrows():
        candidatos_pre = pre[pre['falha_detectada'].astype(str) == str(row['falha_detectada'])]
        idx_pre = _achar_sobreposicao(row, candidatos_pre)
        if idx_pre is not None:
            adicionais.loc[idx, 'classificacao'] = 'compativel_com_preexistente'
            adicionais.loc[idx, 'indice_preexistente'] = int(idx_pre)
    total_injetados = len(inj)
    injetados_detectados = int(inj['detectado'].sum())
    falsos_negativos_injetados = total_injetados - injetados_detectados
    n_preexistentes = len(pre)
    n_adicionais = len(adicionais)
    n_compativeis_pre = int((adicionais['classificacao'] == 'compativel_com_preexistente').sum())
    n_novas = int((adicionais['classificacao'] == 'nova_nao_explicada').sum())
    resumo = {'eventos_injetados': int(total_injetados), 'eventos_injetados_detectados': int(injetados_detectados), 'falsos_negativos_eventos_injetados': int(falsos_negativos_injetados), 'taxa_deteccao_eventos_injetados': injetados_detectados / total_injetados if total_injetados else np.nan, 'padroes_preexistentes_base': int(n_preexistentes), 'deteccoes_pos_injecao_total': int(len(det_pos)), 'deteccoes_adicionais_pos_injecao': int(n_adicionais), 'adicionais_compativeis_com_preexistentes': int(n_compativeis_pre), 'adicionais_novas_nao_explicadas': int(n_novas)}
    return (resumo, inj, pre, det_pos, adicionais)

def salvar_eventos_csv(pasta_var, falha, log_injetados, preexistentes, detectados_pos, comparacao_injetados, adicionais):
    log_injetados.to_csv(os.path.join(pasta_var, f'{falha}_01_eventos_injetados.csv'), index=False)
    pd.DataFrame(preexistentes).to_csv(os.path.join(pasta_var, f'{falha}_02_padroes_preexistentes.csv'), index=False)
    pd.DataFrame(detectados_pos).to_csv(os.path.join(pasta_var, f'{falha}_03_deteccoes_apos_injecao.csv'), index=False)
    comparacao_injetados.to_csv(os.path.join(pasta_var, f'{falha}_04_comparacao_eventos_injetados.csv'), index=False)
    adicionais.to_csv(os.path.join(pasta_var, f'{falha}_05_deteccoes_adicionais_classificadas.csv'), index=False)

def executar(dados_path, saida, inicio, fim):
    global PASTA_SAIDA
    PASTA_SAIDA = str(saida)
    inicio = pd.Timestamp(inicio)
    fim = pd.Timestamp(fim)
    dados = pd.read_csv(dados_path)
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print('\n======================================================')
    print('AVALIAÇÃO DE REGRAS DETERMINÍSTICAS - V2')
    print('Lacuna | Stuck | Stuck-at-zero')
    print('======================================================')
    print('\nEste programa NÃO executa os modelos de ML.')
    print('Ele separa eventos artificiais de padrões preexistentes na série.')
    if dados.empty:
        print('Nenhum dado válido encontrado.')
        return
    resumo_geral = []
    detalhes_injetados_geral = []
    preexistentes_geral = []
    adicionais_geral = []
    configs = {'lacuna': {'duracao': DURACAO_LACUNA, 'min_pontos': MIN_PONTOS_LACUNA}, 'stuck': {'duracao': DURACAO_STUCK, 'min_pontos': MIN_PONTOS_STUCK}, 'stuck_at_zero': {'duracao': DURACAO_STUCK_ZERO, 'min_pontos': MIN_PONTOS_STUCK_ZERO}}
    for idx_var, coluna in enumerate(COLUNAS_ANALISADAS):
        print('\n======================================================')
        print(f'VARIÁVEL: {coluna}')
        print('======================================================')
        df_base = preparar_periodo(dados=dados, coluna=coluna, inicio=inicio, fim=fim)
        if df_base.empty:
            print(f'Sem dados para {coluna}.')
            continue
        pasta_var = os.path.join(PASTA_SAIDA, coluna.replace(',', '').replace('.', ''))
        os.makedirs(pasta_var, exist_ok=True)
        for idx_falha, (falha, cfg) in enumerate(configs.items()):
            print(f'\n--- {falha} ---')
            eventos_preexistentes = detectar_falha(df_base, coluna, falha)
            print(f'Padrões preexistentes detectados na base: {len(eventos_preexistentes)}')
            seed = SEED_BASE + idx_var * 100 + idx_falha
            df_teste, log_injetados = injetar_eventos(df=df_base, coluna=coluna, falha=falha, n_eventos=N_EVENTOS, duracao_range=cfg['duracao'], seed=seed, eventos_preexistentes=eventos_preexistentes)
            eventos_detectados_pos = detectar_falha(df_teste, coluna, falha)
            resumo, detalhe_inj, detalhe_pre, detalhe_det_pos, detalhe_adicionais = avaliar_eventos(log_injetados=log_injetados, eventos_detectados_pos=eventos_detectados_pos, eventos_preexistentes=eventos_preexistentes)
            resumo.update({'variavel': coluna, 'falha': falha, 'min_pontos_regra': int(cfg['min_pontos']), 'minutos_regra': int(cfg['min_pontos'] * 10), 'seed': int(seed)})
            resumo_geral.append(resumo)
            detalhe_inj['variavel'] = coluna
            detalhe_inj['falha_avaliada'] = falha
            detalhes_injetados_geral.append(detalhe_inj)
            if not detalhe_pre.empty:
                detalhe_pre = detalhe_pre.copy()
                detalhe_pre['variavel'] = coluna
                detalhe_pre['falha_avaliada'] = falha
                preexistentes_geral.append(detalhe_pre)
            if not detalhe_adicionais.empty:
                detalhe_adicionais = detalhe_adicionais.copy()
                detalhe_adicionais['variavel'] = coluna
                detalhe_adicionais['falha_avaliada'] = falha
                adicionais_geral.append(detalhe_adicionais)
            salvar_eventos_csv(pasta_var=pasta_var, falha=falha, log_injetados=log_injetados, preexistentes=detalhe_pre, detectados_pos=detalhe_det_pos, comparacao_injetados=detalhe_inj, adicionais=detalhe_adicionais)
            print(f"Eventos artificiais detectados: {resumo['eventos_injetados_detectados']}/{resumo['eventos_injetados']}")
            print(f"Detecções adicionais compatíveis com padrões preexistentes: {resumo['adicionais_compativeis_com_preexistentes']}")
            print(f"Detecções adicionais novas/não explicadas: {resumo['adicionais_novas_nao_explicadas']}")
    df_resumo = pd.DataFrame(resumo_geral)
    path_resumo = os.path.join(PASTA_SAIDA, 'resumo_regras_deterministicas_v2.csv')
    df_resumo.to_csv(path_resumo, index=False)
    if detalhes_injetados_geral:
        df_injetados = pd.concat(detalhes_injetados_geral, ignore_index=True)
    else:
        df_injetados = pd.DataFrame()
    df_injetados.to_csv(os.path.join(PASTA_SAIDA, 'detalhes_eventos_injetados_v2.csv'), index=False)
    if preexistentes_geral:
        df_preexistentes = pd.concat(preexistentes_geral, ignore_index=True)
    else:
        df_preexistentes = pd.DataFrame()
    df_preexistentes.to_csv(os.path.join(PASTA_SAIDA, 'padroes_preexistentes_base_v2.csv'), index=False)
    if adicionais_geral:
        df_adicionais = pd.concat(adicionais_geral, ignore_index=True)
    else:
        df_adicionais = pd.DataFrame()
    df_adicionais.to_csv(os.path.join(PASTA_SAIDA, 'deteccoes_adicionais_classificadas_v2.csv'), index=False)
    if not df_resumo.empty:
        tabela = df_resumo.copy()
        tabela['injetados_detectados'] = tabela['eventos_injetados_detectados'].astype(str) + '/' + tabela['eventos_injetados'].astype(str)
        pivot = tabela.pivot(index='variavel', columns='falha', values='injetados_detectados').reset_index()
        ordem = ['variavel', 'lacuna', 'stuck', 'stuck_at_zero']
        ordem_existente = [c for c in ordem if c in pivot.columns]
        pivot = pivot[ordem_existente]
        path_tabela = os.path.join(PASTA_SAIDA, 'tabela_eventos_artigo_v2.csv')
        pivot.to_csv(path_tabela, index=False)
        cols_analise = ['variavel', 'falha', 'eventos_injetados', 'eventos_injetados_detectados', 'padroes_preexistentes_base', 'deteccoes_pos_injecao_total', 'deteccoes_adicionais_pos_injecao', 'adicionais_compativeis_com_preexistentes', 'adicionais_novas_nao_explicadas', 'min_pontos_regra', 'minutos_regra']
        cols_analise = [c for c in cols_analise if c in tabela.columns]
        tabela_analise = tabela[cols_analise].copy()
        path_tabela_analise = os.path.join(PASTA_SAIDA, 'tabela_analise_completa_v2.csv')
        tabela_analise.to_csv(path_tabela_analise, index=False)
        print('\n======================================================')
        print('TABELA - EVENTOS ARTIFICIAIS DETECTADOS')
        print('======================================================')
        print(pivot.to_string(index=False))
        print('\n======================================================')
        print('ANÁLISE COMPLETA')
        print('======================================================')
        print(tabela_analise.to_string(index=False))
        print('\nArquivos principais gerados:')
        print(path_resumo)
        print(path_tabela)
        print(path_tabela_analise)
    print('\n======================================================')
    print('EXECUÇÃO CONCLUÍDA')
    print('======================================================')
    print('Os padrões preexistentes foram mantidos separados dos eventos artificiais.')
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dados', required=True)
    p.add_argument('--saida', required=True)
    p.add_argument('--inicio', default='2023-10-07 00:00:00')
    p.add_argument('--fim', default='2023-11-09 00:00:00')
    a = p.parse_args()
    executar(a.dados, a.saida, a.inicio, a.fim)
