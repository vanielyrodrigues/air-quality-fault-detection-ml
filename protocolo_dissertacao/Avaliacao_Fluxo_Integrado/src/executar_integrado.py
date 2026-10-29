"""Análise retrospectiva conjunta, com saídas por categoria e método."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('PROTOCOLO_N_JOBS', '1')
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ml_pipeline as ml
from ajustes import reamostrar_e_imputar
from regras_deterministicas import detectar_lacunas, detectar_stuck_e_zero

COLUNAS = {'PM25': 'MP2,5_1', 'PM10': 'MP10_1',
           'Temperatura': 'Temperatura', 'Umidade': 'Umidade'}

def atributos(base, coluna, janela, avanco):
    starts = ml.inicios_janelas(len(base), janela, avanco)
    frame = pd.DataFrame([ml.extrair_atributos_janela(base.iloc[s:s+janela], coluna)
                          for s in starts])
    return starts, frame.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(float)

def predizer(modelo, nome, X):
    if nome in ('xgboost', 'catboost'):
        return (modelo.predict_proba(X)[:, 1] >= (0.30 if nome == 'xgboost' else 0.50)).astype(int)
    return modelo.predict(X).astype(int)

def obter_modelo(args, c, coluna, destino):
    nome, falha = c['modelo'], c['falha']
    arquivo = Path(args.artefatos)/c['variavel']/'modelos'/f'{nome}_{falha}_{args.rodada}.joblib'
    if arquivo.exists():
        return joblib.load(arquivo), {'origem': str(arquivo), 'ajustado_agora': False,
                                     'sha256': hashlib.sha256(arquivo.read_bytes()).hexdigest()}
    if not args.bases_treino:
        raise FileNotFoundError(f'Modelo ausente: {arquivo}. Informe --bases-treino para ajustar somente a configuração fixa.')
    treino = Path(args.bases_treino)/c['variavel']/'bases'/f'rodada_{args.rodada}'/'treino.csv'
    df = pd.read_csv(treino)
    if pd.to_datetime(df.Datetime).max() >= pd.Timestamp(args.inicio):
        raise ValueError('O período analisado deve ser posterior ao treinamento informado.')
    starts, X = atributos(df, coluna, int(c['janela']), int(c['avanco']))
    y = np.array([int(df.iloc[s:s+int(c['janela'])].label.eq(falha).any()) for s in starts])
    if len(np.unique(y)) != 2:
        raise ValueError(f'Treinamento sem ambas as classes: {treino}/{falha}')
    modelo = ml._criar_modelo(nome, 1.5 * (y == 0).sum()/max(1, y.sum()))
    if modelo is None:
        raise RuntimeError(f'Biblioteca indisponível: {nome}')
    if nome == 'catboost':
        modelo.set_params(allow_writing_files=False)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        modelo.fit(X, y)
    novo = destino/'modelos_ajustados'/c['variavel']/arquivo.name
    novo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, novo, compress=3)
    return modelo, {'origem': str(novo), 'ajustado_agora': True,
                    'treino_sha256': hashlib.sha256(treino.read_bytes()).hexdigest(),
                    'avisos': [str(a.message) for a in avisos]}

def executar(args):
    destino = Path(args.saida)
    destino.mkdir(parents=True, exist_ok=False)
    dados = pd.read_csv(args.dados)
    dados['Datetime'] = pd.to_datetime(dados['Datetime'], errors='raise')
    dados = dados[(dados.Datetime >= pd.Timestamp(args.inicio)) &
                  (dados.Datetime < pd.Timestamp(args.fim))].sort_values('Datetime')
    if dados.empty:
        raise ValueError('Período sem leituras.')
    configs = pd.read_csv(args.configuracoes)
    janelas, intervalos, origens, grades = [], [], [], []
    for var in args.variaveis:
        coluna = COLUNAS[var]
        entrada = dados[['Datetime', coluna]].copy()
        entrada[coluna] = pd.to_numeric(entrada[coluna], errors='coerce')
        bruta = entrada.set_index('Datetime').resample('10min').mean().reset_index()
        preparada = reamostrar_e_imputar(entrada, '10min')
        # Lacuna precede o preenchimento. Persistência mantém a preparação publicada.
        regras = detectar_lacunas(bruta, coluna) + detectar_stuck_e_zero(preparada, coluna)
        for e in regras:
            intervalos.append(dict(variavel=var, categoria=e['falha_detectada'],
                metodo='regra', modelo='', inicio=e['inicio'], fim=e['fim'],
                janela='', avanco='', rodada='', unidade='intervalo_regra'))
        base_export = bruta.rename(columns={coluna: 'valor_sem_interpolacao'})
        base_export['valor_preparado'] = preparada[coluna]
        base_export['sem_medicao_original'] = bruta[coluna].isna()
        base_export['variavel'] = var
        grades.append(base_export)
        cc = configs[configs.variavel.eq(var)]
        if len(cc) != 8 or cc.duplicated(['modelo', 'falha']).any():
            raise ValueError(f'Esperadas oito configurações fixas distintas: {var}')
        for c in cc.to_dict('records'):
            j, a = int(c['janela']), int(c['avanco'])
            starts, X = atributos(preparada, coluna, j, a)
            if not starts:
                origens.append(dict(variavel=var, modelo=c['modelo'], falha=c['falha'],
                                    estado='sem_janela_completa', janela=j, avanco=a))
                continue
            modelo, origem = obter_modelo(args, c, coluna, destino)
            n_atributos = (len(modelo.feature_names_) if c['modelo'] == 'catboost'
                           else modelo.n_features_in_)
            if n_atributos != X.shape[1]:
                raise ValueError('Quantidade de atributos incompatível com o modelo.')
            pred = predizer(modelo, c['modelo'], X)
            origens.append(dict(variavel=var, modelo=c['modelo'], falha=c['falha'],
                                rodada=args.rodada, janela=j, avanco=a, **origem))
            for s, p in zip(starts, pred):
                row = dict(variavel=var, categoria=c['falha'], metodo='classificador',
                           modelo=c['modelo'], inicio=preparada.Datetime.iloc[s],
                           fim=preparada.Datetime.iloc[s+j-1], janela=j, avanco=a,
                           rodada=args.rodada, unidade='janela', sinal=int(p))
                janelas.append(row)
                if p:
                    intervalos.append({k: v for k, v in row.items() if k != 'sinal'})
        print(f'{var}: regras e classificadores concluídos', flush=True)
    cols = ['variavel', 'categoria', 'metodo', 'modelo', 'inicio', 'fim', 'janela', 'avanco', 'rodada', 'unidade']
    alertas = pd.DataFrame(intervalos, columns=cols).sort_values(['variavel', 'inicio', 'metodo'])
    alertas.insert(0, 'id', [f'S{i:06d}' for i in range(1, len(alertas)+1)])
    series = pd.concat(grades)
    alertas['posicoes_sem_medicao_original'] = [int(series[
        series.variavel.eq(row.variavel) & (series.Datetime >= row.inicio) &
        (series.Datetime <= row.fim)].sem_medicao_original.sum()) for row in alertas.itertuples()]
    alertas['sobreposicoes_outra_categoria'] = ''
    relacoes = []
    for var, grupo in alertas.groupby('variavel'):
        for idx, row in grupo.iterrows():
            outros = grupo[(grupo.categoria != row.categoria) &
                           (grupo.inicio <= row.fim) & (grupo.fim >= row.inicio)]
            alertas.loc[idx, 'sobreposicoes_outra_categoria'] = ';'.join(outros.id)
            for _, outro in outros.iterrows():
                if row.id < outro.id:
                    relacoes.append(dict(variavel=var, id_a=row.id, id_b=outro.id,
                                         categoria_a=row.categoria, categoria_b=outro.categoria))
    alertas.to_csv(destino/'sinalizacoes_integradas.csv', index=False)
    pd.DataFrame(janelas, columns=cols + ['sinal']).to_csv(destino/'classificacoes_por_janela.csv', index=False)
    pd.DataFrame(relacoes, columns=['variavel', 'id_a', 'id_b', 'categoria_a', 'categoria_b']).to_csv(destino/'sobreposicoes.csv', index=False)
    series.to_csv(destino/'series_analisadas.csv', index=False)
    alertas.groupby(['variavel', 'categoria', 'metodo', 'modelo', 'unidade'], dropna=False).size().rename('quantidade').to_csv(destino/'resumo_sinalizacoes.csv')
    registro = dict(tipo='demonstracao_retrospectiva_integrada', injeção=False,
                    inicio=args.inicio, fim=args.fim, rodada=args.rodada,
                    regras_sobre={'lacuna': 'grade_sem_interpolacao', 'persistencia': 'serie_preparada'},
                    estrategia='multiplas_categorias_sem_prioridade',
                    observacao='Janelas positivas não equivalem a eventos distintos. Nenhum diagnóstico físico é atribuído.',
                    dados_sha256=hashlib.sha256(Path(args.dados).read_bytes()).hexdigest(), modelos=origens)
    (destino/'execucao.json').write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8')
    from consolidar_estado import consolidar
    consolidar(destino)
    gerar_figuras(destino)
    return alertas

def gerar_figuras(destino):
    destino = Path(destino)
    rodada = json.loads((destino/'execucao.json').read_text(encoding='utf-8'))['rodada']
    series = pd.read_csv(destino/'series_analisadas.csv', parse_dates=['Datetime'])
    alertas = pd.read_csv(destino/'sinalizacoes_integradas.csv', parse_dates=['inicio', 'fim']).fillna('')
    nomes = {'oscilacao': 'Oscilação', 'queda': 'Queda', 'lacuna': 'Lacuna',
             'stuck': 'Stuck', 'stuck_at_zero': 'Stuck-at-zero', 'random_forest': 'Random Forest',
             'mlp': 'MLP', 'xgboost': 'XGBoost', 'catboost': 'CatBoost', 'regra': 'Regra'}
    cores = {'oscilacao': '#b26b00', 'queda': '#246ca6', 'lacuna': '#777777',
             'stuck': '#24824a', 'stuck_at_zero': '#8d44ad'}
    for var, base in series.groupby('variavel'):
        var = base.variavel.iloc[0]
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={'height_ratios': [2, 3]})
        ax.plot(base.Datetime, base.valor_sem_interpolacao, lw=.8, color='black')
        ax.set_ylabel({'PM25': 'PM₂,₅ (µg/m³)', 'PM10': 'PM₁₀ (µg/m³)',
                       'Temperatura': 'Temperatura (°C)', 'Umidade': 'Umidade (%)'}[var])
        rows = alertas[alertas.variavel.eq(var)]
        faixas = sorted({f'{r.categoria} | {r.modelo or r.metodo}' for r in rows.itertuples()})
        for r in rows.itertuples():
            y = faixas.index(f'{r.categoria} | {r.modelo or r.metodo}')
            bx.plot([r.inicio, r.fim], [y, y], lw=3, alpha=.55, color=cores[r.categoria])
        labels = [' | '.join(nomes.get(x.strip(), x.strip()) for x in f.split('|')) for f in faixas]
        bx.set_yticks(range(len(faixas)), labels, fontsize=8)
        bx.set_xlabel('Data e hora; intervalos das janelas positivas e das regras')
        ax.set_title(f'{var}: análise retrospectiva conjunta, rodada {rodada}, sem injeção')
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(destino/f'sinalizacoes_{var}.png', dpi=160)
        plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados', required=True)
    p.add_argument('--configuracoes', required=True)
    p.add_argument('--artefatos', required=True)
    p.add_argument('--bases-treino')
    p.add_argument('--saida', required=True)
    p.add_argument('--inicio', required=True)
    p.add_argument('--fim', required=True, help='Limite exclusivo sobre os horários originais das leituras.')
    p.add_argument('--rodada', type=int, choices=range(1, 6), default=1)
    p.add_argument('--variaveis', nargs='+', choices=list(COLUNAS), default=list(COLUNAS))
    executar(p.parse_args())
