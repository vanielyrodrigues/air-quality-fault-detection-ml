"""Treinamento com seleção fixa pelo F1 médio de cinco rodadas de validação."""
import os
import sys
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PROTOCOLO_N_JOBS'] = '1'
import argparse, csv, hashlib, json, platform, time, warnings
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy, sklearn, xgboost, catboost
from sklearn.metrics import confusion_matrix
import ml_pipeline as ml
from protocolo_temporal import separar
from main import config_falhas_aleatoria, criar_base_com_falhas
MODELS = ['random_forest', 'mlp', 'xgboost', 'catboost']
FAULTS = ['oscilacao', 'queda']
WINDOWS = list(range(20, 101, 10))
PAIRS = [(j, a) for j in WINDOWS for a in range(5, j + 1, 5)]

def dump(path, value):
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def metrics(y, p):
    tn, fp, fn, tp = map(int, confusion_matrix(y, p, labels=[0, 1]).ravel())
    div = lambda a, b: a / b if b else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=div(tp, tp + fp), recall=div(tp, tp + fn), f1=div(2 * tp, 2 * tp + fp + fn), accuracy=div(tp + tn, len(y)), balanced_accuracy=(div(tp, tp + fn) + div(tn, tn + fp)) / 2, janelas=len(y), positivas=int(y.sum()), negativas=int(len(y) - y.sum()))

def predict(model, name, X):
    if name in ['xgboost', 'catboost']:
        prob = model.predict_proba(X)[:, 1]
        return ((prob >= (0.3 if name == 'xgboost' else 0.5)).astype(int), prob)
    return (model.predict(X).astype(int), model.predict_proba(X)[:, 1])

@lru_cache(maxsize=32)
def cached(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != 'features'}

def subset(root, part, r, j, a, fault):
    z = cached(str(Path(root) / 'cache' / f'{part}_{r}_{j}.npz'))
    requested = np.array(ml.inicios_janelas(int(z['n']), j, a))
    idx = np.searchsorted(z['starts'], requested)
    assert np.array_equal(z['starts'][idx], requested)
    return (z['X'][idx], z['y_' + fault][idx], z['ini'][idx], z['fim'][idx])

def fit_one(root, model_name, fault, r, j, a):
    X, y, _, _ = subset(root, 'treino', r, j, a, fault)
    model = ml._criar_modelo(model_name, (y == 0).sum() / max(1, y.sum()) * 1.5)
    if model_name == 'catboost':
        model.set_params(allow_writing_files=False)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        model.fit(X, y)
    return (model, [str(w.message) for w in warns], np.unique(y, return_counts=True)[1].tolist())

def validation_task(task):
    root, m, f, r, j, a = task
    dest = Path(root) / 'validacao_individual' / f'{m}_{f}_{r}_{j}_{a}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    start = time.monotonic()
    X, y, _, _ = subset(root, 'treino', r, j, a, f)
    V, v, _, _ = subset(root, 'validacao', r, j, a, f)
    counts = [int((y == i).sum()) for i in [0, 1]]
    row = dict(modelo=m, falha=f, rodada=r, janela=j, avanco=a)
    row['elegivel'] = min(counts) >= 2 and len(np.unique(v)) == 2
    if row['elegivel']:
        model, warns, _ = fit_one(root, m, f, r, j, a)
        p, _ = predict(model, m, V)
        row.update(metrics(v, p))
        row['avisos'] = warns
    else:
        row['motivo'] = 'Treino com menos de 2 por classe ou validação sem ambas as classes'
    row['segundos'] = time.monotonic() - start
    dump(dest, row)
    return row

def cache_partition(base, col, path, j):
    if path.exists():
        return
    starts = ml.inicios_janelas(len(base), j, 5)
    feats = [ml.extrair_atributos_janela(base.iloc[s:s + j], col) for s in starts]
    X = pd.DataFrame(feats).apply(pd.to_numeric, errors='coerce').fillna(0)
    arrays = dict(X=X.to_numpy(dtype=float), features=np.array(X.columns, dtype=str), starts=np.array(starts), n=np.array(len(base)), ini=base.Datetime.iloc[starts].astype(str).to_numpy(dtype=str), fim=base.Datetime.iloc[np.array(starts) + j - 1].astype(str).to_numpy(dtype=str))
    for f in FAULTS:
        c = np.r_[0, np.cumsum(base.label.eq(f).to_numpy())]
        arrays['y_' + f] = np.array([int(c[s + j] > c[s]) for s in starts])
    np.savez_compressed(path, **arrays)

def prepare(root, data, col):
    root = Path(root)
    for d in ['bases', 'cache', 'validacao_individual', 'predicoes', 'modelos']:
        (root / d).mkdir(parents=True, exist_ok=True)
    bases, resumo = separar(pd.read_csv(data), pd.Timestamp('2023-10-07'), pd.Timestamp('2023-11-09'), col)
    pd.DataFrame(resumo).to_csv(root / 'particoes.csv', index=False)
    offset = int.from_bytes(hashlib.sha256(col.encode()).digest()[:4], 'big') % 10000
    total = sum(map(len, bases.values()))
    for r in range(1, 6):
        for i, (part, base) in enumerate(bases.items(), 1):
            folder = root / 'bases' / f'rodada_{r}'
            folder.mkdir(exist_ok=True)
            prepared = folder / f'{part}.csv'
            if prepared.exists():
                sim = pd.read_csv(prepared)
                sim['Datetime'] = pd.to_datetime(sim.Datetime)
            else:
                seed = offset + i * 1000 + r
                cfg = config_falhas_aleatoria(seed, col)
                for p in cfg.values():
                    p['n_eventos'] = max(1, round(p['n_eventos'] * len(base) / total))
                dump(folder / f'config_{part}.json', dict(seed=seed, falhas=cfg))
                sim, log = criar_base_com_falhas(base, col, str(folder), part, False, seed, cfg)
                sim.to_csv(prepared, index=False)
                log.to_csv(folder / f'log_{part}.csv', index=False)
            for j in WINDOWS:
                cache_partition(sim, col, root / 'cache' / f'{part}_{r}_{j}.npz', j)
            if part == 'teste':
                for j in WINDOWS:
                    cache_partition(base, col, root / 'cache' / f'sem_injecao_{r}_{j}.npz', j)
        print(f'BASES/CACHE rodada {r} prontas', flush=True)
    return bases

def select(rows, root):
    by = defaultdict(list)
    for row in rows:
        by[row['modelo'], row['falha'], row['janela'], row['avanco']].append(row)
    agg = []
    for (m, f, j, a), rr in sorted(by.items()):
        assert len(rr) == 5 and {x['rodada'] for x in rr} == set(range(1, 6))
        eligible = all((x['elegivel'] for x in rr))
        row = dict(modelo=m, falha=f, janela=j, avanco=a, rodadas=5, elegivel=eligible)
        if eligible:
            vals = [x['f1'] for x in rr]
            tp, fp, fn = [sum((x[k] for x in rr)) for k in ['tp', 'fp', 'fn']]
            row.update(f1_medio=float(np.mean(vals)), dp_f1=float(np.std(vals, ddof=1)), f1_agregado=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, recall_medio=float(np.mean([x['recall'] for x in rr])))
        agg.append(row)
    pd.DataFrame(agg).to_csv(Path(root) / 'validacao_todas_configuracoes.csv', index=False)
    chosen = []
    for m in MODELS:
        for f in FAULTS:
            options = [x for x in agg if x['modelo'] == m and x['falha'] == f and x['elegivel']]
            if not options:
                raise ValueError(f'Sem configuração elegível nas 5 rodadas: {m}/{f}')
            chosen.append(sorted(options, key=lambda x: (-x['f1_medio'], x['dp_f1'], x['janela'], x['avanco']))[0])
    dump(Path(root) / 'configuracoes_congeladas.json', chosen)
    pd.DataFrame(chosen).to_csv(Path(root) / 'configuracoes_congeladas.csv', index=False)
    return chosen

def final_test(root, chosen):
    import joblib
    final = []
    for c in chosen:
        m, f, j, a = (c['modelo'], c['falha'], c['janela'], c['avanco'])
        for r in range(1, 6):
            target = Path(root) / 'predicoes' / f'{m}_{f}_{r}.csv'
            result = target.with_suffix('.json')
            if result.exists():
                final.extend(json.loads(result.read_text()))
                continue
            model, warns, _ = fit_one(root, m, f, r, j, a)
            joblib.dump(model, Path(root) / 'modelos' / f'{m}_{f}_{r}.joblib', compress=3)
            one = []
            preds = []
            for part, scenario in [('teste', 'com_injecao'), ('sem_injecao', 'sem_injecao')]:
                X, y, ini, fim = subset(root, part, r, j, a, f)
                p, prob = predict(model, m, X)
                row = dict(modelo=m, falha=f, rodada=r, janela=j, avanco=a, cenario=scenario)
                if scenario == 'com_injecao':
                    row.update(metrics(y, p))
                else:
                    row.update(janelas=len(y), sinalizadas=int(p.sum()), proporcao_sinalizada=float(p.mean()))
                row.update(grupos_sinalizados=ml.contar_eventos_binarios(p), avisos=warns)
                one.append(row)
                preds.extend((dict(modelo=m, falha=f, rodada=r, janela=j, avanco=a, cenario=scenario, Datetime_ini=str(s), Datetime_fim=str(e), rotulo=int(t) if scenario == 'com_injecao' else '', predicao=int(q), saida_modelo=float(z)) for s, e, t, q, z in zip(ini, fim, y, p, prob)))
            pd.DataFrame(preds).to_csv(target, index=False)
            dump(result, one)
            final.extend(one)
            print(f'TESTE {m}/{f}/rodada {r}', flush=True)
    pd.DataFrame(final).to_csv(Path(root) / 'resultados_finais.csv', index=False)
    return final

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dados', required=True)
    p.add_argument('--coluna', required=True)
    p.add_argument('--saida', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--somente-preparar', action='store_true')
    a = p.parse_args()
    root = Path(a.saida).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(coluna=a.coluna, dados_sha256=hashlib.sha256(Path(a.dados).read_bytes()).hexdigest(), janela_final=True, selecao='F1 médio aritmético de cinco rodadas na validação', desempate=['menor DP amostral', 'menor janela', 'menor avanço'], elegibilidade='Todas as cinco rodadas: treino >=2 por classe; validação com ambas as classes', versoes={k: v.__version__ for k, v in [('numpy', np), ('pandas', pd), ('scipy', scipy), ('sklearn', sklearn), ('xgboost', xgboost), ('catboost', catboost)]}, python=platform.python_version())
    manifest_file = root / 'protocolo.json'
    if manifest_file.exists():
        assert json.loads(manifest_file.read_text()) == manifest, 'Não misture execuções de protocolos/ambientes diferentes'
    else:
        dump(manifest_file, manifest)
    prepare(root, a.dados, a.coluna)
    if a.somente_preparar:
        return
    tasks = [(str(root), m, f, r, j, s) for m in MODELS for f in FAULTS for r in range(1, 6) for j, s in PAIRS]
    rows = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(validation_task, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 100 == 0:
                print(f'VALIDAÇÃO {i}/{len(tasks)}; {time.monotonic() - start:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(root / 'validacao_por_rodada.csv', index=False)
    chosen = select(rows, root)
    print(pd.DataFrame(chosen).to_string(index=False), flush=True)
    final_test(root, chosen)
    dump(root / 'CONCLUIDO.json', dict(validacoes=len(rows), avaliacoes_teste=80))
    print('CONCLUÍDO', flush=True)
if __name__ == '__main__':
    main()
