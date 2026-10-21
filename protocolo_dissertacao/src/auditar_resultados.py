"""Conferência das configurações e das métricas exportadas."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd

def run(root):
    root = Path(root)
    val = pd.read_csv(root / 'validacao_por_rodada.csv')
    selected = pd.read_csv(root / 'configuracoes_congeladas.csv')
    final = pd.read_csv(root / 'resultados_finais.csv')
    partitions = pd.read_csv(root / 'particoes.csv')
    test = partitions[partitions.particao.eq('teste')].iloc[0]
    start, end = (pd.Timestamp(test.inicio), pd.Timestamp(test.fim))
    grid = set(pd.date_range(start, end, freq='10min'))
    assert len(val) == 4320 and len(final) == 80 and (len(selected) == 8)
    for (model, fault), v in val.groupby(['modelo', 'falha']):
        candidates = []
        for (j, a), r in v.groupby(['janela', 'avanco']):
            assert len(r) == 5 and set(r.rodada) == set(range(1, 6))
            if r.elegivel.all():
                candidates.append((r.f1.mean(), r.f1.std(ddof=1), j, a))
        best = sorted(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))[0]
        c = selected[selected.modelo.eq(model) & selected.falha.eq(fault)].iloc[0]
        assert (int(c.janela), int(c.avanco)) == best[2:]
        np.testing.assert_allclose(c.f1_medio, best[0], atol=1e-12)
    checked = 0
    for file in (root / 'predicoes').glob('*.csv'):
        p = pd.read_csv(file, parse_dates=['Datetime_ini', 'Datetime_fim'])
        for scenario, w in p.groupby('cenario'):
            w = w.sort_values('Datetime_ini')
            one = w.iloc[0]
            assert w.Datetime_ini.iloc[0] == start and w.Datetime_fim.iloc[-1] == end
            assert not w.duplicated(['Datetime_ini', 'Datetime_fim']).any()
            assert (w.Datetime_fim - w.Datetime_ini).eq(pd.Timedelta(minutes=10 * (int(one.janela) - 1))).all()
            covered = set((t for row in w.itertuples() for t in pd.date_range(row.Datetime_ini, row.Datetime_fim, freq='10min')))
            assert covered == grid
            z = final[final.modelo.eq(one.modelo) & final.falha.eq(one.falha) & final.rodada.eq(one.rodada) & final.cenario.eq(scenario)].iloc[0]
            assert (int(z.janela), int(z.avanco)) == (int(one.janela), int(one.avanco))
            assert int(z.janelas) == len(w)
            if scenario == 'com_injecao':
                tp = int((w.rotulo.eq(1) & w.predicao.eq(1)).sum())
                fp = int((w.rotulo.eq(0) & w.predicao.eq(1)).sum())
                fn = int((w.rotulo.eq(1) & w.predicao.eq(0)).sum())
                tn = int((w.rotulo.eq(0) & w.predicao.eq(0)).sum())
                assert (int(z.tp), int(z.fp), int(z.fn), int(z.tn)) == (tp, fp, fn, tn)
                np.testing.assert_allclose(z.f1, 2 * tp / (2 * tp + fp + fn), atol=1e-12)
            else:
                assert int(z.sinalizadas) == int(w.predicao.sum())
                assert w.rotulo.isna().all()
            checked += 1
    assert checked == 80
    result = dict(status='OK', validacoes=4320, configuracoes_fixas=8, avaliacoes_finais=80, verificacoes=['Seleção independente pela média das 5 rodadas', 'Janela e avanço fixos nas 5 rodadas', 'Todas as posições do teste cobertas', 'Sem janelas duplicadas ou fora da partição', 'Matrizes e F1 reproduzidos a partir das predições', 'Sem rótulos artificiais de normalidade no teste sem injeção'])
    (root / 'AUDITORIA_FINAL.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('resultados')
    a = p.parse_args()
    run(a.resultados)
