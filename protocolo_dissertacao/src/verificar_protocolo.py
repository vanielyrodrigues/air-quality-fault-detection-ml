"""Verificações de limites das janelas e seleção de configurações."""
import tempfile
from pathlib import Path
import numpy as np
import ml_pipeline as ml
from executar_revisao import select, MODELS, FAULTS

def run():
    cases = 0
    for n in range(1, 151):
        for j in [20, 40, 100]:
            for a in range(5, j + 1, 5):
                cases += 1
                starts = ml.inicios_janelas(n, j, a)
                if n < j:
                    assert starts == []
                    continue
                assert starts[0] == 0 and starts[-1] == n - j
                assert len(starts) == len(set(starts))
                assert set((i for s in starts for i in range(s, s + j))) == set(range(n))
    assert len(ml.inicios_janelas(144, 20, 5)) == 26
    assert ml.inicios_janelas(144, 100, 100) == [0, 44]
    assert ml.inicios_janelas(140, 20, 5) == list(range(0, 121, 5))
    rows = []
    for model in MODELS:
        for fault in FAULTS:
            for j, a, scores in [(20, 5, [1.0, 1.0, 0.0, 0.0, 0.0]), (40, 10, [0.6] * 5)]:
                for r, score in enumerate(scores, 1):
                    rows.append(dict(modelo=model, falha=fault, rodada=r, janela=j, avanco=a, elegivel=True, f1=score, recall=score, tp=1, fp=1, fn=1))
    with tempfile.TemporaryDirectory() as d:
        selected = select(rows, Path(d))
        assert len(selected) == 8
        assert all((x['janela'] == 40 and x['avanco'] == 10 for x in selected))
        assert all((abs(x['f1_medio'] - 0.6) < 1e-12 for x in selected))
    print(f'OK: {cases} casos de cobertura/limites; seleção usa a média das cinco rodadas, não vencedores individuais.')
if __name__ == '__main__':
    run()
