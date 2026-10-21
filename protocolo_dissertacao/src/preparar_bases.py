"""Exporta as partições e as simulações sem treinar classificadores."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from protocolo_temporal import separar
from main import config_falhas_aleatoria, criar_base_com_falhas

VARIAVEIS = {'PM25': 'MP2,5_1', 'PM10': 'MP10_1', 'Temperatura': 'Temperatura', 'Umidade': 'Umidade'}

def run(dados, saida):
    original = pd.read_csv(dados)
    for nome, coluna in VARIAVEIS.items():
        root = Path(saida) / nome
        root.mkdir(parents=True, exist_ok=True)
        bases, resumo = separar(original, pd.Timestamp('2023-10-07'), pd.Timestamp('2023-11-09'), coluna)
        pd.DataFrame(resumo).to_csv(root / 'particoes.csv', index=False)
        offset = int.from_bytes(hashlib.sha256(coluna.encode()).digest()[:4], 'big') % 10000
        total = sum(map(len, bases.values()))
        for particao, base in bases.items():
            pasta = root / 'bases_sem_injecao'
            pasta.mkdir(exist_ok=True)
            base.to_csv(pasta / (particao + '.csv'), index=False)
        for rodada in range(1, 6):
            pasta = root / 'bases' / f'rodada_{rodada}'
            pasta.mkdir(parents=True, exist_ok=True)
            for i, (particao, base) in enumerate(bases.items(), 1):
                seed = offset + i * 1000 + rodada
                cfg = config_falhas_aleatoria(seed, coluna)
                for parametros in cfg.values():
                    parametros['n_eventos'] = max(1, round(parametros['n_eventos'] * len(base) / total))
                config = {'seed': seed, 'falhas': cfg}
                (pasta / f'config_{particao}.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
                simulada, eventos = criar_base_com_falhas(base, coluna, str(pasta), particao, False, seed, cfg)
                simulada.to_csv(pasta / f'{particao}.csv', index=False)
                eventos.to_csv(pasta / f'log_{particao}.csv', index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dados', required=True)
    p.add_argument('--saida', required=True)
    a = p.parse_args()
    run(a.dados, a.saida)
