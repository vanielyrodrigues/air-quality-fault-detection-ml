"""Conferências de duração, cobertura das janelas e integridade das saídas."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from executar_integrado import ml, detectar_lacunas, detectar_stuck_e_zero
from consolidar_estado import CATEGORIAS, estados_por_instante, consolidar

def verificar(saida):
    # Estados finais: cobertura completa, coexistência, ausência e série curta.
    datas = pd.date_range('2023-11-10', periods=120, freq='10min')
    serie = pd.DataFrame({'variavel': 'PM25', 'Datetime': datas, 'valor_sem_interpolacao': 1.0})
    serie.loc[90, 'valor_sem_interpolacao'] = np.nan
    modelos = [dict(variavel='PM25', modelo=m, falha=f) for m in ['mlp', 'random_forest', 'xgboost', 'catboost'] for f in ['oscilacao', 'queda']]
    jj = pd.DataFrame([dict(variavel='PM25', modelo=m['modelo'], categoria=m['falha'], inicio=datas[0], fim=datas[-1]) for m in modelos])
    ss = pd.DataFrame([dict(variavel='PM25', categoria='lacuna', inicio=datas[10], fim=datas[15]), dict(variavel='PM25', categoria='queda', inicio=datas[14], fim=datas[20])])
    e = estados_por_instante(serie, ss, jj, {'modelos': modelos})
    assert e.loc[0, 'estado'] == 'SEM_SINALIZACAO'
    assert e.loc[10, 'categorias'] == 'lacuna'
    assert e.loc[14, 'estado'] == 'SINALIZACAO_MULTIPLA'
    assert e.loc[14, 'categorias'] == 'lacuna;queda'
    assert e.loc[90, 'estado'] == 'LEITURA_AUSENTE_ABAIXO_DO_LIMIAR'
    curto = estados_por_instante(serie.iloc[:19], ss.iloc[:0], jj.iloc[:0], {'modelos': modelos})
    assert curto.estado.eq('DADOS_INSUFICIENTES').all()
    # Três padrões em uma série: a lacuna deve ser vista antes da interpolação.
    n = 300
    x = 100 + .2*np.arange(n, dtype=float)
    x[60:66] = np.nan
    x[120:144] = 10
    x[200:224] = 0
    base = pd.DataFrame({'Datetime': pd.date_range('2023-11-10', periods=n, freq='10min'), 'valor': x})
    lacunas = detectar_lacunas(base, 'valor')
    persistencias = detectar_stuck_e_zero(base, 'valor')
    assert [(e['inicio_idx'], e['fim_idx']) for e in lacunas] == [(60, 65)]
    assert [(e['falha_detectada'], e['inicio_idx'], e['fim_idx']) for e in persistencias] == [('stuck', 120, 143), ('stuck_at_zero', 200, 223)]
    for n in (19, 20, 21, 140, 951):
        for j in (20, 100):
            for a in (5, j):
                starts = ml.inicios_janelas(n, j, a)
                if n < j:
                    assert starts == []
                else:
                    assert len(starts) == len(set(starts)) and starts[-1]+j == n
                    cobertos = np.zeros(n, bool)
                    for s in starts:
                        cobertos[s:s+j] = True
                    assert cobertos.all()
    d = Path(saida)
    sinais = pd.read_csv(d/'sinalizacoes_integradas.csv').fillna('')
    relacoes = pd.read_csv(d/'sobreposicoes.csv')
    lookup = sinais.set_index('id')
    assert sinais.id.is_unique
    for r in relacoes.itertuples():
        a, b = lookup.loc[r.id_a], lookup.loc[r.id_b]
        assert a.variavel == b.variavel and a.categoria != b.categoria
        assert pd.Timestamp(a.inicio) <= pd.Timestamp(b.fim)
        assert pd.Timestamp(a.fim) >= pd.Timestamp(b.inicio)
        assert b.name in a.sobreposicoes_outra_categoria.split(';')
        assert a.name in b.sobreposicoes_outra_categoria.split(';')
    estados = consolidar(saida)
    assert estados[CATEGORIAS].sum(axis=1).ge(2).equals(estados.estado.eq('SINALIZACAO_MULTIPLA'))
    print('Conferidos: padrões de controle, janelas completas, IDs, sobreposições e estados finais.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--saida', required=True)
    verificar(p.parse_args().saida)
