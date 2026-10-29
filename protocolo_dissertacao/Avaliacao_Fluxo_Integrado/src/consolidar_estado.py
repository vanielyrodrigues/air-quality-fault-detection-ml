"""Resumo retrospectivo multirrótulo das cinco categorias investigadas."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIAS = ['lacuna', 'stuck', 'stuck_at_zero', 'oscilacao', 'queda']

def estados_por_instante(series, sinais, janelas, execucao):
    resultado = []
    for var, base in series.groupby('variavel', sort=False):
        base = base.sort_values('Datetime').reset_index(drop=True)
        datas = pd.to_datetime(base.Datetime)
        n = len(base)
        marcas = {c: np.zeros(n, dtype=bool) for c in CATEGORIAS}
        for r in sinais[sinais.variavel.eq(var)].itertuples():
            marcas[r.categoria] |= ((datas >= pd.Timestamp(r.inicio)) &
                                    (datas <= pd.Timestamp(r.fim))).to_numpy()
        # A ausência de janela não pode ser interpretada como resultado negativo.
        avaliacao_completa = np.full(n, n >= 24, dtype=bool)
        modelos_esperados = [m for m in execucao['modelos'] if m['variavel'] == var]
        for m in modelos_esperados:
            coberto = np.zeros(n, dtype=bool)
            jj = janelas[(janelas.variavel == var) & (janelas.modelo == m['modelo']) &
                         (janelas.categoria == m['falha'])]
            for r in jj.itertuples():
                coberto |= ((datas >= pd.Timestamp(r.inicio)) &
                            (datas <= pd.Timestamp(r.fim))).to_numpy()
            avaliacao_completa &= coberto
        if len(modelos_esperados) != 8:
            avaliacao_completa[:] = False
        for i in range(n):
            categorias = [c for c in CATEGORIAS if marcas[c][i]]
            sem_medicao = pd.isna(base.valor_sem_interpolacao.iloc[i])
            if categorias:
                estado = 'SINALIZACAO_MULTIPLA' if len(categorias) > 1 else 'SINALIZACAO'
            elif not avaliacao_completa[i]:
                estado = 'DADOS_INSUFICIENTES'
            elif sem_medicao:
                estado = 'LEITURA_AUSENTE_ABAIXO_DO_LIMIAR'
            else:
                estado = 'SEM_SINALIZACAO'
            resultado.append(dict(variavel=var, instante_referencia=datas.iloc[i],
                estado=estado, categorias=';'.join(categorias),
                avaliacao_completa=bool(avaliacao_completa[i]),
                sem_medicao_original=bool(sem_medicao),
                **{c: int(marcas[c][i]) for c in CATEGORIAS}))
    return pd.DataFrame(resultado)

def consolidar(saida):
    pasta = Path(saida)
    series = pd.read_csv(pasta/'series_analisadas.csv', parse_dates=['Datetime'])
    sinais = pd.read_csv(pasta/'sinalizacoes_integradas.csv', parse_dates=['inicio', 'fim'])
    janelas = pd.read_csv(pasta/'classificacoes_por_janela.csv', parse_dates=['inicio', 'fim'])
    execucao = json.loads((pasta/'execucao.json').read_text(encoding='utf-8'))
    estado = estados_por_instante(series, sinais, janelas, execucao)
    estado.to_csv(pasta/'estado_por_instante.csv', index=False)
    intervalos = []
    for var, grupo in estado.groupby('variavel', sort=False):
        grupo = grupo.reset_index(drop=True)
        campos = ['estado', 'categorias', 'avaliacao_completa', 'sem_medicao_original']
        novos = grupo[campos].ne(grupo[campos].shift()).any(axis=1)
        for _, trecho in grupo.groupby(novos.cumsum()):
            primeiro = trecho.iloc[0]
            intervalos.append(dict(variavel=var, inicio=primeiro.instante_referencia,
                fim=trecho.instante_referencia.iloc[-1], posicoes=len(trecho),
                **{c: primeiro[c] for c in campos}))
    pd.DataFrame(intervalos).to_csv(pasta/'estado_por_intervalo.csv', index=False)
    (pasta/'criterio_estado.json').write_text(json.dumps(dict(
        categorias_avaliadas=CATEGORIAS,
        combinacao='Uniao dos intervalos positivos; para os classificadores, pelo menos um dos quatro modelos.',
        alcance='Demonstracao retrospectiva; nao se interpreta como fusao validada ou diagnostico fisico.',
        instante_referencia='Instante abrangido pelo intervalo sinalizado, nao horario de emissao de alerta.',
        sem_sinalizacao='Nenhum metodo sinalizou as cinco categorias no instante abrangido; leitura presente e cobertura completa.',
        multiplas_categorias='Preservadas sem prioridade exclusiva.',
        comparacao='Os resultados individuais e metricas anteriores nao sao substituidos por esta regra de uniao.'
    ), ensure_ascii=False, indent=2), encoding='utf-8')
    return estado

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saida', required=True)
    consolidar(p.parse_args().saida)
