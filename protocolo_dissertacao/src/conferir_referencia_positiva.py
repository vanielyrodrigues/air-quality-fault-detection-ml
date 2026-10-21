"""Calcula a referência de todas as janelas positivas nas configurações avaliadas."""
import argparse
from pathlib import Path
import pandas as pd

def run(resultados):
    root = Path(resultados)
    rodadas = []
    for variavel in ['PM25', 'PM10', 'Temperatura', 'Umidade']:
        dados = pd.read_csv(root / variavel / 'resultados_finais.csv')
        dados = dados[dados.cenario.eq('com_injecao')].copy()
        positivos = dados.tp + dados.fn
        negativos = dados.tn + dados.fp
        dados['variavel'] = variavel
        dados['f1_referencia'] = 2 * positivos / (2 * positivos + negativos)
        dados['proporcao_positiva'] = positivos / (positivos + negativos)
        dados['diferenca_f1'] = dados.f1 - dados.f1_referencia
        rodadas.append(dados[['variavel','modelo','falha','rodada','janela','avanco','f1','f1_referencia','proporcao_positiva','diferenca_f1']])
    tabela = pd.concat(rodadas, ignore_index=True)
    destino = root / 'referencia'; destino.mkdir(exist_ok=True)
    tabela.to_csv(destino / 'por_rodada.csv', index=False)
    sintese = tabela.groupby(['variavel','falha','modelo','janela','avanco'], as_index=False).agg(
        f1_medio=('f1','mean'), dp_f1=('f1','std'), f1_referencia=('f1_referencia','mean'),
        proporcao_positiva=('proporcao_positiva','mean'), diferenca_f1=('diferenca_f1','mean'))
    sintese.to_csv(destino / 'sintese.csv', index=False)
    print(sintese.to_string(index=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--resultados', required=True)
    a = p.parse_args(); run(a.resultados)
