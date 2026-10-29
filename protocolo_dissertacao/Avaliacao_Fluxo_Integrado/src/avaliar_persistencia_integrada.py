"""Controles complementares de persistência no fluxo integrado completo."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ajustes import reamostrar_e_imputar
from regras_deterministicas import detectar_stuck_e_zero, injetar_eventos
from executar_integrado import COLUNAS, executar

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados', required=True); p.add_argument('--configuracoes', required=True)
    p.add_argument('--artefatos', required=True); p.add_argument('--saida', required=True)
    args = p.parse_args(); root = Path(args.saida); root.mkdir(parents=True, exist_ok=False)
    original = pd.read_csv(args.dados, parse_dates=['Datetime'])
    inicio, fim = '2023-11-02 09:30:00', '2023-11-09 00:00:00'
    original = original[(original.Datetime >= inicio) & (original.Datetime < fim)]
    resumo, eventos = [], []
    for iv,(var,coluna) in enumerate(COLUNAS.items()):
        base = reamostrar_e_imputar(original[['Datetime',coluna]], '10min')
        anteriores = detectar_stuck_e_zero(base,coluna)
        for k,classe in enumerate(('stuck','stuck_at_zero')):
            seed=20261026+iv*100+k
            serie, log = injetar_eventos(base,coluna,classe,5,(24,36),seed,anteriores)
            d = root/var/classe; d.mkdir(parents=True)
            entrada=d/'entrada_com_injecao.csv'; serie.to_csv(entrada,index=False)
            executar(argparse.Namespace(dados=str(entrada), configuracoes=args.configuracoes,
                artefatos=args.artefatos, bases_treino=None, saida=str(d/'saida'),
                inicio=inicio, fim=fim, rodada=1, variaveis=[var]))
            sinais=pd.read_csv(d/'saida'/'sinalizacoes_integradas.csv',parse_dates=['inicio','fim'])
            alvo=sinais[(sinais.metodo=='regra')&(sinais.categoria==classe)]
            qtd=0
            for e in log.itertuples():
                candidatos=alvo[(alvo.inicio<=e.fim)&(alvo.fim>=e.inicio)]
                coberto=not candidatos.empty; qtd+=int(coberto)
                eventos.append(dict(variavel=var,categoria=classe,evento=e.evento_id,
                    inicio=e.inicio,fim=e.fim,coberto=coberto,seed=seed))
            resumo.append(dict(variavel=var,categoria=classe,injetados=len(log),
                cobertos=qtd,nao_cobertos=len(log)-qtd,seed=seed))
            log.to_csv(d/'eventos_injetados.csv',index=False)
            print(f'Controle conjunto: {var}/{classe}: {qtd}/{len(log)}',flush=True)
    pd.DataFrame(resumo).to_csv(root/'resumo_controles.csv',index=False)
    pd.DataFrame(eventos).to_csv(root/'eventos_controles.csv',index=False)
    (root/'LEIA_PRIMEIRO.txt').write_text('Controles separados para stuck e stuck-at-zero sobre a série de teste já interpolada, com cinco injeções por categoria e variável. Cada evento foi injetado segundo os mesmos critérios usados pela regra. Os demais classificadores foram executados e suas sinalizações preservadas. A cobertura por sobreposição não prova reconhecimento de falhas físicas, nem avalia falsos positivos ou sensibilidade ao limiar. Estes cenários são distintos dos cinco cenários com Oscilação, Queda e Lacuna.\n',encoding='utf-8')

if __name__=='__main__': main()
