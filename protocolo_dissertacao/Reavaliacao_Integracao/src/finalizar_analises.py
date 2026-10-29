"""Resumo descritivo sem injeção e conferência dos controles de persistência."""
import argparse,sys
from pathlib import Path
import pandas as pd
import numpy as np
from reavaliar import persistir,medir
from consolidar_reavaliacao import regras_confirmadas

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fluxo',type=Path,required=True);ap.add_argument('--saida',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(a.fluxo/'src'))
 from executar_integrado import COLUNAS
 from ajustes import reamostrar_e_imputar
 escolhas=pd.read_csv(a.saida/'criterios_escolhidos_validacao.csv')
 pred=pd.read_csv(a.saida/'predicoes_sem_injecao.csv.gz',parse_dates=['tempo'])
 series=pd.read_csv(a.saida/'series_sem_injecao.csv.gz',parse_dates=['tempo'])
 rows=[];estados=[];ctrl=[];controle_eventos=[]
 for var,col in COLUNAS.items():
  serie=series[series.variavel.eq(var)].reset_index(drop=True);tempos=pd.DatetimeIndex(serie.tempo)
  raw=pd.DataFrame({'Datetime':tempos,col:serie.valor_bruto});ready=pd.DataFrame({'Datetime':tempos,col:serie.valor})
  marks=regras_confirmadas(raw,ready,col);disp=[]
  for cat in ['oscilacao','queda']:
   pp=pred[pred.variavel.eq(var)&pred.categoria.eq(cat)];c=escolhas[escolhas.variavel.eq(var)&escolhas.categoria.eq(cat)].iloc[0]
   idx=tempos.get_indexer(pp.tempo);m=np.zeros(len(tempos),bool);m[idx]=persistir(pp.votos,int(c.k),int(c.p));marks[cat]=m
   d=np.zeros(len(tempos),bool);d[idx]=True;disp.append(d)
   for nome,k,p in [('selecionada',int(c.k),int(c.p)),('uniao_sincronizada',1,1)]:
    sinal=persistir(pp.votos,k,p)
    rows.append(dict(variavel=var,categoria=cat,estrategia=nome,rodada=1,k=k,p=p,posicoes_avaliadas=len(pp),posicoes_sinalizadas=int(sinal.sum()),fracao_sinalizada=float(sinal.mean())))
  completo=np.logical_and.reduce(disp);frame=pd.DataFrame(dict(tempo=tempos,variavel=var,rodada=1,ml_disponivel=completo,**marks))
  frame['categorias']=[';'.join(c for c in marks if marks[c][i]) for i in range(len(frame))]
  frame['estado']=['sinalizacao' if c else ('sem_sinalizacao' if ok else 'dados_insuficientes_para_ml') for c,ok in zip(frame.categorias,completo)]
  estados.append(frame)
  for cat in ['stuck','stuck_at_zero']:
   ent=a.fluxo/'avaliacao_persistencia_integrada'/var/cat
   df=pd.read_csv(ent/'entrada_com_injecao.csv',parse_dates=['Datetime']);ev=pd.read_csv(ent/'eventos_injetados.csv',parse_dates=['inicio','fim'])
   base=reamostrar_e_imputar(df[['Datetime',col]],'10min');mm=regras_confirmadas(df,base,col)[cat];t=pd.DatetimeIndex(base.Datetime)
   y=np.zeros(len(t),bool)
   for e in ev.itertuples():y|=(t>=e.inicio)&(t<=e.fim)
   met,det=medir(t,y,mm,ev);ctrl.append(dict(variavel=var,categoria=cat,**met));controle_eventos.extend(dict(variavel=var,categoria=cat,**d) for d in det)
 pd.DataFrame(rows).to_csv(a.saida/'sem_injecao_resumo.csv',index=False)
 pd.concat(estados).to_csv(a.saida/'estado_hibrido_sem_injecao.csv.gz',index=False)
 pd.DataFrame(ctrl).to_csv(a.saida/'controles_persistencia_confirmacao.csv',index=False)
 pd.DataFrame(controle_eventos).to_csv(a.saida/'controles_persistencia_eventos.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(ctrl)[['variavel','categoria','eventos','cobertos']].to_string(index=False))
if __name__=='__main__':main()
