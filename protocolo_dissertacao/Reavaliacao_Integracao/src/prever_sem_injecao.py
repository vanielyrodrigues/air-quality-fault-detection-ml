"""Predições sincronizadas sem injeção, rodada 1; sem medir acurácia física."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,sys
import pandas as pd
import numpy as np
import joblib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fluxo',type=Path,required=True);ap.add_argument('--saida',type=Path,required=True);a=ap.parse_args();sys.path.insert(0,str(a.fluxo/'src'))
 from executar_integrado import atributos,predizer,COLUNAS
 from ajustes import reamostrar_e_imputar
 cfg=pd.read_csv(a.fluxo/'configuracoes_fixas.csv');raw=pd.read_csv(a.fluxo/'dados_ordenados.csv',parse_dates=['Datetime'])
 raw=raw[(raw.Datetime>=pd.Timestamp('2023-11-02 09:30'))&(raw.Datetime<=pd.Timestamp('2023-11-08 23:50'))]
 rows=[];series=[]
 for var,col in COLUNAS.items():
  entrada=raw[['Datetime',col]]
  grade=pd.date_range('2023-11-02 09:30','2023-11-08 23:50',freq='10min',name='Datetime')
  bruta=entrada.set_index('Datetime').resample('10min').mean().reindex(grade).reset_index()
  base=reamostrar_e_imputar(bruta,'10min')
  assert np.array_equal(base.Datetime,bruta.Datetime)
  series.append(pd.DataFrame(dict(tempo=base.Datetime,variavel=var,valor=base[col],valor_bruto=bruta[col])))
  for cat in ['oscilacao','queda']:
   cc=cfg[cfg.variavel.eq(var)&cfg.falha.eq(cat)];ini=int(cc.janela.max())-1;preds={}
   for c in cc.itertuples():
    nome=f'{c.modelo}_{cat}_1.joblib'
    paths=[a.fluxo/pasta/var/'modelos'/nome for pasta in ['modelos_arquivados','modelos_reconstruidos','artefatos_rodada1']]
    path=next(p for p in paths if p.exists() and p.stat().st_size)
    obj=joblib.load(path)
    if c.modelo=='random_forest':obj.set_params(n_jobs=1)
    starts,X=atributos(base,col,int(c.janela),1);fim=np.asarray(starts)+int(c.janela)-1;mask=fim>=ini
    preds[c.modelo]=predizer(obj,c.modelo,X[mask])
   rows.append(pd.DataFrame(dict(tempo=base.Datetime.iloc[ini:].to_numpy(),variavel=var,categoria=cat,rodada=1,votos=np.sum(list(preds.values()),axis=0),**preds)))
  print(f'{var} sem injeção concluído',flush=True)
 pd.concat(rows).to_csv(a.saida/'predicoes_sem_injecao.csv.gz',index=False)
 pd.concat(series).to_csv(a.saida/'series_sem_injecao.csv.gz',index=False)
if __name__=='__main__':main()
