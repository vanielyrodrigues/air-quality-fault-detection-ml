"""Une decisões sincronizadas e regras; mantém indisponibilidade explícita."""
import argparse,sys
from pathlib import Path
import pandas as pd
import numpy as np
from reavaliar import persistir,medir

def regras_confirmadas(bruta,preparada,col):
 from regras_deterministicas import detectar_lacunas,detectar_stuck_e_zero
 n=len(bruta);out={c:np.zeros(n,bool) for c in ['lacuna','stuck','stuck_at_zero']}
 for ev in detectar_lacunas(bruta,col)+detectar_stuck_e_zero(preparada,col):
  cat=ev['falha_detectada']; minimo=6 if cat=='lacuna' else 24
  out[cat][ev['inicio_idx']+minimo-1:ev['fim_idx']+1]=True
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fluxo',type=Path,required=True);ap.add_argument('--entradas',type=Path,required=True);ap.add_argument('--saida',type=Path,required=True);a=ap.parse_args()
 sys.path.insert(0,str(a.fluxo/'src'))
 from executar_integrado import COLUNAS
 from ajustes import reamostrar_e_imputar
 pred=pd.read_csv(a.saida/'predicoes_sincronizadas.csv.gz',parse_dates=['tempo'])
 escolhas=pd.read_csv(a.saida/'criterios_escolhidos_validacao.csv')
 rows=[];intervalos=[];met=[];detalhes=[]
 for var,col in COLUNAS.items():
  for r in range(1,6):
   ent=a.entradas/var/'bases'/f'rodada_{r}'
   df=pd.read_csv(ent/'teste.csv',parse_dates=['Datetime']);ev=pd.read_csv(ent/'log_teste.csv',parse_dates=['inicio','fim'])
   base=reamostrar_e_imputar(df[['Datetime',col]],'10min');tempos=pd.DatetimeIndex(base.Datetime)
   marcas=regras_confirmadas(df,base,col);disponivel={}
   for cat in ['oscilacao','queda']:
    p=pred[pred.variavel.eq(var)&pred.rodada.eq(r)&pred.particao.eq('teste')&pred.categoria.eq(cat)]
    c=escolhas[escolhas.variavel.eq(var)&escolhas.categoria.eq(cat)].iloc[0]
    idx=tempos.get_indexer(p.tempo);assert (idx>=0).all()
    m=np.zeros(len(df),bool);m[idx]=persistir(p.votos.to_numpy(),int(c.k),int(c.p));marcas[cat]=m
    d=np.zeros(len(df),bool);d[idx]=True;disponivel[cat]=d
   completo=np.logical_and.reduce(list(disponivel.values()))
   frame=pd.DataFrame(dict(tempo=tempos,variavel=var,rodada=r,ml_disponivel=completo,**marcas))
   cats=list(marcas)
   frame['categorias']=[';'.join(c for c in cats if marcas[c][i]) for i in range(len(df))]
   frame['estado']=['sinalizacao' if x else ('sem_sinalizacao' if ok else 'dados_insuficientes_para_ml') for x,ok in zip(frame.categorias,completo)]
   rows.append(frame)
   # Qualquer categoria vs qualquer alteração simulada: reconhecimento genérico,
   # não acerto de categoria nem confirmação de falha física.
   uniao=np.logical_or.reduce(list(marcas.values()))
   y=df.label.isin(['oscilacao','queda','lacuna']).to_numpy()
   m,ds=medir(tempos[completo],y[completo],uniao[completo],ev)
   met.append(dict(variavel=var,rodada=r,**m));detalhes.extend(dict(variavel=var,rodada=r,**d) for d in ds)
   for cat in cats:
    mask=marcas[cat];ss=np.flatnonzero(mask&~np.r_[False,mask[:-1]]);ee=np.flatnonzero(mask&~np.r_[mask[1:],False])
    intervalos.extend(dict(variavel=var,rodada=r,categoria=cat,inicio=tempos[s],fim=tempos[e],posicoes=e-s+1) for s,e in zip(ss,ee))
 pd.concat(rows).to_csv(a.saida/'estado_hibrido_teste.csv.gz',index=False)
 pd.DataFrame(intervalos).to_csv(a.saida/'intervalos_hibridos_teste.csv',index=False)
 pd.DataFrame(met).to_csv(a.saida/'qualquer_sinalizacao_teste.csv',index=False)
 pd.DataFrame(detalhes).to_csv(a.saida/'eventos_qualquer_sinalizacao.csv',index=False)
 print(pd.DataFrame(met).groupby('variavel').agg(f1=('f1','mean'),mcc=('mcc','mean'),sinalizada=('fracao_sinalizada','mean'),cobertos=('cobertos','sum'),eventos=('eventos','sum')).to_string())
if __name__=='__main__':main()
