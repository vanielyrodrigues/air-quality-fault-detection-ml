"""Reprodução retrospectiva sincronizada; seleção somente na validação.
Não demonstra causalidade em tempo real nem diagnóstico físico.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELOS=['catboost','mlp','random_forest','xgboost']

def persistir(votos,k,p):
    if not 1<=k<=4 or p<1: raise ValueError('Critério inválido')
    saida=np.zeros(len(votos),dtype=bool); run=0
    for i,v in enumerate(votos):
        run=run+1 if v>=k else 0
        saida[i]=run>=p
    return saida

def medir(tempos,y,pred,eventos):
    y=np.asarray(y,bool); pred=np.asarray(pred,bool)
    tp=int((y&pred).sum());fp=int((~y&pred).sum());fn=int((y&~pred).sum());tn=int((~y&~pred).sum())
    det=[]
    for e in eventos.itertuples():
        achados=tempos[(tempos>=e.inicio)&(tempos<=e.fim)&pred]
        det.append(dict(evento_id=e.evento_id,inicio=str(e.inicio),fim=str(e.fim),
            coberto=bool(len(achados)),atraso_minutos=(achados[0]-e.inicio).total_seconds()/60 if len(achados) else np.nan))
    den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    ini=pred & ~np.r_[False,pred[:-1]]
    finais=np.flatnonzero(pred & ~np.r_[pred[1:],False])
    externos=sum(not y[s:e+1].any() for s,e in zip(np.flatnonzero(ini),finais))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
      mcc=(tp*tn-fp*fn)/den if den else 0,precisao=tp/(tp+fp) if tp+fp else 0,
      recall=tp/(tp+fn) if tp+fn else 0,fracao_sinalizada=float(pred.mean()),
      eventos=len(det),cobertos=sum(d['coberto'] for d in det),episodios=int(ini.sum()),episodios_externos=int(externos),
      posicoes=len(y),positivas=int(y.sum()),f1_sinalizar_tudo=2*int(y.sum())/(len(y)+int(y.sum())) if len(y) else 0),det

def escolher(tab):
    """Cobertura agregada >=90%; depois F1 médio. Fallback maior cobertura."""
    tab=tab.copy(); tab['cobertura']=tab.cobertos/tab.eventos
    aptos=tab[tab.cobertura>=.9]
    if aptos.empty: aptos=tab[tab.cobertura.eq(tab.cobertura.max())]
    return aptos.sort_values(['f1','episodios_externos','fracao_sinalizada','p','k'],ascending=[False,True,True,True,True]).iloc[0]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fluxo',type=Path,required=True);ap.add_argument('--entradas',type=Path,required=True)
    ap.add_argument('--saida',type=Path,required=True)
    a=ap.parse_args();a.saida.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.fluxo/'src'))
    from executar_integrado import atributos,predizer,COLUNAS
    from ajustes import reamostrar_e_imputar
    cfg=pd.read_csv(a.fluxo/'configuracoes_fixas.csv')
    cache={};fontes=[];candidatos=[];eventout=[];preditout=[]
    for var,col in COLUNAS.items():
      for r in range(1,6):
        entrada=a.entradas/var/'bases'/f'rodada_{r}'
        for part in ['validacao','teste']:
          df=pd.read_csv(entrada/f'{part}.csv',parse_dates=['Datetime'])
          ev=pd.read_csv(entrada/f'log_{part}.csv',parse_dates=['inicio','fim'])
          base=reamostrar_e_imputar(df[['Datetime',col]],'10min')
          assert np.array_equal(base.Datetime,df.Datetime),'Grade mudou'
          for cat in ['oscilacao','queda']:
            cc=cfg[cfg.variavel.eq(var)&cfg.falha.eq(cat)]
            assert len(cc)==4 and set(cc.modelo)==set(MODELOS)
            inicio=int(cc.janela.max())-1
            tempos=pd.DatetimeIndex(base.Datetime.iloc[inicio:])
            assert (np.diff(tempos.asi8)==pd.Timedelta('10min').value).all()
            preds={}
            for c in cc.itertuples():
              nome=f'{c.modelo}_{cat}_{r}.joblib';obj=None
              for pasta in ['modelos_arquivados','modelos_reconstruidos','artefatos_rodada1']:
                path=a.fluxo/pasta/var/'modelos'/nome
                if path.exists() and path.stat().st_size:
                  try: obj=joblib.load(path);break
                  except (EOFError,ValueError): pass
              if obj is None: raise FileNotFoundError(nome)
              if hasattr(obj,'set_params') and c.modelo=='random_forest': obj.set_params(n_jobs=1)
              starts,X=atributos(base,col,int(c.janela),1)
              fim=np.asarray(starts)+int(c.janela)-1
              mask=fim>=inicio
              assert np.array_equal(fim[mask],np.arange(inicio,len(base)))
              preds[c.modelo]=predizer(obj,c.modelo,X[mask])
              fontes.append(dict(variavel=var,rodada=r,particao=part,categoria=cat,modelo=c.modelo,arquivo=str(path.relative_to(a.fluxo)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),janela=int(c.janela),avanco_original=int(c.avanco),avanco_reavaliacao=1))
            # Referência por instante: diretamente dos eventos, nunca o rótulo de janela.
            ec=ev[ev.falha.eq(cat)]
            y=np.zeros(len(tempos),bool)
            for e in ec.itertuples(): y|=(tempos>=e.inicio)&(tempos<=e.fim)
            assert np.array_equal(y,df.label.iloc[inicio:].eq(cat).to_numpy())
            votos=np.sum(list(preds.values()),axis=0)
            cache[var,r,part,cat]=(tempos,y,votos,ec)
            frame=pd.DataFrame(dict(tempo=tempos,variavel=var,rodada=r,particao=part,categoria=cat,referencia_instante=y.astype(int),votos=votos,**preds))
            preditout.append(frame)
            if part=='validacao':
              for k in range(1,5):
                for p in [1,2,3]:
                  met,_=medir(tempos,y,persistir(votos,k,p),ec)
                  candidatos.append(dict(variavel=var,categoria=cat,rodada=r,k=k,p=p,**met))
        print(f'{var} rodada {r}: predições sincronizadas concluídas',flush=True)
    cand=pd.DataFrame(candidatos);cand.to_csv(a.saida/'validacao_candidatos_por_rodada.csv',index=False)
    res=cand.groupby(['variavel','categoria','k','p'],as_index=False).agg(f1=('f1','mean'),cobertos=('cobertos','sum'),eventos=('eventos','sum'),episodios_externos=('episodios_externos','sum'),fracao_sinalizada=('fracao_sinalizada','mean'))
    escolhas=pd.DataFrame([escolher(g) for _,g in res.groupby(['variavel','categoria'])]);escolhas.to_csv(a.saida/'criterios_escolhidos_validacao.csv',index=False)
    saida=[]
    for escolha in escolhas.itertuples():
      for r in range(1,6):
        tempos,y,votos,ec=cache[escolha.variavel,r,'teste',escolha.categoria]
        for estrategia,k,p in [('selecionada',int(escolha.k),int(escolha.p)),('uniao_sincronizada',1,1)]:
          met,det=medir(tempos,y,persistir(votos,k,p),ec)
          saida.append(dict(variavel=escolha.variavel,categoria=escolha.categoria,rodada=r,estrategia=estrategia,k=k,p=p,**met))
          eventout.extend(dict(variavel=escolha.variavel,categoria=escolha.categoria,rodada=r,estrategia=estrategia,**d) for d in det)
    out=pd.DataFrame(saida);out.to_csv(a.saida/'teste_por_rodada.csv',index=False)
    resumo=out.groupby(['variavel','categoria','estrategia'],as_index=False).agg(f1=('f1','mean'),f1_dp=('f1','std'),mcc=('mcc','mean'),f1_sinalizar_tudo=('f1_sinalizar_tudo','mean'),fracao_sinalizada=('fracao_sinalizada','mean'),cobertos=('cobertos','sum'),eventos=('eventos','sum'),episodios_externos=('episodios_externos','sum'))
    resumo.to_csv(a.saida/'teste_resumo.csv',index=False)
    pd.concat(preditout).to_csv(a.saida/'predicoes_sincronizadas.csv.gz',index=False)
    pd.DataFrame(eventout).to_csv(a.saida/'eventos_teste.csv',index=False)
    pd.DataFrame(fontes).to_csv(a.saida/'fontes_modelos.csv',index=False)
    (a.saida/'metodo.json').write_text(json.dumps(dict(tipo='reavaliacao_exploratoria_retrospectiva',passo_minutos=10,persistencias=[1,2,3],votos=[1,2,3,4],selecao='validacao_cobertura90_F1medio_fallback_maior_cobertura',aquecimento='maximo_janela_menos1_por_tarefa',intervalos='fechados',modelos_retreinados=False,preparacao='interpolacao_publicada_inclusive_lacunas_simuladas',limites=['periodo_de_teste_ja_inspecionado','sem_validacao_fisica','rotulos_dos_modelos_sao_de_janela_nao_de_instante','nao_e_avaliacao_completa_das_cinco_categorias','interpolacao_pode_usar_valores_futuros']),ensure_ascii=False,indent=2))
    print(resumo.to_string(index=False),flush=True)
if __name__=='__main__':main()
