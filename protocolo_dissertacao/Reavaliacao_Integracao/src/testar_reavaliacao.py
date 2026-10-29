import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"legado"))
from consolidar_reavaliacao import regras_confirmadas
import pandas as pd
import numpy as np
from reavaliar import persistir,medir,escolher
class Conferencia(unittest.TestCase):
 def test_persistencia_nao_retroage(self):
  self.assertEqual(persistir([2,2,2,0,2],2,2).tolist(),[False,True,True,False,False])
 def test_k_exige_concordancia(self):
  self.assertEqual(persistir([1,2,3,4],3,1).tolist(),[False,False,True,True])
 def test_atraso_evento_e_nao_janela(self):
  t=pd.date_range('2023-01-01',periods=6,freq='10min')
  ev=pd.DataFrame([dict(evento_id='a',inicio=t[1],fim=t[2]),dict(evento_id='b',inicio=t[4],fim=t[4])])
  m,e=medir(t,[0,1,1,0,1,0],[0,0,1,0,0,1],ev)
  self.assertEqual((m['cobertos'],m['eventos'],m['tp'],m['fp']),(1,2,1,1))
  self.assertEqual(e[0]['atraso_minutos'],10)
  self.assertEqual(m['episodios_externos'],1)
 def test_fallback_prioriza_cobertura(self):
  df=pd.DataFrame([dict(k=1,p=1,cobertos=2,eventos=3,f1=.1,episodios_externos=1,fracao_sinalizada=.9),dict(k=2,p=1,cobertos=1,eventos=3,f1=.9,episodios_externos=0,fracao_sinalizada=.1)])
  self.assertEqual(escolher(df).k,1)
 def test_cobertura_minima_antes_de_f1(self):
  df=pd.DataFrame([dict(k=1,p=1,cobertos=10,eventos=10,f1=.2,episodios_externos=1,fracao_sinalizada=.9),dict(k=2,p=1,cobertos=8,eventos=10,f1=.9,episodios_externos=0,fracao_sinalizada=.1)])
  self.assertEqual(escolher(df).k,1)
 def test_lacuna_apenas_apos_seis_posicoes(self):
  t=pd.date_range('2023-01-01',periods=8,freq='10min')
  raw=pd.DataFrame({'Datetime':t,'valor':[1,*([np.nan]*6),2]})
  ready=raw.copy();ready['valor']=ready.valor.interpolate()
  m=regras_confirmadas(raw,ready,'valor')
  self.assertEqual(np.flatnonzero(m['lacuna']).tolist(),[6])
 def test_stuck_zero_apenas_apos_24(self):
  t=pd.date_range('2023-01-01',periods=25,freq='10min')
  raw=pd.DataFrame({'Datetime':t,'valor':[0]*24+[2]})
  m=regras_confirmadas(raw,raw,'valor')
  self.assertEqual(np.flatnonzero(m['stuck_at_zero']).tolist(),[23])
  self.assertFalse(m['stuck'].any())
if __name__=='__main__':unittest.main()
