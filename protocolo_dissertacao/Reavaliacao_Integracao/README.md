# Reavaliação complementar da integração

Esta pasta contém código Python, dados utilizados, modelos já treinados, predições e resultados. Não substitui automaticamente as análises por janela da dissertação.

## O que mudou

Os quatro classificadores passam a produzir decisões nos mesmos horários, a cada dez minutos, usando a janela de cada configuração original. O avanço de inferência desta avaliação é uma leitura; os modelos não foram retreinados. Cada decisão usa a janela que termina naquele horário. Uma saída positiva não é estendida para o passado.

São examinados de um a quatro votos e de uma a três decisões consecutivas. Escolhe-se um critério por variável e categoria, utilizando somente a validação das cinco rodadas: exige-se cobertura agregada de pelo menos 90% dos eventos; entre os critérios elegíveis, maior F1 médio por instante. Empates: menos episódios externos, menor fração sinalizada, menor persistência e menos votos. Se nenhum critério alcança 90%, usa-se primeiro a maior cobertura disponível. O conjunto de teste recebe o critério escolhido sem novos ajustes.

A confirmação começa na segunda ou terceira decisão, quando exigida. Episódios agrupam instantes positivos consecutivos da grade de dez minutos. O evento é alcançado se pelo menos um instante positivo estiver dentro de seu intervalo fechado; conta uma única vez. O atraso é medido desde o início do evento até o primeiro instante positivo dentro dele. A referência por instante vem dos intervalos injetados, nunca do rótulo de uma janela inteira.

As primeiras Jmáx−1 posições de cada tarefa não têm comparação completa dos quatro classificadores. São registradas como indisponíveis para ML, não como decisões negativas. As métricas por tarefa usam os horários completos; todos os eventos da partição permanecem no denominador de cobertura. A análise conjunta usa os horários em que ambas as tarefas estão disponíveis. A tabela de eventos preserva os eventos não alcançados.

Lacuna usa a série antes da interpolação e confirma na sexta ausência consecutiva. Stuck e stuck-at-zero mantêm a série preparada do procedimento publicado e confirmam na 24ª posição da sequência. A saída mantém categorias simultâneas. Ausência de sinalização significa somente que nenhum critério disponível foi acionado. Uma sinalização pode coexistir com indisponibilidade de ML; consulte a coluna ml_disponivel.

## Reprodução

Requer Python 3.12 e as versões de `requirements.txt`. Na pasta extraída, execute:

```bash
python src/testar_reavaliacao.py
python src/reavaliar.py --fluxo fontes/fluxo --entradas fontes/entradas --saida reproducao
python src/prever_sem_injecao.py --fluxo fontes/fluxo --saida reproducao
python src/consolidar_reavaliacao.py --fluxo fontes/fluxo --entradas fontes/entradas --saida reproducao
python src/finalizar_analises.py --fluxo fontes/fluxo --saida reproducao
```

Os arquivos de modelos joblib pertencem a esta pesquisa; carregue apenas arquivos cuja origem seja confiável. `manifesto_fontes.json` registra os hashes das fontes. `fontes_modelos.csv` distingue modelos arquivados e reconstruídos no pacote anterior; esta reavaliação não ajusta modelos novos.

## Arquivos de resultados

- `validacao_candidatos_por_rodada.csv`: todas as 12 combinações, por tarefa e rodada.
- `criterios_escolhidos_validacao.csv`: critérios escolhidos e medidas na validação.
- `teste_por_rodada.csv` e `teste_resumo.csv`: teste da regra escolhida e da união sincronizada (um voto, sem persistência).
- `predicoes_sincronizadas.csv.gz`: decisões individuais, votos e referência por instante, na validação e no teste.
- `eventos_teste.csv`: cobertura e atraso por evento e categoria.
- `estado_hibrido_teste.csv.gz`: saídas das cinco categorias, disponibilidade e estado por instante.
- `intervalos_hibridos_teste.csv`: sequências positivas agrupadas, sem projeção para o passado.
- `qualquer_sinalizacao_teste.csv`: qualquer categoria positiva versus qualquer alteração simulada. Esta medida não demonstra acerto da categoria.
- `sem_injecao_resumo.csv`: sinalizações na primeira rodada sem injeção; não fornece acurácia física.
- `controles_persistencia_confirmacao.csv`: conferência das regras nos controles já existentes de stuck e stuck-at-zero; não é nova evidência independente.

F1 e MCC são calculados por instante nesta análise complementar, diferentemente do F1 por janela das tabelas anteriores. Nos resumos, F1, MCC e fração sinalizada são médias das cinco rodadas; as contagens de eventos são somadas. Episódio externo é uma sequência positiva que não intersecta nenhum evento injetado da categoria; não equivale automaticamente a falso diagnóstico físico. Um episódio muito longo pode conter um evento e muitos instantes fora dele; consulte também FP e fração sinalizada.

## Limites

É uma reavaliação exploratória: o período de teste já havia sido inspecionado anteriormente. Não é um novo teste independente. As cinco rodadas compartilham os mesmos períodos físicos. Os modelos foram treinados para reconhecer janelas que contêm um padrão, não para localizar exatamente cada leitura defeituosa; esta avaliação mede a associação temporal das decisões a eventos simulados.

A preparação mantém interpolação linear que pode usar leituras futuras dentro da partição, inclusive para lacunas simuladas. Por isso, decisões datadas no fim da janela não comprovam operação causal em tempo real. Para tal afirmação, seria necessário definir preparação causal, ajustar os modelos somente no treinamento e repetir a avaliação em período ainda não utilizado.

Os testes de ML contêm oscilação, queda e lacuna; stuck e stuck-at-zero são conferidos em controles separados. O estado conjunto testa as cinco saídas, mas não constitui avaliação de todos os tipos ocorrendo juntos em um único conjunto com referência completa. Os controles seguem os limiares das regras por construção. A série sem injeção não tem diagnóstico físico independente. Menos alertas, isoladamente, não demonstra melhor detecção nem aumento de confiabilidade do sensor.

A série sem injeção foi alinhada explicitamente à grade completa do teste (951 posições de 02/11/2023 09h30 a 08/11/2023 23h50). O arquivo original termina às 23h30: as duas posições finais foram mantidas como ausentes antes do preenchimento, sem serem descartadas.
