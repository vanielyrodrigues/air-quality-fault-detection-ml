# Avaliação retrospectiva da decisão integrada

Esta entrega registra a avaliação complementar do fluxo que reúne as regras
e classificadores. Os resultados da dissertação por janela não foram substituídos.

## Pergunta e cenários

O primeiro cenário aplica a união das sinalizações dos quatro classificadores
a cada uma das tarefas Oscilação e Queda, além das regras determinísticas.
Uma categoria é marcada em todas as posições de cada janela positiva ou do
intervalo identificado pela regra. Cada posição é comparada ao rótulo simulado;
um evento injetado é considerado coberto quando alguma posição de seu intervalo
foi sinalizada para a mesma categoria. Essa sobreposição não determina o início
do alerta, não estabelece correspondência exclusiva e não comprova uma falha física.

Foram usadas as quatro variáveis e as cinco rodadas anteriormente executadas,
com Oscilação, Queda e Lacuna simuladas no teste. A entrada da análise integrada
é novamente preparada e interpolada, inclusive em lacunas simuladas, antes da
classificação. Isso difere da preparação utilizada nas avaliações por janela
do texto anterior. As medidas de precisão, recall e F1 são por posição, médias
das cinco rodadas; portanto, não são comparáveis diretamente ao F1 por janela.
Uma posição fora da categoria injetada pode conter outra categoria simulada ou
um padrão preexistente. As contagens de referência são relativas às simulações.

Para stuck e stuck-at-zero foram executados oito controles separados: cinco
eventos de cada categoria por variável, em um conjunto baseado na primeira
rodada do período de teste. As regras e os quatro classificadores foram executados
conjuntamente. Os 40 eventos foram alcançados pela categoria correspondente.
Eles foram gerados de acordo com os mesmos limiares das regras; esse resultado
verifica o funcionamento nas condições de construção, não a capacidade de
identificar falhas físicas nem a taxa de falsos alarmes.

## Alternativa exploratória

Em vez de unir quatro modelos para cada tarefa, a segunda análise utiliza
o classificador com maior F1 médio de validação para cada variável e falha,
mantendo a janela e avanço selecionados. Critérios de desempate: menor desvio
padrão, menor janela, menor avanço e nome do modelo. As regras permanecem.
Essa comparação foi planejada depois que resultados de teste já eram conhecidos:
é exploratória, não uma comprovação independente de superioridade. Em algumas
configurações a área sinalizada diminuiu, mas também houve perda de eventos.

## Conteúdo

- avaliacao_simulada_final/: união dos quatro classificadores; métricas por
  rodada e categoria, eventos, procedência e síntese.
- avaliacao_alternativa_validacao/: alternativa de um classificador por tarefa.
- avaliacao_persistencia_integrada/: 40 controles e saídas completas do fluxo.
- src/: implementação e programas de avaliação.
- configuracoes_fixas.csv: 32 configurações; configuracoes_melhor_validacao.csv:
  oito configurações de maior F1 de validação.
- dados/: séries de treino, teste e logs de injeção das cinco rodadas.
- modelos_arquivados/: modelos disponíveis das três primeiras variáveis.
- modelos_reconstruidos/: modelos de Umidade e um modelo de PM10 reconstruídos
  a partir dos treinamentos arquivados, separados dos modelos anteriores.
- artefatos_rodada1/: modelos para repetir os controles de persistência.
- predicoes_umidade_arquivadas.csv e validacao_umidade_arquivada.csv:
  documentos publicados no commit c33b394, utilizados na conferência e seleção.

Um arquivo local do modelo Random Forest para Oscilação de PM10 na quinta rodada
estava vazio. Ele foi reajustado no treino registrado; todas as predições do teste
com injeção coincidiram individualmente com as arquivadas. Os 32 modelos de
Umidade reajustados para as rodadas 2 a 5 foram conferidos individualmente contra
as predições arquivadas, nos cenários com e sem injeção: 64 comparações sem
divergência de classe. Os oito modelos da primeira rodada de Umidade foram os
preparados na demonstração anterior.

## Reproduzir

Com Python 3.12 e as versões do requirements.txt, a partir desta pasta:

```bash
python -m pip install -r requirements.txt
python src/avaliar_fluxo_simulado.py --bases dados --configuracoes configuracoes_fixas.csv --artefatos-arquivados modelos_arquivados --artefatos-integrados modelos_reconstruidos --saida nova_uniao --rodadas 1 2 3 4 5
python src/avaliar_fluxo_simulado.py --bases dados --configuracoes configuracoes_melhor_validacao.csv --artefatos-arquivados modelos_arquivados --artefatos-integrados modelos_reconstruidos --saida nova_alternativa --rodadas 1 2 3 4 5
python src/avaliar_persistencia_integrada.py --dados dados_ordenados.csv --configuracoes configuracoes_fixas.csv --artefatos artefatos_rodada1 --saida novos_controles
```

Os modelos foram carregados com joblib. Carregue somente arquivos de origem
confiável. As pastas de saída devem ser novas. A demonstração foi realizada no
período de teste já conhecido e não substitui um teste em período ou sensor não
observado, nem uma avaliação com falhas reais anotadas.
