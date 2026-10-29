# Resultado da reavaliação da decisão integrada

A sincronização foi corrigida e a decisão integrada foi avaliada. A regra de concordância e persistência não mostrou melhora uniforme: em algumas tarefas, reduziu sinalizações com perda de eventos. Portanto, esta avaliação resolve a falta de uma comparação explícita da decisão conjunta, mas não comprova desempenho satisfatório em todas as tarefas nem aumento da confiabilidade dos sensores.

## Comparação no teste

As colunas F1 são médias por instante nas cinco rodadas. Cobertura é a soma dos eventos alcançados uma única vez. União significa um voto em horários sincronizados, sem persistência. Estes números não substituem o F1 por janela das análises individuais.

| Variável | Categoria | Votos / decisões | F1 união | F1 selecionada | Eventos união | Eventos selecionada | % sinalizada união → selecionada |
|---|---|---|---:|---:|---:|---:|---:|
| PM₁₀ | Oscilação | 4 / 1 | 0.611 | 0.590 | 27/27 | 25/27 | 66.3 → 47.0 |
| PM₁₀ | Queda | 1 / 1 | 0.347 | 0.347 | 26/28 | 26/28 | 77.0 → 77.0 |
| PM₂,₅ | Oscilação | 4 / 1 | 0.439 | 0.362 | 22/22 | 13/22 | 62.1 → 21.7 |
| PM₂,₅ | Queda | 2 / 1 | 0.263 | 0.296 | 24/26 | 24/26 | 88.5 → 76.4 |
| Temperatura | Oscilação | 3 / 1 | 0.449 | 0.455 | 25/25 | 24/25 | 94.8 → 70.4 |
| Temperatura | Queda | 2 / 2 | 0.269 | 0.229 | 24/26 | 21/26 | 90.2 → 73.5 |
| Umidade | Oscilação | 3 / 1 | 0.465 | 0.489 | 23/24 | 23/24 | 82.8 → 59.5 |
| Umidade | Queda | 1 / 2 | 0.277 | 0.280 | 25/27 | 25/27 | 82.8 → 80.4 |

## O que se conclui

- **Oscilação em PM₂,₅:** a regra passou de 22 para 13 eventos alcançados e reduziu o F1. A redução de sinalizações não caracteriza melhora nesta tarefa.
- **Queda em Temperatura:** o F1 e a cobertura diminuíram; o MCC médio ficou negativo (−0,059). A concordância escolhida na validação não resolveu esta tarefa.
- **Umidade:** houve aumento de F1 mantendo a cobertura da união sincronizada nas duas categorias; em Queda, contudo, ainda foram sinalizados cerca de 80% dos instantes avaliados.
- **Queda em PM₂,₅:** houve aumento de F1 mantendo 24/26 eventos; a fração sinalizada ainda foi elevada, cerca de 76%.
- **Oscilação em PM₁₀:** houve perda de dois eventos e de F1, embora a fração sinalizada tenha diminuído.
- **Queda em PM₁₀:** o critério escolhido foi a própria união; não houve mudança.

Na validação, nenhuma das 12 combinações atingiu 90% de cobertura em Queda de PM₁₀ ou de Umidade. Nesses casos, foi aplicado o fallback declarado de maior cobertura, sem esconder a insuficiência. No teste, a regra selecionada ficou abaixo da referência de sinalizar todos os instantes em Oscilação de PM₂,₅ e Queda de Temperatura. Essa referência é distinta daquela calculada por janela no texto anterior.

A saída conjunta com regras também foi exportada, com categorias simultâneas e indicação de indisponibilidade inicial de ML. As métricas de qualquer sinalização não comprovam classificação correta da categoria: uma regra de stuck-at-zero pode alcançar um trecho simulado como Queda. Uma fração elevada de sinalização também pode elevar artificialmente a cobertura. Por isso, a análise por categoria acima é indispensável.

## Limites e uso no texto

A análise é complementar e exploratória: o período de teste já tinha sido examinado. Os modelos foram treinados para reconhecer padrões em janelas; esta análise mede a associação de suas decisões no fim da janela com os intervalos injetados. A interpolação da preparação pode usar leituras futuras, portanto não há comprovação de operação causal em tempo real. As cinco rodadas não representam cinco períodos físicos independentes.

O fluxo conjunto foi executado no teste com Oscilação, Queda e Lacuna. Stuck e stuck-at-zero foram conferidos nos controles separados já existentes: 40 eventos ao todo, com cinco eventos de cada categoria por variável. A avaliação não contém um único conjunto com as cinco categorias injetadas simultaneamente e referência completa. Não se deve afirmar que esse teste foi realizado.

A conclusão pode afirmar que a integração foi implementada e avaliada, mas deve reconhecer que sua utilidade como detector com sinalização seletiva não foi demonstrada de modo uniforme. Um resultado desfavorável é um resultado científico; não deve ser corrigido por escolhas orientadas pelos números do teste. Uma nova demonstração de generalização exigirá critérios congelados e outro período ainda não utilizado.

## Próximo ajuste na dissertação

Inserir esta avaliação como experimento complementar, preservando as tabelas individuais. O arquivo `avaliacao_complementar.tex` contém uma tabela e uma interpretação curta para revisão. A seção deve informar a mudança do avanço na inferência, a seleção na validação, o aquecimento inicial, a confirmação sem projeção para o passado e o caráter exploratório. A conclusão não deve dizer que a nova decisão foi superior em geral.

O GitHub permanece para depois da conferência do texto e dos resultados, conforme combinado.
