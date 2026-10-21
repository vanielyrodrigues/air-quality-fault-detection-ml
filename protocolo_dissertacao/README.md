# Protocolo experimental da dissertação

Detecção de padrões simulados de falha em séries de PM2,5, PM10, Temperatura e Umidade. São avaliados Random Forest, MLP, XGBoost e CatBoost para Oscilação e Queda, além de regras determinísticas para Lacuna, stuck e stuck-at-zero.

## Organização

| Pasta | Conteúdo |
|---|---|
| `src/` | Programas de preparação, simulação, treinamento e verificação |
| `dados/` | Base de entrada consolidada, anterior à reamostragem e à injeção |
| `resultados/PM25`, `PM10`, `Temperatura`, `Umidade` | Métricas, predições, configurações, bases e figuras |
| `resultados/regras/` | Registros da avaliação determinística |
| `resultados/referencia/` | Referência de classificar todas as janelas como positivas |
| `resultados/sintese/` | Comparações consolidadas entre modelos e variáveis |
| `docs/` | Protocolo, verificações e integridade dos arquivos |

`dados_ordenados.csv` é a base consolidada utilizada como entrada, não uma coleção dos arquivos individuais originais de aquisição. Esses arquivos individuais não estão incluídos nesta pasta. As bases preparadas e simuladas foram reconstruídas com os procedimentos publicados e confrontadas com os registros finais de partições e cobertura de eventos. As predições e métricas são as arquivadas da execução final; não foram substituídas por um novo treinamento.

## Protocolo

- Período de seleção: 07/10/2023 a 09/11/2023; última leitura da grade utilizada no teste: 08/11/2023 às 23h50.
- Resolução de dez minutos e partições cronológicas de 60%, 20% e 20%.
- Cinco sementes de injeção sobre os mesmos períodos, sem cinco períodos independentes.
- 108 pares de janela e avanço: J de 20 a 100, de dez em dez; A de cinco a J, de cinco em cinco.
- Seleção pelo F1 médio positivo nas cinco rodadas de validação. Empates: menor DP amostral, menor J, menor A.
- O par selecionado permanece fixo nas cinco rodadas do teste. Ajuste apenas no treinamento; validação não incorporada ao ajuste final.
- Janela final completa alinhada ao fim quando necessária, sem duplicação nem passagem entre partições.
- 24 atributos fixos acrescidos dos J valores ordenados de cada janela.

As predições positivas sem injeção não têm anotações independentes de falha física. A cobertura de um evento exige somente sobreposição com alguma janela positiva e não comprova delimitação ou alerta distinto. O período de teste foi examinado durante o desenvolvimento; sua independência é limitada. Os experimentos com regras e classificadores foram avaliados separadamente.

## Instalação

Usar Python 3.11 e executar os comandos dentro de `protocolo_dissertacao`:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python src/verificar_protocolo.py
```

`requirements.txt` conserva as versões registradas com o código experimental fornecido. Para a validação de publicação foram verificados sintaxe, funções preservadas, seleção, predições, eventos e regras; não foi repetido o treinamento integral no ambiente de dependências fixadas.

## Consultar e verificar os resultados já publicados

Não é necessário treinar novamente para usar as tabelas e predições incluídas.

```bash
python src/auditar_resultados.py resultados/PM25
python src/auditar_resultados.py resultados/PM10
python src/auditar_resultados.py resultados/Temperatura
python src/auditar_resultados.py resultados/Umidade
python src/conferir_referencia_positiva.py --resultados resultados
```

Cada variável tem 4.320 avaliações de validação e 80 avaliações finais: quatro modelos, duas falhas, cinco rodadas e dois cenários. As matrizes foram conferidas a partir das predições arquivadas. A comparação com a referência é calculada por rodada antes da média.

## Reexecutar os modelos

Utilizar uma pasta NOVA de saída, separada dos resultados arquivados. Os comandos abaixo são demorados e não fazem parte do envio ao GitHub.

```bash
python src/executar_revisao.py --dados dados/dados_ordenados.csv --coluna "MP2,5_1" --saida execucoes/PM25 --workers 2
python src/executar_revisao.py --dados dados/dados_ordenados.csv --coluna MP10_1 --saida execucoes/PM10 --workers 2
python src/executar_revisao.py --dados dados/dados_ordenados.csv --coluna Temperatura --saida execucoes/Temperatura --workers 2
python src/executar_revisao.py --dados dados/dados_ordenados.csv --coluna Umidade --saida execucoes/Umidade --workers 2
```

Ajustar `--workers` à memória e aos processadores disponíveis. Há retomada por tarefas; não iniciar dois processos sobre a mesma pasta. O arquivo `CONCLUIDO.json` é criado pelo programa ao finalizar uma nova execução; não foi fabricado para as evidências arquivadas.

Para gerar o relatório de uma nova execução concluída:

```bash
python src/gerar_relatorio.py --resultados execucoes/PM25 --coluna "MP2,5_1" --variavel PM25
```

Repetir para as outras variáveis com o nome e a coluna correspondentes. As figuras já publicadas em `resultados/*/figuras` foram preservadas da entrega consolidada.

## Preparar somente as bases

```bash
python src/preparar_bases.py --dados dados/dados_ordenados.csv --saida bases_reproduzidas
```

Esse comando não treina modelos e exporta a base sem injeção, as cinco rodadas simuladas, suas sementes e registros de eventos. Não executar sobre resultados de uma execução em andamento.

## Reexecutar as regras

```bash
python src/regras_deterministicas.py --dados dados/dados_ordenados.csv --saida execucoes/regras
```

O período padrão é 07/10/2023 00h00 a 09/11/2023 00h00. As durações e sementes permanecem no código. A verificação de publicação reproduziu 60/60 eventos artificiais, 14 padrões preexistentes e nenhuma detecção adicional nova não explicada.

## Integridade

`docs/SHA256SUMS.txt` registra os hashes dos arquivos do pacote. As rotinas de extração, simulação, criação dos modelos, métricas e seleção retidas foram comparadas estruturalmente com os códigos fornecidos. As alterações operacionais de empacotamento estão descritas em `docs/estrutura_codigo.json`.

A identificação dos autores de métodos, bibliotecas e dados deve ser preservada na dissertação e nas referências correspondentes. Esta pasta não atribui uma licença nova ao código ou aos dados.
