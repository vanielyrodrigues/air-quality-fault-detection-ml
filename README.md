# Air Quality Fault Detection Using Machine Learning

# Fault Detection in Low-Cost Air Quality Sensors using Machine Learning

## Detecção de Falhas de Sensores no Monitoramento de Qualidade do Ar Utilizando Aprendizado de Máquina
---
## Protocolo revisado da dissertação

Os programas Python, a base consolidada de entrada e os resultados do protocolo temporal revisado estão na pasta [protocolo_dissertacao](protocolo_dissertacao/).

Consulte as [instruções de execução e a organização dos materiais](protocolo_dissertacao/README.md).

As demais pastas preservam materiais de etapas anteriores da pesquisa.

## Sobre o projeto

Este repositório contém todo o código-fonte desenvolvido durante a dissertação de Mestrado em Ciência da Computação da Universidade Estadual do Oeste do Paraná (UNIOESTE).

A pesquisa propõe uma metodologia baseada em Aprendizado de Máquina para detecção automática de falhas em sensores de baixo custo utilizados no monitoramento da qualidade do ar.

Além do código-fonte, este repositório disponibiliza todos os scripts utilizados durante o desenvolvimento da pesquisa, incluindo a preparação dos dados, simulação das falhas, treinamento dos modelos de aprendizado de máquina, geração automática das características e produção dos resultados experimentais apresentados na dissertação.

Este repositório constitui o **material suplementar oficial da dissertação**, permitindo a reprodução dos experimentos descritos no trabalho.

---

# Informações acadêmicas

**Título da dissertação**

Detecção de Falhas de Sensores no Monitoramento de Qualidade do Ar Utilizando Aprendizado de Máquina

**Programa**

Programa de Pós-Graduação em Ciência da Computação (PPGComp)

**Instituição**

Universidade Estadual do Oeste do Paraná (UNIOESTE)

**Autora**

Vaniely Rodrigues

**Orientador**

Prof. Dr. Roberto Milton Scheffel

---

# Objetivo

Avaliar classificadores supervisionados e regras determinísticas para identificar padrões indicativos de falhas em séries temporais de sensores ambientais de baixo custo. A avaliação considera falhas simuladas e sinalizações em dados sem injeção artificial, sem confirmação independente de falhas físicas.

---

# Variáveis ambientais analisadas

- PM2.5
- PM10
- Temperatura
- Umidade

---

# Falhas simuladas

O protocolo avalia cinco categorias:

- Oscilação e Queda: classificadores supervisionados de aprendizado de máquina.
- Lacuna, stuck e stuck-at-zero: regras determinísticas de ausência ou persistência das leituras.

As falhas são inseridas artificialmente em séries ambientais. Os resultados avaliam o reconhecimento desses padrões simulados; não confirmam, isoladamente, falhas físicas nos sensores.
---

# Modelos de Aprendizado de Máquina

Foram avaliados os seguintes algoritmos:

- Random Forest
- XGBoost
- CatBoost
- Multi-Layer Perceptron (MLP)

---

# Fluxo geral da metodologia

Para os classificadores de Oscilação e Queda:

1. Ordenação cronológica dos dados.
2. Separação em treinamento (60%), validação (20%) e teste (20%).
3. Preparação dos dados separadamente em cada partição.
4. Injeção das falhas simuladas e formação das janelas com seus atributos e rótulos.
5. Treinamento dos classificadores e seleção das configurações na validação.
6. Avaliação no teste com e sem injeção artificial.

As janelas permanecem dentro de suas respectivas partições. Lacuna, stuck e stuck-at-zero são avaliados separadamente por regras determinísticas.
---

# Estrutura do repositório

```text
air-quality-fault-detection-ml/
│
├── dataError_IC-master/
│   Código principal da dissertação
│
├── graficos_artigo_mp25_1/
│   Figuras utilizadas na dissertação
│
├── raw_data_DUSTAI/
│   Base de dados utilizada nos experimentos
│
├── main e ajustes originais/
│   Versões iniciais do desenvolvimento
│
├── LICENSE
│
└── README.md
```

---

# Código do protocolo revisado

Os programas utilizados no protocolo revisado estão em:

[protocolo_dissertacao/src](protocolo_dissertacao/src/)

Essa pasta reúne a preparação dos dados, a simulação das falhas, a formação das janelas, a extração dos atributos, o treinamento, a seleção na validação e a avaliação dos classificadores e das regras determinísticas.

Os resultados das quatro variáveis estão em:

[protocolo_dissertacao/resultados](protocolo_dissertacao/resultados/)

As instruções de execução estão no [README do protocolo](protocolo_dissertacao/README.md).

As pastas anteriores são preservadas como histórico do desenvolvimento.

---
# Resultados disponibilizados

Os resultados do protocolo revisado estão em
[protocolo_dissertacao/resultados](protocolo_dissertacao/resultados/).

Eles incluem configurações selecionadas, métricas,
predições por janela, registros de eventos injetados,
cobertura dos eventos, alertas externos, matrizes de
confusão, tabelas, figuras e relatórios de conferência.

As demais pastas podem conter resultados de etapas
anteriores e devem ser consultadas como histórico.


---

# Reprodutibilidade

As instruções de execução, as dependências e a organização dos arquivos estão no [README do protocolo revisado](protocolo_dissertacao/README.md).

O pacote inclui a base consolidada de entrada, os programas Python, as configurações, as sementes de simulação e os resultados arquivados. Os arquivos individuais originais de aquisição não estão incluídos nessa pasta.

As rotinas de conferência permitem recalcular métricas a partir das predições publicadas. Essa verificação é distinta de repetir o treinamento completo dos modelos.

Para consultar a versão citada na dissertação, utilize o [commit adfcaf7](https://github.com/vanielyrodrigues/air-quality-fault-detection-ml/tree/adfcaf730c47e0b035a240b3ac73605448ff1e96/protocolo_dissertacao).

---

# Tecnologias utilizadas

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- CatBoost
- Matplotlib
- PyCharm

---

# Como citar este trabalho

CRUZ, Vaniely Rodrigues da. **Detecção de Falhas de Sensores no Monitoramento de Qualidade do Ar Utilizando Aprendizado de Máquina**. 2026. Dissertação (Mestrado em Ciência da Computação) — Universidade Estadual do Oeste do Paraná, Cascavel, 2026.

---

# Contato

Vaniely Rodrigues

Programa de Pós-Graduação em Ciência da Computação

Universidade Estadual do Oeste do Paraná (UNIOESTE)

---

# Licença

Este projeto está disponível sob os termos da licença MIT.

Consulte o arquivo [LICENSE](LICENSE) para mais informações.
