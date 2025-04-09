## Antes de Modelar, É Preciso Explorar: O Mapa Antes da Jornada

Imagine que você recebeu uma base de dados contendo milhares de informações sobre o desempenho de uma plataforma educacional: número de acessos por aluno, tempo médio de estudo por disciplina, notas obtidas, taxa de evasão, participação em fóruns, entre outros.

Seu objetivo é claro: **prever quais alunos estão em risco de abandonar o curso**.

Mas antes de sair aplicando algoritmos e modelos sofisticados, há uma etapa crucial que determina o sucesso de todo o projeto: a **exploração dos dados**.

Sem entender o que os dados *realmente* dizem, você estará **modelando às cegas** — e correrá o risco de tomar decisões com base em ilusões estatísticas, dados corrompidos ou relações que simplesmente não existem.

## O Que é Exploração de Dados?

!!! info "Definição"
    **Exploração de Dados** (*Exploratory Data Analysis – EDA*) é o processo inicial de **investigação e compreensão profunda de um conjunto de dados**. Envolve estatísticas descritivas, visualizações e perguntas orientadoras para revelar padrões, inconsistências, relações e tendências.

É a etapa em que **descobrimos o que sabemos, o que não sabemos, e o que precisamos limpar, corrigir ou transformar**.

## Por Que Explorar Antes de Modelar?

- Para entender a **estrutura geral do dataset**
- Para detectar **erros, inconsistências ou valores ausentes**
- Para identificar **relações entre variáveis**
- Para descobrir **tendências, padrões e agrupamentos naturais**
- Para levantar hipóteses e **orientar o que vale a pena modelar**

Explorar é como **mapear o terreno antes da construção**. Você não começa um prédio sem conhecer o solo.

## Os Primeiros Passos da Exploração

### 1. **Olhar Geral**

```python
df.shape              # tamanho da base (linhas, colunas)
df.info()             # tipos de dados, valores nulos
df.describe()         # estatísticas descritivas numéricas
df.head()             # primeiras linhas da tabela
```

Aqui, já se identifica:

- A presença de variáveis irrelevantes
- Tipos de dados incorretos (ex: datas como texto)
- Colunas com muitos valores nulos

### 2. **Distribuição de Variáveis**

```python
import seaborn as sns
sns.histplot(df['tempo_estudo'], kde=True)
```

- Há valores negativos ou absurdos?
- A distribuição é simétrica, enviesada, multimodal?
- A variável está padronizada?

!!! tip "Dica"
    Variáveis com escalas muito diferentes, outliers extremos ou distribuição estranha podem exigir transformação antes da modelagem.

### 3. **Análise de Valores Faltantes**

```python
df.isnull().sum().sort_values(ascending=False)
```

- Quais colunas têm dados ausentes?
- Esses dados faltam por acaso ou segundo um padrão?
- Eles são removíveis, imputáveis ou informativos?

### 4. **Correlação entre Variáveis Numéricas**

```python
import matplotlib.pyplot as plt
import seaborn as sns

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
```

- Existem variáveis fortemente correlacionadas?
- Há redundância ou multicolinearidade?

### 5. **Relações com a Variável Alvo**

```python
sns.boxplot(x='evadiu', y='tempo_estudo', data=df)
```

- O tempo de estudo médio é diferente entre alunos que evadiram e os que não evadiram?
- Alguma variável se destaca como um forte preditor?

??? question "Reflexão: É possível encontrar insights antes mesmo de modelar?"
    Sim! Um bom EDA pode **revelar padrões poderosos** (ex: alunos que nunca acessaram a plataforma nas primeiras 2 semanas têm taxa de evasão 5x maior) que já orientam ações práticas — antes mesmo de treinar qualquer algoritmo.

## Além dos Gráficos: A Exploração é Investigativa

A exploração de dados é mais do que gráficos bonitos ou comandos de biblioteca. É uma **postura investigativa**. O analista deve fazer perguntas como:

- Esse valor faz sentido no mundo real?
- Existe uma explicação para esse agrupamento?
- O que surpreende? O que parece errado?

!!! example "Exemplo"
    Ao explorar uma base educacional, você descobre que 25% dos alunos têm nota final zero, mas acessaram todos os materiais. Um erro? Ou política da escola? Esse tipo de insight **não aparece automaticamente — é fruto da observação crítica.**

## Explorando com Python: Ferramentas que Ajudam

- `pandas-profiling` / `ydata-profiling`: relatórios automáticos de EDA
- `sweetviz`: compara conjuntos (ex: treino vs teste)
- `dtale`: interface web para exploração interativa

```python
from ydata_profiling import ProfileReport
report = ProfileReport(df, title="Relatório de Exploração")
report.to_file("eda.html")
```

## No Caso da Evasão: O Que Descobrimos com EDA?

No exemplo da plataforma educacional, a equipe descobriu durante a exploração que:

- 70% dos alunos evadidos nunca participaram dos fóruns
- Estudantes com tempo médio de estudo abaixo de 30min/dia tinham taxa de evasão 4x maior
- Havia dezenas de registros com “nota final” negativa — erro na integração dos sistemas

Esses insights permitiram:

- Criar variáveis mais informativas para os modelos
- Corrigir inconsistências antes de modelar
- Apontar ações imediatas para prevenção de evasão — mesmo antes de prever

## Explorar é Dar Voz aos Dados

Explorar dados não é apenas técnica — é um **ato de escuta profunda**.

É a etapa onde deixamos que os dados contem sua história, sem julgamentos, sem forçar modelos ou suposições.  
É onde surgem as boas perguntas, os alertas, os primeiros sinais de valor ou ruído.

**Porque antes de construir previsões, é preciso entender o que os dados querem dizer. E isso começa com a exploração.**