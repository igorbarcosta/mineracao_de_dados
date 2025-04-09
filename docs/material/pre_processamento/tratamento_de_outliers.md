# Tratamento de Outliers

## Fora da Curva: Quando os Dados Gritam e Ninguém Escuta

Imagine que você está desenvolvendo um modelo para prever o preço de aluguel de apartamentos em uma grande cidade. Você coletou um conjunto robusto de dados: número de quartos, metragem, bairro, proximidade do metrô, entre outros.

Mas, ao analisar os dados, você encontra algo curioso:

- Um apartamento de 2 quartos e 50m² no centro listado por **R$ 150.000,00** mensais.
- Outro, com 4 quartos e 300m² em bairro nobre... por apenas **R$ 200,00**.

Esses valores não fazem sentido. Algo claramente está fora do padrão — e talvez fora da realidade.

Esses são os **outliers**: pontos que destoam dos demais, que gritam quando o resto fala baixo.  
E tratá-los com cuidado é essencial para garantir que seu modelo **não aprenda a partir de exceções** ou **erros de digitação**.

## O Que São Outliers, de Verdade?

!!! info "Definição"
    **Outliers** são observações que se afastam significativamente dos outros dados. Eles podem ser causados por erros, exceções reais ou eventos extremos.

Mas atenção: **nem todo ponto incomum é um erro**. Um apartamento de cobertura de luxo por R$ 80 mil pode parecer um outlier, mas talvez represente um segmento real do mercado.

??? question "Reflexão: Todo outlier deve ser removido?"
    Nem sempre! Às vezes, os outliers **contêm sinais valiosos**, como fraudes, comportamentos anômalos ou oportunidades de nicho. O segredo está em entender **por que** eles estão ali.

## Por Que os Outliers Preocupam?

Modelos de machine learning são altamente sensíveis a extremos — especialmente os lineares e os baseados em distância (como KNN e regressão linear).

!!! warning "Atenção"
    Outliers podem:
    
    - Distorcer médias e variâncias  
    - Inflar erros de predição  
    - Prejudicar escalas e normalizações  
    - Gerar *overfitting* se o modelo tentar aprender padrões únicos ou errados

## Detectando Outliers: Como Identificar os Fora da Curva

### 1. **Método do Intervalo Interquartil (IQR)**

Baseado nos quartis (valores que dividem os dados em 4 partes). Um valor é considerado outlier se estiver fora do intervalo:

$$
[Q1 - 1.5 \times IQR,\ Q3 + 1.5 \times IQR]
$$

```python
Q1 = df['preco'].quantile(0.25)
Q3 = df['preco'].quantile(0.75)
IQR = Q3 - Q1

filtro = (df['preco'] >= Q1 - 1.5 * IQR) & (df['preco'] <= Q3 + 1.5 * IQR)
df_filtrado = df[filtro]
```

### 2. **Z-Score**

Mede quantos desvios-padrão um valor está distante da média. Valores com `|z| > 3` geralmente são considerados outliers.

```python
from scipy.stats import zscore

df['zscore_preco'] = zscore(df['preco'])
df_filtrado = df[df['zscore_preco'].abs() < 3]
```

### 3. **Visualizações (sempre!)**

- **Boxplots** destacam valores extremos com facilidade
- **Scatterplots** ajudam a perceber outliers em mais de uma dimensão
- **Histograms** mostram caudas incomuns

```python
import seaborn as sns
sns.boxplot(x=df['preco'])
```

## Tratando Outliers: Remover, Corrigir ou Aceitar?

### Estratégias Comuns:

- **Remoção**: quando são erros claros ou valores impossíveis (ex: aluguel = R$ 1.000.000)
- **Capping (Winsorization)**: substitui valores extremos por limites superiores/inferiores aceitáveis
- **Transformações**: como log ou raiz quadrada, que reduzem o impacto dos valores extremos
- **Modelagem Separada**: tratar segmentos diferentes com modelos próprios (ex: imóveis populares vs. luxo)

```python
# Capping
limite_superior = Q3 + 1.5 * IQR
df['preco_capped'] = df['preco'].apply(lambda x: min(x, limite_superior))
```

!!! danger "Muito Cuidado"
    Nunca aplique tratamento de outliers **antes de entender o contexto**. O que é outlier em um bairro pode ser perfeitamente normal em outro.

## Na Prática: Detectando Outliers em Aluguéis

```python
import pandas as pd
import numpy as np

df = pd.read_csv('imoveis.csv')

# Usando IQR
Q1 = df['aluguel'].quantile(0.25)
Q3 = df['aluguel'].quantile(0.75)
IQR = Q3 - Q1

lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

df['eh_outlier'] = ~df['aluguel'].between(lim_inf, lim_sup)
```

## E Quando o Outlier É a Informação Mais Preciosa?

Imagine que, em um conjunto de dados financeiros, as transações consideradas "fora da curva" representam **fraudes reais**.

Removê-las seria como **cortar as pistas do crime antes da investigação**.

Em problemas como:

- Detecção de fraude
- Manutenção preditiva
- Monitoramento de segurança

...os outliers **são o objetivo final** do modelo. Nesse caso, **identificá-los é mais importante do que removê-los**.

## Outliers: Problema ou Sinal?

O tratamento de outliers exige mais do que código. Exige julgamento, contexto e intuição.

- Se forem **erros**, devem ser corrigidos ou removidos.  
- Se forem **exceções reais**, podem ser modelados separadamente.  
- Se forem **sinais do comportamento que você quer detectar**, são seu ouro.

**Porque às vezes, os dados que mais incomodam... são os que mais revelam.**