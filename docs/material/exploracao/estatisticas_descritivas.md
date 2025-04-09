## Os Primeiros Sinais dos Dados: Enxergando o Retrato Estatístico Antes de Modelar

Imagine que você está colaborando com uma startup de saúde digital. O objetivo é entender o perfil de pacientes com risco de desenvolver doenças crônicas com base em um grande conjunto de dados: idade, frequência de consultas, IMC, nível de glicose, pressão arterial, uso de medicações, entre outros.

Antes mesmo de treinar qualquer modelo, alguém da equipe pergunta:

> “O que os dados estão dizendo, de fato? Como se comportam essas variáveis?”

Essa pergunta é o ponto de partida para qualquer investigação com dados.  
A resposta está no uso consciente de uma das etapas mais subestimadas — e mais reveladoras — da análise: as **estatísticas descritivas**.

## Para Onde os Dados Apontam? A Intuição das Medidas Centrais

O primeiro passo para compreender um conjunto de dados é descobrir **onde os valores tendem a se concentrar**.

### A média: a tendência do grupo

!!! info "Definição"
    A **média aritmética** é a soma de todos os valores dividida pela quantidade total de observações:

    $$
    \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
    $$

É amplamente utilizada, mas altamente sensível a valores extremos. Um único outlier pode puxar a média para cima ou para baixo, mascarando o padrão verdadeiro dos dados.

### A mediana: o centro real

!!! info "Definição"
    A **mediana** é o valor que divide o conjunto ordenado ao meio — 50% dos dados estão abaixo dela, 50% acima.

Mais robusta do que a média diante de assimetrias ou outliers.

### A moda: o valor mais comum

!!! info "Definição"
    A **moda** é o valor mais frequente em uma variável.

É especialmente útil para variáveis categóricas, mas também revela padrões importantes em variáveis discretas (como número de visitas, doses, etc.).

!!! example "Exemplo"
    Em uma base com os seguintes níveis de glicose (mg/dL):  
    [92, 94, 93, 120, 88, 94, 92, 94, 250]

    - Média: 114 (afetada pelo 250)
    - Mediana: 94
    - Moda: 94

    A moda e a mediana representam melhor a tendência geral nesse caso.

## O Quão Espalhados Estamos? As Medidas de Dispersão

Conhecer a tendência central **não basta**. Duas variáveis podem ter a mesma média e comportamentos completamente diferentes. É preciso saber o quanto os dados se afastam do centro.

### Variância e desvio padrão

!!! info "Definição"
    A **variância** mede a média dos desvios quadráticos em relação à média. O **desvio padrão** é a raiz quadrada da variância — e representa a dispersão **na mesma unidade da variável**.

!!! note "Fórmula"
    $$
    \text{Var}(X) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \quad , \quad \text{Desvio padrão} = \sqrt{\text{Var}(X)}
    $$

Altos desvios padrão indicam que os dados estão espalhados; baixos, que estão concentrados em torno da média.

### Intervalo interquartil (IQR)

!!! info "Definição"
    O **intervalo interquartil (IQR)** representa a amplitude entre o primeiro (Q1) e o terceiro quartil (Q3):

    $$
    IQR = Q3 - Q1
    $$

É uma medida robusta à presença de outliers, pois foca nos 50% centrais da distribuição.

!!! tip "Dica"
    O IQR é frequentemente usado para detectar **valores atípicos**: valores abaixo de \( Q1 - 1.5 \times IQR \) ou acima de \( Q3 + 1.5 \times IQR \).

## Formato da Distribuição: Assimetria e Curtose

Até aqui, sabemos onde os dados se concentram e o quanto se dispersam. Mas como eles se **comportam** graficamente?

### Assimetria (skewness)

!!! info "Definição"
    A **assimetria** mede o grau de simetria de uma distribuição:

    - Assimetria ≈ 0: distribuição simétrica  
    - Assimetria > 0: cauda à direita (assimetria positiva)  
    - Assimetria < 0: cauda à esquerda (assimetria negativa)

!!! example "Exemplo"
    Em uma variável como “tempo desde o último check-up”, é comum que muitos valores estejam próximos de 0 e poucos muito altos — gerando uma **assimetria positiva**.

### Curtose

!!! info "Definição"
    A **curtose** indica o grau de concentração de dados no centro da distribuição e o peso das caudas:

    - Curtose ≈ 3: distribuição normal (mesocúrtica)  
    - Curtose > 3: caudas pesadas e pico acentuado (leptocúrtica)  
    - Curtose < 3: distribuição achatada e caudas leves (platicúrtica)

Curtose elevada sugere que há **mais valores extremos** do que se esperaria em uma distribuição normal.

## Como Interpretar: O Que as Estatísticas Realmente Revelam?

As estatísticas descritivas não são apenas números: **elas contam uma história** sobre os dados.

- Se a média e a mediana diferem muito → **possível assimetria**
- Se o desvio padrão é alto → **dispersão significativa**
- Se a curtose é elevada → **muitos outliers em potencial**
- Se a moda é diferente da média e da mediana → **a distribuição pode ser multimodal**

## Python na Prática

```python
import pandas as pd

# Resumo estatístico padrão
df['glicose'].describe()

# Medidas adicionais
df['glicose'].skew()     # assimetria
df['glicose'].kurt()     # curtose
```

Para IQR:

```python
q1 = df['glicose'].quantile(0.25)
q3 = df['glicose'].quantile(0.75)
iqr = q3 - q1
```

## Cenário da Saúde Digital: O Que Descobrimos?

Após aplicar as estatísticas descritivas sobre os dados clínicos, a equipe identificou que:

- A variável **IMC** apresentava média 28, mas com desvio padrão de 9, indicando grande variação.
- A distribuição de **nível de glicose** tinha assimetria positiva acentuada e curtose > 4: **muitos valores extremos** e cauda longa — típica de pacientes com risco glicêmico elevado.
- A mediana do número de consultas era 1, enquanto a média era 3 — o que revelou que **a maioria dos pacientes faz poucas visitas**, mas alguns têm dezenas, puxando a média para cima.

Esses insights **guiaram a criação de variáveis derivadas** (como faixas de risco) e também serviram como base para alertar os gestores sobre **grupos específicos de atenção**.

## Estatísticas Descritivas: O Diagnóstico Antes da Prescrição

Antes de aplicar qualquer técnica sofisticada, as estatísticas descritivas mostram **onde está o centro da história, onde ela se espalha, e onde ela se desvia do esperado**.

São o alicerce de toda análise confiável.  
Elas não apenas **resumem** os dados — elas **revelam os padrões que direcionam as próximas perguntas**.

**Porque entender os dados começa por escutá-los. E as estatísticas descritivas são a primeira tradução confiável dessa voz.**