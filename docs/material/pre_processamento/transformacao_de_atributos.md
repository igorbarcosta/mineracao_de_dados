# Transformação de Atributos

## Medidas Fora de Escala: Quando os Dados Falam Línguas Diferentes

Imagine que você está desenvolvendo um sistema de detecção de fraudes bancárias. Seu modelo precisa analisar variáveis como:

- **valor da transação** (em reais),
- **tempo desde a última compra** (em segundos),
- **número de transações nas últimas 24h**,
- **idade do cliente**.

O modelo está treinado e... os resultados são péssimos. Mas por quê?

Ao investigar, você percebe algo curioso: os valores das transações variam de **R$ 10 a R$ 20.000**, enquanto o “número de transações” raramente passa de 5. Já o “tempo desde a última compra” tem valores entre 0 e 86.400 segundos.

Seu modelo está tentando aprender a partir de variáveis em **escalas completamente diferentes** – como se estivesse misturando metros, litros e temperaturas numa mesma equação.

E aqui entra uma etapa fundamental, porém muitas vezes negligenciada: o **redimensionamento e transformação de dados** (*Feature Scaling and Transformation*).

## Escalas Distorcem Modelos: O Problema das Unidades Desalinhadas

Algoritmos de machine learning funcionam a partir de **operações matemáticas** entre atributos. Quando as escalas são diferentes, variáveis com valores absolutos maiores tendem a dominar o cálculo.

!!! warning "Atenção"
    Em modelos que dependem de distância (como KNN, SVM) ou gradientes (como regressão logística, redes neurais), **não escalar os dados pode comprometer completamente o desempenho**.

!!! info "Definição"
    **Redimensionamento de atributos** é o processo de ajustar os valores numéricos de diferentes variáveis para uma escala comum, sem distorcer suas relações internas.

## Normalizar ou Padronizar? Eis a Questão

### 1. **Min-Max Scaling** (Normalização)

Esse método transforma os valores para uma faixa entre 0 e 1.

!!! note "Fórmula"
    $$ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} $$

!!! example "Exemplo"
    Se a idade varia entre 18 e 60, um valor de 30 vira:
    $$ \frac{30 - 18}{60 - 18} = \frac{12}{42} ≈ 0.286 $$

!!! tip "Dica"
    Use Min-Max Scaling quando quiser preservar a **distribuição original dos dados**, especialmente para algoritmos que usam distâncias.

### 2. **Standard Scaling** (Padronização)

Transforma os dados para terem **média 0 e desvio padrão 1**. 

!!! note "Fórmula"
    $$ x_{pad} = \frac{x - \mu}{\sigma} $$  
    Onde \( \mu \) é a média e \( \sigma \) é o desvio padrão da variável.

!!! example "Exemplo"
    Se o valor de uma transação foi R$ 2000, e a média é R$ 1000 com desvio de R$ 500:
    $$ \frac{2000 - 1000}{500} = 2 $$

!!! tip "Dica"
    Use Standard Scaling quando os dados tiverem **distribuições Gaussianas** ou quando o modelo presume centramento dos dados (ex: PCA, regressão logística).

### 3. **Robust Scaling**

Baseado na **mediana** e no intervalo interquartílico. É mais resistente a *outliers*.

!!! note "Fórmula"
    $$ x_{robust} = \frac{x - \text{mediana}}{IQR} $$

!!! tip "Dica"
    Use quando seus dados tiverem valores extremos que você não pode (ou não quer) remover.

## Transformações que Vão Além da Escala

Além de alinhar escalas, às vezes precisamos **transformar a distribuição** dos dados. Por exemplo, se os valores estão muito concentrados ou enviesados para um lado.

### 1. **Transformação Logarítmica**

Reduz a amplitude de variáveis assimétricas (como rendas, volumes ou preços).

```python
df['valor_log'] = np.log1p(df['valor'])  # log(1 + x) para evitar log(0)
```

!!! warning "Atenção"
    Só use log em valores **positivos**. Negativos e zeros precisam de ajustes (como `log1p`).

### 2. **Box-Cox e Yeo-Johnson**

Transformações paramétricas que **normalizam a distribuição**.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
df[['valor_transformado']] = pt.fit_transform(df[['valor']])
```

!!! tip "Dica"
    Use quando quiser aplicar algoritmos que presumem normalidade (ex: LDA, regressão linear clássica).

## Implementando no Python: Uma Pipeline Escalável

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Exemplo de dataframe
numericas = ['idade', 'valor_transacao', 'tempo_ultima_compra']

scaler = ColumnTransformer(transformers=[
    ('standard', StandardScaler(), ['idade']),
    ('minmax', MinMaxScaler(), ['tempo_ultima_compra']),
], remainder='passthrough')

df_scaled = scaler.fit_transform(df[numericas])
```

??? question "Reflexão: Por que usar diferentes escalas para diferentes variáveis?"
    Porque o comportamento dos dados pode ser diferente. Idade pode seguir uma distribuição aproximadamente normal (ideal para padronização), enquanto tempo pode ter limites conhecidos (ideal para normalização), e valores monetários podem conter *outliers* (ideal para escaladores robustos).

## De Volta à Fraude: Agora em Escala Real

Ao aplicar a transformação correta para cada variável:

- **Tempo desde a última compra** foi normalizado entre 0 e 1.
- **Valor da transação** foi transformado com log.
- **Idade** foi padronizada com média 0 e desvio 1.

Com isso, o modelo parou de ser “enganado” pelas escalas distintas. Os algoritmos passaram a enxergar **padrões reais**, não apenas diferenças numéricas arbitrárias.

## Escalar é Nivelar o Jogo

Modelos de machine learning são como atletas de elite: eles têm grande capacidade, mas dependem do ambiente. Se os dados estão em campos inclinados e com obstáculos invisíveis, o desempenho despenca.

**Escalar e transformar os dados é como nivelar o terreno e ajustar a iluminação**: só assim o modelo consegue jogar com clareza.

Porque no fim das contas, **não é o tamanho do número que importa — e sim o que ele representa.**