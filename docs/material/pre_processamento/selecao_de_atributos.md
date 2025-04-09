# Seleção de Atributos

## O Poder de Escolher: Quando Menos é Mais em Machine Learning

Você foi contratado por uma empresa de e-commerce para prever quais clientes provavelmente farão uma nova compra no próximo mês. A equipe coletou mais de **300 variáveis**: cliques, tempo na página, número de produtos visualizados, localização, dispositivo usado, horário de navegação, e muito mais.

Mas ao treinar o modelo, ele não performa bem. Ele é lento, difícil de interpretar e — pior — não melhora com mais dados.

Surge então uma pergunta fundamental:  
**"Será que estamos alimentando o modelo com informação... ou com ruído?"**

A resposta está em uma etapa crítica do pipeline de mineração de dados: o **Feature Selection**, ou **seleção de atributos**.

## Informação Não é o Mesmo que Utilidade

Na era dos big data, coletar dados é fácil. Mas cada nova variável tem um custo: complexidade, ruído, tempo computacional, risco de overfitting.

!!! info "Definição"
    **Feature Selection** é o processo de identificar e manter **apenas as variáveis mais relevantes** para um modelo preditivo, descartando aquelas que são redundantes, irrelevantes ou prejudiciais.

!!! warning "Atenção"
    Nem toda variável que parece "interessante" é útil para prever o que queremos. Algumas apenas **correlacionam-se por acaso**, ou estão fortemente correlacionadas com outras — o que pode prejudicar a estabilidade do modelo.

## Por Que Fazer Feature Selection?

- **Reduz overfitting**: menos variáveis = menos chance do modelo aprender padrões falsos.
- **Aumenta a interpretabilidade**: modelos com menos atributos são mais fáceis de entender.
- **Melhora o desempenho**: menos colunas = mais velocidade e menor custo computacional.
- **Facilita a manutenção do sistema**: você depende de menos dados no futuro.

## Estratégias para Escolher com Inteligência

### 1. **Seleção Baseada em Estatísticas Simples**

#### Variância Baixa

Atributos com quase nenhum valor variando (ex: 99% das linhas são “sim”) **não ajudam a classificar ou prever**.

```python
from sklearn.feature_selection import VarianceThreshold

seletor = VarianceThreshold(threshold=0.01)
df_filtrado = seletor.fit_transform(df)
```

### 2. **Correlação com a Variável Alvo**

Analisar a correlação entre cada variável e o que você quer prever.  
Para problemas de classificação, pode-se usar *Information Gain*, *Chi-quadrado*, ou correlação com `target` binário.

```python
from sklearn.feature_selection import SelectKBest, chi2

X_numerico = df.drop('comprou', axis=1)
y = df['comprou']
melhores = SelectKBest(score_func=chi2, k=10)
X_selecionado = melhores.fit_transform(X_numerico, y)
```

!!! tip "Dica"
    Essa abordagem funciona bem como **primeiro filtro**, mas não captura interações entre variáveis.

### 3. **Eliminação Recursiva de Atributos (RFE)**

Um método que testa diferentes subconjuntos de atributos treinando modelos e eliminando iterativamente os menos importantes.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()
rfe = RFE(estimator=modelo, n_features_to_select=10)
rfe.fit(X, y)
```

### 4. **Importância do Modelo**

Algoritmos como Random Forest e XGBoost conseguem medir **a importância das variáveis** diretamente com base no quanto elas contribuem para a performance do modelo.

```python
from sklearn.ensemble import RandomForestClassifier

modelo = RandomForestClassifier()
modelo.fit(X, y)

importancias = modelo.feature_importances_
```

!!! example "Exemplo"
    Se o modelo usa 200 variáveis mas 90% das decisões estão baseadas em apenas 12 delas, talvez você deva focar nessas 12.

### 5. **Seleção com Regularização (L1)**

Alguns modelos (como Lasso) naturalmente forçam os coeficientes de variáveis irrelevantes a zero.

```python
from sklearn.linear_model import Lasso

modelo = Lasso(alpha=0.1)
modelo.fit(X, y)
```

## Python na Prática: Um Pipeline de Seleção

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('selecao', SelectFromModel(RandomForestClassifier(n_estimators=100))),
    ('classificador', RandomForestClassifier())
])

pipeline.fit(X_treino, y_treino)
```

??? question "Reflexão: Existe risco em remover variáveis demais?"
    Sim. Remover variáveis importantes pode **empobrecer o modelo**, limitando sua capacidade de capturar padrões complexos. Feature Selection é um equilíbrio entre simplificação e preservação da informação.

## No E-commerce: A Escolha Certa Salva o Modelo

No caso do e-commerce, o time usava 300 variáveis, mas após aplicar RFE com uma árvore de decisão, descobriu-se que **apenas 15 atributos explicavam mais de 90% da variância** na decisão de compra.

Além de acelerar o treinamento, o modelo ficou mais estável e interpretável — o que permitiu aos analistas de negócio **entender os fatores-chave de conversão de clientes**.

## Feature Selection Não É Poda Cega — É Curadoria

Reduzir variáveis não é sobre jogar dados fora. É sobre escolher **com intenção**, entendendo **o que realmente importa** para o modelo e para o negócio.

É um processo investigativo, cuidadoso, que exige conhecimento técnico e sensibilidade analítica.

**Porque no fim, mais dados não significam mais valor. E sim, mais decisões a tomar.**  
**E saber escolher é o que separa análises medianas de modelos memoráveis.**