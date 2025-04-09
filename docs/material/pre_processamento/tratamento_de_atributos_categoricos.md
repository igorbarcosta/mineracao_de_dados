# Tratamento de Atributos Categóricos

## O Dilema dos Rótulos: Quando a Máquina Não Entende Palavras

Imagine que você está construindo um modelo de machine learning para prever se um cliente vai cancelar a assinatura de um serviço online. Os dados parecem promissores: idade, tempo de uso, número de acessos... mas aí você se depara com colunas como:

- “Cidade”: *São Paulo*, *Curitiba*, *Recife*...
- “Plano”: *Básico*, *Premium*, *Empresarial*
- “Gênero”: *Feminino*, *Masculino*, *Outro*

Esses valores fazem total sentido para nós, humanos. Mas para um algoritmo de aprendizado de máquina? Tudo isso é apenas... **texto**.

E algoritmos, por padrão, não falam português, nem inglês. Eles falam números.

É nesse ponto que entra uma etapa crucial da preparação dos dados: o **pré-processamento de atributos categóricos**.

## As Palavras Dizem Muito – Mas Não Como Estão

Antes de alimentar um modelo com os dados, precisamos garantir que *todo o conteúdo seja interpretável matematicamente*. E isso inclui transformar atributos categóricos em algo que a máquina possa calcular distâncias, probabilidades, pesos.

!!! info "Definição"
    **Atributos categóricos** são variáveis que representam categorias ou grupos distintos, e não quantidades numéricas. Exemplos comuns incluem gênero, cidade, estado civil ou tipo de plano.

!!! warning "Atenção"
    A forma como esses atributos são transformados pode influenciar diretamente a qualidade e imparcialidade do modelo. Transformações inadequadas podem introduzir **viés** ou **relações inexistentes** entre categorias.

## De Categorias a Números: Como Traduzir Sem Corromper o Sentido

### A Tentação do Label Encoding: Quando o Número Engana

A primeira ideia de muitos iniciantes é simplesmente converter as categorias em números inteiros. Parece fácil, né?

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['plano_codificado'] = le.fit_transform(df['plano'])
```

Isso transforma:

- *Básico* → 0
- *Empresarial* → 1
- *Premium* → 2

Mas atenção...

!!! danger "Muito Cuidado"
    Label Encoding **implica uma ordem numérica** entre categorias. O modelo pode interpretar que *Premium > Empresarial > Básico*, mesmo que não haja essa hierarquia real. Isso é perigoso, especialmente em algoritmos que consideram distância ou magnitude (como regressão linear ou KNN).

### One-Hot Encoding: Dando a Cada Categoria Seu Lugar

A solução mais segura (e popular) para variáveis nominais é o **One-Hot Encoding**.

!!! info "Definição"
    **One-Hot Encoding** cria uma nova coluna para cada categoria, marcando com 1 se a instância pertence àquela categoria, e 0 caso contrário.

```python
df = pd.get_dummies(df, columns=['plano'])
```

Isso resulta em:

| plano_Básico | plano_Premium | plano_Empresarial |
|--------------|----------------|--------------------|
| 1            | 0              | 0                  |
| 0            | 1              | 0                  |
| 0            | 0              | 1                  |

!!! tip "Dica"
    Em modelos lineares, é comum remover uma das colunas (a "variável de referência") para evitar colinearidade. Isso é chamado de *drop first*.

```python
df = pd.get_dummies(df, columns=['plano'], drop_first=True)
```

### Frequência ou Ordinal Encoding: Quando a Ordem Existe

Se a variável categórica tem uma **ordem natural** (como *ruim < regular < bom < excelente*), podemos usar codificações que preservem essa relação.

```python
from sklearn.preprocessing import OrdinalEncoder

ord_encoder = OrdinalEncoder(categories=[['ruim', 'regular', 'bom', 'excelente']])
df['avaliacao_ord'] = ord_encoder.fit_transform(df[['avaliacao']])
```

!!! example "Exemplo"
    Para uma variável “satisfação” com os valores [“ruim”, “regular”, “bom”, “excelente”], a codificação ordinal respeita a hierarquia esperada.

!!! warning "Atenção"
    Nunca use codificação ordinal em variáveis nominais (como cidade, cor dos olhos ou nome do produto). Isso induz o modelo a ver relações de ordem inexistentes.

## Impact Encoding e Target Encoding: O Caminho Estatístico

Essas técnicas avançadas substituem cada categoria por uma **estatística agregada** (como a média da variável alvo para cada grupo). São úteis quando há muitas categorias e os métodos anteriores causariam explosão dimensional.

```python
mean_churn = df.groupby('cidade')['cancelou'].mean()
df['cidade_encoded'] = df['cidade'].map(mean_churn)
```

!!! warning "Atenção"
    Esse tipo de codificação **usa informação do alvo**. Se não for feito com validação cruzada adequada, pode causar *data leakage* (vazamento de informação).

## Python na Prática: Um Exemplo Completo

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Exemplo de DataFrame
df = pd.DataFrame({
    'plano': ['Básico', 'Premium', 'Empresarial', 'Premium'],
    'cidade': ['SP', 'RJ', 'SP', 'MG']
})

# Codificação One-Hot
col_transform = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['plano', 'cidade'])
    ],
    remainder='passthrough'
)

df_encoded = col_transform.fit_transform(df)
```

??? question "Reflexão: Quando o One-Hot pode atrapalhar?"
    Quando temos muitas categorias (ex: nomes de produtos ou cidades com centenas de valores distintos), o One-Hot pode criar um número enorme de colunas, dificultando a modelagem e aumentando o risco de overfitting.

## De Volta ao Churn: Categorizando para Prever Melhor

Na análise de churn que estávamos desenvolvendo, as variáveis “plano” e “cidade” eram cruciais.

- A equipe percebeu que usar Label Encoding nas cidades fazia o modelo atribuir significado numérico à geografia – um erro.
- Ao aplicar One-Hot Encoding, a performance melhorou.
- Para “plano”, testou-se também uma codificação ordinal, dado que o plano *Premium* geralmente oferecia mais recursos que o *Básico*.

Essa decisão refletiu o conhecimento de negócio dentro da modelagem — e trouxe um ganho real na capacidade de previsão.

## Códigos que Falam: Categorização é Tradução

A tarefa de pré-processar atributos categóricos vai muito além de "transformar letras em números". Trata-se de **tradução de contexto para lógica computacional**. E, como em qualquer tradução, nuances importam.

Uma má tradução gera ruído, distorções e mal-entendidos. Já uma codificação bem feita garante que o modelo ouça a verdade por trás das palavras.

**Porque mesmo os melhores algoritmos erram, quando as categorias não falam a linguagem certa.**