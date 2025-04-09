# Tratamento de Dados Ausentes

## Quando os Dados Somem: O Dilema Invisível nos Sistemas Inteligentes

Imagine que você está desenvolvendo um sistema de recomendação para uma plataforma de streaming. Sua missão é criar um modelo capaz de sugerir filmes e séries com base no perfil dos usuários: idade, histórico de visualização, preferências de gênero, avaliações anteriores, entre outros.

Mas ao examinar o banco de dados... surpresa! Vários usuários não preencheram suas idades. Outros nunca avaliaram nenhum filme. Alguns assistiram a apenas dois episódios aleatórios. O sistema, que parecia estar pronto para brilhar, agora tropeça em buracos invisíveis na base de dados.

O que fazer com essas lacunas? Ignorá-las pode distorcer o modelo. Preenchê-las mal pode ser ainda pior. E é aqui que entra uma das etapas mais críticas – e frequentemente negligenciadas – da mineração de dados: o **pré-processamento de dados ausentes**.

## Antes de Decidir, Entenda o Porquê: A Natureza dos Dados Faltantes

### Dados ausentes são todos iguais?

Nem todo dado ausente é criado da mesma forma. Para tratar essas lacunas de maneira inteligente, primeiro precisamos entender sua origem.

!!! info "Definição"
    Em ciência de dados, chamamos de **dados ausentes** (ou *missing data*) qualquer valor que *deveria estar disponível*, mas não está. Eles podem surgir por falhas técnicas, decisões do usuário ou critérios de coleta.

Existem três principais causas:

- **MCAR (Missing Completely at Random)**: a ausência é totalmente aleatória, sem relação com outras variáveis.  
- **MAR (Missing at Random)**: a ausência depende de outras variáveis observadas.  
- **MNAR (Missing Not at Random)**: a ausência depende do próprio valor ausente (ou de algo não observado).

!!! example "Exemplo"
    Imagine um aplicativo de saúde onde a variável “peso” está faltando. Se os dados sumiram porque houve erro no sensor, temos MCAR.  
    Se os dados estão faltando apenas para pessoas com mais de 60 anos, temos MAR.  
    Agora, se pessoas com sobrepeso preferiram não informar esse valor, temos MNAR.

!!! warning "Atenção"
    Ignorar a causa dos dados ausentes pode comprometer toda a análise. O tratamento adequado depende diretamente desse diagnóstico inicial.

## Estratégias de Sobrevivência: Como Lidar com Dados Ausentes

Agora que sabemos o que estamos enfrentando, vamos explorar as principais táticas para lidar com dados ausentes. E como em qualquer missão de resgate, nem todas as abordagens são igualmente eficazes – cada uma tem prós, contras e armadilhas escondidas.

### Remoção de Dados

A primeira ideia (e a mais tentadora) é simplesmente ignorar as linhas com valores faltantes.

!!! example "Exemplo"
    Ao treinar um modelo com 1000 usuários, 150 deles não preencheram a idade. Você decide remover essas linhas.

!!! danger "Muito Cuidado"
    Essa estratégia só é segura se os dados estiverem *completamente ao acaso* (MCAR) e se a exclusão não reduzir demais o tamanho da amostra ou introduzir viés.

### Imputação: Preencher com Cuidado

Em vez de ignorar os dados ausentes, podemos tentar **preenchê-los** com estimativas razoáveis. A isso damos o nome de *imputação*.

!!! info "Definição"
    **Imputação** é o processo de substituir dados ausentes por valores estimados com base em alguma estratégia ou modelo.

#### Técnicas Simples:

- **Média, Moda ou Mediana** (para variáveis numéricas ou categóricas)
- **Valor constante** (ex: “desconhecido”, 0)
- **Forward Fill / Backward Fill** (em séries temporais)

!!! warning "Atenção"
    Preencher todos os valores com a média pode distorcer a variabilidade dos dados e ocultar padrões importantes.

#### Técnicas Avançadas:

- **KNN Imputation**: preenche valores com base em vizinhos mais próximos.
- **Regressão Múltipla**: usa outras variáveis para prever o valor ausente.
- **MICE (Multiple Imputation by Chained Equations)**: realiza várias imputações e agrega os resultados.

!!! tip "Dica"
    Técnicas mais sofisticadas tendem a preservar melhor a estrutura estatística dos dados, mas também exigem mais cuidado e poder computacional.

## Mãos na Massa: Lidando com Dados Ausentes no Python

Vamos ver como essas ideias se materializam no Python, usando `pandas` e `scikit-learn`.

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# Suponha que temos o seguinte DataFrame
df = pd.DataFrame({
    'idade': [25, 30, None, 40, None],
    'nota': [4.5, None, 3.0, 5.0, 4.0]
})

# Verificando valores ausentes
print(df.isnull().sum())

# Imputação simples com a média
imputador_media = SimpleImputer(strategy='mean')
df[['idade', 'nota']] = imputador_media.fit_transform(df[['idade', 'nota']])

# Imputação usando KNN
imputador_knn = KNNImputer(n_neighbors=2)
df_knn = pd.DataFrame(imputador_knn.fit_transform(df), columns=df.columns)
```
??? question "Por que a imputação com KNN pode ser mais adequada do que preencher com a média?"
    KNN considera a similaridade entre os registros para estimar valores ausentes, preservando melhor a estrutura local dos dados. Isso pode evitar que o modelo aprenda padrões artificiais que a imputação com média pode introduzir.

## Revisitando a Plataforma de Streaming: Decisões com Dados Completos

Lembre-se do sistema de recomendação do início? Após analisar as variáveis com valores ausentes, descobriu-se que:

- A idade estava ausente principalmente em usuários que se registraram via login social (MAR).
- As avaliações faltantes vinham de novos usuários sem histórico (MAR/MNAR).

A equipe optou por:

- Imputar a idade com base em perfis semelhantes usando KNN.
- Tratar a ausência de avaliações como uma *feature*, indicando que o usuário é novo – o que, por si só, é uma informação útil para o modelo.

Graças ao tratamento cuidadoso dos dados ausentes, o sistema conseguiu manter sua performance e oferecer recomendações relevantes mesmo com informações incompletas.

## Quando os Dados Faltam, o Raciocínio Não Pode Faltar

Dados ausentes são inevitáveis. Mas tratá-los de maneira descuidada é opcional – e perigoso. Eles carregam informações implícitas sobre o processo de coleta, o comportamento do usuário e até os vieses do sistema.

Dominar o pré-processamento de dados ausentes é como afiar suas ferramentas antes de entrar na floresta da análise. Não garante o sucesso por si só, mas aumenta – e muito – suas chances de não se perder no caminho.

**Porque, no fim, os dados que não vemos podem ser justamente os que mais influenciam o que descobrimos.**