# Engenharia de Atributos

## O Segredo por Trás dos Modelos Poderosos: A Arte da Criação de Atributos

Imagine que você trabalha em uma fintech que oferece crédito para microempreendedores. Seu time está desenvolvendo um modelo para prever inadimplência com base em dados como:

- Renda mensal declarada  
- Número de dependentes  
- Valor do último empréstimo  
- Data de abertura da conta  

Você aplica os algoritmos mais avançados — redes neurais, XGBoost, ensemble models — mas a performance ainda não impressiona. A AUC não passa de 0.65, e os erros são altos.

Até que alguém da equipe propõe:  
**“E se a gente criasse uma variável chamada ‘renda por dependente’?**  
Ou talvez: **‘tempo de relacionamento com o banco’?”**

Poucos minutos depois, o modelo salta para 0.78 de AUC.

Esse salto de qualidade não veio do modelo, nem de um ajuste técnico. Veio de uma habilidade estratégica: **Feature Engineering** ou **Engenharia de Atributos**.

## Quando o Valor Não Está nos Dados, Mas no que Você Faz com Eles

!!! info "Definição"
    **Feature Engineering** (ou engenharia de atributos) é o processo de **criar, transformar, combinar ou selecionar variáveis** (atributos ou *features*) a partir dos dados brutos, com o objetivo de **melhorar o desempenho dos modelos de machine learning**.

Esse processo é onde o conhecimento do domínio encontra a técnica. Não basta saber programação ou estatística. É preciso entender o problema, o contexto e os comportamentos escondidos nos dados.

## Revelando o Invisível: Criando Valor Onde Não Existia

A beleza da engenharia de atributos está na possibilidade de **extrair significado adicional** a partir de dados existentes. Vamos explorar algumas estratégias comuns (e poderosas):

### 1. **Combinações Matemáticas Simples**

- `renda / número de dependentes`  
- `valor do empréstimo / renda`  
- `saldo médio / idade`

!!! tip "Dica"
    Relações entre variáveis muitas vezes são mais informativas do que as variáveis isoladas. Especialmente em finanças, razão e proporção são atributos extremamente úteis.

### 2. **Transformações Temporais**

A partir de datas, você pode extrair:

- Tempo desde um evento (`hoje - data_abertura_conta`)
- Mês, dia da semana, hora (`data_transacao.dt.month`)
- Se a transação ocorreu em final de semana

```python
df['tempo_cliente'] = (pd.Timestamp.today() - df['data_abertura_conta']).dt.days
```

### 3. **Agregações Baseadas em Grupo**

Imagine que cada cliente tem múltiplas transações. Você pode agregar:

- Valor médio de compra por cliente
- Desvio padrão do valor gasto
- Frequência de compra por semana

```python
df_agg = df.groupby('id_cliente')['valor_compra'].agg(['mean', 'std', 'count'])
```

!!! example "Exemplo"
    Um cliente que faz poucas compras, mas sempre no mesmo valor, pode se comportar muito diferente de outro que compra frequentemente com grande variabilidade.

### 4. **Contagens e Frequências**

Quantas vezes uma categoria aparece? Com que frequência?

```python
frequencia_bairro = df['bairro'].value_counts()
df['freq_bairro'] = df['bairro'].map(frequencia_bairro)
```

!!! tip "Dica"
    Contagens e frequências ajudam o modelo a capturar padrões comuns ou raros, o que pode ser útil para identificar comportamentos anômalos.

### 5. **Binários e Flags Estratégicas**

- `é_novo_cliente = dias_desde_abertura < 90`
- `tem_dependente = dependentes > 0`
- `cliente_renda_alta = renda > 5000`

!!! warning "Atenção"
    Variáveis booleanas simples (0 ou 1) ajudam muito algoritmos lineares e tornam regras de decisão mais interpretáveis.

## Python na Prática: Criando Novas Features

```python
import pandas as pd

# Suponha que temos este DataFrame:
df = pd.DataFrame({
    'renda': [3000, 5000, 2000],
    'dependentes': [2, 0, 3],
    'data_abertura': pd.to_datetime(['2019-01-01', '2020-05-15', '2021-11-10']),
    'valor_emprestimo': [1000, 15000, 5000]
})

# 1. Nova feature: renda por dependente
df['renda_por_dependente'] = df['renda'] / (df['dependentes'] + 1)

# 2. Tempo como cliente (em dias)
df['tempo_cliente'] = (pd.Timestamp.today() - df['data_abertura']).dt.days

# 3. Relação entre empréstimo e renda
df['emprestimo_sobre_renda'] = df['valor_emprestimo'] / df['renda']
```

??? question "Reflexão: Qual dessas features você acha que mais ajuda a prever inadimplência?"
    Depende do contexto, mas a variável `emprestimo_sobre_renda` costuma ser um forte preditor, pois relaciona o comprometimento financeiro diretamente com a capacidade de pagamento.

## Feature Engineering ≠ Feature Selection

Não confunda! Enquanto **Feature Engineering** é o processo criativo de construir atributos, **Feature Selection** é o processo analítico de **escolher quais dessas features manter**.

As duas se complementam: criamos atributos inteligentes, depois deixamos os algoritmos (ou análises estatísticas) decidirem quais de fato ajudam.

## E Quando Engenharia Vai Mal?

!!! danger "Muito Cuidado"
    Criar muitas features **irrelevantes ou redundantes** pode:

    - Aumentar o custo computacional
    - Gerar *overfitting*
    - Confundir modelos lineares
    - Dificultar a interpretabilidade

A boa engenharia de atributos é guiada por **intuição, contexto e testes contínuos**. Não existe uma receita pronta — cada problema exige uma abordagem própria.

## Voltando à Fintech: Atributos que Revelam a Verdade

No cenário inicial, a equipe criou apenas três novas variáveis derivadas de dados que já estavam disponíveis. O resultado?

- Redução de 15% no erro do modelo
- Aumento significativo na interpretabilidade
- Facilidade em justificar decisões de crédito com base em atributos compreensíveis

O aprendizado foi claro: **os dados originais tinham valor, mas estavam adormecidos**. Foi preciso fazer as perguntas certas e construir os atributos certos para que o modelo pudesse enxergar.

## A Criatividade é a Nova Força Bruta

Feature Engineering é o coração invisível dos modelos robustos. Não adianta usar as técnicas mais sofisticadas se as variáveis que você entrega ao algoritmo não carregam o **sinal real do comportamento que você quer prever**.

**Modelos preditivos são como esculturas: os dados brutos precisam ser talhados com técnica, contexto e criatividade.**

Porque, no fim, a inteligência não está apenas no algoritmo. Está nas perguntas que fazemos — e nas variáveis que criamos para respondê-las.