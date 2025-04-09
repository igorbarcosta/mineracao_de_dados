# Detecção e Correção de Inconsistências

## Quando os Dados Se Contradizem

Você está trabalhando em um projeto para prever o risco de crédito de clientes. O banco de dados parece robusto: milhares de registros com idade, profissão, valor da renda, status da dívida, e muito mais.

Mas, ao explorar os dados, você encontra:

- Um cliente com **idade 12** e **renda de R$ 15.000**.
- Outro com **profissão = “Aposentado”** e **idade = 22**.
- Um terceiro com **renda declarada de R$ 0**, mas que pagou **em dia um empréstimo de R$ 50.000**.

Nada disso parece fazer sentido. Pior: se essas contradições forem ignoradas, seu modelo poderá aprender padrões errados — e tomar decisões perigosas.

Esses são os sinais de um problema silencioso, mas devastador: **inconsistência nos dados**.

## O Perigo das Incongruências Invisíveis

!!! info "Definição"
    **Inconsistências em dados** ocorrem quando **valores conflitantes ou logicamente incompatíveis** aparecem em um mesmo registro ou conjunto. Elas podem surgir por erros de digitação, falhas de integração entre sistemas ou preenchimento incorreto.

Diferente de *outliers*, que são pontos extremos, **inconsistências são contradições internas**.

??? question "Reflexão: Por que inconsistências são mais perigosas que dados ausentes?"
    Porque elas podem **parecer corretas**, mas escondem erros que **confundem algoritmos**, **enganam análises** e **geram decisões erradas com aparente legitimidade**.

## De Onde Vêm as Inconsistências?

- **Erros manuais** (idade = 222)
- **Integrações mal feitas** (campo de “profissão” vindo de sistema diferente do de “idade”)
- **Falta de regras de validação** (aceitar qualquer valor em campos numéricos)
- **Sistemas legados com lógicas antigas e conflitantes**

## Como Detectar Inconsistências? Estratégias Fundamentais

### 1. **Regras de Negócio e Lógica**

A primeira e mais poderosa abordagem é aplicar o **conhecimento do domínio**. Exemplo:

- Se profissão = “Aposentado” ⇒ idade ≥ 50
- Se estado civil = “Casado” ⇒ idade ≥ 15
- Se “renda” > 0 ⇒ “possui emprego” ≠ “não”

```python
# Exemplo: regra para flagar clientes menores de idade com profissões formais
df['inconsistente'] = ((df['idade'] < 18) & (df['profissao'].notna()))
```

### 2. **Regras Cruzadas Entre Variáveis**

Exemplo: valores negativos de atributos que não deveriam ser negativos:

- “valor da dívida” < 0  
- “número de filhos” < 0

```python
df['divida_invalida'] = df['divida'] < 0
```

### 3. **Detecção por Distribuição Condicional**

Verificar se a distribuição de uma variável muda drasticamente dentro de subgrupos onde ela *não deveria* mudar.

!!! example "Exemplo"
    A média salarial entre pessoas da mesma profissão não deveria variar tanto entre idades próximas (ex: entre 30 e 35 anos). Se houver grandes saltos, algo pode estar errado.

### 4. **Verificações de Formato e Domínio**

- CEPs com menos de 8 dígitos  
- Nomes com números ou símbolos  
- Datas impossíveis (ex: “32/01/2022”)  
- Sexo = “X” (em banco que só aceita “M” ou “F”)

```python
df['cep_valido'] = df['cep'].str.len() == 8
```

## Corrigindo Inconsistências: Cada Caso, Uma Solução

### 1. **Correção com Regras**

- Substituir valores incorretos por valores plausíveis
- Corrigir erros simples de digitação (ex: "masculuno" → "masculino")

```python
df['sexo'] = df['sexo'].replace({'masculuno': 'masculino'})
```

### 2. **Correção por Inferência**

Se a idade está faltando ou inválida, pode-se estimar com base em outras variáveis (como escolaridade ou tempo de trabalho).

```python
idade_media_por_profissao = df.groupby('profissao')['idade'].mean()
df['idade_corrigida'] = df.apply(
    lambda row: idade_media_por_profissao[row['profissao']] if row['idade'] < 0 else row['idade'], axis=1
)
```

### 3. **Remoção de Registros Irrecuperáveis**

Se a inconsistência for grave e não houver como corrigi-la com segurança, pode ser necessário excluir o registro.

!!! danger "Muito Cuidado"
    Só remova dados como **última opção**, e **nunca em massa sem análise detalhada**. Às vezes, registros errados são os únicos com certo padrão raro, e a remoção pode enviesar o modelo.

## Ferramentas para Automatizar Verificações

- **Pandas Profiling / ydata-profiling**: gera relatórios automáticos com alertas de inconsistência.
- **Great Expectations**: permite escrever *testes unitários* para dados, validando formatos, faixas, categorias.
- **PyDeequ**: biblioteca de qualidade de dados inspirada na ferramenta Deequ da Amazon.

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Relatório de Qualidade de Dados")
profile.to_file("relatorio.html")
```

## No Caso do Risco de Crédito: Corrigir Antes de Prever

No cenário inicial, o modelo de risco de crédito estava apresentando comportamento errático. Após uma investigação cuidadosa:

- Descobriu-se que mais de **8% dos registros tinham contradições internas**, como renda negativa, faixas etárias incoerentes e vínculos de trabalho incompatíveis com a idade.
- Após aplicar um pipeline de **validação + correção + exclusão pontual**, o desempenho do modelo melhorou em mais de 20%.

A conclusão foi clara: **não era o algoritmo que estava errado. Eram os dados que estavam mentindo — sutilmente.**

## Detectar Inconsistências é Ler nas Entrelinhas

Modelos de machine learning aprendem o que você ensina. Se você alimenta um modelo com dados contraditórios, ele **vai aprender a contradizer a realidade**.

Detectar e corrigir inconsistências não é apenas um passo técnico — é uma responsabilidade.

**Porque nos dados, como na vida, o perigo raramente está no que falta. Está no que parece certo, mas não é.**