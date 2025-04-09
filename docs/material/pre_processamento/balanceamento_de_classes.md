# Balanceamento de Classes

## A Justiça dos Dados: Quando o Desequilíbrio Engana o Modelo

Imagine que você está desenvolvendo um modelo para **detectar transações bancárias fraudulentas**. O conjunto de dados é enorme: um milhão de registros. Mas, ao fazer a contagem dos casos de fraude, você descobre que apenas **1.000 transações são fraudulentas**.

Ou seja, **99,9% dos dados são “normais”**.

Você treina o modelo, roda as métricas e... 99,9% de acurácia! Parece ótimo, até perceber que o modelo está **simplesmente dizendo "não é fraude" para todo mundo**.

Esse é o perigo silencioso do **desbalanceamento de classes**: quando a maioria dos exemplos pertence a uma única classe, e o modelo aprende a ignorar o que realmente importa.

## O Que é Balanceamento de Classes?

!!! info "Definição"
    O **balanceamento de classes** é o processo de ajustar a proporção entre categorias de uma variável alvo (geralmente em problemas de classificação), de forma que o modelo **não seja enviesado pela classe majoritária**.

Em problemas **desbalanceados**, o modelo tende a favorecer a classe mais comum — mesmo que ela não seja a mais importante.

??? question "Reflexão: O que é pior — errar uma transação normal ou deixar passar uma fraude?"
    Em muitos contextos (saúde, segurança, finanças), errar a **minoria crítica** tem um custo muito maior. Por isso, **precisamos forçar o modelo a aprender com os casos raros.**

## Detectando o Desequilíbrio

```python
df['classe'].value_counts(normalize=True)
```

Se uma das classes representa menos de 10% dos casos, **o problema provavelmente é desbalanceado**.

Exemplos comuns:

- Fraude financeira  
- Diagnóstico de doenças raras  
- Cancelamento de serviços (churn)  
- Reconhecimento de falhas em sensores industriais

## Estratégias de Balanceamento: Forçando o Modelo a Prestar Atenção

### 1. **Undersampling** (Redução da classe majoritária)

Reduz artificialmente a quantidade de exemplos da classe dominante para se igualar à minoria.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)
```

!!! warning "Atenção"
    Você pode perder dados valiosos! Use com cuidado, principalmente se tiver pouco volume.

### 2. **Oversampling** (Aumento da classe minoritária)

Duplica ou sintetiza novos exemplos da classe rara.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X, y)
```

### 3. **SMOTE (Synthetic Minority Oversampling Technique)**

Cria **novos exemplos sintéticos** da classe minoritária com base em seus vizinhos mais próximos.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
```

!!! tip "Dica"
    O SMOTE é mais eficaz que duplicar amostras, pois **cria variações realistas** que ajudam o modelo a generalizar melhor.

### 4. **Estratégias Baseadas em Peso**

Alguns algoritmos (como regressão logística, árvores, redes neurais) permitem **atribuir pesos maiores** às classes minoritárias.

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(class_weight='balanced')
```

!!! example "Exemplo"
    Em um dataset com 95% de classe 0 e 5% de classe 1, o modelo com `class_weight='balanced'` trata os erros da classe 1 como **20 vezes mais graves** que os da classe 0.

## Avaliação em Dados Desbalanceados: A Acurácia Engana

Em datasets desbalanceados, **acurácia é uma métrica perigosa**, pois pode estar alta mesmo que o modelo esteja ignorando a classe rara.

Prefira:

- **Precisão (Precision)**  
- **Revocação (Recall)**  
- **F1-Score**  
- **Matriz de Confusão**  
- **Curvas ROC e AUC**

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

??? question "Reflexão: O que priorizar — precisão ou revocação?"
    Depende do contexto. Se o erro de falso negativo é mais grave (ex: deixar uma fraude passar), priorize **revocação**. Se falso positivo for mais custoso (ex: acusar um cliente inocente), priorize **precisão**.

## No Caso da Fraude: A Lógica do Desequilíbrio

No nosso cenário inicial, o modelo que acertava 99,9% dos casos **não detectava fraudes reais**.

Após aplicar **SMOTE** e treinar com pesos balanceados, o modelo passou a identificar corretamente **87% das fraudes**, com uma queda insignificante na acurácia geral.

Mais importante: agora o modelo **prestava atenção onde realmente importava**.

## Balancear É Valorizar o Que Importa

A realidade dos dados é desigual — e isso se reflete nos conjuntos que analisamos. Mas permitir que essa desigualdade conduza o aprendizado **é um erro técnico e ético**.

Balancear as classes é dar voz aos dados raros.  
É garantir que padrões importantes não sejam sufocados pela maioria.

**Porque em muitos problemas, o que é raro não é ruído. É o sinal que vale ouro.**