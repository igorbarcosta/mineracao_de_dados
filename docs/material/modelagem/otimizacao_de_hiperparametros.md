## O Labirinto dos Hiperparâmetros: Otimizar sem Trapacear

### Como Melhorar um Modelo sem Enganar a Avaliação

Na seção anterior, aprendemos a validar modelos com honestidade, medindo sua capacidade de generalizar para dados desconhecidos.

Agora, vamos explorar um desafio diferente, mas conectado: **como melhorar o desempenho de um modelo** sem abrir mão do rigor estatístico e da ética experimental.

Imagine que você está refinando um modelo de classificação para detecção de fraudes financeiras. O algoritmo escolhido foi uma floresta aleatória.

Alguém pergunta:
> “Quantas árvores você usou?”

Você responde: “Cem.”  
A outra pessoa rebate:
> “Mas por que cem? Você testou outras quantidades? E a profundidade das árvores?”

Esse tipo de pergunta revela uma dimensão crucial da modelagem: **o impacto das escolhas externas ao aprendizado** — os chamados **hiperparâmetros**.

---

## Afinal, o que são Hiperparâmetros?

!!! info "Parâmetros vs. Hiperparâmetros"

    - **Parâmetros**: aprendidos automaticamente durante o treinamento (ex: coeficientes de uma regressão).
    - **Hiperparâmetros**: definidos manualmente antes do treinamento; controlam como o modelo se comporta.

    Em outras palavras: hiperparâmetros moldam *como* o modelo aprende. Parâmetros são *o que* ele aprende.

!!! example "Analogia: o fogão e a receita"

    Treinar um modelo é como cozinhar.  
    Os dados são os ingredientes.  
    Os hiperparâmetros são os botões do fogão: temperatura, tempo, intensidade da chama.

    Eles não são aprendidos pela receita. Você precisa regulá-los com cuidado, ou corre o risco de deixar tudo cru (underfitting) ou queimar (overfitting).

---

## O Papel dos Hiperparâmetros na Generalização

Hiperparâmetros afetam diretamente o equilíbrio entre **viés** e **variância** do modelo:

!!! info "Impacto no Viés e na Variância"

    - Hiperparâmetros que limitam a complexidade do modelo (ex: profundidade rasa em árvores) → aumentam o viés e reduzem a variância.  
    - Hiperparâmetros que permitem modelos mais complexos → reduzem o viés, mas aumentam a variância.

    Ajustá-los corretamente significa encontrar o ponto ideal entre **modelo fraco demais** e **modelo que decora os dados**.

---

## Panorama do Ajuste de Hiperparâmetros

!!! info "Etapas do Processo de Tuning"

    1. Escolher os hiperparâmetros a testar  
    2. Definir a estratégia de busca (grid, random, bayesiana)  
    3. Usar validação cruzada (com pipeline) para avaliar cada combinação  
    4. Selecionar a melhor configuração  
    5. Avaliar o modelo final no conjunto de teste (intocado)

Cada passo deve ser conduzido com cuidado. Pequenos erros podem comprometer toda a avaliação.

---

## Armadilhas a Evitar

!!! danger "Ajustar com Acesso ao Teste Final"

    Se você escolhe os hiperparâmetros com base nos resultados no conjunto de teste, está **violando o princípio da generalização**.  
    O teste final perde o valor como medida real de desempenho.

!!! warning "Otimizar Sem Intuição"

    Testar combinações aleatórias sem entender o papel de cada hiperparâmetro é como **girar botões no escuro esperando um resultado mágico**.

    Sempre que possível, estude o efeito de cada hiperparâmetro antes de iniciar a busca.

---

## Estratégias de Busca: Como Encontrar o Melhor Caminho

Existem diferentes formas de explorar o espaço de hiperparâmetros. Vamos começar pela mais sistemática.

### 1. Busca em Grade (Grid Search)

Você define uma lista de valores para cada hiperparâmetro e o algoritmo testa **todas as combinações possíveis**.

!!! example "Exemplo de busca em grade"

    Parâmetros:
    - n_estimators = [100, 200, 300]
    - max_depth = [5, 10]

    Total de combinações testadas: 3 × 2 = 6

!!! tip "Quando usar Grid Search"

    - Poucos hiperparâmetros envolvidos  
    - Número de combinações pequeno  
    - Tempo de treino aceitável

---

### 2. Busca Aleatória (Random Search)

O algoritmo sorteia aleatoriamente algumas combinações dentro do espaço definido.

!!! warning "Random Search não é inferior"

    Em muitos casos, Random Search encontra boas soluções mais rapidamente que Grid Search.  
    Especialmente útil quando:
    
    - Nem todos os hiperparâmetros influenciam igualmente  
    - O espaço de busca é amplo  
    - Tempo é um recurso escasso

---

### 3. Otimização Bayesiana e Métodos Avançados

Esses métodos constroem uma “função de expectativa” para prever quais combinações devem dar bons resultados, com base nas já testadas.

Ferramentas populares:

- `Optuna`  
- `Hyperopt`  
- `BayesianSearchCV`

!!! tip "Quando usar Otimização Bayesiana"

    - Modelos caros de treinar  
    - Muitos hiperparâmetros  
    - Necessidade de eficiência computacional

---

## Comparativo entre Estratégias

| Estratégia        | Quando usar                            | Prós                                | Contras                            |
|-------------------|-----------------------------------------|-------------------------------------|-------------------------------------|
| Grid Search       | Poucos hiperparâmetros e combinações    | Controle total, sistemático         | Custo computacional alto            |
| Random Search     | Espaço grande, pouco tempo              | Eficiente, cobre bem o espaço       | Não testa tudo                     |
| Otimização Bayesiana | Processos caros, tuning avançado     | Adaptativa, aprende com os testes   | Requer mais configuração            |

---

## Aplicando com Validação Cruzada

O jeito certo de ajustar hiperparâmetros é combinar cada tentativa com **validação cruzada**, garantindo uma avaliação robusta.

!!! example "Busca com validação cruzada em Scikit-Learn"

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10]
}

model = RandomForestClassifier(random_state=42)

grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)
```

!!! warning "Atenção ao Overfitting na Validação Cruzada"

    Mesmo com validação cruzada, é possível ajustar tanto o modelo ao processo de avaliação que ele se especializa nos folds — e perde generalização real.

    Por isso, o conjunto de teste final deve **sempre ser preservado** para a última avaliação.

---

## Dica Avançada: Validação Cruzada Aninhada

!!! tip "Validação Aninhada (Nested CV)"

    Em contextos mais exigentes (como competições ou pesquisa), pode-se usar validação cruzada aninhada:  
    - Uma validação externa (para estimar desempenho final)  
    - E uma interna (para ajustar os hiperparâmetros)

    Essa técnica elimina o risco de sobreajuste ao processo de tuning — mas é mais custosa.

---

## Checklist Final: Tuning com Consciência

- [ ] O conjunto de teste foi mantido completamente separado?  
- [ ] A busca usou validação cruzada (preferencialmente estratificada)?  
- [ ] O espaço de busca foi bem definido, com lógica e limites?  
- [ ] A métrica de avaliação está alinhada com o objetivo do projeto?  
- [ ] A complexidade do modelo final faz sentido para os dados?

---

## Desafio Final

??? question "E se você só tivesse 20 minutos para ajustar hiperparâmetros em um dataset grande?"

<details>
<summary>Possível abordagem</summary>

- Usaria Random Search com 5 a 10 combinações  
- Limitando os hiperparâmetros a dois ou três mais relevantes  
- Validando com 3-fold cross-validation estratificada  
- Escolheria uma métrica simples e representativa (ex: F1-score)  
- Reservaria um conjunto de teste final, mesmo pequeno

</details>

---

## Encerramento: Um Número Pode Mudar Tudo

Ajustar hiperparâmetros **não é uma formalidade técnica**.  
É uma etapa onde decisões sutis afetam a confiabilidade do modelo e, por consequência, as decisões que virão a seguir.

Modelos mal ajustados são como máquinas com peças frouxas: funcionam, mas falham quando mais importa.

Um bom cientista de dados trata o ajuste com seriedade, parcimônia e consciência.  
Porque até a escolha de um número carrega **consequências reais**.
