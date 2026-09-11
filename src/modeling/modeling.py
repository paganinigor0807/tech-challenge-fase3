# ============================================================
# MODELAGEM
# ============================================================

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# VALIDAÇÃO CRUZADA (K-FOLD)
# ============================================================

def validar_cruzado(
    modelo,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    nome_modelo="Modelo"
):
    """
    Executa validação cruzada (K-Fold) em um modelo,
    usando somente o conjunto de treino.

    Parâmetros:
        modelo: estimador do sklearn (sem treinar ainda)
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds
        scoring: métrica utilizada
        nome_modelo: nome para exibição

    Retorna:
        scores: array com o score de cada fold
    """

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print(f"\n===== VALIDAÇÃO CRUZADA - {nome_modelo} =====")
    print(f"Scores por fold ({scoring}): {scores.round(4)}")
    print(f"Média:         {scores.mean():.4f}")
    print(f"Desvio padrão: {scores.std():.4f}")

    return scores


# ============================================================
# DECISION TREE
# ============================================================

def treinar_decision_tree(
    X_train,
    y_train,
    cv=5,
    scoring="f1"
):
    """
    Treina uma Decision Tree utilizando validação cruzada
    e GridSearchCV para ajuste de hiperparâmetros.

    Parâmetros:
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds da validação cruzada
        scoring: métrica utilizada pelo GridSearch

    Retorna:
        modelo_decision_tree
        grid_search
    """

    modelo = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    # --------------------------------------------------------
    # Validação cruzada com parâmetros padrão, antes do ajuste
    # de hiperparâmetros, só para termos uma referência inicial
    # --------------------------------------------------------
    validar_cruzado(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        nome_modelo="Decision Tree (parâmetros padrão)"
    )

    parametros = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    modelo_decision_tree = grid_search.best_estimator_

    print("\n===== DECISION TREE (após GridSearch) =====")
    print("Melhores parâmetros:")
    print(grid_search.best_params_)

    print("\nMelhor score (média da validação cruzada):")
    print(grid_search.best_score_)

    return modelo_decision_tree, grid_search



def treinar_decision_tree_randomized(
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_iter=20,
    random_state=42
):
    """
    Treina uma Decision Tree utilizando validação cruzada
    e RandomizedSearchCV para ajuste de hiperparâmetros.

    Parâmetros:
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds da validação cruzada
        scoring: métrica utilizada pelo RandomizedSearch
        n_iter: número de combinações aleatórias testadas
        random_state: semente para reprodutibilidade

    Retorna:
        modelo_decision_tree
        random_search
    """

    modelo = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    # --------------------------------------------------------
    # Validação cruzada com parâmetros padrão, antes do ajuste
    # de hiperparâmetros, só para termos uma referência inicial
    # --------------------------------------------------------
    validar_cruzado(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        nome_modelo="Decision Tree (parâmetros padrão)"
    )

    parametros = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    modelo_decision_tree = random_search.best_estimator_

    print("\n===== DECISION TREE (após RandomizedSearch) =====")
    print("Melhores parâmetros:")
    print(random_search.best_params_)

    print("\nMelhor score (média da validação cruzada):")
    print(random_search.best_score_)

    return modelo_decision_tree, random_search


# ============================================================
# XGBOOST
# ============================================================

def treinar_xgboost_randomized(
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_iter=15,
    random_state=42
):
    """
    Treina um XGBoost utilizando validação cruzada
    e RandomizedSearchCV para ajuste de hiperparâmetros.

    Como o dataset é desbalanceado, usamos scale_pos_weight
    (equivalente ao class_weight="balanced" do sklearn) para
    dar mais peso à classe minoritária.

    Parâmetros:
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds da validação cruzada
        scoring: métrica utilizada pelo RandomizedSearch
        n_iter: número de combinações aleatórias testadas
        random_state: semente para reprodutibilidade

    Retorna:
        modelo_xgboost
        random_search
    """

    # --------------------------------------------------------
    # scale_pos_weight = qtd classe majoritária / qtd classe minoritária
    # --------------------------------------------------------
    qtd_negativos = (y_train == 0).sum()
    qtd_positivos = (y_train == 1).sum()
    scale_pos_weight = qtd_negativos / qtd_positivos

    modelo = XGBClassifier(
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist" 
    )

    # --------------------------------------------------------
    # Validação cruzada com parâmetros padrão, antes do ajuste
    # de hiperparâmetros, só para termos uma referência inicial
    # --------------------------------------------------------
    validar_cruzado(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        nome_modelo="XGBoost (parâmetros padrão)"
    )

    parametros = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],       
    "gamma": [0, 0.1, 0.3],              
    "reg_alpha": [0, 0.1, 1],            
    "reg_lambda": [1, 1.5, 2] 
    }

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=2
    )

    random_search.fit(X_train, y_train)

    modelo_xgboost = random_search.best_estimator_

    print("\n===== XGBOOST (após RandomizedSearch) =====")
    print("Melhores parâmetros:")
    print(random_search.best_params_)

    print("\nMelhor score (média da validação cruzada):")
    print(random_search.best_score_)

    return modelo_xgboost, random_search


# ============================================================
# RANDOM FOREST
# ============================================================

def treinar_random_forest(
    X_train,
    y_train,
    cv=5,
    scoring="f1"
):
    """
    Treina uma Random Forest utilizando validação cruzada
    e GridSearchCV para ajuste de hiperparâmetros.

    Parâmetros:
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds da validação cruzada
        scoring: métrica utilizada pelo GridSearch

    Retorna:
        modelo_random_forest
        grid_search
    """

    modelo = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    validar_cruzado(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        nome_modelo="Random Forest (parâmetros padrão)"
    )

    parametros = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    }

    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    modelo_random_forest = grid_search.best_estimator_

    print("\n===== RANDOM FOREST (após GridSearch) =====")
    print("Melhores parâmetros:")
    print(grid_search.best_params_)

    print("\nMelhor score (média da validação cruzada):")
    print(grid_search.best_score_)

    return modelo_random_forest, grid_search



def treinar_random_forest_randomized(
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_iter=10,
    random_state=42
):
    """
    Treina uma Random Forest utilizando validação cruzada
    e RandomizedSearchCV para ajuste de hiperparâmetros.

    Parâmetros:
        X_train: dados de treinamento
        y_train: variável alvo
        cv: número de folds da validação cruzada
        scoring: métrica utilizada pelo RandomizedSearch
        n_iter: número de combinações aleatórias testadas
        random_state: semente para reprodutibilidade

    Retorna:
        modelo_random_forest
        random_search
    """

    modelo = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    validar_cruzado(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        nome_modelo="Random Forest (parâmetros padrão)"
    )

    parametros = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5]
    }

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros,
        n_iter=8,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=2
    )

    random_search.fit(X_train, y_train)

    modelo_random_forest = random_search.best_estimator_

    print("\n===== RANDOM FOREST (após RandomizedSearch) =====")
    print("Melhores parâmetros:")
    print(random_search.best_params_)

    print("\nMelhor score (média da validação cruzada):")
    print(random_search.best_score_)

    return modelo_random_forest, random_search


# ============================================================
# AVALIAÇÃO DO MODELO
# ============================================================

def avaliar_modelo(
    modelo,
    X,
    y,
    nome_modelo="Modelo"
):
    """
    Avalia um modelo de classificação.

    Retorna um DF com as principais métricas.
    """

    y_pred = modelo.predict(X)

    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = None

    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    matriz = confusion_matrix(y, y_pred)

    print(f"\n===== {nome_modelo} =====")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    if auc is not None:
        print(f"ROC-AUC:   {auc:.4f}")

    print("\nMatriz de confusão:")
    print(matriz)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            zero_division=0
        )
    )

    resultados = pd.DataFrame({
        "modelo": [nome_modelo],
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1": [f1],
        "roc_auc": [auc]
    })

    return resultados


# ============================================================
# COMPARAÇÃO DOS MODELOS
# ============================================================

def comparar_modelos(resultados):
    """
    Compara os resultados dos modelos.

    Parâmetro:
        resultados: lista de dataframes retornados por
                    avaliar_modelo()

    Retorna um DF consolidado.
    """

    comparacao = pd.concat(
        resultados,
        ignore_index=True
    )

    comparacao = comparacao.sort_values(
        by="f1",
        ascending=False
    )

    return comparacao