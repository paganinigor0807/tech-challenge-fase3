# ============================================================
# EVALUATION
# ============================================================



import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ============================================================
# SHAP - ANÁLISE COMPLETA
# ============================================================


def criar_shap_explainer(modelo, X, amostra=5000, random_state=42):
    """
    Cria o TreeExplainer e calcula os SHAP values.

    Parâmetros:
        modelo: modelo treinado (baseado em árvore)
        X: DataFrame processado
        amostra: tamanho da amostra (recomendado para datasets grandes)
        random_state: semente para reprodutibilidade

    Retorna:
        explainer, shap_values, expected_value, X_amostra
    """

    if amostra is not None and len(X) > amostra:
        X_amostra = X.sample(amostra, random_state=random_state)
    else:
        X_amostra = X

    explainer = shap.TreeExplainer(modelo)

    shap_values = explainer.shap_values(X_amostra)

    # --------------------------------------------------------
    # Classificação binária: pega a classe positiva (1)
    #
    # Versões antigas do shap retornam uma LISTA [classe_0, classe_1]
    # Versões novas retornam um ARRAY 3D (n_amostras, n_features, n_classes)
    # --------------------------------------------------------

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    expected_value = explainer.expected_value

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    print(f"Shape dos SHAP values: {shap_values.shape}")
    print(f"Valor base (expected value): {expected_value:.4f}")

    return explainer, shap_values, expected_value, X_amostra


def verificar_eficiencia_shap(modelo, shap_values, expected_value, X_amostra, n=5):
    """
    Verifica a propriedade de eficiência do SHAP:
    soma dos SHAP values + valor base = previsão do modelo.

    Serve como teste de sanidade, não como gráfico para apresentação.
    """

    for i in range(n):
        prediction = modelo.predict_proba(X_amostra.iloc[[i]])[0][1]
        shap_sum = shap_values[i].sum() + expected_value

        print(f"Instância {i}:")
        print(f"  Previsão do modelo:  {prediction:.6f}")
        print(f"  Valor base + Σ SHAP: {shap_sum:.6f}")
        print(f"  Diferença:           {abs(prediction - shap_sum):.10f}")
        print()


def plot_shap_waterfall(shap_values, expected_value, X_amostra, indice=0, max_display=10, salvar=True):
    """
    Gera o Waterfall Plot para uma instância específica.

    Parâmetros:
        indice: posição da instância dentro de X_amostra
        max_display: quantidade de features exibidas
        salvar: se True, salva em ../image
    """

    explanation = shap.Explanation(
        values=shap_values[indice],
        base_values=expected_value,
        data=X_amostra.iloc[indice].values,
        feature_names=X_amostra.columns.tolist()
    )

    shap.waterfall_plot(explanation, max_display=max_display, show=False)

    if salvar:
        plt.savefig(f"../image/waterfall_instancia_{indice}.png", bbox_inches="tight", dpi=150)

    plt.show()
    plt.close()


def plot_shap_summary(shap_values, X_amostra, salvar=True):
    """
    Gera o Summary Plot (beeswarm) - visão global do modelo.
    """

    plt.figure(figsize=(10, 8))

    shap.summary_plot(shap_values, X_amostra, show=False)

    if salvar:
        plt.savefig("../image/summary_plot.png", bbox_inches="tight", dpi=150)

    plt.show()
    plt.close()


def plot_shap_dependence(feature, shap_values, X_amostra, salvar=True):
    """
    Gera o Dependence Plot para uma feature específica.

    Útil para revelar relações não-lineares que Pearson não captura.
    """

    plt.figure(figsize=(8, 6))

    shap.dependence_plot(feature, shap_values, X_amostra, show=False)

    if salvar:
        nome_arquivo = feature.replace(" ", "_").replace("/", "_")
        plt.savefig(f"../image/dependence_{nome_arquivo}.png", bbox_inches="tight", dpi=150)

    plt.show()
    plt.close()