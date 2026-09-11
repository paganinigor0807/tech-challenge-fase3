# ============================================================
# VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

sns.set_style("whitegrid")


# ============================================================
# DISTRIBUIÇÃO DA TARGET
# ============================================================

def plot_distribuicao_target(y_train, titulo="Distribuição da Target (Treino)", salvar=True):
    """
    Plota a distribuição das classes da variável alvo.
    """

    contagem = y_train.value_counts().sort_index()
    percentual = y_train.value_counts(normalize=True).sort_index() * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    barras = ax.bar(
        contagem.index.astype(str),
        contagem.values,
        color=["#e74c3c", "#2ecc71"]
    )

    for i, barra in enumerate(barras):
        altura = barra.get_height()
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            altura,
            f"{contagem.values[i]:,}\n({percentual.values[i]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Classe (0 = Não alfabetizado | 1 = Alfabetizado)")
    ax.set_ylabel("Quantidade de alunos")

    plt.tight_layout()

    if salvar:
        plt.savefig("../image/distribuicao_target.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# MAPA DE CORRELAÇÃO (EDA)
# ============================================================

def plot_mapa_correlacao(df, colunas_numericas=None, titulo="Mapa de Correlação", salvar=True):
    """
    Plota um heatmap de correlação entre variáveis numéricas.

    Parâmetros:
        df: DataFrame (bruto ou processado)
        colunas_numericas: lista de colunas a incluir (se None, usa todas numéricas)
        titulo: título do gráfico
    """

    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    correlacao = df[colunas_numericas].corr()

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        correlacao,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
        ax=ax
    )

    ax.set_title(titulo, fontsize=13, fontweight="bold")

    plt.tight_layout()

    if salvar:
        plt.savefig("../image/mapa_correlacao.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# COMPARAÇÃO DE MÉTRICAS ENTRE MODELOS
# ============================================================

def plot_comparacao_modelos(comparacao_final, metricas=None, salvar=True):
    """
    Plota um gráfico de barras comparando as métricas dos
    modelos testados no conjunto de teste.
    """

    if metricas is None:
        metricas = ["accuracy", "f1", "roc_auc"]

    dados = comparacao_final[
        comparacao_final["modelo"].str.contains("Teste")
    ].copy()

    dados["modelo"] = dados["modelo"].str.replace(" - Teste", "")

    dados = dados.set_index("modelo")[metricas]

    fig, ax = plt.subplots(figsize=(8, 5))

    dados.plot(kind="bar", ax=ax, colormap="viridis", rot=0)

    ax.set_title("Comparação de Modelos (Conjunto de Teste)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Métrica")

    plt.tight_layout()

    if salvar:
        plt.savefig("../image/comparacao_modelos.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# TOP VARIÁVEIS MAIS IMPORTANTES
# ============================================================

def plot_top_features(importancias, coluna_valor="importancia", top_n=10, titulo=None, salvar=True):
    """
    Plota as variáveis mais importantes do modelo final.

    Parâmetros:
        importancias: DataFrame com colunas "feature" e a métrica de importância
        coluna_valor: "importancia" (nativa) ou "importancia_shap" (SHAP)
    """

    dados = importancias.head(top_n).sort_values(coluna_valor)

    if titulo is None:
        titulo = "Top Variáveis Mais Importantes - Random Forest"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(dados["feature"], dados[coluna_valor], color="#3498db")

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importância")

    plt.tight_layout()

    if salvar:
        nome_arquivo = coluna_valor.replace(" ", "_")
        plt.savefig(f"../image/top_features_{nome_arquivo}.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# CURVA ROC - COMPARAÇÃO DE MODELOS
# ============================================================

def plot_curva_roc(modelos_dict, X_test, y_test, salvar=True):
    """
    Plota a curva ROC de múltiplos modelos no mesmo gráfico.

    Parâmetros:
        modelos_dict: dicionário {"nome do modelo": modelo_treinado}
        X_test: dados de teste
        y_test: variável alvo de teste
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    for nome, modelo in modelos_dict.items():

        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = modelo.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(fpr, tpr, label=f"{nome} (AUC = {auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório (AUC = 0.5)")

    ax.set_title("Curva ROC - Comparação de Modelos", fontsize=13, fontweight="bold")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.legend()

    plt.tight_layout()

    if salvar:
        plt.savefig("../image/curva_roc.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# MATRIZ DE CONFUSÃO (HEATMAP)
# ============================================================

def plot_matriz_confusao(modelo, X, y, nome_modelo="Modelo", salvar=True):
    """
    Plota a matriz de confusão como heatmap.

    Parâmetros:
        modelo: modelo treinado
        X: dados (treino ou teste)
        y: variável alvo verdadeira
        nome_modelo: nome para exibição no título
    """

    y_pred = modelo.predict(X)

    matriz = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Não alfabetizado (0)", "Alfabetizado (1)"],
        yticklabels=["Não alfabetizado (0)", "Alfabetizado (1)"],
        ax=ax
    )

    ax.set_title(f"Matriz de Confusão - {nome_modelo}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")

    plt.tight_layout()

    if salvar:
        nome_arquivo = nome_modelo.lower().replace(" ", "_")
        plt.savefig(f"../image/matriz_confusao_{nome_arquivo}.png", bbox_inches="tight", dpi=150)

    plt.show()


# ============================================================
# DISTRIBUIÇÃO DAS PROBABILIDADES PREVISTAS
# ============================================================

def plot_distribuicao_probabilidades(modelo, X_test, y_test, nome_modelo="Modelo", salvar=True):
    """
    Plota a distribuição (densidade) das probabilidades previstas
    pelo modelo, separadas por classe real.
    """

    y_prob = modelo.predict_proba(X_test)[:, 1]

    df_plot = pd.DataFrame({
        "probabilidade": y_prob,
        "classe_real": y_test.map({0: "Não alfabetizado", 1: "Alfabetizado"}).values
    })

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.kdeplot(
        data=df_plot,
        x="probabilidade",
        hue="classe_real",
        fill=True,
        alpha=0.4,
        palette=["#e74c3c", "#2ecc71"],
        ax=ax
    )

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)

    ax.set_title(f"Distribuição das Probabilidades Previstas - {nome_modelo}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Probabilidade Prevista (classe 1)")
    ax.set_ylabel("Densidade")

    plt.tight_layout()

    if salvar:
        nome_arquivo = nome_modelo.lower().replace(" ", "_")
        plt.savefig(f"../image/distribuicao_probabilidades_{nome_arquivo}.png", bbox_inches="tight", dpi=150)

    plt.show()