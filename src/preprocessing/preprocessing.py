# Iniciando etapa de pré-processamento. iremos separar as variáveis explicativas da target, treino e teste, ajustar valores nulos
# ajustar variáveis categóricas, aplicar técnicas de escalonamento de dados e integrar isso à modelagem.

import pandas as pd

from sklearn.model_selection import train_test_split

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    OneHotEncoder
)

from sklearn.feature_selection import chi2


def separar_variaveis(df, target):
    """
    Separa as variáveis preditoras (X) da variável alvo (y).

    df = nome do dataframe
    target = nome da variavel target
    """

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def dividir_dados(X, y, test_size=0.20, random_state=42):
    """
    Divide os dados em treino e teste.

    X : Variáveis preditoras.
    y : Variável alvo.
    test_size : Proporção destinada ao conjunto de teste.
    random_state : Semente para garantir reprodutibilidade.

    Vai nos retornar: X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def identificar_tipos_colunas(X):
    """
    Identifica automaticamente as variáveis numéricas e categóricas.

    X : Variáveis preditoras.
    """

    colunas_numericas = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    colunas_categoricas = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    return colunas_numericas, colunas_categoricas


def preprocessar_dados(
        X_train,
        X_test,
        y_train,
        y_test,
        colunas_robust,
        colunas_minmax,
        id_municipio="id_municipio_alunos",
        id_uf="id_uf",
        rede="rede_alunos",
        threshold_pearson=0.10,
        threshold_spearman=0.20,
        alpha_chi2=0.05
):
    """
    Realiza o pré-processamento dos dados.

    ============================================================
    IMPUTAÇÃO
    ============================================================

    Variáveis numéricas:
        1. Município + Rede -> mediana
        2. UF + Rede        -> mediana
        3. Global           -> mediana do TREINO


    ============================================================
    FEATURE SELECTION (apenas análise, não exclui colunas)
    ============================================================

    Numéricas:
        - Pearson
        - Spearman

    Categóricas:
        - Qui-quadrado

    ============================================================
    TRANSFORMAÇÕES
    ============================================================

    Numéricas:
        - MinMaxScaler
        - StandardScaler disponível para teste
        - RobustScaler disponível para teste

    Categóricas:
        - OneHotEncoder
        - Dummy Encoding disponível através de drop="first"

    ============================================================
    DATA LEAKAGE
    ============================================================

    Todas as estatísticas são calculadas SOMENTE no X_train.

    X_test apenas recebe transformações calculadas
    anteriormente no conjunto de treino.

    Retorna:
        X_train_processado
        X_test_processado
        y_train
        y_test
        preprocessor
        features_selecionadas
        correlacoes_pearson
        resultados_spearman
        resultados_chi2
        distribuicao_target
    """

    # ============================================================
    # CÓPIAS
    # ============================================================

    X_train = X_train.copy()
    X_test = X_test.copy()

    y_train = y_train.copy()
    y_test = y_test.copy()


    # ============================================================
    # VERIFICAR COLUNAS DE GRANULARIDADE
    # ============================================================

    colunas_granularidade = [
        id_municipio,
        id_uf,
        rede
    ]

    for coluna in colunas_granularidade:

        if coluna not in X_train.columns:

            raise ValueError(
                f"A coluna '{coluna}' não foi encontrada no X_train."
            )


    # ============================================================
    # IDENTIFICAR VARIÁVEIS
    # ============================================================

    colunas_numericas, colunas_categoricas = (
        identificar_tipos_colunas(X_train)
    )


    # ============================================================
    # IDs NÃO SÃO FEATURES
    #
    # Mas continuam nos DataFrames porque serão utilizados
    # durante a imputação contextual.
    # ============================================================

    colunas_numericas = [
        coluna
        for coluna in colunas_numericas
        if coluna not in [id_municipio, id_uf, rede]
    ]

    # ============================================================
    # CAPITAL_UF E META_ATINGIDA É BINÁRIA/CATEGÓRICA
    #
    # Mesmo sendo 0/1, representa uma categoria.
    # ela ja esta tratada.
    # ============================================================

    if "capital_uf" in colunas_numericas:

        colunas_numericas.remove("capital_uf")

    if "meta_atingida" in colunas_numericas:
    
        colunas_numericas.remove("meta_atingida")



    # ============================================================
    # IMPUTAÇÃO DOS VALORES NULOS
    # 1. Município + Rede
    # 2. UF + Rede
    # 3. Mediana global do treino
    # ============================================================

    for coluna in colunas_numericas:

        # --------------------------------------------------------
        # 1. Mediana por Município + Rede
        # --------------------------------------------------------
        mediana_municipio_rede = (
            X_train
            .groupby([id_municipio, rede])[coluna]
            .median()
        )

        X_train[coluna] = X_train[coluna].fillna(
            pd.Series(
                X_train.set_index([id_municipio, rede]).index.map(mediana_municipio_rede),
                index=X_train.index
            )
        )

        X_test[coluna] = X_test[coluna].fillna(
            pd.Series(
                X_test.set_index([id_municipio, rede]).index.map(mediana_municipio_rede),
                index=X_test.index
            )
        )

        # --------------------------------------------------------
        # 2. Mediana por UF + Rede
        # --------------------------------------------------------
        mediana_uf_rede = (
            X_train
            .groupby([id_uf, rede])[coluna]
            .median()
        )

        X_train[coluna] = X_train[coluna].fillna(
            pd.Series(
                X_train.set_index([id_uf, rede]).index.map(mediana_uf_rede),
                index=X_train.index
            )
        )

        X_test[coluna] = X_test[coluna].fillna(
            pd.Series(
                X_test.set_index([id_uf, rede]).index.map(mediana_uf_rede),
                index=X_test.index
            )
        )

        # --------------------------------------------------------
        # 3. Mediana global do TREINO
        # --------------------------------------------------------
        mediana_global = X_train[coluna].median()

        X_train[coluna] = X_train[coluna].fillna(mediana_global)
        X_test[coluna] = X_test[coluna].fillna(mediana_global)


    # ============================================================
    # VERIFICAR NULOS APÓS IMPUTAÇÃO
    # ============================================================

    print("\nNulos após imputação:")

    print(
        f"Train: {X_train.isnull().sum().sum()}"
    )

    print(
        f"Test: {X_test.isnull().sum().sum()}"
    )


    # ============================================================
    # FEATURE SELECTION
    #
    # SOMENTE X_train + y_train
    # ============================================================

    resultados_feature_selection = {}


    # ============================================================
    # 1. PEARSON
    # ============================================================

    correlacoes_pearson = (
        X_train[colunas_numericas]
        .corrwith(y_train)
        .abs()
        .sort_values(
            ascending=False
        )
    )


    print("\n" + "=" * 60)
    print("PEARSON")
    print("=" * 60)

    print(correlacoes_pearson)


    resultados_feature_selection["pearson"] = (
        correlacoes_pearson
    )


    colunas_pearson = correlacoes_pearson[
        correlacoes_pearson >= threshold_pearson
    ].index.tolist()


    # ============================================================
    # 2. SPEARMAN
    # ============================================================

    resultados_spearman = []


    for col in colunas_numericas:

        corr, pval = spearmanr(
            X_train[col],
            y_train
        )

        if abs(corr) >= threshold_spearman:

            resultados_spearman.append({

                "feature": col,

                "correlacao": corr,

                "p_valor": pval

            })


    resultados_spearman = pd.DataFrame(
        resultados_spearman
    )


    print("\n" + "=" * 60)
    print("SPEARMAN")
    print("=" * 60)

    print(resultados_spearman)


    resultados_feature_selection["spearman"] = (
        resultados_spearman
    )


    if not resultados_spearman.empty:

        colunas_spearman = (
            resultados_spearman[
                "feature"
            ].tolist()
        )

    else:

        colunas_spearman = []


    # ============================================================
    # 3. QUI-QUADRADO
    #
    # SOMENTE VARIÁVEIS CATEGÓRICAS
    # ============================================================

    X_cat_train = X_train[
        colunas_categoricas
    ].copy()


    X_cat_codificado = pd.DataFrame(
        index=X_cat_train.index
    )


    for coluna in colunas_categoricas:

        X_cat_codificado[coluna] = (
            X_cat_train[coluna]
            .astype("category")
            .cat.codes
        )


    scores_chi2, pvalues_chi2 = chi2(
        X_cat_codificado,
        y_train
    )


    resultados_chi2 = pd.DataFrame({

        "feature": colunas_categoricas,

        "chi2": scores_chi2,

        "p_valor": pvalues_chi2

    }).sort_values(
        "p_valor"
    )


    print("\n" + "=" * 60)
    print("QUI-QUADRADO")
    print("=" * 60)

    print(resultados_chi2)


    resultados_feature_selection["chi2"] = (
        resultados_chi2
    )


    colunas_chi2 = resultados_chi2.loc[
        resultados_chi2["p_valor"] < alpha_chi2,
        "feature"
    ].tolist()


    # ============================================================
    # FEATURES SELECIONADAS (apenas análise/relatório)
    # ============================================================

    features_selecionadas = {

        "pearson": colunas_pearson,

        "spearman": colunas_spearman,

        "chi2": colunas_chi2

    }


    # ============================================================
    # Nenhuma coluna numérica é excluída por esses testes —
    # todas as colunas numéricas originais seguem no pipeline.
    # ============================================================

    colunas_numericas_selecionadas = colunas_numericas.copy()


    # ============================================================
    # Nenhuma coluna categórica é excluída por esses testes —
    # todas as colunas categóricas originais seguem no pipeline.
    # ============================================================

    colunas_categoricas_selecionadas = colunas_categoricas.copy()

    # ============================================================
    # COLUNAS FINAIS
    # ============================================================

    colunas_finais = (
        colunas_numericas_selecionadas
        + colunas_categoricas_selecionadas
        + ["capital_uf", "meta_atingida"]
    )


    X_train = X_train[
        colunas_finais
    ]

    X_test = X_test[
        colunas_finais
    ]


    # ============================================================
    # ATUALIZAR LISTAS
    # ============================================================

    colunas_numericas = (
        colunas_numericas_selecionadas
    )

    colunas_categoricas = (
        colunas_categoricas_selecionadas
    )


    # ============================================================
    # PREPROCESSOR
    #
    # NUMÉRICAS -> MINMAX
    # CATEGÓRICAS -> ONE HOT
    # ============================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numericas_minmax",
                MinMaxScaler(),
                # Para testar posteriormente:
                # StandardScaler()
                colunas_minmax
            ),

            (
                "numericas_robust",
                RobustScaler(),
                colunas_robust
            ),

            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                colunas_categoricas
            ),
            (
                "passthrough",
                "passthrough",
                ["capital_uf", "meta_atingida"]
            )

        ],

        remainder="drop"

    )


    # ============================================================
    # FIT SOMENTE NO TREINO
    # ============================================================

    X_train_processado = (
        preprocessor.fit_transform(
            X_train
        )
    )


    # ============================================================
    # TEST -> SOMENTE TRANSFORM
    # ============================================================

    X_test_processado = (
        preprocessor.transform(
            X_test
        )
    )


    # ============================================================
    # NOMES DAS FEATURES
    # ============================================================

    nomes_features = (
        preprocessor
        .get_feature_names_out()
    )


    # ============================================================
    # DATAFRAME FINAL
    # ============================================================

    X_train_processado = pd.DataFrame(

        X_train_processado,

        columns=nomes_features,

        index=X_train.index

    )


    X_test_processado = pd.DataFrame(

        X_test_processado,

        columns=nomes_features,

        index=X_test.index

    )


    # ============================================================
    # CHECK-UP
    # ============================================================

    print("\n" + "=" * 60)
    print("CHECK-UP DO DATASET FINAL")
    print("=" * 60)


    print("\nDimensões:")

    print(
        f"Train: {X_train_processado.shape}"
    )

    print(
        f"Test:  {X_test_processado.shape}"
    )


    print("\nValores nulos:")

    print(
        f"Train: {X_train_processado.isnull().sum().sum()}"
    )

    print(
        f"Test:  {X_test_processado.isnull().sum().sum()}"
    )


    print("\nTipos das variáveis finais:")

    print(
        X_train_processado
        .dtypes
        .value_counts()
    )


    # ============================================================
    # DISTRIBUIÇÃO DA TARGET
    # ============================================================

    print("\n" + "=" * 60)
    print("DISTRIBUIÇÃO DA TARGET")
    print("=" * 60)


    distribuicao_target = pd.DataFrame({

        "quantidade":
            y_train
            .value_counts()
            .sort_index(),

        "percentual":
            y_train
            .value_counts(
                normalize=True
            )
            .sort_index()
            .mul(100)

    })


    print("\nDistribuição no TRAIN:")

    print(
        distribuicao_target
    )


    # ============================================================
    # BALANCEAMENTO
    # ============================================================

    percentual_minoria = (
        distribuicao_target[
            "percentual"
        ].min()
    )


    print("\nAvaliação do balanceamento:")


    if percentual_minoria >= 40:

        print(
            "✓ Target relativamente balanceada."
        )

    elif percentual_minoria >= 30:

        print(
            "⚠️ Leve desbalanceamento."
        )

    elif percentual_minoria >= 20:

        print(
            "⚠️ Desbalanceamento moderado."
        )

    else:

        print(
            "⚠️ Forte desbalanceamento."
        )


    # ============================================================
    # DISTRIBUIÇÃO NOS CONJUNTOS
    # ============================================================

    print(
        "\nDistribuição da target por conjunto:"
    )


    distribuicao_conjuntos = pd.DataFrame({

        "Train":
            y_train
            .value_counts(
                normalize=True
            ),

        "Test":
            y_test
            .value_counts(
                normalize=True
            )

    }).T * 100


    print(
        distribuicao_conjuntos.round(2)
    )


    # ============================================================
    # RETORNO
    # ============================================================

    return (

        X_train_processado,

        X_test_processado,

        y_train,

        y_test,

        preprocessor,

        features_selecionadas,

        correlacoes_pearson,

        resultados_spearman,

        resultados_chi2,

        distribuicao_target

    )