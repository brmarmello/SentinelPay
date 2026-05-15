from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from sentinelpay.config import DEFAULT_THRESHOLD, ModelConfig, TARGET
from sentinelpay.data import dataset_summary, load_transactions
from sentinelpay.evaluation import evaluate_predictions, precision_recall_points
from sentinelpay.features import split_and_scale
from sentinelpay.model import predict_with_threshold, train_classifier


st.set_page_config(
    page_title="SentinelPay",
    page_icon=":shield:",
    layout="wide",
)


@st.cache_data(show_spinner="Carregando dataset publico de transacoes...")
def get_data() -> pd.DataFrame:
    return load_transactions()


@st.cache_resource(show_spinner="Treinando modelo XGBoost...")
def get_trained_assets(use_smote: bool, n_estimators: int):
    config = ModelConfig(use_smote=use_smote, n_estimators=n_estimators)
    df = get_data()
    X_train, X_test, y_train, y_test = split_and_scale(df, test_size=config.test_size)
    model = train_classifier(X_train, y_train, config)
    y_prob, _ = predict_with_threshold(model, X_test, DEFAULT_THRESHOLD)
    return model, X_train, X_test, y_train, y_test, y_prob


def plot_confusion_matrix(matrix) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Legitima", "Fraude"])
    ax.set_yticks([0, 1], labels=["Legitima", "Fraude"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color="#0f172a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


st.title("SentinelPay")
st.caption("Monitor inteligente para risco de fraude em transacoes financeiras.")

with st.sidebar:
    st.header("Configuracao")
    threshold = st.slider("Limiar de decisao", 0.01, 0.99, DEFAULT_THRESHOLD, 0.01)
    use_smote = st.toggle("Usar SMOTE no treino", value=True)
    n_estimators = st.slider("Arvores do XGBoost", 100, 500, 250, 50)
    shap_sample_size = st.slider("Amostras para SHAP", 20, 200, 60, 20)

try:
    raw_df = get_data()
    model, X_train, X_test, y_train, y_test, cached_prob = get_trained_assets(use_smote, n_estimators)
except Exception as exc:
    st.error(f"Nao foi possivel carregar dados ou treinar o modelo: {exc}")
    st.stop()

y_prob, y_pred = predict_with_threshold(model, X_test, threshold)
metrics = evaluate_predictions(y_test, y_prob, y_pred)
summary = dataset_summary(raw_df)

overview_tab, threshold_tab, explain_tab, data_tab = st.tabs(
    ["Visao Geral", "Limiar e Metricas", "Explicabilidade", "Dados"]
)

with overview_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transacoes", f"{summary['total_transactions']:,}")
    col2.metric("Fraudes", f"{summary['fraud_transactions']:,}")
    col3.metric("Taxa de fraude", f"{summary['fraud_rate']:.3%}")
    col4.metric("Average precision", f"{metrics['average_precision']:.3f}")

    class_counts = (
        raw_df[TARGET]
        .map({0: "Legitimas", 1: "Fraudes"})
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="quantidade")
    )
    fig = px.bar(
        class_counts,
        x="classe",
        y="quantidade",
        color="classe",
        color_discrete_map={"Legitimas": "#2563eb", "Fraudes": "#dc2626"},
        text_auto=True,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Quantidade")
    st.plotly_chart(fig, use_container_width=True)

with threshold_tab:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Relatorio de classificacao")
        st.dataframe(metrics["classification_report"].round(3), use_container_width=True)
    with col2:
        st.subheader("Matriz de confusao")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]), use_container_width=True)

    pr_df = precision_recall_points(y_test, y_prob)
    fig = px.line(pr_df, x="recall", y="precision", hover_data=["threshold"])
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

with explain_tab:
    st.subheader("Importancia global com SHAP")
    with st.spinner("Calculando explicacoes para uma amostra do teste..."):
        try:
            import shap

            sample = X_test.sample(min(shap_sample_size, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            fig, _ = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            fig2, _ = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, sample, show=False)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
        except Exception as exc:
            st.warning(f"SHAP nao ficou disponivel nesta execucao: {exc}")

with data_tab:
    st.subheader("Amostra dos dados")
    st.dataframe(raw_df.head(100), use_container_width=True)
    st.subheader("Top transacoes por probabilidade de fraude no teste")
    top_cases = X_test.copy()
    top_cases["fraud_probability"] = y_prob
    top_cases["actual_class"] = y_test
    st.dataframe(top_cases.sort_values("fraud_probability", ascending=False).head(25), use_container_width=True)
