from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_utils import ensure_data_file, get_base_dataframes

st.set_page_config(page_title="Análise de Indicadores - Passos Mágicos", layout="wide")

INDICADORES = ["IAA", "IEG", "IPS", "IDA"]
INDICADOR_HELP_MAP = {
    "IAA": "Indicador de Autoavaliação do Aluno",
    "IEG": "Indicador de Engajamento",
    "IPS": "Indicador Psicossocial",
    "IDA": "Indicador de Aprendizagem",
}

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genero_norm"] = df["genero"].map(
        {
            "Feminino": "Feminino",
            "Menina": "Feminino",
            "Masculino": "Masculino",
            "Menino": "Masculino",
        }
    ).fillna("Outro")

    for col in INDICADORES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ano_base" in df.columns:
        df["ano_base"] = pd.to_numeric(df["ano_base"], errors="coerce")

    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filtros")

        anos = sorted(df["ano_base"].dropna().astype(int).unique().tolist())
        sel_anos = st.multiselect("Ano base", options=anos, default=anos)

        generos = sorted(df["genero_norm"].dropna().unique().tolist())
        sel_genero = st.multiselect("Gênero", options=generos, default=generos)

        risco_options = ["Todos", "Sem risco", "Com risco"]
        sel_risco = st.selectbox("Classe de risco", options=risco_options, index=0)

    mask = df["ano_base"].isin(sel_anos) & df["genero_norm"].isin(sel_genero)

    if sel_risco == "Sem risco":
        mask &= df["risco_defasagem"] == 0
    elif sel_risco == "Com risco":
        mask &= df["risco_defasagem"] == 1

    return df[mask].copy()

def _melt_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in INDICADORES if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["indicador", "valor"])

    melted = df.melt(
        id_vars=["ano_base", "risco_defasagem", "genero_norm"],
        value_vars=cols,
        var_name="indicador",
        value_name="valor",
    )
    return melted.dropna(subset=["valor"])

def _fig_evolucao_ano(df: pd.DataFrame) -> go.Figure:
    evo = (
        df.groupby("ano_base")[INDICADORES]
        .mean(numeric_only=True)
        .reset_index()
        .dropna(subset=["ano_base"])
    )

    fig = go.Figure()
    cores = {"IAA": "#4EA8DE", "IEG": "#F45B69", "IPS": "#F4D35E", "IDA": "#2EC4B6"}

    for ind in INDICADORES:
        if ind in evo.columns:
            fig.add_trace(
                go.Scatter(
                    x=evo["ano_base"].astype(int),
                    y=evo[ind],
                    mode="lines+markers",
                    name=ind,
                    line=dict(width=3, color=cores[ind]),
                    marker=dict(size=8),
                )
            )

    fig.update_layout(
        title="Evolução dos Indicadores por Ano",
        xaxis_title="Ano",
        yaxis_title="Valor médio",
        hovermode="x unified",
        height=460,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_comparacao_risco(df_melt: pd.DataFrame) -> go.Figure:
    base = (
        df_melt.groupby(["indicador", "risco_defasagem"])["valor"]
        .mean()
        .reset_index()
        .assign(classe=lambda x: x["risco_defasagem"].map({0: "Sem Risco", 1: "Com Risco"}))
    )

    fig = px.bar(
        base,
        x="indicador",
        y="valor",
        color="classe",
        barmode="group",
        color_discrete_map={"Sem Risco": "#4EA8DE", "Com Risco": "#F45B69"},
        text=base["valor"].round(2),
        title="Média dos Indicadores: Sem Risco vs Com Risco",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Indicadores",
        yaxis_title="Média",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _fig_distribuicao(df_melt: pd.DataFrame) -> go.Figure:
    base = df_melt.copy()
    base["classe"] = base["risco_defasagem"].map({0: "Sem Risco", 1: "Com Risco"})

    fig = px.box(
        base,
        x="indicador",
        y="valor",
        color="classe",
        color_discrete_map={"Sem Risco": "#4EA8DE", "Com Risco": "#F45B69"},
        title="Distribuição dos Indicadores por Classe de Risco",
        points=False,
    )
    fig.update_layout(
        xaxis_title="Indicadores",
        yaxis_title="Valor",
        height=430,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def main() -> None:
    ensure_data_file()
    base, _ = get_base_dataframes()

    st.title("Análise de Indicadores")
    st.caption("Página dedicada aos indicadores IAA, IEG, IPS e IDA.")

    df = _prepare(base)
    df = _apply_filters(df)

    if df.empty:
        st.warning("Nenhum registro encontrado com os filtros selecionados.")
        return

    st.subheader("Métricas Gerais", text_alignment='center')
    k_cols = st.columns(4)
    for i, ind in enumerate(INDICADORES):
        val = float(df[ind].mean()) if ind in df.columns else float("nan")
        k_cols[i].metric(
            f"{ind} médio",
            f"{val:.2f}" if pd.notna(val) else "N/A",
            help=INDICADOR_HELP_MAP.get(ind),
        )

    st.divider()

    st.subheader("Evolução no Tempo", text_alignment='center')
    st.plotly_chart(_fig_evolucao_ano(df))
    st.caption("Mostra a tendência dos indicadores ao longo dos anos para acompanhar melhora ou queda de desempenho.", text_alignment='center')

    melted = _melt_indicadores(df)
    if melted.empty:
        st.info("Não foi possível gerar os gráficos de indicadores por falta de dados válidos.")
        return

    st.subheader("Comparativo e Distribuição dos Indicadores", text_alignment="center")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_fig_comparacao_risco(melted))
        st.caption("Compara a média dos indicadores entre alunos sem risco e com risco de defasagem.", text_alignment='center')
    with col2:
        st.plotly_chart(_fig_distribuicao(melted))
        st.caption("Mostra dispersão e mediana dos indicadores para identificar variações entre grupos.", text_alignment='center')


if __name__ == "__main__":
    main()
