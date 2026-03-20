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