from __future__ import annotations

import io
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from matplotlib.backends.backend_pdf import PdfPages
from plotly.subplots import make_subplots
from ml_pipeline import predict_dataframe
from streamlit_utils import ensure_data_file, get_base_dataframes, get_bundle

st.set_page_config(page_title="Dashboard Geral - Passos Mágicos", layout="wide")

COR_RISCO = "#E84855"
COR_SEM_RISCO = "#2E86AB"

COR_ANO = {"2022": "#2E7D32", "2023": "#FBC02D", "2024": "#1E88E5"}
COR_GENERO = {"Feminino": "#EC407A", "Masculino": "#1E88E5", "Outro": "#9E9E9E"}
COR_BARRAS_INST = ["#8E44AD", "#E67E22", "#16A085", "#2E86DE", "#D35400", "#27AE60"]

INDICADORES = ["IAA", "IEG", "IPS", "IDA"]
NOMES_IND = {
    "IAA": "IAA\n(Autoaval.)",
    "IEG": "IEG\n(Engajam.)",
    "IPS": "IPS\n(Psicossoc.)",
    "IDA": "IDA\n(Aprendiz.)",
}
NIVEL_ORDEM = ["Sem Risco", "Atencao", "Risco Moderado", "Risco Alto"]
NIVEL_EMOJIS = {"Sem Risco": "🟢", "Atencao": "🟡", "Risco Moderado": "🟠", "Risco Alto": "🔴"}

PLOTLY_HEIGHT_PADRAO = 460
PLOTLY_HEIGHT_MENOR = 420

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genero_norm"] = df["genero"].map(
        {"Feminino": "Feminino", "Menina": "Feminino",
         "Masculino": "Masculino", "Menino": "Masculino"}
    ).fillna("Outro")
    df["idade_num"] = pd.to_numeric(df["idade"], errors="coerce")

    def _cat_inst(v: str) -> str:
        v_low = str(v).lower()
        if "pública" in v_low or "publica" in v_low or "escola p" in v_low:
            return "Escola Publica"
        if any(x in v_low for x in ["privada", "decisão", "decisao", "jp ii"]):
            return "Escola Privada/Parceira"
        if any(x in v_low for x in ["universitário", "universitario", "3º em", "3o em"]):
            return "Ensino Superior/Formado"
        return "Outro"

    df["inst_cat"] = df["instituicao_aluno"].fillna("Nao Informado").apply(_cat_inst)
    return df

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("🔎 Filtros")

        anos = sorted(df["ano_base"].dropna().unique().tolist())
        sel_anos = st.multiselect("Ano base", options=anos, default=anos)

        generos = sorted(df["genero_norm"].unique().tolist())
        sel_gen = st.multiselect("Gênero", options=generos, default=generos)

        insts = sorted(df["inst_cat"].unique().tolist())
        sel_inst = st.multiselect("Instituição", options=insts, default=insts)

        i_min = int(df["idade_num"].min(skipna=True)) if df["idade_num"].notna().any() else 7
        i_max = int(df["idade_num"].max(skipna=True)) if df["idade_num"].notna().any() else 27
        sel_idade = st.slider("Faixa de Idade", min_value=i_min, max_value=i_max, value=(i_min, i_max))

    mask = (
        df["ano_base"].isin(sel_anos)
        & df["genero_norm"].isin(sel_gen)
        & df["inst_cat"].isin(sel_inst)
    )
    if df["idade_num"].notna().any():
        mask &= df["idade_num"].between(sel_idade[0], sel_idade[1], inclusive="both").fillna(False)
    return df[mask].copy()