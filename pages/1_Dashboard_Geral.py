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

def _apply_plotly_layout(fig: go.Figure, title: str, y_title: str, x_title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=40, t=80, b=50),
    )
    return fig

def _fig_risco_ano(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("ano_base")["risco_defasagem"].agg(["mean", "count"]).reset_index()
    anos_str = g["ano_base"].astype(str)
    cores_ano = [COR_ANO.get(a, COR_RISCO) for a in anos_str]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=anos_str,
            y=g["mean"] * 100,
            name="% Risco",
            marker_color=cores_ano,
            opacity=0.85,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=anos_str,
            y=g["count"],
            mode="lines+markers",
            name="Nº Alunos",
            line=dict(color=COR_SEM_RISCO, width=3),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="% com Risco", range=[0, 100], secondary_y=False)
    fig.update_yaxes(title_text="Total de Alunos", secondary_y=True)
    fig = _apply_plotly_layout(fig, "Taxa de Risco e Volume por Ano", "% com Risco", "Ano", PLOTLY_HEIGHT_PADRAO)
    return fig


def _fig_risco_genero(df: pd.DataFrame) -> go.Figure:
    g = (
        df.groupby("genero_norm")["risco_defasagem"]
        .mean()
        .reset_index()
        .sort_values("risco_defasagem", ascending=False)
    )
    g["taxa_risco"] = g["risco_defasagem"] * 100
    g = g[g["taxa_risco"] > 0]

    labels = g["genero_norm"].tolist()
    values = g["taxa_risco"].tolist()
    cores = [COR_GENERO.get(lbl, "#9E9E9E") for lbl in labels]

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=cores),
            textinfo="label+percent",
            hovertemplate="%{label}<br>Taxa de risco: %{value:.1f}%<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(
        title="Taxa de Risco por Gênero",
        height=PLOTLY_HEIGHT_PADRAO,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def _fig_risco_inst(df: pd.DataFrame) -> go.Figure:
    g = (df.groupby("inst_cat")["risco_defasagem"]
           .agg(["mean", "count"]).reset_index().sort_values("mean"))
    media = df["risco_defasagem"].mean() * 100
    texto = [f"{v:.1f}% " for v, n in zip(g["mean"] * 100, g["count"])]
    cores_inst = [COR_BARRAS_INST[i % len(COR_BARRAS_INST)] for i in range(len(g))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=g["inst_cat"],
            x=g["mean"] * 100,
            orientation="h",
            name="% Risco",
            marker_color=cores_inst,
            text=texto,
            textposition="outside",
        )
    )
    fig.update_xaxes(range=[0, 70])
    fig = _apply_plotly_layout(fig, "Taxa de Risco por Categoria de Instituição", "% com Risco de Defasagem", "", PLOTLY_HEIGHT_PADRAO)
    return fig


def _fig_indicadores(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in INDICADORES if c in df.columns]
    sem = df[df["risco_defasagem"] == 0][cols].mean()
    com = df[df["risco_defasagem"] == 1][cols].mean()
    labels = [NOMES_IND.get(c, c) for c in cols]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=sem.values,
            name="Sem Risco",
            marker_color=COR_SEM_RISCO,
            text=[f"{v:.1f}" for v in sem.values],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=com.values,
            name="Com Risco",
            marker_color=COR_RISCO,
            text=[f"{v:.1f}" for v in com.values],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 12])
    fig = _apply_plotly_layout(fig, "Média dos Indicadores — Sem Risco vs Com Risco", "Média", "Indicadores", height=400)
    return fig


def _fig_idade(df: pd.DataFrame) -> go.Figure:
    sem = df[df["risco_defasagem"] == 0]["idade_num"].dropna()
    com = df[df["risco_defasagem"] == 1]["idade_num"].dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=sem,
            name="Sem Risco",
            marker_color=COR_SEM_RISCO,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=com,
            name="Com Risco",
            marker_color=COR_RISCO,
            opacity=0.7,
        )
    )
    fig.update_layout(barmode="overlay")
    fig = _apply_plotly_layout(fig, "Distribuição de Idades por Classe de Risco", "Quantidade de Alunos", "Idade", PLOTLY_HEIGHT_MENOR)
    return fig


def _fig_niveis(df: pd.DataFrame) -> go.Figure:
    counts = df["nivel_risco"].value_counts().reindex(NIVEL_ORDEM, fill_value=0)
    cores = [COR_SEM_RISCO, "#F4D03F", "#E67E22", COR_RISCO]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=cores,
            text=[str(int(v)) for v in counts.values],
            textposition="outside",
            name="Alunos",
        )
    )
    fig = _apply_plotly_layout(fig, "Distribuição por Nível de Risco", "Quantidade de Alunos", "Nível de Risco", PLOTLY_HEIGHT_MENOR)
    return fig


def _fig_comparacao(row_min: pd.Series, row_max: pd.Series) -> go.Figure:
    cols  = [c for c in INDICADORES if c in row_min.index]
    v_min = [float(row_min[c]) if pd.notna(row_min.get(c)) else 0.0 for c in cols]
    v_max = [float(row_max[c]) if pd.notna(row_max.get(c)) else 0.0 for c in cols]
    labels = [NOMES_IND.get(c, c) for c in cols]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=v_min,
            name="Menor Risco",
            marker_color=COR_SEM_RISCO,
            text=[f"{v:.1f}" for v in v_min],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=v_max,
            name="Maior Risco",
            marker_color=COR_RISCO,
            text=[f"{v:.1f}" for v in v_max],
            textposition="outside",
        )
    )
    fig.add_hline(y=7.0, line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 12])
    fig = _apply_plotly_layout(fig, "Comparação de Indicadores: Menor Risco vs Maior Risco", "Nota / Indicador", "Indicadores", PLOTLY_HEIGHT_PADRAO)
    return fig


def _collect_insights(df: pd.DataFrame, total: int, taxa_risco: float, media_prob: float) -> list[tuple[str, str, str]]:
    insights: list[tuple[str, str, str]] = []

    media_ieg = float(df["IEG"].mean()) if "IEG" in df.columns else float("nan")
    media_iaa = float(df["IAA"].mean()) if "IAA" in df.columns else float("nan")
    media_ida = float(df["IDA"].mean()) if "IDA" in df.columns else float("nan")

    if pd.notna(media_ieg):
        if media_ieg >= 8:
            insights.append(("sucesso", "✅", f"Engajamento forte dos alunos (IEG médio: {media_ieg:.1f})"))
        elif media_ieg >= 6:
            insights.append(("atencao", "⚠️", f"Engajamento moderado dos alunos (IEG médio: {media_ieg:.1f})"))
        else:
            insights.append(("critico", "🚨", f"Engajamento baixo dos alunos (IEG médio: {media_ieg:.1f})"))

    if "ano_base" in df.columns and df["ano_base"].nunique() >= 2:
        evolucao = df.groupby("ano_base")["risco_defasagem"].mean().sort_index()
        variacao_pp = (evolucao.iloc[-1] - evolucao.iloc[0]) * 100
        if variacao_pp <= -2:
            insights.append(("sucesso", "📉", f"Boa evolução temporal: risco caiu {abs(variacao_pp):.1f} p.p. no período"))
        elif variacao_pp >= 2:
            insights.append(("critico", "📈", f"Alerta temporal: risco subiu {variacao_pp:.1f} p.p. no período"))
        else:
            insights.append(("info", "➖", f"Risco estável no período (variação de {variacao_pp:+.1f} p.p.)"))


    if pd.notna(media_ida):
        if media_ida >= 7:
            insights.append(("sucesso", "🎓", f"Desempenho de aprendizagem consistente (IDA médio: {media_ida:.1f})"))
        else:
            insights.append(("atencao", "📚", f"Aprendizagem pede reforço (IDA médio: {media_ida:.1f})"))



    return insights[:3]


def _render_insights(insights: list[tuple[str, str, str]]) -> None:
    st.subheader("🔥 Insights Automáticos")
    st.markdown(
        """
        <style>
            .insight-card {
                border-radius: 10px;
                padding: 14px 16px;
                margin: 8px 0;
                border: 1px solid rgba(255,255,255,0.08);
                font-size: 1.06rem;
                font-weight: 600;
            }
            .insight-info {
                background: linear-gradient(90deg, rgba(31,86,140,0.55), rgba(31,86,140,0.35));
                color: #79b8ff;
            }
            .insight-sucesso {
                background: linear-gradient(90deg, rgba(20,110,65,0.55), rgba(20,110,65,0.35));
                color: #5af78e;
            }
            .insight-atencao {
                background: linear-gradient(90deg, rgba(145,125,15,0.55), rgba(145,125,15,0.35));
                color: #ffe88a;
            }
            .insight-critico {
                background: linear-gradient(90deg, rgba(139,42,50,0.55), rgba(139,42,50,0.35));
                color: #ff9ea8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    insights_top3 = insights[:3]
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            if i < len(insights_top3):
                tipo, emoji, texto = insights_top3[i]
                st.markdown(
                    f"<div class='insight-card insight-{tipo}'>{emoji} {texto}</div>",
                    unsafe_allow_html=True,
                )