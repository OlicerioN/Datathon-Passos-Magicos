from __future__ import annotations
import pandas as pd
import streamlit as st

from ml_pipeline import DATA_FILE, MODEL_FILE, load_base, load_bundle, train_and_persist

def ensure_data_file() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Base nao encontrada: {DATA_FILE}")
    
@st.cache_resource(show_spinner=False)
def get_bundle(force_retrain: bool = False):
    if force_retrain or not MODEL_FILE.exists():
        return train_and_persist(MODEL_FILE)
    return load_bundle(MODEL_FILE)

@st.cache_data(show_spinner=False)
def get_base_dataframe() -> tuple[pd.DataFrame, pd.DataFrame]:
    "Retorna (base_processada, base_processada). Ambos apontam para o mesmo CSV que foi usado no notebook"
    base = load_base()
    return base, base
