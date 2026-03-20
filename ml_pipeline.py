from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data") / "base_processada.csv"
MODEL_FILE = Path("artifacts") / "modelo_risco_defasagem.joblib"
TARGET_COL = "risco_defasagem"

def load_base() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df

def _risk_level(probability: float) -> str:
    if probability < 0.25:
        return "Sem Risco"
    elif probability < 0.50:
        return "Atencao"
    elif probability < 0.75:
        return "Risco Moderado"
    else:
        return "Risco Alto"
    
@dataclass
class TrainedBundle:
    model: RandomForestClassifier
    feature_columns: list[str]
    raw_feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    default_values: dict[str, Any]
    category_options: dict[str, list[Any]]
    metrics: dict[str, Any]

def train_bundle(random_state: int = 42, n_estimators: int = 300) -> TrainedBundle:
    base = load_base()

    X = base.drop(columns=[TARGET_COL])
    y = base[TARGET_COL]

    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    default_values: dict[str, Any] = {}
    category_options: dict[str, list[str]] = {}

    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        med = X[col].median()
        default_values[col] = float(med) if pd.notna(med) else 0.0

    for col in categorical_columns:
        mode = X[col].mode(dropna=True)
        default_values[col] = mode.iloc[0] if not mode.empty else "Não Informado"
        category_options[col] = sorted(X[col].dropna().astype(str).unique().tolist())
        X[col] = X[col].fillna(default_values[col]).astype(str)
    
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, pred, output_dict=True)
    accuracy = float(accuracy_score(y_test, pred))

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "n_registros": int(len(base)),
        "target_balance": y.value_counts().to_dict(),
    }

    return TrainedBundle(
        model=model,
        feature_columns=X_encoded.columns.tolist(),
        raw_feature_columns=X.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        default_values=default_values,
        category_options=category_options,
        metrics=metrics,
    )

def save_bundle(bundle: TrainedBundle, model_path: Path = MODEL_FILE) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

def save_bundle(bundle: TrainedBundle, model_path: Path = MODEL_FILE) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

def load_bundle(model_path: Path = MODEL_FILE) -> TrainedBundle:
    return joblib.load(model_path)

def train_and_persist(model_path: Path = MODEL_FILE) -> TrainedBundle:
    bundle = train_bundle()
    save_bundle(bundle, model_path=model_path)
    return bundle

def _prepare_inference_frame(df: pd.DataFrame, bundle: TrainedBundle) -> pd.DataFrame:
    frame = df.copy()
    for col in bundle.raw_feature_columns:
        if col not in frame.columns:
            frame[col] = bundle.default_values.get(col, "Nao Informado")
    
    frame = frame[bundle.raw_feature_columns]

    for col in bundle.numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame[col] = frame[col].fillna(bundle.default_values.get(col, 0.0))
    
    for col in bundle.categorical_columns:
        frame[col] = frame[col].fillna(bundle.default_values.get(col, "Nao Informado")).astype(str)
    
    encoded = pd.get_dummies(frame, drop_first=True)
    encoded = encoded.reindex(columns=bundle.feature_columns, fill_value=0)
    return encoded

def predict_dataframe(df: pd.DataFrame, bundle: TrainedBundle) -> pd.DataFrame:
    X = _prepare_inference_frame(df, bundle)
    probabilities = bundle.model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = df.copy()
    result["predicao_binaria"] = predictions
    result["probabilidade_risco"] = probabilities
    result["nivel_risco"] = result["probabilidade_risco"].apply(_risk_level)
    return result