"""
CALORIQ — AI Calories Predictor
================================
A Streamlit dashboard around a trained Multiple Linear Regression model.

Model features (exactly as trained in the notebook, in this order):
    Age, Height (cm), Weight (kg), Duration (min), Heart_Rate (bpm), Body_Temp (°C)

No scaler or encoder was used in training, so none is applied here.
The saved model is only loaded and used for inference. It is never retrained.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CALORIQ — AI Calories Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="auto",
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Preferred model file names (matched case-insensitively). If none of these exist,
# the app falls back to ANY .pki / .pkl / .joblib file in the project folder, so a
# renamed model file still works.
MODEL_CANDIDATES = [
    "Calories.pki",
    "mobile_charging_time_prediction_model.pki",
    "model.pkl",
    "model.joblib",
    "model.pki",
    "calories_model.pkl",
    "calories_model.joblib",
]
MODEL_EXTENSIONS = {".pki", ".pkl", ".joblib"}
MODEL_FOLDERS = ["", "model", "models"]
DATA_CANDIDATES = ["calories.csv", "data/calories.csv", "dataset/calories.csv"]

TARGET = "Calories"
FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]  # training order

LABELS = {
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "Duration": "Duration",
    "Heart_Rate": "Heart rate",
    "Body_Temp": "Body temperature",
}
UNITS = {
    "Age": "years",
    "Height": "cm",
    "Weight": "kg",
    "Duration": "min",
    "Heart_Rate": "bpm",
    "Body_Temp": "°C",
}
# Hard limits: values outside these are rejected with an error.
HARD_LIMITS = {
    "Age": (10, 100),
    "Height": (100.0, 230.0),
    "Weight": (25.0, 200.0),
    "Duration": (1, 300),
    "Heart_Rate": (40, 220),
    "Body_Temp": (35.0, 42.0),
}
DEFAULTS = {
    "Age": 30,
    "Height": 170.0,
    "Weight": 70.0,
    "Duration": 20,
    "Heart_Rate": 100,
    "Body_Temp": 40.0,
}

DEFAULT_TZ = "Asia/Kolkata"  # used only if the browser timezone can't be detected

ACCENT = "#00F5A0"
ACCENT_2 = "#00D9F5"
BG = "#070B14"

# ──────────────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(0,245,160,0.10), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(0,217,245,0.10), transparent 55%),
        {BG};
    color: #E6EDF7;
}}
/* Hide Streamlit chrome, but keep the header itself so the sidebar open/close arrow stays reachable */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {{ visibility: visible !important; color: {ACCENT}; }}
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapseButton"] svg {{ color: {ACCENT}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: rgba(10,16,30,0.92);
    border-right: 1px solid rgba(255,255,255,0.06);
}}
.brand {{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.55rem; letter-spacing:.14em;
         background:linear-gradient(90deg,{ACCENT},{ACCENT_2}); -webkit-background-clip:text;
         -webkit-text-fill-color:transparent; margin-bottom:0; }}
.brand-sub {{ color:#9AA8C2; font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:1.2rem; }}

/* Glass cards */
.glass {{
    background: linear-gradient(145deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    margin-bottom: 1rem;
}}
.hero {{ padding: 2.2rem 2rem; position: relative; overflow: hidden; }}
.hero::after {{ content:""; position:absolute; right:-80px; top:-80px; width:260px; height:260px; border-radius:50%;
               background: radial-gradient(circle, rgba(0,245,160,0.25), transparent 65%); }}
.hero h1 {{ font-family:'Space Grotesk',sans-serif; font-size:2.5rem; font-weight:700; margin:0 0 .2rem 0; line-height:1.1; }}
.hero h1 span {{ background:linear-gradient(90deg,{ACCENT},{ACCENT_2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.hero .tag {{ color:{ACCENT}; font-weight:600; letter-spacing:.18em; text-transform:uppercase; font-size:.85rem; margin-bottom:.9rem; }}
.hero p {{ color:#C3CEE0; max-width:760px; margin:0; line-height:1.65; }}

.section-title {{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; margin:1.4rem 0 .6rem 0; }}
.section-title::before {{ content:""; display:inline-block; width:4px; height:1.05rem; margin-right:.6rem; vertical-align:-2px;
                         border-radius:3px; background:linear-gradient(180deg,{ACCENT},{ACCENT_2}); }}

/* Status pill */
.pill {{ display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .85rem; border-radius:999px; font-size:.8rem; font-weight:600; }}
.pill.ok  {{ background:rgba(0,245,160,.12); color:{ACCENT}; border:1px solid rgba(0,245,160,.35); }}
.pill.bad {{ background:rgba(255,92,92,.12); color:#FF7A7A; border:1px solid rgba(255,92,92,.35); }}
.pill.warn{{ background:rgba(255,196,0,.12); color:#FFC400; border:1px solid rgba(255,196,0,.35); }}
.dot {{ width:8px; height:8px; border-radius:50%; background:currentColor; box-shadow:0 0 10px currentColor; }}

/* Metric cards */
.metric {{ text-align:left; }}
.metric .k {{ color:#A3B1C9; font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; font-weight:600; }}
.metric .v {{ font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700; margin-top:.2rem; }}
.metric .s {{ color:#93A1BA; font-size:.78rem; margin-top:.1rem; }}

/* Result card */
.result {{ text-align:center; padding:2rem 1.5rem; border:1px solid rgba(0,245,160,.35);
          background: linear-gradient(145deg, rgba(0,245,160,0.12), rgba(0,217,245,0.05));
          box-shadow: 0 0 60px rgba(0,245,160,.12), 0 10px 40px rgba(0,0,0,.4); }}
.result .k {{ color:#B8C8DB; letter-spacing:.16em; text-transform:uppercase; font-size:.8rem; font-weight:600; }}
.result .big {{ font-family:'Space Grotesk',sans-serif; font-size:4.2rem; font-weight:700; line-height:1.05; margin:.4rem 0;
               background:linear-gradient(90deg,{ACCENT},{ACCENT_2}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.result .unit {{ font-size:1.4rem; color:#CFE; font-weight:500; -webkit-text-fill-color:#CFE; }}
.result .note {{ color:#B4C2D8; font-size:.9rem; max-width:560px; margin:.6rem auto 0 auto; line-height:1.55; }}

.disclaimer {{ border-left:3px solid {ACCENT_2}; padding:.7rem 1rem; background:rgba(0,217,245,.06);
              border-radius:0 12px 12px 0; color:#C3CEE0; font-size:.85rem; }}

/* Inputs */
div[data-testid="stForm"] {{ background:linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
    border:1px solid rgba(255,255,255,0.09); border-radius:20px; padding:1.4rem; }}
.stButton > button, .stFormSubmitButton > button {{
    border-radius:12px; font-weight:700; border:1px solid rgba(255,255,255,.15);
    background: rgba(255,255,255,.04); color:#E6EDF7; transition: all .2s ease; }}
.stFormSubmitButton > button[kind="primary"], .stButton > button[kind="primary"] {{
    background: linear-gradient(90deg,{ACCENT},{ACCENT_2}); color:#04121A; border:none;
    box-shadow:0 8px 28px rgba(0,245,160,.28); padding:.7rem 1rem; font-size:1rem; }}
.stFormSubmitButton > button:hover, .stButton > button:hover {{ transform: translateY(-1px); border-color:{ACCENT}; }}

.footer {{ text-align:center; color:#8B9AB5; font-size:.82rem; margin-top:3rem; padding-top:1.2rem;
          border-top:1px solid rgba(255,255,255,.06); }}

/* Readability: keep widgets dark & high-contrast even if the phone is in light mode */
:root, .stApp {{ color-scheme: dark; }}
.stApp label, .stApp [data-testid="stWidgetLabel"] p {{ color:#E6EDF7 !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color:#B4C2D8 !important; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color:#B4C2D8 !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stThumbValue"] {{ color:{ACCENT} !important; font-weight:700; }}
div[data-baseweb="input"], div[data-baseweb="base-input"] {{
    background:rgba(255,255,255,.07) !important; border-color:rgba(255,255,255,.2) !important; }}
div[data-baseweb="input"] input, div[data-baseweb="base-input"] input {{
    color:#F2F6FC !important; -webkit-text-fill-color:#F2F6FC !important; }}
div[data-testid="stNumberInput"] button {{ background:rgba(255,255,255,.08) !important; color:#E6EDF7 !important; }}
.stDownloadButton > button {{ border-radius:12px; font-weight:700; border:1px solid rgba(255,255,255,.2);
    background:rgba(255,255,255,.06); color:#E6EDF7; }}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{ color:#E6EDF7 !important; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color:#B4C2D8 !important; }}

@media (max-width: 768px) {{
    .block-container {{ padding: 3.2rem 1rem 2rem 1rem; }}
    .glass {{ padding:1.1rem 1.1rem; border-radius:16px; backdrop-filter:none; -webkit-backdrop-filter:none; }}
    .hero {{ padding:1.4rem 1.2rem; }}
    .hero h1 {{ font-size:1.7rem; }}
    .hero::after {{ width:150px; height:150px; right:-60px; top:-60px; }}
    .hero p {{ font-size:.95rem; }}
    .section-title {{ font-size:1.05rem; }}
    .metric .v {{ font-size:1.5rem; }}
    .result {{ padding:1.5rem 1rem; }}
    .result .big {{ font-size:2.8rem; }}
    .result .unit {{ font-size:1.1rem; }}
    div[data-testid="stForm"] {{ padding:1rem; }}
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{ min-height:48px; width:100%; }}
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data / model loading
# ──────────────────────────────────────────────────────────────────────────────
def _first_existing(candidates: list[str]) -> Path | None:
    """Return the first candidate path that exists next to app.py."""
    for name in candidates:
        path = BASE_DIR / name
        if path.is_file():
            return path
    return None


def find_model_file() -> Path | None:
    """Find the saved model: known names first (case-insensitive), then any model-like file."""
    folders = [BASE_DIR / f for f in MODEL_FOLDERS if (BASE_DIR / f).is_dir()]
    wanted = [n.lower() for n in MODEL_CANDIDATES]
    for name in wanted:
        for folder in folders:
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.name.lower() == name:
                    return f
    for folder in folders:
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in MODEL_EXTENSIONS:
                return f
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the saved model. Returns (model, path, error_message)."""
    path = find_model_file()
    if path is None:
        return None, None, (
            "No model file found. Put your saved model (e.g. `Calories.pki`) in the same "
            "folder as app.py. Files ending in .pki, .pkl or .joblib are detected automatically."
        )
    try:
        model = joblib.load(path)
    except ModuleNotFoundError as exc:
        return None, path, f"A package needed to open the model is missing: {exc}. Run the pip install command again."
    except Exception as exc:  # corrupt file, sklearn version mismatch, ...
        return None, path, (
            f"Could not open `{path.name}`: {exc}. If this is a scikit-learn version mismatch, "
            "install the same version that was used in Colab."
        )

    if not hasattr(model, "predict"):
        return None, path, f"`{path.name}` loaded, but it is not a trained model (no predict method)."
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != len(FEATURES):
        return None, path, (
            f"The model expects {n_in} features but this app is built for {len(FEATURES)} "
            f"({', '.join(FEATURES)}). Make sure this is the calories model."
        )
    return model, path, None


def model_feature_order(model) -> list[str]:
    """Use the model's own feature names (if stored) so column order always matches training."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None and set(names) == set(FEATURES):
        return list(names)
    return FEATURES


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the optional training dataset. Returns (df, path, error_message)."""
    path = _first_existing(DATA_CANDIDATES)
    if path is None:
        return None, None, "Dataset not found. Add `calories.csv` next to app.py to unlock data-driven charts."
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, path, f"Could not read `{path.name}`: {exc}"
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        return None, path, f"`{path.name}` is missing required columns: {', '.join(missing)}."
    df = df.drop_duplicates()  # same cleaning step as the notebook
    return df, path, None


@st.cache_data(show_spinner=False)
def compute_evaluation(_model, df: pd.DataFrame, feature_order: tuple[str, ...]):
    """
    Re-create the notebook's split (test_size=0.2, random_state=42) and score the
    SAVED model on it. Nothing is retrained.
    """
    X = df[list(feature_order)]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    out = {}
    for name, (Xs, ys) in {"Train": (X_train, y_train), "Test": (X_test, y_test)}.items():
        pred = _model.predict(Xs)
        out[name] = {
            "R²": r2_score(ys, pred),
            "MAE": mean_absolute_error(ys, pred),
            "RMSE": float(np.sqrt(mean_squared_error(ys, pred))),
            "n": len(ys),
        }
    out["test_actual"] = y_test.to_numpy()
    out["test_pred"] = _model.predict(X_test)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Validation & prediction
# ──────────────────────────────────────────────────────────────────────────────
def validate_inputs(values: dict, df: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block prediction; warnings do not."""
    errors, warnings = [], []
    for feat, (lo, hi) in HARD_LIMITS.items():
        v = values[feat]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{LABELS[feat]} is required.")
        elif not (lo <= v <= hi):
            hint = ""
            if feat == "Height" and v < 100:
                hint = " Height must be in centimetres (e.g. 170), not feet or metres."
            errors.append(f"{LABELS[feat]} must be between {lo} and {hi} {UNITS[feat]}.{hint}")

    # Linear models extrapolate poorly: warn when outside what the model was trained on.
    if not errors and df is not None:
        for feat in FEATURES:
            lo, hi = float(df[feat].min()), float(df[feat].max())
            if not (lo <= values[feat] <= hi):
                warnings.append(
                    f"{LABELS[feat]} ({values[feat]} {UNITS[feat]}) is outside the training data range "
                    f"({lo:g}–{hi:g}). The estimate may be less reliable."
                )
    return errors, warnings


def predict_calories(model, values: dict) -> float:
    """Build a one-row DataFrame in training column order and predict. No refitting of anything."""
    order = model_feature_order(model)
    row = pd.DataFrame([[values[f] for f in order]], columns=order)
    return float(model.predict(row)[0])


def contribution_table(model, values: dict) -> pd.DataFrame:
    """Per-feature contribution (coefficient × input) from the fitted linear model."""
    order = model_feature_order(model)
    coefs = np.ravel(model.coef_)
    rows = [{"Feature": LABELS[f], "Contribution": float(c * values[f])} for f, c in zip(order, coefs)]
    rows.insert(0, {"Feature": "Intercept", "Contribution": float(np.ravel(model.intercept_)[0])})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers (Plotly, shared dark style)
# ──────────────────────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E1E8F4", size=13),
        dragmode=False,  # lets the page scroll when a finger is on the chart
        margin=dict(l=20, r=20, t=55, b=30),
        height=height,
        title_font=dict(family="Space Grotesk, sans-serif", size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.14)", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.14)", zeroline=False, automargin=True)
    return fig


def fig_gauge(pred: float, df: pd.DataFrame | None) -> go.Figure:
    axis_max = float(df[TARGET].max()) if df is not None else max(500.0, pred * 1.4)
    axis_max = max(axis_max, pred * 1.1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number=dict(suffix=" kcal", font=dict(size=34, color="#E6EDF7")),
        gauge=dict(
            axis=dict(range=[0, axis_max], tickcolor="#9FB0C8", tickfont=dict(color="#CBD5E6")),
            bar=dict(color=ACCENT),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
        ),
    ))
    fig.update_layout(title="Calories burned — prediction summary")
    return style_fig(fig, 300)


def fig_contributions(table: pd.DataFrame, pred: float) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(table) + ["total"],
        x=list(table["Feature"]) + ["Prediction"],
        y=list(table["Contribution"]) + [0],
        increasing=dict(marker=dict(color=ACCENT)),
        decreasing=dict(marker=dict(color="#FF6B8B")),
        totals=dict(marker=dict(color=ACCENT_2)),
        connector=dict(line=dict(color="rgba(255,255,255,0.2)")),
        hovertemplate="%{x}: %{y:.2f} kcal<extra></extra>",
    ))
    fig.update_layout(title="How the model built this estimate (coefficient × input)",
                      yaxis_title="kcal", showlegend=False)
    fig.update_xaxes(tickangle=-30)
    return style_fig(fig, 380)


def fig_actual_vs_pred(actual: np.ndarray, pred: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers", name="Test samples",
                             marker=dict(color=ACCENT, size=5, opacity=0.55),
                             hovertemplate="Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>"))
    lo, hi = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line=dict(color=ACCENT_2, dash="dash")))
    fig.update_layout(title="Actual vs predicted calories (test set)",
                      xaxis_title="Actual calories (kcal)", yaxis_title="Predicted calories (kcal)")
    return style_fig(fig, 420)


def fig_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale=[[0, "#FF6B8B"], [0.5, "#0B1220"], [1, ACCENT]],
                    zmin=-1, zmax=1)
    fig.update_layout(title="Feature correlation heatmap (training dataset)")
    return style_fig(fig, 460)


def fig_coefficients(model, df: pd.DataFrame | None) -> go.Figure:
    order = model_feature_order(model)
    coefs = np.ravel(model.coef_)
    if df is not None:
        # Standardised effect = coefficient × feature std: comparable across different units.
        vals = [c * float(df[f].std()) for f, c in zip(order, coefs)]
        title = "Feature influence (coefficient × feature std. dev.)"
        xlab = "kcal change per 1 std. dev. increase"
    else:
        vals = list(coefs)
        title = "Raw regression coefficients (units differ per feature)"
        xlab = "kcal per unit of feature"
    plot_df = pd.DataFrame({"Feature": [LABELS[f] for f in order], "Value": vals}).sort_values("Value")
    fig = px.bar(plot_df, x="Value", y="Feature", orientation="h",
                 color="Value", color_continuous_scale=[[0, "#FF6B8B"], [0.5, ACCENT_2], [1, ACCENT]])
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title="", coloraxis_showscale=False)
    return style_fig(fig, 380)


def fig_history(hist: pd.DataFrame) -> go.Figure:
    fig = px.line(hist, x="Run", y="Predicted kcal", markers=True,
                  hover_data=["Time", "Age", "Weight", "Duration", "Heart_Rate"])
    fig.update_traces(line=dict(color=ACCENT, width=3), marker=dict(size=9, color=ACCENT_2))
    fig.update_layout(title="Prediction history (this session)", xaxis_title="Prediction #",
                      yaxis_title="Calories (kcal)")
    fig.update_xaxes(dtick=1)
    return style_fig(fig, 340)


# ──────────────────────────────────────────────────────────────────────────────
# Reusable UI blocks
# ──────────────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="glass metric"><div class="k">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def status_pill(model, err) -> None:
    if model is not None:
        st.markdown('<span class="pill ok"><span class="dot"></span>Model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill bad"><span class="dot"></span>Model not loaded</span>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer"><b>Note:</b> This is a statistical estimate from a Multiple Linear Regression model, '
        "not a medical measurement. Actual calories burned vary from person to person.</div>",
        unsafe_allow_html=True,
    )


def init_state() -> None:
    for feat, default in DEFAULTS.items():
        st.session_state.setdefault(f"in_{feat}", default)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last", None)


def reset_inputs() -> None:
    """Callback for the reset button."""
    for feat, default in DEFAULTS.items():
        st.session_state[f"in_{feat}"] = default
    st.session_state["last"] = None


def now_local() -> datetime:
    """
    Real current time in the VIEWER's timezone.
    The server (e.g. Streamlit Cloud) usually runs in UTC, so a plain datetime.now() shows the wrong time.
    st.context.timezone gives the browser's timezone; if unavailable we fall back to India time (IST).
    """
    tz_name = None
    try:
        tz_name = st.context.timezone
    except Exception:
        pass
    for name in (tz_name, DEFAULT_TZ):
        if not name:
            continue
        try:
            return datetime.now(ZoneInfo(name))
        except Exception:
            continue
    return datetime.now(timezone(timedelta(hours=5, minutes=30), "IST"))


def show_chart(fig: go.Figure) -> None:
    """Plotly chart that behaves on phones: full width, no toolbar, no scroll-trapping."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": False})


def history_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state["history"])


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
def render_hero(model, err) -> None:
    st.markdown(
        """
        <div class="glass hero">
            <div class="tag">Predict. Analyze. Optimize.</div>
            <h1><span>CALORIQ</span> — AI Calories Predictor</h1>
            <p>CALORIQ uses a Multiple Linear Regression model trained on workout data to estimate how many
            calories you burn from your age, height, weight, exercise duration, heart rate and body temperature.
            Enter your numbers, get an instant estimate, and explore how each factor drives the result.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_pill(model, err)
    if err:
        st.error(err)


def page_dashboard(model, model_err, df, df_err) -> None:
    render_hero(model, model_err)
    hist = history_df()

    section("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Model", "Linear Regression", "Multiple features")
    with c2:
        metric_card("Input features", str(len(FEATURES)), ", ".join(LABELS[f] for f in FEATURES[:3]) + "…")
    with c3:
        metric_card("Predictions this session", str(len(hist)))
    with c4:
        latest = f"{hist['Predicted kcal'].iloc[-1]:.1f} kcal" if len(hist) else "—"
        metric_card("Latest estimate", latest, "Run one on the Predict page" if not len(hist) else "")

    if len(hist) >= 1:
        section("Your prediction history")
        show_chart(fig_history(hist))
    else:
        st.info("No predictions yet. Open **Predict** in the sidebar to generate your first estimate.")

    if df is not None:
        section("Dataset snapshot")
        d1, d2, d3 = st.columns(3)
        with d1:
            metric_card("Training rows", f"{len(df):,}", "after removing duplicates")
        with d2:
            metric_card("Avg calories", f"{df[TARGET].mean():.1f} kcal", "dataset mean")
        with d3:
            metric_card("Avg duration", f"{df['Duration'].mean():.1f} min", "dataset mean")
    elif df_err:
        st.caption(df_err)


def page_predict(model, model_err, df, df_err) -> None:
    render_hero(model, model_err)
    section("Prediction input")

    with st.form("predict_form", border=False):
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age (years)", min_value=1, max_value=120, step=1, key="in_Age",
                                  help="Age in whole years.")
            height = st.number_input("Height (cm)", min_value=0.0, max_value=272.0, step=1.0, key="in_Height",
                                     help="Height in centimetres — the model was trained on cm, not feet.")
            weight = st.slider("Weight (kg)", 20.0, 250.0, step=0.5, key="in_Weight",
                               help="Body weight in kilograms.")
        with b:
            duration = st.slider("Exercise duration (minutes)", 1, 300, key="in_Duration",
                                 help="Length of the workout session in minutes.")
            heart_rate = st.slider("Heart rate (bpm)", 30, 230, key="in_Heart_Rate",
                                   help="Average heart rate during exercise.")
            body_temp = st.slider("Body temperature (°C)", 34.0, 43.0, step=0.1, key="in_Body_Temp",
                                  help="Body temperature at the end of the workout.")
        submitted = st.form_submit_button("Predict Calories Burned", type="primary", width="stretch")

    st.button("Reset inputs", on_click=reset_inputs, help="Restore the default values.")

    values = {
        "Age": age, "Height": height, "Weight": weight,
        "Duration": duration, "Heart_Rate": heart_rate, "Body_Temp": body_temp,
    }

    if submitted:
        if model is None:
            st.error("The model is not loaded, so a prediction can't be made. See the message above.")
        else:
            errors, warns = validate_inputs(values, df)
            if errors:
                for msg in errors:
                    st.error(msg)
            else:
                try:
                    pred = predict_calories(model, values)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                else:
                    if not np.isfinite(pred) or pred < 0:
                        st.error(
                            f"The model returned {pred:.1f} kcal, which isn't physically meaningful. "
                            "These inputs are probably too far from the data the model was trained on."
                        )
                    else:
                        prev = st.session_state["history"][-1]["Predicted kcal"] if st.session_state["history"] else None
                        stamp = now_local()
                        st.session_state["history"].append({
                            "Run": len(st.session_state["history"]) + 1,
                            "Date": stamp.strftime("%d %b %Y"),
                            "Time": stamp.strftime("%I:%M:%S %p"),
                            **values,
                            "Predicted kcal": round(pred, 2),
                        })
                        st.session_state["last"] = {"values": values, "pred": pred, "warns": warns, "prev": prev,
                                                    "at": stamp.strftime("%d %b %Y, %I:%M:%S %p %Z")}

    last = st.session_state.get("last")
    if last:
        pred = last["pred"]
        section("Result")
        st.markdown(
            f"""
            <div class="glass result">
                <div class="k">Estimated calories burned</div>
                <div class="big">{pred:,.1f} <span class="unit">kcal</span></div>
                <div class="note">Based on {last['values']['Duration']} min of exercise at an average
                heart rate of {last['values']['Heart_Rate']} bpm for a {last['values']['Age']}-year-old weighing
                {last['values']['Weight']:g} kg.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for w in last["warns"]:
            st.warning(w)
        if last["prev"] is not None:
            delta = pred - last["prev"]
            st.caption(f"Compared with your previous prediction: {delta:+.1f} kcal.")
        if last.get("at"):
            st.caption(f"Predicted at {last['at']}")
        disclaimer()

        left, right = st.columns([1, 1.4])
        with left:
            show_chart(fig_gauge(pred, df))
        with right:
            show_chart(fig_contributions(contribution_table(model, last["values"]), pred))

        hist = history_df()
        if len(hist) > 1:
            show_chart(fig_history(hist))

    hist = history_df()
    if len(hist):
        section("History")
        st.dataframe(hist, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                               file_name="caloriq_prediction_history.csv", mime="text/csv", width="stretch")
        with d2:
            if st.button("Clear history", width="stretch"):
                st.session_state["history"] = []
                st.session_state["last"] = None
                st.rerun()


def page_insights(model, model_err, df, df_err) -> None:
    section("Model insights")
    status_pill(model, model_err)
    if model is None:
        st.error(model_err)
        return

    section("Model performance")
    if df is None:
        st.info(
            "Performance metrics need the original dataset so the saved model can be scored on its held-out test "
            f"split. {df_err} No numbers are shown until then."
        )
    else:
        try:
            ev = compute_evaluation(model, df, tuple(model_feature_order(model)))
        except Exception as exc:
            st.error(f"Could not evaluate the model on the dataset: {exc}")
            ev = None
        if ev:
            st.caption(
                "Scores are computed by applying your saved model to the same 80/20 split used in the notebook "
                "(random_state=42). Test-set scores measure performance on unseen data; training scores are "
                "shown only for comparison."
            )
            t = ev["Test"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("R² (test)", f"{t['R²']:.4f}", f"train: {ev['Train']['R²']:.4f}")
            with c2:
                metric_card("MAE (test)", f"{t['MAE']:.2f} kcal", f"train: {ev['Train']['MAE']:.2f}")
            with c3:
                metric_card("RMSE (test)", f"{t['RMSE']:.2f} kcal", f"train: {ev['Train']['RMSE']:.2f}")
            with c4:
                metric_card("Test samples", f"{t['n']:,}", f"train: {ev['Train']['n']:,}")
            show_chart(fig_actual_vs_pred(ev["test_actual"], ev["test_pred"]))

    section("What drives the prediction")
    try:
        show_chart(fig_coefficients(model, df))
        eq = pd.DataFrame({
            "Feature": [LABELS[f] for f in model_feature_order(model)],
            "Coefficient": np.ravel(model.coef_),
            "Unit": [UNITS[f] for f in model_feature_order(model)],
        })
        st.dataframe(eq, hide_index=True, width="stretch")
        st.caption(f"Intercept: {float(np.ravel(model.intercept_)[0]):.4f}")
    except Exception as exc:
        st.error(f"Could not read the model coefficients: {exc}")

    if df is not None:
        section("Feature correlations")
        show_chart(fig_heatmap(df))

        section("Dataset overview")
        st.write(f"**{len(df):,} rows** × **{df.shape[1]} columns** (duplicates removed, as in the notebook).")
        st.dataframe(df.head(10), hide_index=True, width="stretch")
        st.dataframe(df[FEATURES + [TARGET]].describe().T.round(2), width="stretch")
    elif df_err:
        st.caption(df_err)

    hist = history_df()
    if len(hist) > 1:
        section("Prediction history")
        show_chart(fig_history(hist))


def page_about() -> None:
    section("About CALORIQ")
    st.markdown(
        """
        <div class="glass">
        <p><b>CALORIQ</b> is a Machine Learning project built for a BCA course. It predicts calories burned during
        exercise with a <b>Multiple Linear Regression</b> model trained in scikit-learn.</p>
        <p><b>Features used:</b> Age, Height (cm), Weight (kg), Duration (min), Heart rate (bpm), Body temperature (°C).</p>
        <p><b>How it works:</b> your inputs are arranged into one row in the same column order as training and passed to
        the saved model. No scaling or encoding was used in training, so none is applied here, and the model is never
        retrained inside the app.</p>
        <p><b>Limitations:</b> a linear model works best inside the range of its training data. Estimates for extreme
        values are less reliable, and results are never a medical measurement.</p>
        <p><b>Stack:</b> Python · Streamlit · Pandas · NumPy · scikit-learn · Plotly · Joblib</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    model, model_path, model_err = load_model()
    df, data_path, df_err = load_dataset()

    with st.sidebar:
        st.markdown('<div class="brand">CALORIQ</div><div class="brand-sub">AI Calories Predictor</div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigation", ["Dashboard", "Predict", "Model Insights", "About"],
                        label_visibility="collapsed", key="page")
        st.divider()
        status_pill(model, model_err)
        if model_path:
            st.caption(f"Model file: `{model_path.name}`")
        if data_path and df is not None:
            st.caption(f"Dataset: `{data_path.name}` ({len(df):,} rows)")
        else:
            st.caption("Dataset: not loaded (optional)")
        st.divider()
        st.caption("Estimates only — not medical advice.")

    if page == "Dashboard":
        page_dashboard(model, model_err, df, df_err)
    elif page == "Predict":
        page_predict(model, model_err, df, df_err)
    elif page == "Model Insights":
        page_insights(model, model_err, df, df_err)
    else:
        page_about()

    st.markdown(
        '<div class="footer">CALORIQ — AI Calories Predictor · Built with Python &amp; Machine Learning</div>',
        unsafe_allow_html=True,
    )


main()
