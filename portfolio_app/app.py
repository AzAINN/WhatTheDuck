"""
Portfolio Optimizer - Clean Dark Theme

Highlights:
  - Dark background with high contrast text
  - Clean, professional UI without emojis
  - Dual data modes: CSV upload or ticker fetch (yfinance)
  - Progressive form: core controls + collapsible advanced panel
  - Multi-stage progress indicator during optimization
  - Metrics grid (VaR, CVaR, Sharpe, Volatility)
  - Allocation chart, risk contributions, loss distribution, correlation heatmap
  - Scenario compare scaffold (keeps last two runs) and CSV export of weights

Run:
    PYTHONPATH=OpenPhiSolve:. streamlit run portfolio_app/app.py
"""

from __future__ import annotations

import io
import platform
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional market data source
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except Exception:
    _YF_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Portfolio QIHD imports (deferred to allow graceful failure)
# ---------------------------------------------------------------------------
_SOLVER_AVAILABLE = False
try:
    from portfolio_qihd import (
        PortfolioSpec,
        solve_portfolio,
        synthetic_factor_model,
        compute_mu_cov,
        caps_from_constant,
    )
    _SOLVER_AVAILABLE = True
except ImportError:
    pass


# ===========================================================================
# Design tokens - Dark theme with high contrast
# ===========================================================================
_BG_DARK    = "#0D1117"      # Deep dark background
_BG_CARD    = "#161B22"      # Card/surface background
_BG_HOVER   = "#21262D"      # Hover state
_BORDER     = "#30363D"      # Border color
_TEXT       = "#E6EDF3"      # Primary text (high contrast)
_TEXT_SEC   = "#8B949E"      # Secondary text
_TEXT_MUTE  = "#6E7681"      # Muted text
_ACCENT     = "#58A6FF"      # Primary accent (blue)
_ACCENT_LT  = "#79C0FF"      # Light accent
_SUCCESS    = "#3FB950"      # Success green
_SUCCESS_LT = "#56D364"      # Light success
_DANGER     = "#F85149"      # Danger red
_DANGER_LT  = "#FF7B72"      # Light danger
_WARNING    = "#D29922"      # Warning orange
_WARNING_LT = "#E3B341"      # Light warning

# Chart palette (vibrant on dark)
_CHART_PAL = [
    "#58A6FF",  # blue
    "#3FB950",  # green
    "#F85149",  # red
    "#A371F7",  # purple
    "#D29922",  # orange
    "#79C0FF",  # light blue
    "#56D364",  # light green
    "#FF7B72",  # light red
    "#BC8CFF",  # light purple
    "#E3B341",  # light orange
]


# ===========================================================================
# Custom CSS - Dark theme with clean aesthetics
# ===========================================================================
_CSS = f"""
<style>
/* --- Global --- */
.stApp {{
    background-color: {_BG_DARK};
    color: {_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

/* --- Header --- */
header[data-testid="stHeader"] {{
    background-color: {_BG_DARK};
    border-bottom: 1px solid {_BORDER};
}}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {{
    background-color: {_BG_CARD};
    border-right: 1px solid {_BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown p {{
    font-size: 0.88rem;
    color: {_TEXT_SEC};
    line-height: 1.5;
}}
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {_TEXT};
}}

/* --- Card containers (expanders) --- */
div[data-testid="stExpander"] {{
    border: 1px solid {_BORDER};
    border-radius: 8px;
    background-color: {_BG_CARD};
}}
div[data-testid="stExpander"]:hover {{
    border-color: {_TEXT_MUTE};
}}
div[data-testid="stExpander"] summary {{
    color: {_TEXT};
}}

/* --- Metric cards --- */
div[data-testid="stMetric"] {{
    background-color: {_BG_CARD};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 16px 20px;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {_TEXT_MUTE};
}}
div[data-testid="stMetric"] label {{
    color: {_TEXT_SEC};
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: {_TEXT};
    font-weight: 700;
    font-size: 1.6rem;
}}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
    color: {_TEXT_SEC};
}}

/* --- Buttons --- */
button[kind="primary"] {{
    background-color: {_ACCENT} !important;
    color: {_BG_DARK} !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
}}
button[kind="primary"]:hover {{
    background-color: {_ACCENT_LT} !important;
}}
button[kind="secondary"] {{
    background-color: {_BG_HOVER} !important;
    color: {_TEXT} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}

/* --- Inputs --- */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {{
    background-color: {_BG_HOVER} !important;
    color: {_TEXT} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}

/* --- Slider --- */
div[data-testid="stSlider"] > div > div {{
    background-color: {_BG_HOVER};
}}
div[data-testid="stSlider"] span {{
    color: {_TEXT_SEC};
}}

/* --- Dataframes --- */
div[data-testid="stDataFrame"] {{
    border: 1px solid {_BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
div[data-testid="stDataFrame"] th {{
    background-color: {_BG_HOVER} !important;
    color: {_TEXT} !important;
}}
div[data-testid="stDataFrame"] td {{
    background-color: {_BG_CARD} !important;
    color: {_TEXT_SEC} !important;
}}

/* --- Divider --- */
hr {{
    border: none;
    height: 1px;
    background-color: {_BORDER};
    margin: 1.5rem 0;
}}

/* --- Section headers --- */
.section-header {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {_TEXT};
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}}
.section-subtext {{
    font-size: 0.9rem;
    color: {_TEXT_SEC};
    margin-bottom: 1rem;
    line-height: 1.5;
}}

/* --- Status badges --- */
.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-ok {{
    background-color: {_SUCCESS};
    color: {_BG_DARK};
}}
.badge-warn {{
    background-color: {_WARNING};
    color: {_BG_DARK};
}}
.badge-err {{
    background-color: {_DANGER};
    color: {_BG_DARK};
}}

/* --- Alert boxes --- */
div[data-testid="stAlert"] {{
    background-color: {_BG_CARD};
    border: 1px solid {_BORDER};
    border-radius: 6px;
}}

/* --- Empty state --- */
.empty-state {{
    text-align: center;
    padding: 3rem 2rem;
    background-color: {_BG_CARD};
    border: 1px dashed {_BORDER};
    border-radius: 12px;
    margin: 2rem 0;
}}

/* --- Radio buttons --- */
div[data-testid="stRadio"] label {{
    color: {_TEXT_SEC} !important;
}}
div[data-testid="stRadio"] label[data-checked="true"] {{
    color: {_TEXT} !important;
}}

/* --- Download button --- */
button[data-testid="stDownloadButton"] {{
    background-color: {_BG_HOVER} !important;
    color: {_TEXT} !important;
    border: 1px solid {_BORDER} !important;
}}

/* --- File uploader --- */
div[data-testid="stFileUploader"] {{
    background-color: {_BG_HOVER};
    border: 1px dashed {_BORDER};
    border-radius: 8px;
}}
div[data-testid="stFileUploader"] label {{
    color: {_TEXT_SEC} !important;
}}

/* --- Tabs --- */
button[data-baseweb="tab"] {{
    color: {_TEXT_SEC} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {_TEXT} !important;
    border-bottom-color: {_ACCENT} !important;
}}
</style>
"""


# ===========================================================================
# Helpers
# ===========================================================================

def _detect_gpu() -> dict:
    """Detect GPU availability, WSL-aware."""
    info = {"jax_backend": None, "torch_cuda": False, "torch_device": None,
            "recommended": "cpu", "wsl": False}

    # WSL detection
    uname = platform.uname()
    if "microsoft" in uname.release.lower() or "wsl" in uname.release.lower():
        info["wsl"] = True

    # JAX
    try:
        import jax
        backend = jax.default_backend()
        info["jax_backend"] = backend
        if backend == "gpu":
            info["recommended"] = "gpu"
    except Exception:
        pass

    # PyTorch / CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda"] = True
            info["torch_device"] = torch.cuda.get_device_name(0)
            info["recommended"] = "gpu"
    except Exception:
        pass

    return info


def _load_returns(csv_buf: Optional[io.BytesIO], n_assets: int, seed: int) -> pd.DataFrame:
    """Load returns from CSV buffer or generate synthetic data."""
    if csv_buf is not None:
        csv_buf.seek(0)
        df = pd.read_csv(csv_buf)

        # Ensure numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
            numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            raise ValueError("CSV has no numeric columns. Expected returns data (rows=time, cols=assets).")

        df = df[numeric_cols]
        n_assets = min(n_assets, df.shape[1])
        return df.iloc[:, :n_assets]

    if not _SOLVER_AVAILABLE:
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal((2048, n_assets)) * 0.02
        cols = [f"A{i}" for i in range(n_assets)]
        return pd.DataFrame(arr, columns=cols)

    arr = synthetic_factor_model(n_samples=2048, n_assets=n_assets, n_factors=5, seed=seed)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(arr, columns=cols)


def _fetch_ticker_returns(tickers: str, start: datetime, end: datetime) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch adjusted close prices via yfinance and return daily pct-change matrix."""
    if not _YF_AVAILABLE:
        return pd.DataFrame(), "yfinance not installed. Run: pip install yfinance"

    symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not symbols:
        return pd.DataFrame(), "Please enter at least one ticker symbol."

    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame(), "No data returned for given tickers/date range."
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            prices = data["Adj Close"]
        elif isinstance(data.columns, pd.MultiIndex) and "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"]
        else:
            prices = data
        returns = prices.pct_change().dropna(how="all")
        returns = returns.dropna(axis=1, how="all")
        if returns.empty:
            return pd.DataFrame(), "Not enough price history to compute returns."
        return returns, None
    except Exception as exc:
        return pd.DataFrame(), f"Data fetch failed: {exc}"


def _historical_var(pnl: np.ndarray, alpha: float) -> float:
    """Historical VaR as quantile of losses."""
    return float(np.quantile(-pnl, 1 - alpha))


def _historical_cvar(pnl: np.ndarray, alpha: float) -> float:
    """Historical CVaR (Expected Shortfall)."""
    losses = -pnl
    var_threshold = np.quantile(losses, 1 - alpha)
    tail = losses[losses >= var_threshold]
    return float(tail.mean()) if len(tail) > 0 else float(var_threshold)


def _mc_var(returns: np.ndarray, weights: np.ndarray, alpha: float,
            n_scenarios: int = 50_000, seed: int = 42):
    """Monte Carlo VaR/CVaR using resampled returns."""
    rng = np.random.default_rng(seed)
    n_obs = returns.shape[0]
    idx = rng.integers(0, n_obs, size=n_scenarios)
    sim_returns = returns[idx]
    pnl = sim_returns @ weights
    losses = -pnl
    var_val = float(np.quantile(losses, 1 - alpha))
    tail = losses[losses >= var_val]
    cvar_val = float(tail.mean()) if len(tail) > 0 else var_val
    return var_val, cvar_val


def _correlation_heatmap(df: pd.DataFrame):
    """Return a matplotlib figure with correlation heatmap (dark theme)."""
    if df.shape[1] < 2:
        return None
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.6), facecolor=_BG_CARD)
    ax.set_facecolor(_BG_CARD)
    cax = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_SEC)
    cbar.outline.set_edgecolor(_BORDER)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_SEC)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8.5, color=_TEXT_SEC)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8.5, color=_TEXT_SEC)
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="700", color=_TEXT, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def _progress_runner(stages):
    """Iterate through stage tuples (label, percent_increment) yielding cumulative percent."""
    progress = st.progress(0, text="Initializing...")
    current = 0
    for label, inc in stages:
        current = min(100, current + inc)
        progress.progress(current, text=label)
        time.sleep(0.1)
    return progress


def _export_weights_csv(asset_names, weights):
    buf = io.StringIO()
    pd.DataFrame({"Asset": asset_names, "Weight": weights}).to_csv(buf, index=False)
    return buf.getvalue()


# ===========================================================================
# Chart helpers (dark theme)
# ===========================================================================

def _style_fig(fig, ax):
    """Apply consistent dark chart styling."""
    fig.patch.set_facecolor(_BG_CARD)
    ax.set_facecolor(_BG_DARK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_BORDER)
    ax.spines["bottom"].set_color(_BORDER)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=_TEXT_SEC, labelsize=9, width=1)
    ax.xaxis.label.set_color(_TEXT_SEC)
    ax.yaxis.label.set_color(_TEXT_SEC)
    ax.title.set_color(_TEXT)
    ax.grid(True, linestyle='--', alpha=0.2, color=_BORDER, linewidth=0.6)


def _plot_weights(asset_names, weights, max_show=20):
    """Horizontal bar chart of portfolio weights (dark theme)."""
    order = np.argsort(weights)[::-1]
    n = min(max_show, len(order))
    idx = order[:n]
    names = [asset_names[i] for i in idx]
    vals = weights[idx]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, n * 0.35)), facecolor=_BG_CARD)
    _style_fig(fig, ax)

    y_pos = np.arange(n)
    for i, (y, val) in enumerate(zip(y_pos, vals)):
        color = _CHART_PAL[i % len(_CHART_PAL)]
        ax.barh(y, val, height=0.65, color=color, edgecolor=_BG_CARD, linewidth=1, alpha=0.9)
        if val > 0.005:
            ax.text(val + 0.003, y, f"{val:.1%}", va="center", fontsize=8.5,
                    color=_TEXT_SEC, fontweight="500")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="500", color=_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Weight", fontsize=11, fontweight="600", color=_TEXT_SEC)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_title("Portfolio Allocation", fontsize=14, fontweight="700", pad=14, color=_TEXT)

    fig.tight_layout()
    return fig


def _plot_loss_distribution(returns, weights, alpha, n_scenarios=50_000, seed=42):
    """Histogram of simulated portfolio losses with VaR/CVaR lines (dark theme)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, returns.shape[0], size=n_scenarios)
    pnl = returns[idx] @ weights
    losses = -pnl

    var_val = np.quantile(losses, 1 - alpha)
    tail = losses[losses >= var_val]
    cvar_val = tail.mean() if len(tail) > 0 else var_val

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=_BG_CARD)
    _style_fig(fig, ax)

    n_hist, bins_hist, patches = ax.hist(losses, bins=120, density=True, alpha=0.8,
                                          edgecolor=_BG_CARD, linewidth=0.3, color=_ACCENT)

    ax.axvline(var_val, color=_DANGER, linestyle="--", linewidth=2.5, alpha=0.95,
               label=f"VaR ({1-alpha:.0%}) = {var_val:.4f}", zorder=10)
    ax.axvline(cvar_val, color=_WARNING, linestyle="-.", linewidth=2.5, alpha=0.95,
               label=f"CVaR = {cvar_val:.4f}", zorder=10)

    ax.set_xlabel("Portfolio Loss", fontsize=11, fontweight="600", color=_TEXT_SEC)
    ax.set_ylabel("Density", fontsize=11, fontweight="600", color=_TEXT_SEC)
    ax.set_title("Loss Distribution (Monte Carlo)", fontsize=14, fontweight="700",
                 pad=14, color=_TEXT)
    leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor=_BORDER,
                    fontsize=9.5, loc="upper right", facecolor=_BG_CARD, labelcolor=_TEXT)

    fig.tight_layout()
    return fig


def _plot_risk_contribution(asset_names, weights, cov):
    """Bar chart of marginal risk contributions (dark theme)."""
    sel = weights > 1e-6
    if sel.sum() == 0:
        return None
    w_sel = weights[sel]
    cov_sel = cov[np.ix_(sel, sel)]
    names_sel = [asset_names[i] for i in np.where(sel)[0]]

    sigma_w = cov_sel @ w_sel
    port_vol = np.sqrt(max(w_sel @ sigma_w, 1e-12))
    rc = w_sel * sigma_w / port_vol
    rc_pct = rc / rc.sum() * 100 if rc.sum() > 0 else rc * 0

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, len(names_sel) * 0.38)),
                           facecolor=_BG_CARD)
    _style_fig(fig, ax)

    y_pos = np.arange(len(names_sel))
    colors = [_CHART_PAL[i % len(_CHART_PAL)] for i in range(len(names_sel))]

    for i, (y, pct, color) in enumerate(zip(y_pos, rc_pct, colors)):
        ax.barh(y, pct, height=0.65, color=color, edgecolor=_BG_CARD, linewidth=1, alpha=0.9)
        if pct > 2:
            ax.text(pct + 0.5, y, f"{pct:.1f}%", va="center", fontsize=8.5,
                    color=_TEXT_SEC, fontweight="500")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sel, fontsize=9.5, fontweight="500", color=_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Risk Contribution (%)", fontsize=11, fontweight="600", color=_TEXT_SEC)
    ax.set_title("Marginal Risk Contributions", fontsize=14, fontweight="700",
                 pad=14, color=_TEXT)
    fig.tight_layout()
    return fig


def _store_scenario(state: dict, label: str, asset_names, weights, metrics: dict):
    """Keep last two scenarios in session state for comparison."""
    entry = {
        "label": label,
        "asset_names": asset_names,
        "weights": weights.tolist(),
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    history = state.setdefault("scenarios", [])
    history.append(entry)
    state["scenarios"] = history[-2:]


def _render_comparison(state: dict):
    scenarios = state.get("scenarios", [])
    if len(scenarios) < 2:
        return
    st.subheader("Strategy Comparison")
    a, b = scenarios[-2], scenarios[-1]
    cols = st.columns(2)
    for col, scenario in zip(cols, [a, b]):
        col.markdown(f"**{scenario['label']}** | {scenario['timestamp']}")
        m = scenario["metrics"]
        col.metric("VaR 95%", f"{m['var']:.4f}")
        col.metric("CVaR 95%", f"{m['cvar']:.4f}")
        col.metric("Sharpe", f"{m['sharpe']:.3f}")
        col.metric("Volatility", f"{m['vol']:.2%}")
    c1, c2 = st.columns(2)
    for col, scenario in zip((c1, c2), (a, b)):
        fig = _plot_weights(scenario["asset_names"], np.array(scenario["weights"]), max_show=15)
        col.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ===========================================================================
# Page config and CSS injection
# ===========================================================================
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_CSS, unsafe_allow_html=True)

st.session_state.setdefault("scenarios", [])

# ===========================================================================
# Sidebar - Input Parameters
# ===========================================================================
gpu_info = _detect_gpu()
today = datetime.today()
default_start = today - timedelta(days=750)
default_end = today

with st.sidebar:
    st.markdown("### Portfolio Optimizer")
    st.markdown(
        f'<p class="section-subtext">'
        f'Optimize allocations and measure VaR / CVaR. Fetch market data or upload CSV.</p>',
        unsafe_allow_html=True,
    )

    # --- System status ---
    st.markdown("**System Status**")
    cols_status = st.columns(2)
    with cols_status[0]:
        if _SOLVER_AVAILABLE:
            st.markdown('<span class="badge badge-ok">QIHD Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-err">QIHD Missing</span>', unsafe_allow_html=True)
    with cols_status[1]:
        if gpu_info["recommended"] == "gpu":
            label = (gpu_info.get("torch_device") or gpu_info.get("jax_backend", "GPU")).replace("NVIDIA ", "").replace("GeForce ", "")[:12]
            st.markdown(f'<span class="badge badge-ok">{label}</span>', unsafe_allow_html=True)
        else:
            wsl_note = " WSL" if gpu_info["wsl"] else ""
            st.markdown(f'<span class="badge badge-warn">CPU{wsl_note}</span>', unsafe_allow_html=True)

    st.divider()

    # --- Data selection ---
    data_mode = st.radio("Data source", ["Upload CSV", "Fetch tickers", "Synthetic"], index=1)
    asset_cap = st.slider("Max assets to use", 5, 150, 50, step=5)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    csv_file = None
    tickers = ""
    date_range = (default_start, default_end)
    if data_mode == "Upload CSV":
        csv_file = st.file_uploader("Upload returns CSV", type=["csv"], help="Rows=time, cols=assets. Numeric only.")
    elif data_mode == "Fetch tickers":
        tickers = st.text_input("Tickers", "AAPL, MSFT, GOOGL, AMZN, NVDA")
        date_range = st.date_input("Date range", value=(default_start, default_end))
        if not _YF_AVAILABLE:
            st.warning("yfinance not installed. Ticker fetch unavailable.")
    else:
        n_assets = st.slider("Synthetic assets", 5, 300, 40, step=5)

    st.divider()

    # --- Portfolio ---
    st.markdown("**Portfolio Constraints**")
    max_k = st.slider("Max holdings (K)", 1, 60, 15, help="Cardinality constraint: hold at most K assets.")
    cap = st.slider("Per-asset cap", 0.02, 0.50, 0.10, step=0.01, format="%.2f", help="Upper bound per asset weight.")
    lambda_risk = st.slider("Risk aversion", 0.1, 20.0, 5.0, step=0.1, help="Higher = more conservative.")

    st.divider()

    # --- Optimization and risk controls ---
    st.markdown("**Optimization**")
    mode = st.radio("Mode", ["Minimize Variance", "Minimize CVaR (display-only)"], index=0)
    device = st.selectbox("Device", ["gpu", "cpu"], index=0 if gpu_info["recommended"] == "gpu" else 1)
    dt = st.number_input("Step size (dt)", 0.01, 1.0, 0.2, step=0.01, format="%.2f")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_shots = st.number_input("Shots", 10, 5000, 500, step=50, help="Parallel trajectories.")
    with col_s2:
        n_steps = st.number_input("Steps", 200, 20000, 5000, step=200, help="Integration steps.")

    with st.expander("Advanced settings", expanded=False):
        lookback_days = st.selectbox("Historical lookback (days)", [63, 126, 252, 500, 750], index=2)
        solver_choice = st.selectbox("Solver backend", ["QIHD + Gurobi (default)", "CVXPY (experimental)"], index=0)
        risk_free = st.number_input("Risk-free rate (for Sharpe)", -0.02, 0.15, 0.02, step=0.005, format="%.3f")
        alpha = st.select_slider("VaR confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95, format_func=lambda x: f"{x:.1%}")
        mc_scenarios = st.number_input("MC scenarios", 10_000, 500_000, 50_000, step=10_000)

    st.divider()

    run_btn = st.button("Optimize Portfolio", type="primary", use_container_width=True)

    if not _SOLVER_AVAILABLE:
        st.info("QIHD solver not found. Install with: pip install -e OpenPhiSolve")


# ===========================================================================
# Main content
# ===========================================================================

# Hero section
hero = st.container()
with hero:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Portfolio Optimization and VaR")
        st.markdown(
            f'<p class="section-subtext">Clean, explainable optimization with real-time risk metrics. '
            f'Built for quantitative analysis with high-performance computing.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("Upload CSV or fetch tickers | Min-variance and VaR/CVaR | Export weights")
        st.markdown('<span class="badge badge-ok">Fast QIHD</span> <span class="badge badge-warn">VaR Ready</span>', unsafe_allow_html=True)
    with c2:
        fig_pulse, axp = plt.subplots(figsize=(3.8, 3.2), facecolor=_BG_CARD)
        axp.axis("off")
        axp.set_facecolor(_BG_CARD)
        axp.add_patch(Rectangle((0, 0), 1, 1, color=_ACCENT, alpha=0.15, lw=0))
        axp.text(0.5, 0.6, "Optimize", ha="center", va="center", fontsize=16, color=_ACCENT, fontweight="700")
        axp.text(0.5, 0.4, "VaR | CVaR | Sharpe", ha="center", va="center", fontsize=10, color=_TEXT_SEC)
        st.pyplot(fig_pulse, use_container_width=True)
        plt.close(fig_pulse)


# --- Load data based on selection ---
data_load_error = None
source_label = data_mode
returns_df = pd.DataFrame()
asset_names = []

try:
    if data_mode == "Upload CSV":
        returns_df = _load_returns(csv_file, asset_cap, seed)
        source_label = "CSV Upload" if csv_file else "Synthetic fallback"
    elif data_mode == "Fetch tickers":
        start_dt, end_dt = date_range if isinstance(date_range, tuple) else (default_start, default_end)
        returns_df, err = _fetch_ticker_returns(tickers, start=start_dt, end=end_dt + timedelta(days=1))
        if err:
            data_load_error = err
        else:
            returns_df = returns_df.iloc[:, :asset_cap]
            source_label = f"Tickers ({len(returns_df.columns)})"
    else:
        returns_df = _load_returns(None, n_assets, seed)
        returns_df = returns_df.iloc[:, :asset_cap]
        source_label = "Synthetic factor model"

    asset_names = list(returns_df.columns)
    n_actual = returns_df.shape[1]
except Exception as e:
    data_load_error = str(e)
    returns_df = pd.DataFrame()
    asset_names = []
    n_actual = 0

if data_load_error:
    st.error(f"Data loading failed: {data_load_error}")
    st.stop()

# Apply lookback if enough rows
if returns_df.shape[0] > lookback_days:
    returns_df = returns_df.tail(lookback_days)

# --- Data preview / stats ---
st.markdown("#### Data Summary")
ds1, ds2, ds3, ds4 = st.columns(4)
ds1.metric("Assets", n_actual)
ds2.metric("Observations", returns_df.shape[0])
ds3.metric("Source", source_label)
ds4.metric("Lookback (days)", returns_df.shape[0])

if returns_df.shape[0] < 252:
    st.warning("Sample size under 252 trading days. Risk estimates may be unstable.")

with st.expander("Preview and Diagnostics", expanded=False):
    st.dataframe(returns_df.head(12), use_container_width=True, height=260)
    heat = _correlation_heatmap(returns_df)
    if heat is not None:
        st.pyplot(heat, use_container_width=True)
        plt.close(heat)

# ===========================================================================
# Run optimization
# ===========================================================================
if run_btn:
    if not _SOLVER_AVAILABLE:
        st.error("Cannot run: QIHD solver (OpenPhiSolve) is not installed. Run: pip install -e OpenPhiSolve")
        st.stop()

    if n_actual == 0 or returns_df.empty:
        st.error("No data available to optimize. Please upload or fetch returns.")
        st.stop()

    returns_np = returns_df.values
    alpha_tail = 1 - alpha

    # Progress
    stages = [("Preparing data...", 15), ("Computing covariance...", 20), ("Running QIHD...", 45), ("Calculating VaR...", 20)]
    progress = _progress_runner(stages)

    mu, cov = compute_mu_cov(returns_np)
    caps = caps_from_constant(cap, n_actual)

    if max_k > n_actual:
        st.warning(f"K={max_k} exceeds asset count {n_actual}. Clamping to {n_actual}.")
        max_k = n_actual

    spec = PortfolioSpec(mu=mu, cov=cov, upper=caps, max_positions=max_k, lambda_risk=lambda_risk)
    backend_kwargs = dict(n_shots=int(n_shots), n_steps=int(n_steps), dt=float(dt), device=device)

    t0 = time.perf_counter()
    try:
        result = solve_portfolio(spec, backend_kwargs=backend_kwargs, if_refine=False)
    except Exception as exc:
        progress.empty()
        st.error(f"Solver failed: {exc}")
        st.stop()
    solve_time = time.perf_counter() - t0
    progress.progress(100, text="Done")
    progress.empty()

    weights = np.clip(result.weights, 0, None)
    wsum = weights.sum()
    if wsum > 0:
        weights = weights / wsum
    selected = np.nonzero(result.selection > 0.5)[0]

    # Metrics
    pnl = returns_np @ weights
    hist_var = _historical_var(pnl, alpha)
    hist_cvar = _historical_cvar(pnl, alpha)
    mc_var, mc_cvar = _mc_var(returns_np, weights, alpha, n_scenarios=mc_scenarios, seed=seed)
    vol = float(np.std(pnl))
    exp_ret = float(np.mean(pnl))
    sharpe = (exp_ret - risk_free) / vol if vol > 1e-9 else 0.0

    metrics = {"var": hist_var, "cvar": hist_cvar, "sharpe": sharpe, "vol": vol}
    label = f"{mode.split()[1]} | K={max_k}"
    _store_scenario(st.session_state, label, asset_names, weights, metrics)

    st.divider()
    st.markdown('<p class="section-header">Optimization Results</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Assets", f"{len(selected)} / {n_actual}")
    m2.metric("Sum of Weights", f"{weights.sum():.4f}")
    m3.metric("Solve Time", f"{solve_time:.2f}s")
    m4.metric("Mode", mode)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Historical VaR ({alpha:.0%})", f"{hist_var:.4f}", delta=None)
    r2.metric("Historical CVaR", f"{hist_cvar:.4f}")
    r3.metric(f"MC VaR ({mc_scenarios//1000}k sims)", f"{mc_var:.4f}")
    r4.metric("Sharpe", f"{sharpe:.3f}", delta=f"Vol {vol:.2%}")

    # Allocation + table
    col_chart, col_table = st.columns([1, 1], gap="large")
    with col_chart:
        fig_w = _plot_weights(asset_names, weights)
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)
        csv_data = _export_weights_csv(asset_names, weights)
        st.download_button("Download weights CSV", csv_data, "weights.csv", mime="text/csv", use_container_width=True)

    with col_table:
        wdf = pd.DataFrame({"Asset": asset_names, "Weight": weights, "Selected": weights > 1e-6, "Return (avg)": mu}).sort_values("Weight", ascending=False)
        wdf_display = wdf[wdf["Selected"]].copy()
        wdf_display["Weight"] = wdf_display["Weight"].apply(lambda x: f"{x:.2%}")
        wdf_display["Return (avg)"] = wdf_display["Return (avg)"].apply(lambda x: f"{x:.6f}")
        wdf_display = wdf_display.drop(columns=["Selected"]).reset_index(drop=True)
        st.dataframe(wdf_display, use_container_width=True, height=380)

    st.divider()
    st.markdown('<p class="section-header">Risk Analysis</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subtext">VaR at {alpha:.0%} confidence, tail {alpha_tail:.0%}.</p>', unsafe_allow_html=True)

    col_dist, col_risk = st.columns([1, 1], gap="large")
    with col_dist:
        fig_loss = _plot_loss_distribution(returns_np, weights, alpha, n_scenarios=mc_scenarios, seed=seed)
        st.pyplot(fig_loss, use_container_width=True)
        plt.close(fig_loss)
    with col_risk:
        fig_rc = _plot_risk_contribution(asset_names, weights, cov)
        if fig_rc is not None:
            st.pyplot(fig_rc, use_container_width=True)
            plt.close(fig_rc)

    with st.expander("Solver Details", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Device", device.upper())
        d2.metric("Shots", n_shots)
        d3.metric("Steps", n_steps)
        d4.metric("dt", f"{dt:.2f}")
        st.markdown(f"**Risk aversion** = {lambda_risk:.1f}")
        st.markdown(f"**Cardinality** K = {max_k}")
        st.markdown(f"**Per-asset cap** u = {cap:.2f}")
        st.markdown(f"Solver: {solver_choice}")
        if gpu_info["wsl"]:
            st.info("Running on WSL2. Ensure nvidia-smi works in WSL for GPU acceleration.")

    st.divider()
    _render_comparison(st.session_state)

else:
    st.markdown(
        f'<div class="empty-state">'
        f'<p style="font-size: 1.2rem; font-weight: 600; color: {_TEXT}; margin-bottom: 0.5rem;">'
        f'Ready to optimize your portfolio</p>'
        f'<p style="font-size: 0.95rem; color: {_TEXT_SEC}; margin-bottom: 1.5rem; line-height: 1.6;">'
        f'Configure parameters in the sidebar and click <strong>Optimize Portfolio</strong>. '
        f'The solver will find optimal weights using quantum-inspired Hamiltonian dynamics, then compute VaR metrics.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
