import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# ============================================================
# AMZN LIMIT ORDER BOOK PROBABILITY DASHBOARD
# Statistical CTMC Model
# ============================================================

st.set_page_config(
    page_title="AMZN Limit Order Book Dashboard",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------------------
# 1. LOAD EXPORTED DATA
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "AMZN_Dashboard_Data"

FILES = {
    "first_passage": "first_passage_results.csv",
    "sensitivity": "sensitivity_results.csv",
    "limit": "limit_rates.csv",
    "market": "market_rates.csv",
    "cancellation": "cancellation_rates.csv"
}

@st.cache_data
def load_data():
    data = {}

    for key, filename in FILES.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        data[key] = pd.read_csv(path)

    return data


st.title("📈 AMZN Limit Order Book Probability Dashboard")

st.markdown(
    """
    **Statistical modelling of limit order book dynamics**

    This dashboard uses fitted order-flow rates and a finite-state
    continuous-time Markov chain (CTMC) to examine eventual
    upward and downward first-passage probabilities.
    """
)

try:
    data = load_data()
except Exception as error:
    st.error(str(error))
    st.info(
        "Ensure app.py and the AMZN_Dashboard_Data folder "
        "are in the same directory."
    )
    st.stop()

first_passage = data["first_passage"]
sensitivity = data["sensitivity"]
limit_rates = data["limit"]
market_rates = data["market"]
cancellation_rates = data["cancellation"]

# ------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# ------------------------------------------------------------

st.sidebar.header("Model Controls")

spread_options = sorted(
    sensitivity["Spread_Ticks"].dropna().unique().tolist()
)

selected_spread = st.sidebar.selectbox(
    "Select spread (ticks)",
    spread_options,
    index=0
)

spread_data = sensitivity[
    sensitivity["Spread_Ticks"] == selected_spread
].copy()

ask_options = sorted(
    spread_data["Initial_Ask_Queue_Units"]
    .dropna().unique().tolist()
)

selected_ask = st.sidebar.selectbox(
    "Initial ask queue (units)",
    ask_options
)

bid_options = sorted(
    spread_data.loc[
        spread_data["Initial_Ask_Queue_Units"] == selected_ask,
        "Initial_Bid_Queue_Units"
    ].dropna().unique().tolist()
)

if not bid_options:
    st.error("No precomputed bid queue scenarios are available.")
    st.stop()

selected_bid = st.sidebar.selectbox(
    "Initial bid queue (units)",
    bid_options
)

# ------------------------------------------------------------
# 3. RETRIEVE EXACT PRECOMPUTED PROBABILITIES
# ------------------------------------------------------------

selected_result = sensitivity[
    (sensitivity["Spread_Ticks"] == selected_spread)
    & (
        sensitivity["Initial_Ask_Queue_Units"]
        == selected_ask
    )
    & (
        sensitivity["Initial_Bid_Queue_Units"]
        == selected_bid
    )
]

st.subheader("Selected Model State")

col1, col2, col3 = st.columns(3)

col1.metric("Spread", f"{selected_spread} ticks")
col2.metric("Initial ask queue", f"{selected_ask} units")
col3.metric("Initial bid queue", f"{selected_bid} units")

if selected_result.empty:
    st.warning(
        "This queue combination has no precomputed result. "
        "No probability has been estimated for this selection."
    )
    st.stop()

result = selected_result.iloc[0]

upward = float(result["Upward_Probability"])
downward = float(result["Downward_Probability"])

valid = (
    np.isfinite(upward)
    and np.isfinite(downward)
    and 0 <= upward <= 1
    and 0 <= downward <= 1
    and np.isclose(upward + downward, 1.0, atol=1e-6)
)

if not valid:
    st.error("The selected probability result failed validation.")
    st.stop()

# ------------------------------------------------------------
# 4. FIRST-PASSAGE PROBABILITIES
# ------------------------------------------------------------

st.divider()
st.header("First-Passage Probabilities")

st.caption(
    "These are eventual first-passage probabilities from the "
    "finite-state CTMC model, not probabilities within a fixed "
    "time horizon."
)

col1, col2 = st.columns(2)

col1.metric(
    "Upward probability",
    f"{upward * 100:.2f}%"
)

col2.metric(
    "Downward probability",
    f"{downward * 100:.2f}%"
)

st.markdown("**Probability distribution**")

probability_chart = pd.DataFrame({
    "Direction": ["Upward", "Downward"],
    "Probability (%)": [
        upward * 100,
        downward * 100
    ]
}).set_index("Direction")

st.bar_chart(
    probability_chart,
    y="Probability (%)"
)

st.success(
    f"Probability sum = {(upward + downward):.6f}"
)

# ------------------------------------------------------------
# 5. ORIGINAL BASELINE COMPARISON
# ------------------------------------------------------------

st.divider()
st.header("Original Model Baseline")

baseline = first_passage[
    first_passage["Spread_Ticks"] == selected_spread
]

if not baseline.empty:

    baseline = baseline.iloc[0]

    baseline_col1, baseline_col2 = st.columns(2)

    baseline_col1.metric(
        "Baseline upward probability",
        f"{baseline['Upward_Probability'] * 100:.2f}%"
    )

    baseline_col2.metric(
        "Baseline downward probability",
        f"{baseline['Downward_Probability'] * 100:.2f}%"
    )

    st.write(
        "Original ask queue:",
        int(baseline["Initial_Ask_Queue_Units"])
    )

    st.write(
        "Original bid queue:",
        int(baseline["Initial_Bid_Queue_Units"])
    )

    st.write(
        "Observed spread-state snapshots:",
        int(baseline["Snapshot_Count"])
    )

    st.write(
        "Queue cap:",
        int(baseline["Queue_Cap"])
    )

    st.write(
        "Reliable cap result:",
        bool(baseline["Reliable_Cap_Result"])
    )

# ------------------------------------------------------------
# 6. QUEUE SENSITIVITY ANALYSIS
# ------------------------------------------------------------

st.divider()
st.header("Queue Sensitivity Analysis")

st.markdown(
    """
    The following table and heatmap show how the model's
    eventual upward probability changes with initial ask
    and bid queue sizes.
    """
)

heatmap_data = spread_data.pivot_table(
    index="Initial_Ask_Queue_Units",
    columns="Initial_Bid_Queue_Units",
    values="Upward_Probability",
    aggfunc="first"
)

st.subheader("Upward Probability Heatmap")

fig, ax = plt.subplots(figsize=(9, 5))

image = ax.imshow(
    heatmap_data.values * 100,
    aspect="auto",
    cmap="viridis",
    origin="lower"
)

ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns)

ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

ax.set_xlabel("Initial Bid Queue (units)")
ax.set_ylabel("Initial Ask Queue (units)")
ax.set_title(
    f"Upward First-Passage Probability — Spread {selected_spread}"
)

fig.colorbar(
    image,
    ax=ax,
    label="Upward probability (%)"
)

for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]

        if pd.notna(value):
            ax.text(
                j,
                i,
                f"{value * 100:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8
            )

st.pyplot(fig)
plt.close(fig)

st.subheader("Sensitivity Results")

st.dataframe(
    spread_data[
        [
            "Initial_Ask_Queue_Units",
            "Initial_Bid_Queue_Units",
            "Upward_Probability",
            "Downward_Probability",
            "Is_Original_Queue"
        ]
    ].sort_values(
        [
            "Initial_Ask_Queue_Units",
            "Initial_Bid_Queue_Units"
        ]
    ),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# 7. FITTED LIMIT ORDER RATES
# ------------------------------------------------------------

st.divider()
st.header("Fitted Order-Flow Parameters")

st.subheader("Limit Order Arrival Rates")

limit_display = limit_rates[
    (limit_rates["Spread_Ticks"] == selected_spread)
    & (
        limit_rates["Distance_Ticks"]
        == selected_spread
    )
].copy()

if not limit_display.empty:
    st.dataframe(
        limit_display[
            [
                "Side",
                "Direction",
                "Distance_Ticks",
                "Arrival_Count",
                "Average_Limit_Size",
                "Time_in_State",
                "Arrival_Rate",
                "Reliable_Count"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No best-quote limit-order rate records "
        "are available for this spread."
    )

# ------------------------------------------------------------
# 8. MARKET ORDER RATES
# ------------------------------------------------------------

st.subheader("Market Order Rates")

market_display = market_rates[
    market_rates["Spread_Ticks"] == selected_spread
].copy()

if not market_display.empty:
    st.dataframe(
        market_display[
            [
                "Side",
                "Market_Direction",
                "Market_Count",
                "Average_Market_Size",
                "Market_Rate_Raw",
                "Market_Rate_Adjusted",
                "Reliable_Count"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No market-order rate records are available "
        "for this spread."
    )

# ------------------------------------------------------------
# 9. CANCELLATION RATES
# ------------------------------------------------------------

st.subheader("Cancellation Rates")

cancellation_display = cancellation_rates[
    (cancellation_rates["Spread_Ticks"] == selected_spread)
    & (
        cancellation_rates["Distance_Ticks"]
        == selected_spread
    )
].copy()

if not cancellation_display.empty:
    st.dataframe(
        cancellation_display[
            [
                "Side",
                "Direction",
                "Cancellation_Count",
                "Cancelled_Volume",
                "Average_Cancellation_Size",
                "Average_Queue_Volume",
                "Time_in_State",
                "Cancellation_Rate",
                "Reliable_Count"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No best-quote cancellation records are available "
        "for this spread."
    )

# ------------------------------------------------------------
# 10. DATA RELIABILITY WARNINGS
# ------------------------------------------------------------

st.divider()
st.header("Model Reliability and Limitations")

st.info(
    """
    **Interpretation and limitations**

    1. Results are based on the supplied AMZN LOBSTER trading-day data.

    2. The CTMC uses a finite queue-state space with a specified
       queue cap.

    3. The displayed first-passage probabilities are eventual
       probabilities, not fixed-horizon probabilities.

    4. Results are conditional on the fitted spread-state rates
       and the selected initial queue sizes.

    5. Sparse or unreliable state estimates should be interpreted
       cautiously.

    6. The current dashboard does not claim validated order-fill
       probabilities for best-quote or deeper queue positions.
    """
)

st.caption(
    "Research demonstration — statistical model estimates, "
    "not investment advice."
)

st.divider()

st.write(
    "AMZN Limit Order Book Statistical Modelling Project"
)
