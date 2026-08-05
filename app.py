import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="GoldForex AI",
    page_icon="📈",
    layout="wide",
)

st.title("GoldForex AI")
st.caption(
    "Gold market analysis and risk-planning tool. "
    "Signals are educational and are not guaranteed."
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
GOLD_CONTRACT_SIZE = 100.0  # Common XAU/USD specification: 100 oz per 1.00 lot


def get_api_key() -> str:
    """Read the API key securely from Streamlit Secrets."""
    try:
        return str(st.secrets["TWELVE_DATA_API_KEY"])
    except (KeyError, FileNotFoundError):
        st.error(
            "Twelve Data API key is missing. Add it to Streamlit Secrets as:\n\n"
            'TWELVE_DATA_API_KEY = "your-new-key"'
        )
        st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_data(
    symbol: str,
    interval: str,
    outputsize: int,
    api_key: str,
) -> pd.DataFrame:
    """Download OHLC price data from Twelve Data."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
    }

    try:
        response = requests.get(
            TWELVE_DATA_URL,
            params=params,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not connect to Twelve Data: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Twelve Data returned an invalid response.") from exc

    if payload.get("status") == "error":
        message = payload.get("message", "Unknown Twelve Data error")
        raise RuntimeError(message)

    values = payload.get("values")

    if not values:
        raise RuntimeError(
            f"No price data was returned for {symbol}. "
            "Your Twelve Data plan may not support this symbol or interval."
        )

    frame = pd.DataFrame(values)

    required_columns = {"datetime", "open", "high", "low", "close"}

    if not required_columns.issubset(frame.columns):
        raise RuntimeError("The returned market data is missing required columns.")

    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = ["open", "high", "low", "close"]

    if "volume" in frame.columns:
        numeric_columns.append("volume")

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = (
        frame.dropna(subset=["datetime", "open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )

    if len(frame) < 60:
        raise RuntimeError(
            "Not enough candles were returned to calculate the indicators."
        )

    return frame


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate EMA, RSI, MACD and ATR indicators."""
    data = frame.copy()

    data["ema20"] = data["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    data["ema50"] = data["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    price_change = data["close"].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + relative_strength))
    data["rsi"] = data["rsi"].fillna(50)

    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()

    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    data["recent_high"] = data["high"].rolling(20).max()
    data["recent_low"] = data["low"].rolling(20).min()

    return data.dropna().reset_index(drop=True)


def analyse_setup(data: pd.DataFrame) -> dict:
    """Score the current setup and create a trade plan."""
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    bullish_points = 0
    bearish_points = 0
    reasons = []

    # Trend
    if latest["close"] > latest["ema20"] > latest["ema50"]:
        bullish_points += 30
        reasons.append("Price and EMAs show an upward trend.")
    elif latest["close"] < latest["ema20"] < latest["ema50"]:
        bearish_points += 30
        reasons.append("Price and EMAs show a downward trend.")
    else:
        reasons.append("The EMA trend is mixed.")

    # Momentum
    if latest["macd"] > latest["macd_signal"]:
        bullish_points += 20
        reasons.append("MACD momentum is bullish.")
    else:
        bearish_points += 20
        reasons.append("MACD momentum is bearish.")

    # RSI
    if 52 <= latest["rsi"] <= 70:
        bullish_points += 20
        reasons.append("RSI supports buying momentum.")
    elif 30 <= latest["rsi"] <= 48:
        bearish_points += 20
        reasons.append("RSI supports selling momentum.")
    elif latest["rsi"] > 70:
        bearish_points += 5
        reasons.append("RSI is overbought; buying now may be risky.")
    elif latest["rsi"] < 30:
        bullish_points += 5
        reasons.append("RSI is oversold; selling now may be risky.")
    else:
        reasons.append("RSI is neutral.")

    # Candle direction
    if latest["close"] > latest["open"]:
        bullish_points += 10
        reasons.append("The latest completed candle is bullish.")
    elif latest["close"] < latest["open"]:
        bearish_points += 10
        reasons.append("The latest completed candle is bearish.")

    # Market structure
    if (
        latest["high"] > previous["high"]
        and latest["low"] > previous["low"]
    ):
        bullish_points += 20
        reasons.append("Price formed a higher high and higher low.")
    elif (
        latest["high"] < previous["high"]
        and latest["low"] < previous["low"]
    ):
        bearish_points += 20
        reasons.append("Price formed a lower high and lower low.")
    else:
        reasons.append("Short-term market structure is unclear.")

    strongest_score = max(bullish_points, bearish_points)

    if bullish_points >= 65 and bullish_points >= bearish_points + 20:
        direction = "BUY"
        confidence = bullish_points
    elif bearish_points >= 65 and bearish_points >= bullish_points + 20:
        direction = "SELL"
        confidence = bearish_points
    else:
        direction = "NO TRADE"
        confidence = strongest_score

    entry = float(latest["close"])
    atr = float(latest["atr"])
    stop_distance = max(atr * 1.5, entry * 0.001)

    if direction == "BUY":
        stop_loss = entry - stop_distance
        tp1 = entry + stop_distance
        tp2 = entry + (stop_distance * 2)
        tp3 = entry + (stop_distance * 3)
    elif direction == "SELL":
        stop_loss = entry + stop_distance
        tp1 = entry - stop_distance
        tp2 = entry - (stop_distance * 2)
        tp3 = entry - (stop_distance * 3)
    else:
        stop_loss = np.nan
        tp1 = np.nan
        tp2 = np.nan
        tp3 = np.nan

    return {
        "direction": direction,
        "confidence": min(int(confidence), 100),
        "entry": entry,
        "stop_loss": stop_loss,
        "stop_distance": stop_distance,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rsi": float(latest["rsi"]),
        "atr": atr,
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "reasons": reasons,
        "candle_time": latest["datetime"],
    }


def create_risk_table(
    balance_gbp: float,
    gbp_usd: float,
    entry: float,
    stop_distance: float,
    leverage: int,
) -> pd.DataFrame:
    """Calculate loss, lot size, profit targets and margin."""
    rows = []

    for risk_percentage in [1, 2, 3, 5, 10]:
        risk_gbp = balance_gbp * risk_percentage / 100
        risk_usd = risk_gbp * gbp_usd

        loss_per_one_lot_usd = stop_distance * GOLD_CONTRACT_SIZE

        if loss_per_one_lot_usd <= 0:
            lot_size = 0.0
        else:
            lot_size = risk_usd / loss_per_one_lot_usd

        # Round down to 0.01 so displayed risk is not exceeded.
        lot_size = math.floor(lot_size * 100) / 100
        lot_size = max(lot_size, 0.0)

        actual_loss_usd = (
            lot_size * GOLD_CONTRACT_SIZE * stop_distance
        )
        actual_loss_gbp = actual_loss_usd / gbp_usd

        margin_usd = (
            entry * GOLD_CONTRACT_SIZE * lot_size / leverage
        )
        margin_gbp = margin_usd / gbp_usd

        rows.append(
            {
                "Risk": f"{risk_percentage}%",
                "Maximum loss": f"£{risk_gbp:,.2f}",
                "Lot size": f"{lot_size:.2f}",
                "Estimated actual loss": f"£{actual_loss_gbp:,.2f}",
                "TP1 profit (1:1)": f"£{actual_loss_gbp:,.2f}",
                "TP2 profit (1:2)": f"£{actual_loss_gbp * 2:,.2f}",
                "TP3 profit (1:3)": f"£{actual_loss_gbp * 3:,.2f}",
                "Estimated margin": f"£{margin_gbp:,.2f}",
            }
        )

    return pd.DataFrame(rows)


def make_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.tail(100)

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=chart_data["datetime"],
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="XAU/USD",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data["datetime"],
            y=chart_data["ema20"],
            mode="lines",
            name="EMA 20",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data["datetime"],
            y=chart_data["ema50"],
            mode="lines",
            name="EMA 50",
        )
    )

    figure.update_layout(
        title="XAU/USD 15-minute chart",
        xaxis_title="Time (UTC)",
        yaxis_title="Gold price in USD",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return figure


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------

api_key = get_api_key()

with st.sidebar:
    st.header("Account settings")

    account_balance = st.number_input(
        "Account balance (£)",
        min_value=10.0,
        value=200.0,
        step=10.0,
    )

    leverage = st.selectbox(
        "Broker leverage",
        options=[30, 50, 100, 200, 500],
        index=4,
    )

    st.caption(
        "Lot-size calculations assume 1.00 Gold lot equals 100 ounces. "
        "Confirm this in your broker's XAU/USD contract specification."
    )

    refresh = st.button(
        "Refresh analysis",
        use_container_width=True,
    )

if refresh:
    st.cache_data.clear()

try:
    with st.spinner("Loading live Gold market data..."):
        gold_data = fetch_market_data(
            symbol="XAU/USD",
            interval="15min",
            outputsize=250,
            api_key=api_key,
        )

        gbp_usd_data = fetch_market_data(
            symbol="GBP/USD",
            interval="15min",
            outputsize=60,
            api_key=api_key,
        )

        gold_data = add_indicators(gold_data)
        analysis = analyse_setup(gold_data)

        gbp_usd = float(gbp_usd_data.iloc[-1]["close"])

except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

direction = analysis["direction"]

if direction == "BUY":
    st.success(
        f"Current result: BUY — setup score {analysis['confidence']}/100"
    )
elif direction == "SELL":
    st.error(
        f"Current result: SELL — setup score {analysis['confidence']}/100"
    )
else:
    st.warning(
        f"Current result: NO TRADE — strongest score "
        f"{analysis['confidence']}/100"
    )

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Gold price",
    f"${analysis['entry']:,.2f}",
)

metric2.metric(
    "RSI",
    f"{analysis['rsi']:.1f}",
)

metric3.metric(
    "ATR",
    f"${analysis['atr']:.2f}",
)

metric4.metric(
    "GBP/USD",
    f"{gbp_usd:.5f}",
)

st.plotly_chart(
    make_chart(gold_data),
    use_container_width=True,
)

st.subheader("Setup checklist")

for reason in analysis["reasons"]:
    st.write(f"• {reason}")

if direction != "NO TRADE":
    st.subheader("Trade plan")

    plan1, plan2, plan3 = st.columns(3)

    plan1.metric("Entry", f"{analysis['entry']:,.2f}")
    plan1.metric("Stop loss", f"{analysis['stop_loss']:,.2f}")

    plan2.metric("TP1 — 1:1", f"{analysis['tp1']:,.2f}")
    plan2.metric("TP2 — 1:2", f"{analysis['tp2']:,.2f}")

    plan3.metric("TP3 — 1:3", f"{analysis['tp3']:,.2f}")
    plan3.metric(
        "Stop distance",
        f"${analysis['stop_distance']:.2f}",
    )

    st.subheader("Risk and lot-size calculator")

    risk_table = create_risk_table(
        balance_gbp=account_balance,
        gbp_usd=gbp_usd,
        entry=analysis["entry"],
        stop_distance=analysis["stop_distance"],
        leverage=leverage,
    )

    st.dataframe(
        risk_table,
        hide_index=True,
        use_container_width=True,
    )

    st.warning(
        "A 10% risk level is extremely aggressive. A short run of losses "
        "could reduce a small account rapidly. The table shows options, "
        "not a recommendation to use the largest one."
    )

else:
    st.info(
        "No stop loss, take profit or lot size is being suggested because "
        "the setup has not passed the minimum signal score."
    )

updated_time = datetime.now(timezone.utc)

st.caption(
    f"Page checked at {updated_time:%d %b %Y, %H:%M:%S} UTC. "
    "Twelve Data candles may be delayed depending on your subscription."
)

st.divider()

st.caption(
    "Important: This tool cannot predict the market or guarantee profit. "
    "Prices and contract specifications can differ between Twelve Data "
    "and your broker. Always verify the entry price, contract size, "
    "minimum lot, spread and stop-loss amount in your broker platform."
)