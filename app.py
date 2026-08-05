import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Gold & Forex AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Gold & Forex Trading Assistant")
st.caption(
    "Live market analysis using Twelve Data. "
    "Signals are estimates from technical indicators, not guaranteed predictions."
)


# =========================================================
# MARKET SETTINGS
# =========================================================

MARKETS = {
    "Gold — XAU/USD": {
        "symbols": ["XAU/USD", "XAUUSD"],
        "digits": 2,
        "category": "gold",
    },
    "EUR/USD": {
        "symbols": ["EUR/USD"],
        "digits": 5,
        "category": "forex",
    },
    "GBP/USD": {
        "symbols": ["GBP/USD"],
        "digits": 5,
        "category": "forex",
    },
    "USD/JPY": {
        "symbols": ["USD/JPY"],
        "digits": 3,
        "category": "forex",
    },
    "USD/CHF": {
        "symbols": ["USD/CHF"],
        "digits": 5,
        "category": "forex",
    },
    "AUD/USD": {
        "symbols": ["AUD/USD"],
        "digits": 5,
        "category": "forex",
    },
    "NZD/USD": {
        "symbols": ["NZD/USD"],
        "digits": 5,
        "category": "forex",
    },
    "USD/CAD": {
        "symbols": ["USD/CAD"],
        "digits": 5,
        "category": "forex",
    },
    "EUR/GBP": {
        "symbols": ["EUR/GBP"],
        "digits": 5,
        "category": "forex",
    },
    "GBP/JPY": {
        "symbols": ["GBP/JPY"],
        "digits": 3,
        "category": "forex",
    },
    "S&P 500 Index": {
        "symbols": ["SPX", "GSPC", "SPY"],
        "digits": 2,
        "category": "index",
    },
    "E-mini S&P 500": {
        "symbols": ["ES", "ES1!", "SPY"],
        "digits": 2,
        "category": "index",
    },
    "Micro E-mini S&P 500": {
        "symbols": ["MES", "MES1!", "SPY"],
        "digits": 2,
        "category": "index",
    },
}

INTERVALS = {
    "5 minutes": "5min",
    "15 minutes": "15min",
    "1 hour": "1h",
    "4 hours": "4h",
}


# =========================================================
# API KEY
# =========================================================

def get_api_key() -> Optional[str]:
    possible_names = [
        "TWELVE_DATA_API_KEY",
        "TWELVEDATA_API_KEY",
        "twelve_data_api_key",
        "API_KEY",
        "api_key",
    ]

    for name in possible_names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass

    return None


API_KEY = get_api_key()

if not API_KEY:
    st.error(
        "Twelve Data API key not found. In Streamlit Secrets, add:\n\n"
        'TWELVE_DATA_API_KEY = "your_key_here"'
    )
    st.stop()


# =========================================================
# DATA FUNCTIONS
# =========================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(
    candidate_symbols: tuple,
    interval: str,
    output_size: int,
    api_key: str,
):
    url = "https://api.twelvedata.com/time_series"
    last_error = "No market data returned."

    for symbol in candidate_symbols:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": output_size,
            "apikey": api_key,
            "format": "JSON",
            "timezone": "UTC",
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            last_error = f"Connection error: {exc}"
            continue
        except ValueError:
            last_error = "Twelve Data returned an invalid response."
            continue

        if payload.get("status") == "error":
            last_error = payload.get("message", "Twelve Data API error.")
            continue

        values = payload.get("values")

        if not values:
            last_error = f"No candles were returned for {symbol}."
            continue

        frame = pd.DataFrame(values)

        required_columns = ["datetime", "open", "high", "low", "close"]

        if not all(column in frame.columns for column in required_columns):
            last_error = f"Incomplete candle data returned for {symbol}."
            continue

        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(
                frame["volume"],
                errors="coerce",
            )

        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            errors="coerce",
            utc=True,
        )

        frame = (
            frame.dropna(subset=required_columns)
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if len(frame) < 55:
            last_error = (
                f"Not enough candle history was returned for {symbol}. "
                "Try again later or choose another interval."
            )
            continue

        return frame, symbol, None

    return None, None, last_error


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    change = close.diff()

    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, float("nan"))

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi.fillna(50)


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)

    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    true_range = ranges.max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema_200"] = data["close"].ewm(span=200, adjust=False).mean()

    data["rsi"] = calculate_rsi(data["close"], 14)

    data["macd"] = (
        data["close"].ewm(span=12, adjust=False).mean()
        - data["close"].ewm(span=26, adjust=False).mean()
    )

    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["macd_histogram"] = (
        data["macd"] - data["macd_signal"]
    )

    data["atr"] = calculate_atr(data, 14)

    return data.dropna().reset_index(drop=True)


# =========================================================
# SUPPORT AND RESISTANCE
# =========================================================

def calculate_support_resistance(
    data: pd.DataFrame,
    lookback: int = 50,
):
    recent = data.tail(min(lookback, len(data)))

    support = float(recent["low"].min())
    resistance = float(recent["high"].max())

    return support, resistance


# =========================================================
# MAIN SIGNAL
# =========================================================

def analyse_market(data: pd.DataFrame):
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    bullish_points = 0
    bearish_points = 0
    bullish_reasons = []
    bearish_reasons = []

    price = float(latest["close"])
    atr = float(latest["atr"])

    if price > latest["ema_20"]:
        bullish_points += 15
        bullish_reasons.append("Price is above the 20 EMA")
    else:
        bearish_points += 15
        bearish_reasons.append("Price is below the 20 EMA")

    if latest["ema_20"] > latest["ema_50"]:
        bullish_points += 20
        bullish_reasons.append("20 EMA is above the 50 EMA")
    else:
        bearish_points += 20
        bearish_reasons.append("20 EMA is below the 50 EMA")

    if latest["ema_9"] > latest["ema_20"]:
        bullish_points += 10
        bullish_reasons.append("Short-term EMA momentum is bullish")
    else:
        bearish_points += 10
        bearish_reasons.append("Short-term EMA momentum is bearish")

    if latest["macd"] > latest["macd_signal"]:
        bullish_points += 15
        bullish_reasons.append("MACD momentum is bullish")
    else:
        bearish_points += 15
        bearish_reasons.append("MACD momentum is bearish")

    if latest["macd_histogram"] > previous["macd_histogram"]:
        bullish_points += 10
        bullish_reasons.append("Bullish momentum is increasing")
    else:
        bearish_points += 10
        bearish_reasons.append("Bearish momentum is increasing")

    rsi = float(latest["rsi"])

    if 52 <= rsi <= 70:
        bullish_points += 15
        bullish_reasons.append(f"RSI supports buyers at {rsi:.1f}")
    elif 30 <= rsi <= 48:
        bearish_points += 15
        bearish_reasons.append(f"RSI supports sellers at {rsi:.1f}")
    elif rsi > 70:
        bearish_points += 5
        bearish_reasons.append(
            f"RSI is overbought at {rsi:.1f}"
        )
    elif rsi < 30:
        bullish_points += 5
        bullish_reasons.append(
            f"RSI is oversold at {rsi:.1f}"
        )

    recent_closes = data["close"].tail(6)

    if recent_closes.iloc[-1] > recent_closes.iloc[0]:
        bullish_points += 15
        bullish_reasons.append("Recent price structure is rising")
    else:
        bearish_points += 15
        bearish_reasons.append("Recent price structure is falling")

    total_points = bullish_points + bearish_points

    if total_points == 0:
        total_points = 1

    bullish_percentage = round(
        bullish_points / total_points * 100
    )

    bearish_percentage = round(
        bearish_points / total_points * 100
    )

    score_difference = abs(
        bullish_percentage - bearish_percentage
    )

    if (
        bullish_percentage >= 62
        and score_difference >= 20
    ):
        signal = "BUY"
        confidence = bullish_percentage
        reasons = bullish_reasons
    elif (
        bearish_percentage >= 62
        and score_difference >= 20
    ):
        signal = "SELL"
        confidence = bearish_percentage
        reasons = bearish_reasons
    else:
        signal = "NO TRADE"
        confidence = max(
            bullish_percentage,
            bearish_percentage,
        )
        reasons = [
            "Indicators are conflicting",
            "There is not enough confirmation for a quality setup",
        ]

    support, resistance = calculate_support_resistance(data)

    if signal == "BUY":
        entry = price
        limit_entry = max(
            float(latest["ema_20"]),
            price - (atr * 0.45),
        )
        stop_loss = min(
            support - (atr * 0.15),
            entry - (atr * 1.35),
        )

        risk_distance = entry - stop_loss

        tp1 = entry + risk_distance
        tp2 = entry + (risk_distance * 2)
        tp3 = entry + (risk_distance * 3)

    elif signal == "SELL":
        entry = price
        limit_entry = min(
            float(latest["ema_20"]),
            price + (atr * 0.45),
        )
        stop_loss = max(
            resistance + (atr * 0.15),
            entry + (atr * 1.35),
        )

        risk_distance = stop_loss - entry

        tp1 = entry - risk_distance
        tp2 = entry - (risk_distance * 2)
        tp3 = entry - (risk_distance * 3)

    else:
        entry = price
        limit_entry = float(latest["ema_20"])
        stop_loss = None
        risk_distance = None
        tp1 = None
        tp2 = None
        tp3 = None

    return {
        "signal": signal,
        "confidence": confidence,
        "bullish_score": bullish_percentage,
        "bearish_score": bearish_percentage,
        "price": price,
        "entry": entry,
        "limit_entry": limit_entry,
        "stop_loss": stop_loss,
        "risk_distance": risk_distance,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "support": support,
        "resistance": resistance,
        "rsi": rsi,
        "atr": atr,
        "reasons": reasons[:6],
    }


# =========================================================
# HISTORICAL REVERSAL ANALYSIS
# =========================================================

def detect_reversal(data: pd.DataFrame):
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    price = float(latest["close"])
    atr = float(latest["atr"])

    support, resistance = calculate_support_resistance(
        data,
        lookback=80,
    )

    bullish_points = 0
    bearish_points = 0

    bullish_reasons = []
    bearish_reasons = []

    distance_to_support = abs(price - support)
    distance_to_resistance = abs(resistance - price)

    if distance_to_support <= atr * 1.5:
        bullish_points += 25
        bullish_reasons.append(
            "Price is close to an historical support zone"
        )

    if distance_to_resistance <= atr * 1.5:
        bearish_points += 25
        bearish_reasons.append(
            "Price is close to an historical resistance zone"
        )

    if latest["rsi"] < 35:
        bullish_points += 20
        bullish_reasons.append(
            f"RSI is oversold at {latest['rsi']:.1f}"
        )

    if latest["rsi"] > 65:
        bearish_points += 20
        bearish_reasons.append(
            f"RSI is overbought at {latest['rsi']:.1f}"
        )

    bullish_candle = (
        latest["close"] > latest["open"]
        and latest["close"] > previous["high"]
    )

    bearish_candle = (
        latest["close"] < latest["open"]
        and latest["close"] < previous["low"]
    )

    if bullish_candle:
        bullish_points += 25
        bullish_reasons.append(
            "Bullish breakout candle has formed"
        )

    if bearish_candle:
        bearish_points += 25
        bearish_reasons.append(
            "Bearish breakdown candle has formed"
        )

    recent = data.tail(12)

    price_change = (
        recent["close"].iloc[-1]
        - recent["close"].iloc[0]
    )

    rsi_change = (
        recent["rsi"].iloc[-1]
        - recent["rsi"].iloc[0]
    )

    if price_change < 0 and rsi_change > 4:
        bullish_points += 30
        bullish_reasons.append(
            "Possible bullish RSI divergence detected"
        )

    if price_change > 0 and rsi_change < -4:
        bearish_points += 30
        bearish_reasons.append(
            "Possible bearish RSI divergence detected"
        )

    if (
        latest["macd_histogram"] > previous["macd_histogram"]
        and latest["macd_histogram"] < 0
    ):
        bullish_points += 15
        bullish_reasons.append(
            "Selling momentum appears to be weakening"
        )

    if (
        latest["macd_histogram"] < previous["macd_histogram"]
        and latest["macd_histogram"] > 0
    ):
        bearish_points += 15
        bearish_reasons.append(
            "Buying momentum appears to be weakening"
        )

    strongest_score = max(
        bullish_points,
        bearish_points,
    )

    confidence = min(strongest_score, 95)

    if bullish_points >= 50 and bullish_points > bearish_points:
        reversal_type = "Potential bullish reversal"
        zone_low = support
        zone_high = support + atr
        confirmation = price + (atr * 0.35)
        invalidation = support - (atr * 0.4)
        reasons = bullish_reasons

    elif bearish_points >= 50 and bearish_points > bullish_points:
        reversal_type = "Potential bearish reversal"
        zone_low = resistance - atr
        zone_high = resistance
        confirmation = price - (atr * 0.35)
        invalidation = resistance + (atr * 0.4)
        reasons = bearish_reasons

    else:
        reversal_type = "No strong reversal setup detected"
        zone_low = None
        zone_high = None
        confirmation = None
        invalidation = None
        reasons = [
            "Historical price action does not currently show "
            "enough reversal confirmation"
        ]

    return {
        "type": reversal_type,
        "confidence": confidence,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "reasons": reasons[:5],
    }


# =========================================================
# LOT SIZE ESTIMATE
# =========================================================

def estimate_lot_size(
    category: str,
    risk_amount_gbp: float,
    stop_distance: Optional[float],
):
    if not stop_distance or stop_distance <= 0:
        return None, None

    # These are approximate common contract values.
    # Broker specifications can differ.
    if category == "gold":
        value_per_price_unit_per_lot_gbp = 79.0

    elif category == "forex":
        value_per_price_unit_per_lot_gbp = 79000.0

    else:
        value_per_price_unit_per_lot_gbp = 39.5

    loss_per_lot = (
        stop_distance
        * value_per_price_unit_per_lot_gbp
    )

    if loss_per_lot <= 0:
        return None, None

    raw_lot = risk_amount_gbp / loss_per_lot
    rounded_lot = math.floor(raw_lot * 100) / 100

    return max(rounded_lot, 0.01), loss_per_lot


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Trade settings")

    selected_market = st.selectbox(
        "Market",
        list(MARKETS.keys()),
    )

    selected_interval_name = st.selectbox(
        "Chart timeframe",
        list(INTERVALS.keys()),
        index=1,
    )

    account_balance = st.number_input(
        "Account balance (£)",
        min_value=1.0,
        value=200.0,
        step=10.0,
    )

    risk_percentage = st.slider(
        "Risk per trade (%)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    history_size = st.select_slider(
        "Candle history",
        options=[100, 150, 200, 300, 500],
        value=200,
    )

    st.warning(
        "10% risk per trade is extremely high. "
        "A small losing run can reduce the account quickly."
    )

    refresh_pressed = st.button(
        "🔄 Refresh analysis",
        use_container_width=True,
    )

if refresh_pressed:
    st.cache_data.clear()


# =========================================================
# LOAD DATA
# =========================================================

market_settings = MARKETS[selected_market]
interval = INTERVALS[selected_interval_name]

with st.spinner("Loading live market candles..."):
    candles, working_symbol, data_error = fetch_candles(
        tuple(market_settings["symbols"]),
        interval,
        history_size,
        API_KEY,
    )

if data_error or candles is None:
    st.error(
        f"Could not load {selected_market}.\n\n"
        f"Reason: {data_error}\n\n"
        "Some S&P futures symbols require a paid market-data "
        "subscription. Try S&P 500 Index or another market."
    )
    st.stop()

data = add_indicators(candles)

if len(data) < 30:
    st.error("Not enough processed candle data to run the analysis.")
    st.stop()

analysis = analyse_market(data)
reversal = detect_reversal(data)

digits = market_settings["digits"]

format_price = lambda value: (
    "—"
    if value is None
    else f"{value:,.{digits}f}"
)


# =========================================================
# LIVE MARKET HEADER
# =========================================================

st.subheader(f"{selected_market} · {selected_interval_name}")
st.caption(f"Data symbol currently working: `{working_symbol}`")

top_col1, top_col2, top_col3, top_col4 = st.columns(4)

top_col1.metric(
    "Current price",
    format_price(analysis["price"]),
)

top_col2.metric(
    "Signal",
    analysis["signal"],
)

top_col3.metric(
    "Setup confidence",
    f"{analysis['confidence']}%",
)

top_col4.metric(
    "RSI",
    f"{analysis['rsi']:.1f}",
)


# =========================================================
# CHART
# =========================================================

chart_data = data.tail(120)

figure = go.Figure()

figure.add_trace(
    go.Candlestick(
        x=chart_data["datetime"],
        open=chart_data["open"],
        high=chart_data["high"],
        low=chart_data["low"],
        close=chart_data["close"],
        name="Price",
    )
)

figure.add_trace(
    go.Scatter(
        x=chart_data["datetime"],
        y=chart_data["ema_20"],
        mode="lines",
        name="EMA 20",
    )
)

figure.add_trace(
    go.Scatter(
        x=chart_data["datetime"],
        y=chart_data["ema_50"],
        mode="lines",
        name="EMA 50",
    )
)

figure.add_hline(
    y=analysis["support"],
    line_dash="dot",
    annotation_text="Support",
)

figure.add_hline(
    y=analysis["resistance"],
    line_dash="dot",
    annotation_text="Resistance",
)

figure.update_layout(
    height=570,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=25, b=10),
    legend=dict(orientation="h"),
)

st.plotly_chart(
    figure,
    use_container_width=True,
)


# =========================================================
# TRADE PLAN
# =========================================================

st.subheader("🎯 Trade setup")

if analysis["signal"] == "NO TRADE":
    st.warning(
        "NO TRADE — the indicators do not currently provide "
        "enough confirmation."
    )

else:
    signal_col1, signal_col2, signal_col3 = st.columns(3)

    signal_col1.metric(
        "Suggested market entry",
        format_price(analysis["entry"]),
    )

    signal_col2.metric(
        f"{analysis['signal'].title()} limit",
        format_price(analysis["limit_entry"]),
    )

    signal_col3.metric(
        "Stop loss",
        format_price(analysis["stop_loss"]),
    )

    tp_col1, tp_col2, tp_col3 = st.columns(3)

    tp_col1.metric(
        "TP1 · 1:1",
        format_price(analysis["tp1"]),
    )

    tp_col2.metric(
        "TP2 · 1:2",
        format_price(analysis["tp2"]),
    )

    tp_col3.metric(
        "TP3 · 1:3",
        format_price(analysis["tp3"]),
    )


# =========================================================
# RISK AND LOT SIZE
# =========================================================

st.subheader("💷 Risk and position size")

risk_amount = account_balance * (
    risk_percentage / 100
)

estimated_lot, loss_per_lot = estimate_lot_size(
    market_settings["category"],
    risk_amount,
    analysis["risk_distance"],
)

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric(
    "Account risk",
    f"£{risk_amount:,.2f}",
)

risk_col2.metric(
    "Stop distance",
    format_price(analysis["risk_distance"]),
)

risk_col3.metric(
    "Estimated maximum lot",
    "—" if estimated_lot is None else f"{estimated_lot:.2f}",
)

st.info(
    "Lot size is an estimate because contract sizes, tick values, "
    "currency conversion and symbol specifications vary by broker. "
    "Check the exact loss shown in MT5 before placing the trade."
)


# =========================================================
# SUPPORT, RESISTANCE AND SCORES
# =========================================================

st.subheader("📊 Market structure")

structure_col1, structure_col2 = st.columns(2)

structure_col1.metric(
    "Historical support",
    format_price(analysis["support"]),
)

structure_col2.metric(
    "Historical resistance",
    format_price(analysis["resistance"]),
)

score_col1, score_col2 = st.columns(2)

score_col1.metric(
    "Bullish evidence",
    f"{analysis['bullish_score']}%",
)

score_col2.metric(
    "Bearish evidence",
    f"{analysis['bearish_score']}%",
)

st.markdown("#### Why this signal?")

for reason in analysis["reasons"]:
    st.write(f"• {reason}")


# =========================================================
# REVERSAL EXTRA
# =========================================================

st.subheader("🔄 Potential reversal analysis")
st.caption(
    "This is an additional historical-chart assessment. "
    "It does not replace the main setup."
)

st.markdown(f"### {reversal['type']}")

if reversal["type"] == "No strong reversal setup detected":
    st.info(reversal["reasons"][0])

else:
    reversal_col1, reversal_col2 = st.columns(2)

    reversal_col1.metric(
        "Historical pattern score",
        f"{reversal['confidence']}%",
    )

    reversal_col2.metric(
        "Potential zone",
        (
            f"{format_price(reversal['zone_low'])} – "
            f"{format_price(reversal['zone_high'])}"
        ),
    )

    confirmation_col1, confirmation_col2 = st.columns(2)

    confirmation_col1.metric(
        "Confirmation level",
        format_price(reversal["confirmation"]),
    )

    confirmation_col2.metric(
        "Invalidation level",
        format_price(reversal["invalidation"]),
    )

    st.markdown("#### Historical evidence")

    for reason in reversal["reasons"]:
        st.write(f"• {reason}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Educational trading research only. Prices may be delayed. "
    "Always confirm the live broker price, spread, contract size, "
    "stop-loss cost and upcoming economic news before trading."
)