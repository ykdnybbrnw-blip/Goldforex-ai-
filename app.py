import math
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Gold, Forex & Index Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Gold, Forex & Index Trading Assistant")
st.caption(
    "Live Twelve Data market analysis. Signals are technical estimates, "
    "not guaranteed predictions or financial advice."
)


MARKETS = {
    "Gold — XAU/USD": {
        "symbols": ["XAU/USD", "XAUUSD"],
        "digits": 2,
        "category": "gold",
    },
    "EUR/USD": {"symbols": ["EUR/USD"], "digits": 5, "category": "forex"},
    "GBP/USD": {"symbols": ["GBP/USD"], "digits": 5, "category": "forex"},
    "USD/JPY": {"symbols": ["USD/JPY"], "digits": 3, "category": "forex"},
    "USD/CHF": {"symbols": ["USD/CHF"], "digits": 5, "category": "forex"},
    "AUD/USD": {"symbols": ["AUD/USD"], "digits": 5, "category": "forex"},
    "NZD/USD": {"symbols": ["NZD/USD"], "digits": 5, "category": "forex"},
    "USD/CAD": {"symbols": ["USD/CAD"], "digits": 5, "category": "forex"},
    "EUR/GBP": {"symbols": ["EUR/GBP"], "digits": 5, "category": "forex"},
    "GBP/JPY": {"symbols": ["GBP/JPY"], "digits": 3, "category": "forex"},
    "US100 — Nasdaq 100": {
        "symbols": ["NDX", "QQQ"],
        "digits": 2,
        "category": "index",
    },
    "US500 — S&P 500": {
        "symbols": ["SPX", "SPY"],
        "digits": 2,
        "category": "index",
    },
    "E-mini S&P 500 — ES": {
        "symbols": ["ES", "ES1!", "SPY"],
        "digits": 2,
        "category": "index",
    },
    "Micro E-mini S&P 500 — MES": {
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
        "Twelve Data API key not found.\n\n"
        "Add this in Streamlit Secrets:\n\n"
        'TWELVE_DATA_API_KEY = "your_key_here"'
    )
    st.stop()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(
    candidate_symbols: tuple[str, ...],
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
        required = ["datetime", "open", "high", "low", "close"]

        if not all(column in frame.columns for column in required):
            last_error = f"Incomplete candle data returned for {symbol}."
            continue

        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if "volume" in frame.columns:
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")

        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            errors="coerce",
            utc=True,
        )

        frame = (
            frame.dropna(subset=required)
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if len(frame) < 60:
            last_error = (
                f"Not enough candle history was returned for {symbol}. "
                "Try another interval or market."
            )
            continue

        return frame, symbol, None

    return None, None, last_error


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
    return (100 - (100 / (1 + relative_strength))).fillna(50)


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
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_histogram"] = data["macd"] - data["macd_signal"]
    data["atr"] = calculate_atr(data, 14)

    return data.dropna().reset_index(drop=True)


def calculate_support_resistance(
    data: pd.DataFrame,
    lookback: int = 50,
) -> tuple[float, float]:
    recent = data.tail(min(lookback, len(data)))
    return float(recent["low"].min()), float(recent["high"].max())


def analyse_market(data: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    bullish_points = 0
    bearish_points = 0
    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    price = float(latest["close"])
    atr = float(latest["atr"])
    rsi = float(latest["rsi"])

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
        bullish_reasons.append("Bullish momentum is improving")
    else:
        bearish_points += 10
        bearish_reasons.append("Bearish momentum is improving")

    if 52 <= rsi <= 70:
        bullish_points += 15
        bullish_reasons.append(f"RSI supports buyers at {rsi:.1f}")
    elif 30 <= rsi <= 48:
        bearish_points += 15
        bearish_reasons.append(f"RSI supports sellers at {rsi:.1f}")
    elif rsi > 70:
        bearish_points += 5
        bearish_reasons.append(f"RSI is overbought at {rsi:.1f}")
    elif rsi < 30:
        bullish_points += 5
        bullish_reasons.append(f"RSI is oversold at {rsi:.1f}")

    recent_closes = data["close"].tail(6)

    if recent_closes.iloc[-1] > recent_closes.iloc[0]:
        bullish_points += 15
        bullish_reasons.append("Recent price structure is rising")
    else:
        bearish_points += 15
        bearish_reasons.append("Recent price structure is falling")

    total = max(bullish_points + bearish_points, 1)
    bullish_percentage = round(bullish_points / total * 100)
    bearish_percentage = round(bearish_points / total * 100)
    difference = abs(bullish_percentage - bearish_percentage)

    if bullish_percentage >= 62 and difference >= 20:
        signal = "BUY"
        confidence = bullish_percentage
        reasons = bullish_reasons
    elif bearish_percentage >= 62 and difference >= 20:
        signal = "SELL"
        confidence = bearish_percentage
        reasons = bearish_reasons
    else:
        signal = "NO TRADE"
        confidence = max(bullish_percentage, bearish_percentage)
        reasons = [
            "Indicators are conflicting",
            "There is not enough confirmation for a quality setup",
        ]

    support, resistance = calculate_support_resistance(data)

    entry = price
    limit_entry = float(latest["ema_20"])
    stop_loss = None
    risk_distance = None
    tp1 = None
    tp2 = None
    tp3 = None

    if signal == "BUY":
        limit_entry = max(float(latest["ema_20"]), price - atr * 0.45)
        stop_loss = min(support - atr * 0.15, entry - atr * 1.35)
        risk_distance = entry - stop_loss
        tp1 = entry + risk_distance
        tp2 = entry + risk_distance * 2
        tp3 = entry + risk_distance * 3

    elif signal == "SELL":
        limit_entry = min(float(latest["ema_20"]), price + atr * 0.45)
        stop_loss = max(resistance + atr * 0.15, entry + atr * 1.35)
        risk_distance = stop_loss - entry
        tp1 = entry - risk_distance
        tp2 = entry - risk_distance * 2
        tp3 = entry - risk_distance * 3

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


def detect_reversal(data: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    previous = data.iloc[-2]

    price = float(latest["close"])
    atr = float(latest["atr"])
    support, resistance = calculate_support_resistance(data, lookback=80)

    bullish_points = 0
    bearish_points = 0
    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []

    if abs(price - support) <= atr * 1.5:
        bullish_points += 25
        bullish_reasons.append("Price is near a historical support zone")

    if abs(resistance - price) <= atr * 1.5:
        bearish_points += 25
        bearish_reasons.append("Price is near a historical resistance zone")

    if latest["rsi"] < 35:
        bullish_points += 20
        bullish_reasons.append(f"RSI is oversold at {latest['rsi']:.1f}")

    if latest["rsi"] > 65:
        bearish_points += 20
        bearish_reasons.append(f"RSI is overbought at {latest['rsi']:.1f}")

    bullish_confirmation_candle = (
        latest["close"] > latest["open"]
        and latest["close"] > previous["high"]
    )

    bearish_confirmation_candle = (
        latest["close"] < latest["open"]
        and latest["close"] < previous["low"]
    )

    if bullish_confirmation_candle:
        bullish_points += 25
        bullish_reasons.append("Bullish confirmation candle has formed")

    if bearish_confirmation_candle:
        bearish_points += 25
        bearish_reasons.append("Bearish confirmation candle has formed")

    recent = data.tail(12)
    price_change = recent["close"].iloc[-1] - recent["close"].iloc[0]
    rsi_change = recent["rsi"].iloc[-1] - recent["rsi"].iloc[0]

    if price_change < 0 and rsi_change > 4:
        bullish_points += 30
        bullish_reasons.append("Possible bullish RSI divergence detected")

    if price_change > 0 and rsi_change < -4:
        bearish_points += 30
        bearish_reasons.append("Possible bearish RSI divergence detected")

    if (
        latest["macd_histogram"] > previous["macd_histogram"]
        and latest["macd_histogram"] < 0
    ):
        bullish_points += 15
        bullish_reasons.append("Selling momentum appears to be weakening")

    if (
        latest["macd_histogram"] < previous["macd_histogram"]
        and latest["macd_histogram"] > 0
    ):
        bearish_points += 15
        bearish_reasons.append("Buying momentum appears to be weakening")

    if latest["ema_20"] > latest["ema_50"]:
        main_trend = "Bullish"
    elif latest["ema_20"] < latest["ema_50"]:
        main_trend = "Bearish"
    else:
        main_trend = "Neutral"

    confidence = min(max(bullish_points, bearish_points), 95)

    direction = "NONE"
    stage = "NO SETUP"
    action = "NO TRADE"
    title = "No strong reversal setup detected"
    zone_low = None
    zone_high = None
    confirmation = None
    invalidation = None
    reasons = [
        "Historical price action does not currently provide enough reversal evidence"
    ]

    if bullish_points >= 50 and bullish_points > bearish_points:
        direction = "BULLISH"
        zone_low = support
        zone_high = support + atr
        confirmation = max(float(previous["high"]), price + atr * 0.25)
        invalidation = support - atr * 0.4

        if latest["close"] < invalidation:
            stage = "INVALIDATED"
            action = "NO BUY"
            title = "Bullish reversal invalidated"
        elif bullish_confirmation_candle:
            stage = "CONFIRMED"
            action = "CONSIDER BUY"
            title = "Bullish reversal confirmed"
        else:
            stage = "FORMING"
            action = "WAIT FOR CONFIRMATION"
            title = "Potential bullish reversal forming"

        reasons = bullish_reasons

    elif bearish_points >= 50 and bearish_points > bullish_points:
        direction = "BEARISH"
        zone_low = resistance - atr
        zone_high = resistance
        confirmation = min(float(previous["low"]), price - atr * 0.25)
        invalidation = resistance + atr * 0.4

        if latest["close"] > invalidation:
            stage = "INVALIDATED"
            action = "NO SELL"
            title = "Bearish reversal invalidated"
        elif bearish_confirmation_candle:
            stage = "CONFIRMED"
            action = "CONSIDER SELL"
            title = "Bearish reversal confirmed"
        else:
            stage = "FORMING"
            action = "WAIT FOR CONFIRMATION"
            title = "Potential bearish reversal forming"

        reasons = bearish_reasons

    against_trend = (
        direction == "BULLISH" and main_trend == "Bearish"
    ) or (
        direction == "BEARISH" and main_trend == "Bullish"
    )

    if against_trend:
        reasons.append(
            f"This reversal is against the current {main_trend.lower()} trend"
        )

    return {
        "title": title,
        "direction": direction,
        "stage": stage,
        "action": action,
        "confidence": confidence,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "main_trend": main_trend,
        "against_trend": against_trend,
        "reasons": reasons[:6],
    }


def estimate_lot_size(
    category: str,
    risk_amount_gbp: float,
    stop_distance: Optional[float],
):
    if not stop_distance or stop_distance <= 0:
        return None

    if category == "gold":
        estimated_value_per_price_unit_per_lot_gbp = 79.0
    elif category == "forex":
        estimated_value_per_price_unit_per_lot_gbp = 79000.0
    else:
        estimated_value_per_price_unit_per_lot_gbp = 39.5

    loss_per_lot = (
        stop_distance * estimated_value_per_price_unit_per_lot_gbp
    )

    if loss_per_lot <= 0:
        return None

    raw_lot = risk_amount_gbp / loss_per_lot
    rounded_lot = math.floor(raw_lot * 100) / 100
    return max(rounded_lot, 0.01)



def quality_grade(score: int) -> tuple[str, str]:
    """Convert a score into the app's visible quality grade."""
    if score >= 95:
        return "A+", "⭐⭐⭐⭐⭐"
    if score >= 90:
        return "A", "⭐⭐⭐⭐"
    if score >= 80:
        return "B", "⭐⭐⭐"
    if score >= 70:
        return "C", "⭐⭐"
    return "REJECT", "❌"


def decision_from_analysis(analysis: dict, reversal: dict) -> str:
    """
    Step 1 decision layer.
    This is deliberately strict: it can say WAIT or NO TRADE even when
    the underlying direction is BUY/SELL.
    """
    signal = analysis["signal"]
    score = int(analysis["confidence"])
    atr = float(analysis["atr"])
    price = float(analysis["price"])
    limit_entry = float(analysis["limit_entry"])

    if signal == "NO TRADE" or score < 80:
        return "NO TRADE"

    reversal_conflict = (
        signal == "BUY"
        and reversal["direction"] == "BEARISH"
        and reversal["stage"] in {"FORMING", "CONFIRMED"}
    ) or (
        signal == "SELL"
        and reversal["direction"] == "BULLISH"
        and reversal["stage"] in {"FORMING", "CONFIRMED"}
    )

    if reversal_conflict:
        return "WAIT"

    distance_to_limit = abs(price - limit_entry)

    # If the preferred pullback is still meaningfully away, use a limit order.
    if distance_to_limit > atr * 0.20:
        return "SET BUY LIMIT" if signal == "BUY" else "SET SELL LIMIT"

    if score >= 90:
        return "ENTER BUY NOW" if signal == "BUY" else "ENTER SELL NOW"

    return "WAIT"


def proxy_warning(selected_name: str, working_symbol: str) -> Optional[str]:
    if "US100" in selected_name and working_symbol == "QQQ":
        return "US100 is currently using QQQ as a proxy because direct Nasdaq-100 index data was unavailable."
    if (
        ("US500" in selected_name or "E-mini" in selected_name or "Micro E-mini" in selected_name)
        and working_symbol == "SPY"
    ):
        return "This market is currently using SPY as a proxy, not the actual futures/index contract."
    return None


CORE_SCAN_MARKETS = [
    "Gold — XAU/USD",
    "US100 — Nasdaq 100",
    "US500 — S&P 500",
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
]


def scan_core_markets(
    interval: str,
    output_size: int,
    api_key: str,
) -> list[dict]:
    """
    Scan a compact list to stay within common Twelve Data API limits.
    Results are cached by fetch_candles.
    """
    rows: list[dict] = []

    for market_name in CORE_SCAN_MARKETS:
        settings = MARKETS[market_name]
        candles, symbol, error = fetch_candles(
            tuple(settings["symbols"]),
            interval,
            output_size,
            api_key,
        )

        if error or candles is None:
            rows.append(
                {
                    "Market": market_name,
                    "Rating": "Unavailable",
                    "Score": 0,
                    "Action": "NO DATA",
                    "Direction": "—",
                    "Symbol used": "—",
                    "_analysis": None,
                    "_reversal": None,
                }
            )
            continue

        processed = add_indicators(candles)
        if len(processed) < 30:
            continue

        market_analysis = analyse_market(processed)
        market_reversal = detect_reversal(processed)
        grade, stars = quality_grade(int(market_analysis["confidence"]))
        action = decision_from_analysis(market_analysis, market_reversal)

        rows.append(
            {
                "Market": market_name,
                "Rating": f"{stars} {grade}",
                "Score": int(market_analysis["confidence"]),
                "Action": action,
                "Direction": market_analysis["signal"],
                "Symbol used": symbol,
                "_analysis": market_analysis,
                "_reversal": market_reversal,
            }
        )

    return sorted(rows, key=lambda row: row["Score"], reverse=True)


with st.sidebar:
    st.header("Trade settings")

    selected_market = st.selectbox(
        "Market",
        list(MARKETS.keys()),
    )

    selected_interval_name = st.selectbox(
        "Chart timeframe",
        list(INTERVALS.keys()),
        index=2,
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
        "A short losing run can reduce the account quickly."
    )

    scan_best_trade = st.checkbox(
        "Scan core markets for best setup",
        value=True,
        help="Scans Gold, US100, US500 and selected major Forex pairs. "
        "Turn this off if your Twelve Data plan reaches its API limit.",
    )

    refresh_pressed = st.button(
        "🔄 Refresh analysis",
        use_container_width=True,
    )

if refresh_pressed:
    st.cache_data.clear()


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
        "Some futures symbols need a paid data subscription. "
        "The app may fall back to SPY."
    )
    st.stop()

data = add_indicators(candles)

if len(data) < 30:
    st.error("Not enough processed candle data to run the analysis.")
    st.stop()

analysis = analyse_market(data)
reversal = detect_reversal(data)

digits = market_settings["digits"]


def format_price(value):
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"



# =========================================================
# BEST TRADE AVAILABLE — STEP 1
# =========================================================

ranking_rows = []

if scan_best_trade:
    with st.spinner("Comparing the core markets..."):
        ranking_rows = scan_core_markets(
            interval=interval,
            output_size=min(history_size, 200),
            api_key=API_KEY,
        )

valid_rankings = [
    row for row in ranking_rows
    if row.get("_analysis") is not None
]

st.markdown("# 🏆 Best trade available")

if valid_rankings:
    best = valid_rankings[0]
    best_analysis = best["_analysis"]
    best_reversal = best["_reversal"]
    best_grade, best_stars = quality_grade(int(best_analysis["confidence"]))
    best_action = decision_from_analysis(best_analysis, best_reversal)

    st.markdown(
        f"## {best['Market']} — {best_stars} {best_grade} "
        f"({best_analysis['confidence']}/100)"
    )

    if best_action.startswith("ENTER BUY"):
        st.success(f"🟢 ACTION: {best_action}")
    elif best_action.startswith("ENTER SELL"):
        st.error(f"🔴 ACTION: {best_action}")
    elif "LIMIT" in best_action:
        st.warning(f"🟡 ACTION: {best_action}")
    elif best_action == "WAIT":
        st.warning("⏳ ACTION: WAIT")
    else:
        st.info("❌ ACTION: NO TRADE")

    best_digits = MARKETS[best["Market"]]["digits"]

    def best_price(value):
        if value is None:
            return "—"
        return f"{value:,.{best_digits}f}"

    best_entry = (
        best_analysis["limit_entry"]
        if "LIMIT" in best_action
        else best_analysis["entry"]
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Entry", best_price(best_entry))
    b2.metric("Stop loss", best_price(best_analysis["stop_loss"]))
    b3.metric("TP1", best_price(best_analysis["tp1"]))
    b4.metric("TP2", best_price(best_analysis["tp2"]))

    st.caption(
        f"Reversal: {best_reversal['title']} · "
        f"Data symbol: {best['Symbol used']}"
    )

    best_proxy = proxy_warning(best["Market"], best["Symbol used"])
    if best_proxy:
        st.warning(best_proxy)

    if best_grade not in {"A+", "A"}:
        st.info(
            "The highest-ranked setup is below A quality. "
            "Treat the result as WAIT/NO TRADE rather than forcing an entry."
        )

    display_rows = [
        {
            "Rank": index + 1,
            "Market": row["Market"],
            "Rating": row["Rating"],
            "Score": row["Score"],
            "Action": row["Action"],
            "Symbol": row["Symbol used"],
        }
        for index, row in enumerate(valid_rankings)
    ]

    st.markdown("### Market rankings")
    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        hide_index=True,
    )
else:
    selected_grade, selected_stars = quality_grade(int(analysis["confidence"]))
    selected_action = decision_from_analysis(analysis, reversal)

    st.markdown(
        f"## {selected_market} — {selected_stars} {selected_grade} "
        f"({analysis['confidence']}/100)"
    )
    st.info(
        f"ACTION: {selected_action}. "
        "Core-market scanning is switched off or no ranking data was available."
    )

st.divider()

st.subheader(f"{selected_market} · {selected_interval_name}")
st.caption(f"Data symbol currently working: `{working_symbol}`")

selected_proxy_warning = proxy_warning(selected_market, working_symbol)
if selected_proxy_warning:
    st.warning(selected_proxy_warning)

top_col1, top_col2, top_col3, top_col4 = st.columns(4)

top_col1.metric("Current price", format_price(analysis["price"]))
top_col2.metric("Signal", analysis["signal"])
top_col3.metric("Setup confidence", f"{analysis['confidence']}%")
top_col4.metric("RSI", f"{analysis['rsi']:.1f}")


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

st.plotly_chart(figure, use_container_width=True)


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

    tp_col1.metric("TP1 · 1:1", format_price(analysis["tp1"]))
    tp_col2.metric("TP2 · 1:2", format_price(analysis["tp2"]))
    tp_col3.metric("TP3 · 1:3", format_price(analysis["tp3"]))


st.subheader("💷 Risk and position size")

risk_amount = account_balance * (risk_percentage / 100)

estimated_lot = estimate_lot_size(
    market_settings["category"],
    risk_amount,
    analysis["risk_distance"],
)

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric("Account risk", f"£{risk_amount:,.2f}")
risk_col2.metric(
    "Stop distance",
    format_price(analysis["risk_distance"]),
)
risk_col3.metric(
    "Estimated maximum lot",
    "—" if estimated_lot is None else f"{estimated_lot:.2f}",
)

st.info(
    "Lot size is only an estimate because contract sizes, tick values "
    "and currency conversion vary by broker. Confirm the exact loss in MT5."
)


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


st.subheader("🔄 Potential reversal analysis")
st.caption(
    "This is additional historical-chart analysis and does not "
    "replace the main trade setup."
)

st.markdown(f"## {reversal['title']}")

status_col1, status_col2, status_col3 = st.columns(3)

status_col1.metric("Reversal stage", reversal["stage"])
status_col2.metric("Suggested action", reversal["action"])
status_col3.metric("Main trend", reversal["main_trend"])

if reversal["stage"] == "NO SETUP":
    st.info(
        "No reliable bullish or bearish reversal is currently detected."
    )
else:
    detail_col1, detail_col2 = st.columns(2)

    detail_col1.metric(
        "Historical pattern score",
        f"{reversal['confidence']}%",
    )
    detail_col2.metric(
        "Potential reversal zone",
        (
            f"{format_price(reversal['zone_low'])} – "
            f"{format_price(reversal['zone_high'])}"
        ),
    )

    level_col1, level_col2 = st.columns(2)

    level_col1.metric(
        "Candle-close confirmation",
        format_price(reversal["confirmation"]),
    )
    level_col2.metric(
        "Candle-close invalidation",
        format_price(reversal["invalidation"]),
    )

    if reversal["stage"] == "FORMING":
        st.warning(
            f"{reversal['action']} — wait for the current "
            f"{selected_interval_name.lower()} candle to close beyond "
            "the confirmation level."
        )
    elif reversal["stage"] == "CONFIRMED":
        st.success(
            f"{reversal['action']} — the latest candle has provided "
            "reversal confirmation. Check the main trade setup, stop "
            "loss and risk-to-reward before entering."
        )
    elif reversal["stage"] == "INVALIDATED":
        st.error(
            f"{reversal['action']} — price has moved beyond the "
            "invalidation level, so the reversal idea is no longer valid."
        )

    if reversal["against_trend"]:
        st.warning(
            "This reversal is against the current main trend, so it "
            "may be less reliable and needs stronger confirmation."
        )

    st.markdown("#### Historical evidence")
    for reason in reversal["reasons"]:
        st.write(f"• {reason}")


st.divider()
st.caption(
    "Educational trading research only. Prices may be delayed. "
    "Always confirm the broker price, spread, contract size, stop-loss "
    "cost and upcoming economic news before trading."
)
