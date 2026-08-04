
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="GoldForex AI",
    page_icon="📈",
    layout="wide",
)

MARKETS = {
    "Gold": {
        "ticker": "GC=F",
        "contract_size": 100.0,
        "price_decimals": 2,
    },
    "EUR/USD": {
        "ticker": "EURUSD=X",
        "contract_size": 100000.0,
        "price_decimals": 5,
    },
    "GBP/USD": {
        "ticker": "GBPUSD=X",
        "contract_size": 100000.0,
        "price_decimals": 5,
    },
    "USD/JPY": {
        "ticker": "JPY=X",
        "contract_size": 100000.0,
        "price_decimals": 3,
    },
}

TIMEFRAMES = {
    "15 minutes": ("15m", "60d"),
    "30 minutes": ("30m", "60d"),
    "1 hour": ("1h", "2y"),
}

st.title("GoldForex AI")
st.caption("Research dashboard only. It does not place trades and cannot guarantee profits.")


@st.cache_data(ttl=120)
def download_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("No data was returned by Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    return data[required].dropna().copy()


@st.cache_data(ttl=120)
def latest_gbpusd() -> float:
    data = download_data("GBPUSD=X", "15m", "5d")
    return float(data["Close"].iloc[-1])


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()

    previous_close = data["Close"].shift(1)
    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR"] = true_range.rolling(14).mean()
    data["RecentHigh"] = data["High"].rolling(10).max().shift(1)
    data["RecentLow"] = data["Low"].rolling(10).min().shift(1)
    data["Support"] = data["Low"].rolling(20).min()
    data["Resistance"] = data["High"].rolling(20).max()

    return data.dropna().copy()


def uk_session() -> tuple[str, bool, datetime]:
    now = datetime.now(ZoneInfo("Europe/London"))
    minutes = now.hour * 60 + now.minute

    if 8 * 60 <= minutes <= 11 * 60:
        return "London focus window", True, now

    if 13 * 60 + 30 <= minutes <= 16 * 60 + 30:
        return "London/New York overlap", True, now

    return "Outside preferred hours", False, now


def score_setup(lower: pd.DataFrame, higher: pd.DataFrame) -> dict:
    row = lower.iloc[-1]
    hrow = higher.iloc[-1]

    bull = 0
    bear = 0
    reasons = []
    cautions = []

    if hrow["EMA20"] > hrow["EMA50"] > hrow["EMA200"]:
        bull += 25
        reasons.append("Higher timeframe trend is bullish")
    elif hrow["EMA20"] < hrow["EMA50"] < hrow["EMA200"]:
        bear += 25
        reasons.append("Higher timeframe trend is bearish")
    else:
        cautions.append("Higher timeframe trend is mixed")

    if row["EMA20"] > row["EMA50"] > row["EMA200"]:
        bull += 20
        reasons.append("Current timeframe EMAs are bullish")
    elif row["EMA20"] < row["EMA50"] < row["EMA200"]:
        bear += 20
        reasons.append("Current timeframe EMAs are bearish")
    else:
        cautions.append("Current timeframe EMAs are mixed")

    if row["RSI"] >= 55:
        bull += 15
        reasons.append("RSI confirms bullish momentum")
    elif row["RSI"] <= 45:
        bear += 15
        reasons.append("RSI confirms bearish momentum")
    else:
        cautions.append("RSI is neutral")

    if row["MACD"] > row["MACD_SIGNAL"]:
        bull += 15
        reasons.append("MACD is bullish")
    else:
        bear += 15
        reasons.append("MACD is bearish")

    if row["Close"] > row["RecentHigh"]:
        bull += 15
        reasons.append("Price broke above a recent high")
    elif row["Close"] < row["RecentLow"]:
        bear += 15
        reasons.append("Price broke below a recent low")
    else:
        cautions.append("No fresh market-structure break")

    if row["Close"] > row["EMA20"]:
        bull += 10
    else:
        bear += 10

    direction = "WAIT"
    score = max(bull, bear)

    if bull >= 75 and bull > bear:
        direction = "BUY"
    elif bear >= 75 and bear > bull:
        direction = "SELL"

    return {
        "direction": direction,
        "score": int(score),
        "bull": bull,
        "bear": bear,
        "price": float(row["Close"]),
        "atr": float(row["ATR"]),
        "rsi": float(row["RSI"]),
        "support": float(row["Support"]),
        "resistance": float(row["Resistance"]),
        "reasons": reasons,
        "cautions": cautions,
    }


def make_trade_plan(
    result: dict,
    market_name: str,
    balance_gbp: float,
    risk_percent: float,
    reward_ratio: float,
    leverage: int,
) -> dict | None:
    if result["direction"] == "WAIT":
        return None

    price = result["price"]
    atr = result["atr"]

    if result["direction"] == "BUY":
        stop = price - 1.5 * atr
        target = price + (price - stop) * reward_ratio
    else:
        stop = price + 1.5 * atr
        target = price - (stop - price) * reward_ratio

    risk_gbp = balance_gbp * risk_percent / 100
    gbpusd = latest_gbpusd()
    risk_usd = risk_gbp * gbpusd

    contract_size = MARKETS[market_name]["contract_size"]
    stop_distance = abs(price - stop)
    raw_lots = risk_usd / (stop_distance * contract_size)
    lots = max(0.0, math.floor(raw_lots * 100) / 100)

    notional_usd = lots * contract_size * price
    margin_gbp = (notional_usd / max(leverage, 1)) / gbpusd

    return {
        "entry": price,
        "stop": stop,
        "target": target,
        "risk_gbp": risk_gbp,
        "reward_gbp": risk_gbp * reward_ratio,
        "raw_lots": raw_lots,
        "lots": lots,
        "margin_gbp": margin_gbp,
    }


def candlestick_chart(data: pd.DataFrame, market_name: str) -> go.Figure:
    chart = data.tail(180)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="Price",
        )
    )

    fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA20"], name="EMA 20"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA50"], name="EMA 50"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA200"], name="EMA 200"))

    fig.update_layout(
        title=f"{market_name} price and trend",
        height=560,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


with st.sidebar:
    st.header("Settings")

    market_name = st.selectbox("Market", list(MARKETS.keys()), index=0)
    timeframe_name = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=0)

    balance_gbp = st.number_input(
        "Account balance (£)",
        min_value=1.0,
        value=150.0,
        step=10.0,
    )

    risk_percent = st.slider(
        "Risk per trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    reward_ratio = st.slider(
        "Reward-to-risk",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.25,
    )

    leverage = st.number_input(
        "Leverage",
        min_value=1,
        max_value=1000,
        value=500,
        step=1,
    )

    use_session_filter = st.checkbox(
        "Use UK daytime filter",
        value=True,
    )

    analyse = st.button("Analyse market", type="primary", use_container_width=True)


if analyse:
    try:
        interval, period = TIMEFRAMES[timeframe_name]
        ticker = MARKETS[market_name]["ticker"]

        with st.spinner("Downloading market data and analysing the setup..."):
            lower = add_indicators(download_data(ticker, interval, period))

            higher_interval = "1h" if interval in ("15m", "30m") else "1d"
            higher_period = "2y"
            higher = add_indicators(
                download_data(ticker, higher_interval, higher_period)
            )

            result = score_setup(lower, higher)
            session_name, session_ok, now = uk_session()

            final_signal = result["direction"]
            if use_session_filter and not session_ok:
                final_signal = "WAIT"

            plan = None
            if final_signal != "WAIT":
                plan = make_trade_plan(
                    result,
                    market_name,
                    balance_gbp,
                    risk_percent,
                    reward_ratio,
                    int(leverage),
                )

        signal_col, score_col, price_col, session_col = st.columns(4)

        signal_col.metric("Research signal", final_signal)
        score_col.metric("Edge score", f"{result['score']}/100")
        price_col.metric("Yahoo price", f"{result['price']:.{MARKETS[market_name]['price_decimals']}f}")
        session_col.metric("UK session", session_name)

        if final_signal == "WAIT":
            st.warning(
                "No entry plan is shown because the setup does not currently pass every filter."
            )
        elif plan is not None:
            st.subheader("Trade plan")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", f"{plan['entry']:.{MARKETS[market_name]['price_decimals']}f}")
            c2.metric("Stop", f"{plan['stop']:.{MARKETS[market_name]['price_decimals']}f}")
            c3.metric("Target", f"{plan['target']:.{MARKETS[market_name]['price_decimals']}f}")
            c4.metric("Suggested lot size", f"{plan['lots']:.2f}")

            c5, c6, c7 = st.columns(3)
            c5.metric("Planned risk", f"£{plan['risk_gbp']:.2f}")
            c6.metric("Potential reward", f"£{plan['reward_gbp']:.2f}")
            c7.metric("Approx. margin", f"£{plan['margin_gbp']:.2f}")

            if plan["lots"] < 0.01:
                st.error(
                    "The calculated size is below 0.01 lots. A 0.01-lot trade may risk more than your selected amount."
                )

        st.subheader("Why")

        for reason in result["reasons"]:
            st.write(f"✅ {reason}")

        if use_session_filter and not session_ok:
            st.write("⚠️ Outside your preferred UK trading hours")

        for caution in result["cautions"]:
            st.write(f"⚠️ {caution}")

        st.plotly_chart(
            candlestick_chart(lower, market_name),
            use_container_width=True,
        )

        st.subheader("Important limitations")
        st.info(
            "Yahoo Finance gold uses GC=F futures rather than your broker's exact XAU/USD quote. "
            "The lot-size calculator assumes 100 oz per standard Gold lot and 100,000 units per standard Forex lot. "
            "Check the contract specification inside 1xTrade before using any lot-size figure live."
        )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

else:
    st.info("Choose your settings, then press **Analyse market**.")
