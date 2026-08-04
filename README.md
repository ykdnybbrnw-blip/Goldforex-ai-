
# GoldForex AI

A mobile-friendly Streamlit research dashboard for Gold and major Forex pairs.

## Features

- BUY / SELL / WAIT research signal
- Edge score
- Multi-timeframe trend confirmation
- RSI, MACD, ATR and market-structure checks
- UK daytime trading filter
- Entry, stop and target research levels
- Suggested lot size
- Planned risk, potential reward and approximate margin
- Interactive candlestick chart

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is an educational research tool, not financial advice.

Yahoo Finance Gold data uses `GC=F`, which is Gold futures rather than your broker's exact XAU/USD quote. Contract sizes, spreads, margin rules and minimum lot steps can vary by broker.
