# AI Trader - Intraday Research MVP

This project is an intraday trade-advisor sandbox for NSE/BSE stocks. It is built for research, backtesting, and paper trading first. Do not connect it to live orders until you have tested it with realistic costs, slippage, and strict risk limits.

## Best First Setup

- Market: NSE large-cap equities
- Universe: Nifty 50 or Nifty 100
- Timeframe: 5-minute or 15-minute candles
- Direction: long-only first
- Mode: backtest and paper trading only

## Data Format

Place CSV files in `data/raw/`. Each file should contain one symbol with these columns:

```csv
timestamp,symbol,open,high,low,close,volume
2026-05-20 09:15:00,RELIANCE,1430.00,1434.50,1428.20,1432.80,120000
```

Use exchange-local timestamps. For Indian equities, that means IST market time.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Zerodha Kite Setup

Create a Kite Connect app and set credentials in your shell:

```powershell
$env:KITE_API_KEY="your_api_key"
$env:KITE_API_SECRET="your_api_secret"
```

Print the daily login URL:

```powershell
python -m src.aitrader.kite_data login-url
```

Open the URL, log in, and copy the `request_token` from the redirect URL. Then create the local session:

```powershell
python -m src.aitrader.kite_data session --request-token "request_token_from_redirect"
```

Download NSE instrument metadata:

```powershell
python -m src.aitrader.kite_data instruments
```

Download 15-minute candles:

```powershell
python -m src.aitrader.kite_data candles --symbols RELIANCE,TCS,INFY --from "2026-01-01" --to "2026-05-21"
```

This saves files such as `data/raw/RELIANCE_15m.csv`, which the model can train on.

## FYERS Setup

Create a FYERS app with these permissions:

- Profile Details
- Transaction Info
- Historical Data
- Quotes & Market data

Store credentials in `src/aitrader/.env`:

```text
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=your_redirect_uri
FYERS_ACCESS_TOKEN=
```

Print the FYERS login URL:

```powershell
python -m src.aitrader.fyers_data login-url
```

Open the URL, log in, and copy either the `auth_code` or the full redirect URL. Then save the access token:

```powershell
python -m src.aitrader.fyers_data token --auth-code "auth_code_or_full_redirect_url"
```

Download 15-minute candles:

```powershell
python -m src.aitrader.fyers_data candles --symbols RELIANCE,TCS,INFY --from "2026-01-01" --to "2026-05-22"
```

Download, train, and print paper trade suggestions:

```powershell
python -m src.aitrader.run_fyers_pipeline --symbols RELIANCE,TCS,INFY --from "2026-01-01" --to "2026-05-22"
```

Train from already downloaded files:

```powershell
python -m src.aitrader.run_fyers_pipeline --from "2026-01-01" --to "2026-05-22" --skip-download
```

## Run A Backtest

```powershell
python -m src.aitrader.train --config config.yaml --data data/raw
```

## Get Latest Suggestion

```powershell
python -m src.aitrader.suggest --config config.yaml --data data/raw
```

`data/sample_intraday.csv` is included only to show the required CSV shape. For real testing, use at least several weeks of intraday candles. More is better.

## What The Model Does

The first version trains a simple classifier to estimate whether the next intraday holding window has enough upside after estimated costs. It uses technical features such as returns, moving averages, RSI, volatility, VWAP distance, volume change, and time of day.

The output is a decision-support signal:

- `BUY`: setup passed model confidence and risk filters
- `AVOID`: setup did not pass

It also prints confidence, suggested stop-loss, target, and rough position sizing based on config risk.

## Important Risk Notes

Intraday trading is noisy. A model can look good in a weak backtest and still fail live because of overfitting, brokerage, taxes, slippage, poor fills, latency, and sudden news. Start with paper trading and keep position sizing small.

For Indian markets, use SEBI-registered intermediaries and understand fees, margins, and risk before trading.
