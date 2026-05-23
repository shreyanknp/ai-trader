from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from kiteconnect import KiteConnect


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interval_from_minutes(minutes: int) -> str:
    supported = {1, 3, 5, 10, 15, 30, 60}
    if minutes not in supported:
        raise ValueError(f"Kite candle interval must be one of {sorted(supported)} minutes.")
    return "minute" if minutes == 1 else f"{minutes}minute"


def env_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set environment variable {name} first.")
    return value


def make_client(config: dict, require_access_token: bool = True) -> KiteConnect:
    kite = KiteConnect(api_key=env_value("KITE_API_KEY"))
    if require_access_token:
        session_path = Path(config["kite"]["session_path"])
        if not session_path.exists():
            raise FileNotFoundError(
                f"No Kite session found at {session_path}. Run the session command first."
            )
        session = json.loads(session_path.read_text(encoding="utf-8"))
        kite.set_access_token(session["access_token"])
    return kite


def print_login_url(config: dict) -> None:
    kite = make_client(config, require_access_token=False)
    print(kite.login_url())


def create_session(config: dict, request_token: str) -> None:
    kite = make_client(config, require_access_token=False)
    data = kite.generate_session(request_token, api_secret=env_value("KITE_API_SECRET"))
    session = {
        "access_token": data["access_token"],
        "public_token": data.get("public_token"),
        "user_id": data.get("user_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    session_path = Path(config["kite"]["session_path"])
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    print(f"Saved Kite session to {session_path}")


def download_instruments(config: dict) -> pd.DataFrame:
    kite = make_client(config)
    exchange = config["market"]["exchange"]
    instruments = pd.DataFrame(kite.instruments(exchange))
    if instruments.empty:
        raise ValueError(f"No instruments returned for exchange {exchange}.")

    path = Path(config["kite"]["instruments_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    instruments.to_csv(path, index=False)
    print(f"Saved {len(instruments)} {exchange} instruments to {path}")
    return instruments


def load_or_download_instruments(config: dict) -> pd.DataFrame:
    path = Path(config["kite"]["instruments_path"])
    if path.exists():
        return pd.read_csv(path)
    return download_instruments(config)


def lookup_instrument_token(instruments: pd.DataFrame, symbol: str) -> int:
    matches = instruments[instruments["tradingsymbol"].str.upper() == symbol.upper()]
    if matches.empty:
        raise ValueError(f"Could not find symbol {symbol} in instruments file.")
    equity_matches = matches[matches["instrument_type"].eq("EQ")]
    row = equity_matches.iloc[0] if not equity_matches.empty else matches.iloc[0]
    return int(row["instrument_token"])


def normalise_candles(candles: list[dict], symbol: str) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(candles)
    timestamp = pd.to_datetime(df["date"])
    if getattr(timestamp.dt, "tz", None) is not None:
        timestamp = timestamp.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol.upper(),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
        }
    )
    return out


def download_candles(config: dict, symbols: list[str], from_date: str, to_date: str) -> None:
    kite = make_client(config)
    instruments = load_or_download_instruments(config)
    interval = interval_from_minutes(int(config["market"]["timeframe_minutes"]))
    from_date = normalise_datetime_arg(from_date, config["market"]["session_start"])
    to_date = normalise_datetime_arg(to_date, config["market"]["session_end"])
    raw_dir = Path(config["kite"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        token = lookup_instrument_token(instruments, symbol)
        candles = kite.historical_data(token, from_date, to_date, interval)
        df = normalise_candles(candles, symbol)
        path = raw_dir / f"{symbol.upper()}_{config['market']['timeframe_minutes']}m.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} candles for {symbol.upper()} to {path}")


def normalise_datetime_arg(value: str, fallback_time: str) -> str:
    if len(value.strip()) == 10:
        return f"{value.strip()} {fallback_time}:00"
    return value


def parse_symbols(value: str) -> list[str]:
    symbols = [symbol.strip().upper() for symbol in value.split(",")]
    return [symbol for symbol in symbols if symbol]


def main() -> None:
    parser = argparse.ArgumentParser(description="Zerodha Kite data utilities")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login-url", help="Print the Kite login URL")

    session_parser = subparsers.add_parser("session", help="Create and save today's access token")
    session_parser.add_argument("--request-token", required=True)

    subparsers.add_parser("instruments", help="Download and cache exchange instruments")

    candles_parser = subparsers.add_parser("candles", help="Download historical candles")
    candles_parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g. RELIANCE,TCS")
    candles_parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    candles_parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "login-url":
        print_login_url(config)
    elif args.command == "session":
        create_session(config, args.request_token)
    elif args.command == "instruments":
        download_instruments(config)
    elif args.command == "candles":
        download_candles(config, parse_symbols(args.symbols), args.from_date, args.to_date)


if __name__ == "__main__":
    main()
