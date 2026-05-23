from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    load_env_file(config["fyers"].get("env_path", "src/aitrader/.env"))
    return config


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_value(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} in your environment or src/aitrader/.env first.")
    return value


def import_fyers_model():
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as exc:
        raise ImportError("Install FYERS SDK first: pip install fyers-apiv3") from exc
    return fyersModel


def clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if ":" not in cleaned:
        cleaned = f"NSE:{cleaned}"
    if "-" not in cleaned:
        cleaned = f"{cleaned}-EQ"
    return cleaned


def file_symbol(symbol: str) -> str:
    return clean_symbol(symbol).replace(":", "_").replace("-", "_")


def resolution_from_minutes(minutes: int) -> str:
    if minutes not in {1, 2, 3, 5, 10, 15, 20, 30, 60}:
        raise ValueError("FYERS intraday resolution must be 1, 2, 3, 5, 10, 15, 20, 30, or 60.")
    return str(minutes)


def print_login_url(config: dict) -> None:
    fyers_model = import_fyers_model()
    session = fyers_model.SessionModel(
        client_id=env_value("FYERS_CLIENT_ID"),
        secret_key=env_value("FYERS_SECRET_KEY"),
        redirect_uri=env_value("FYERS_REDIRECT_URI"),
        response_type="code",
        grant_type="authorization_code",
        state="ai-trader",
    )
    print(session.generate_authcode())


def extract_auth_code(value: str) -> str:
    if "://" not in value:
        return value
    parsed = urlparse(value)
    params = parse_qs(parsed.query)
    auth_code = params.get("auth_code") or params.get("code")
    if not auth_code:
        raise ValueError("Could not find auth_code/code in redirect URL.")
    return auth_code[0]


def create_access_token(config: dict, auth_code_or_url: str) -> None:
    fyers_model = import_fyers_model()
    session = fyers_model.SessionModel(
        client_id=env_value("FYERS_CLIENT_ID"),
        secret_key=env_value("FYERS_SECRET_KEY"),
        redirect_uri=env_value("FYERS_REDIRECT_URI"),
        response_type="code",
        grant_type="authorization_code",
        state="ai-trader",
    )
    session.set_token(extract_auth_code(auth_code_or_url))
    response = session.generate_token()
    access_token = response.get("access_token")
    if not access_token:
        raise ValueError(f"FYERS did not return access_token. Response: {response}")

    env_path = Path(config["fyers"].get("env_path", "src/aitrader/.env"))
    update_env_file(env_path, "FYERS_ACCESS_TOKEN", access_token)
    print(f"Saved FYERS_ACCESS_TOKEN to {env_path}")


def update_env_file(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated = False
    result = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            result.append(f"{key}={value}")
            updated = True
        else:
            result.append(line)
    if not updated:
        result.append(f"{key}={value}")
    path.write_text("\n".join(result) + "\n", encoding="utf-8")


def make_client():
    fyers_model = import_fyers_model()
    return fyers_model.FyersModel(
        client_id=env_value("FYERS_CLIENT_ID"),
        token=env_value("FYERS_ACCESS_TOKEN"),
        is_async=False,
        log_path="",
    )


def split_date_ranges(from_date: str, to_date: str, days: int = 90) -> list[tuple[str, str]]:
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date, "%Y-%m-%d").date()
    ranges = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        ranges.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return ranges


def download_candles(config: dict, symbols: list[str], from_date: str, to_date: str) -> None:
    fyers = make_client()
    resolution = resolution_from_minutes(int(config["market"]["timeframe_minutes"]))
    raw_dir = Path(config["fyers"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        fyers_symbol = clean_symbol(symbol)
        frames = []
        for chunk_from, chunk_to in split_date_ranges(from_date, to_date):
            payload = {
                "symbol": fyers_symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": chunk_from,
                "range_to": chunk_to,
                "cont_flag": "1",
            }
            response = fyers.history(data=payload)
            if response.get("s") != "ok":
                raise ValueError(f"FYERS history failed for {fyers_symbol}: {response}")
            frames.append(normalise_history(response.get("candles", []), fyers_symbol))

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df = df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")
        path = raw_dir / f"{file_symbol(fyers_symbol)}_{config['market']['timeframe_minutes']}m.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} candles for {fyers_symbol} to {path}")


def normalise_history(candles: list[list], symbol: str) -> pd.DataFrame:
    columns = ["epoch", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(candles, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    timestamp = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return pd.DataFrame(
        {
            "timestamp": timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "volume": df["volume"],
        }
    )


def parse_symbols(value: str | None, config: dict) -> list[str]:
    if value:
        return [clean_symbol(symbol) for symbol in value.split(",") if symbol.strip()]
    return [clean_symbol(symbol) for symbol in config["fyers"]["default_symbols"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="FYERS data utilities")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login-url", help="Print FYERS login URL")

    token_parser = subparsers.add_parser("token", help="Create FYERS access token from auth code or redirect URL")
    token_parser.add_argument("--auth-code", required=True)

    candles_parser = subparsers.add_parser("candles", help="Download historical candles")
    candles_parser.add_argument("--symbols", help="Comma-separated symbols, e.g. RELIANCE,TCS or NSE:RELIANCE-EQ")
    candles_parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    candles_parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "login-url":
        print_login_url(config)
    elif args.command == "token":
        create_access_token(config, args.auth_code)
    elif args.command == "candles":
        download_candles(config, parse_symbols(args.symbols, config), args.from_date, args.to_date)


if __name__ == "__main__":
    main()
