from __future__ import annotations

import argparse

from .fyers_data import download_candles, load_config, parse_symbols
from .suggest import latest_suggestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FYERS candles, train, and print paper suggestions")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", help="Comma-separated symbols, e.g. RELIANCE,TCS or NSE:RELIANCE-EQ")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--data", default="data/raw", help="CSV file, folder, glob, or comma-separated paths")
    parser.add_argument("--skip-download", action="store_true", help="Train from existing local CSV files")
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.skip_download:
        download_candles(config, parse_symbols(args.symbols, config), args.from_date, args.to_date)

    latest_suggestion(args.data, args.config)


if __name__ == "__main__":
    main()
