"""
run_backtest.py — CLI entry point for local execution.

Usage:
    python run_backtest.py                    # default: 5 years, both strategies
    python run_backtest.py --years 3          # 3 years of data
    python run_backtest.py --csv my_data.csv  # load from local CSV
"""

import argparse
from src.backtest import NiftyDataLoader, DataEngineering, SMA9Strategy, EMA9Strategy, Backtester


def main():
    parser = argparse.ArgumentParser(description="Nifty 50 Options Backtest")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")
    parser.add_argument("--csv",   type=str, default=None, help="Path to local CSV file (optional)")
    parser.add_argument("--capital", type=int, default=500_000, help="Starting capital in INR (default: 500000)")
    args = parser.parse_args()

    # ── Step 1: Load Data ──────────────────────────────────
    loader = NiftyDataLoader(years=args.years)
    df_raw = loader.load_from_csv(args.csv) if args.csv else loader.load()

    # ── Step 2: Feature Engineering ────────────────────────
    eng = DataEngineering(df_raw)
    df  = eng.build()

    # ── Step 3: Backtest ────────────────────────────────────
    bt = Backtester(df)
    bt.INITIAL_CAPITAL = args.capital
    bt.add_strategy(SMA9Strategy())
    bt.add_strategy(EMA9Strategy())
    bt.run()

    # ── Step 4: Results ─────────────────────────────────────
    bt.print_summary()
    bt.save_trade_logs(out_dir="results")
    bt.plot_results(save_path="results/backtest_comparison.png")

    print("\n✅ Done. Check the 'results/' folder for CSVs and chart.")


if __name__ == "__main__":
    main()
