"""
=============================================================
  NIFTY 50 EOD OPTIONS BACKTEST — OOP FRAMEWORK
=============================================================
  Class 1 : NiftyDataLoader      — fetch / load EOD data
  Class 2 : DataEngineering      — feature engineering
  Class 3a: SMA9Strategy         — 9-day Simple MA signals
  Class 3b: EMA9Strategy         — 9-day Exp MA signals
  Class 4 : Backtester           — run + analyse results
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CLASS 1 : NiftyDataLoader
# ─────────────────────────────────────────────────────────────
class NiftyDataLoader:
    """
    Fetches Nifty 50 EOD (End-of-Day) OHLCV data.

    Usage
    -----
    loader = NiftyDataLoader(years=5)
    df = loader.load()

    You can also load from a local CSV:
    df = loader.load_from_csv("nifty_data.csv")
    """

    TICKER = "^NSEI"   # Yahoo Finance symbol for Nifty 50

    def __init__(self, years: int = 5):
        self.years  = years
        self.end    = datetime.today()
        self.start  = self.end - timedelta(days=years * 365)

    # ── public ──────────────────────────────────────────────
    def load(self) -> pd.DataFrame:
        """Download data from Yahoo Finance and return clean OHLCV DataFrame."""
        print(f"[DataLoader] Downloading Nifty 50 data ({self.years} years) …")
        raw = yf.download(
            self.TICKER,
            start=self.start.strftime("%Y-%m-%d"),
            end=self.end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError("Download failed — check internet connection.")

        df = self._clean(raw)
        print(f"[DataLoader] Loaded {len(df)} trading days  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load EOD data from a local CSV (must have Date + Close columns)."""
        print(f"[DataLoader] Reading from {path} …")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.columns = [c.capitalize() for c in df.columns]
        df = df[["Open", "Close", "High", "Low", "Volume"]].dropna()
        df.sort_index(inplace=True)
        print(f"[DataLoader] Loaded {len(df)} rows from CSV")
        return df

    # ── private ─────────────────────────────────────────────
    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level columns, keep OHLCV, drop NaN."""
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df


# ─────────────────────────────────────────────────────────────
# CLASS 2 : DataEngineering
# ─────────────────────────────────────────────────────────────
class DataEngineering:
    """
    Feature engineering on raw OHLCV data.

    Adds
    ----
    SMA_9, EMA_9          : moving averages
    ATR_14                : average true range (for option sizing)
    Monthly_Expiry        : bool flag — last Thursday of each month
    ATM_Strike            : nearest 50-pt strike to Close
    Daily_Return          : % change day-on-day
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def build(self) -> pd.DataFrame:
        print("[DataEngineering] Building features …")
        df = self.df

        # Moving averages
        df["SMA_9"] = df["Close"].rolling(window=9).mean()
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()

        # ATR (Average True Range)
        high_low   = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close  = (df["Low"]  - df["Close"].shift()).abs()
        df["ATR_14"] = pd.concat([high_low, high_close, low_close], axis=1)\
                         .max(axis=1).rolling(14).mean()

        # Daily return
        df["Daily_Return"] = df["Close"].pct_change() * 100

        # ATM strike (nearest 50-point round)
        df["ATM_Strike"] = (df["Close"] / 50).round() * 50

        # Monthly expiry flag (last Thursday of each month)
        df["Monthly_Expiry"] = df.index.to_series().apply(self._last_thursday)

        df.dropna(inplace=True)
        print(f"[DataEngineering] Features ready — {len(df)} rows, "
              f"{df['Monthly_Expiry'].sum()} monthly expiry dates")
        return df

    # ── private ─────────────────────────────────────────────
    @staticmethod
    def _last_thursday(date: pd.Timestamp) -> bool:
        """True if 'date' is the last Thursday of its calendar month."""
        if date.weekday() != 3:       # 3 = Thursday
            return False
        # Check if next Thursday falls in a different month
        next_thu = date + timedelta(weeks=1)
        return next_thu.month != date.month


# ─────────────────────────────────────────────────────────────
# BASE STRATEGY (abstract)
# ─────────────────────────────────────────────────────────────
class _BaseStrategy:
    """
    Abstract base — subclasses implement generate_signals().
    Signal column: +1 = Buy Call ATM, -1 = Buy Put ATM, 0 = No trade
    """

    # ── option premium model ─────────────────────────────────
    @staticmethod
    def estimate_premium(spot: float, atr: float, days_to_expiry: int) -> float:
        """
        Simple ATR-based ATM option premium estimate.
        Premium ≈ 0.4 × ATR × sqrt(DTE)
        (rough Black-Scholes approximation for backtesting)
        """
        dte   = max(days_to_expiry, 1)
        prem  = 0.4 * atr * np.sqrt(dte)
        return round(max(prem, spot * 0.004), 2)   # floor at 0.4% of spot

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# CLASS 3a : SMA9Strategy
# ─────────────────────────────────────────────────────────────
class SMA9Strategy(_BaseStrategy):
    """
    Strategy 1 — Simple Moving Average (9-day)
    -------------------------------------------
    Entry rule  : on every monthly expiry day
      Close > SMA_9  →  Buy ATM Call
      Close < SMA_9  →  Buy ATM Put
    Exit  rule  : hold until next monthly expiry (exit at open)
    """

    name = "SMA-9 Strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[{self.name}] Generating signals …")
        trades = []
        expiry_dates = df[df["Monthly_Expiry"]].index.tolist()

        for i, entry_date in enumerate(expiry_dates[:-1]):
            exit_date = expiry_dates[i + 1]
            row       = df.loc[entry_date]

            close, sma = row["Close"], row["SMA_9"]
            signal = 1 if close > sma else -1        # 1=Call, -1=Put
            option = "CALL" if signal == 1 else "PUT"
            strike = row["ATM_Strike"]
            dte    = (exit_date - entry_date).days

            entry_prem = self.estimate_premium(close, row["ATR_14"], dte)
            exit_close = df.loc[exit_date, "Close"]
            exit_prem  = self._exit_premium(close, exit_close, signal,
                                             entry_prem, df.loc[exit_date, "ATR_14"])

            pnl        = (exit_prem - entry_prem) * 50   # 1 lot = 50 qty
            pnl_pct    = (exit_prem - entry_prem) / entry_prem * 100

            trades.append({
                "strategy"    : self.name,
                "entry_date"  : entry_date,
                "exit_date"   : exit_date,
                "signal"      : option,
                "strike"      : int(strike),
                "entry_spot"  : round(close, 2),
                "sma9"        : round(sma, 2),
                "entry_prem"  : round(entry_prem, 2),
                "exit_prem"   : round(exit_prem, 2),
                "pnl"         : round(pnl, 2),
                "pnl_pct"     : round(pnl_pct, 2),
                "dte"         : dte,
            })

        return pd.DataFrame(trades)

    # ── private ─────────────────────────────────────────────
    @staticmethod
    def _exit_premium(entry_spot, exit_spot, signal, entry_prem, exit_atr):
        """Estimate exit premium based on how much spot moved in signal direction."""
        move   = (exit_spot - entry_spot) * signal   # positive = favourable
        profit = move * 0.6                           # delta ≈ 0.5 ATM, decayed
        return max(entry_prem + profit, 0)


# ─────────────────────────────────────────────────────────────
# CLASS 3b : EMA9Strategy
# ─────────────────────────────────────────────────────────────
class EMA9Strategy(_BaseStrategy):
    """
    Strategy 2 — Exponential Moving Average (9-day)
    -------------------------------------------------
    Same logic as SMA9Strategy but uses EMA_9 instead.
    EMA reacts faster to recent price changes — fewer false signals
    in trending markets.

    Entry rule  : on every monthly expiry day
      Close > EMA_9  →  Buy ATM Call
      Close < EMA_9  →  Buy ATM Put
    Exit  rule  : hold until next monthly expiry
    """

    name = "EMA-9 Strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[{self.name}] Generating signals …")
        trades = []
        expiry_dates = df[df["Monthly_Expiry"]].index.tolist()

        for i, entry_date in enumerate(expiry_dates[:-1]):
            exit_date = expiry_dates[i + 1]
            row       = df.loc[entry_date]

            close, ema = row["Close"], row["EMA_9"]
            signal = 1 if close > ema else -1
            option = "CALL" if signal == 1 else "PUT"
            strike = row["ATM_Strike"]
            dte    = (exit_date - entry_date).days

            entry_prem = self.estimate_premium(close, row["ATR_14"], dte)
            exit_close = df.loc[exit_date, "Close"]
            exit_prem  = self._exit_premium(close, exit_close, signal,
                                             entry_prem, df.loc[exit_date, "ATR_14"])

            pnl     = (exit_prem - entry_prem) * 50
            pnl_pct = (exit_prem - entry_prem) / entry_prem * 100

            trades.append({
                "strategy"    : self.name,
                "entry_date"  : entry_date,
                "exit_date"   : exit_date,
                "signal"      : option,
                "strike"      : int(strike),
                "entry_spot"  : round(close, 2),
                "ema9"        : round(ema, 2),
                "entry_prem"  : round(entry_prem, 2),
                "exit_prem"   : round(exit_prem, 2),
                "pnl"         : round(pnl, 2),
                "pnl_pct"     : round(pnl_pct, 2),
                "dte"         : dte,
            })

        return pd.DataFrame(trades)

    @staticmethod
    def _exit_premium(entry_spot, exit_spot, signal, entry_prem, exit_atr):
        move   = (exit_spot - entry_spot) * signal
        profit = move * 0.6
        return max(entry_prem + profit, 0)


# ─────────────────────────────────────────────────────────────
# CLASS 4 : Backtester
# ─────────────────────────────────────────────────────────────
class Backtester:
    """
    Runs one or more strategies and produces comparative results.

    Usage
    -----
    bt = Backtester(df_engineered)
    bt.add_strategy(SMA9Strategy())
    bt.add_strategy(EMA9Strategy())
    bt.run()
    bt.print_summary()
    bt.plot_results()
    """

    INITIAL_CAPITAL = 500_000   # ₹5 lakh starting capital

    def __init__(self, df: pd.DataFrame):
        self.df         = df
        self.strategies = []
        self.results    = {}    # strategy_name → trades DataFrame

    # ── public ──────────────────────────────────────────────
    def add_strategy(self, strategy: _BaseStrategy):
        self.strategies.append(strategy)

    def run(self):
        """Run all strategies and cache trade logs."""
        if not self.strategies:
            raise ValueError("No strategies added. Use add_strategy() first.")
        for strat in self.strategies:
            trades          = strat.generate_signals(self.df)
            trades          = self._add_cumulative(trades)
            self.results[strat.name] = trades
        print("[Backtester] All strategies complete.")

    def print_summary(self):
        """Print comparison table for all strategies."""
        print("\n" + "═" * 62)
        print("  BACKTEST RESULTS — NIFTY 50  |  MONTHLY EXPIRY OPTIONS")
        print("═" * 62)
        for name, trades in self.results.items():
            m = self._metrics(trades)
            self._print_strategy(name, m)
        print("═" * 62)

    def get_metrics_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of all strategy metrics (useful in Colab)."""
        rows = []
        for name, trades in self.results.items():
            m = self._metrics(trades)
            m["strategy"] = name
            rows.append(m)
        return pd.DataFrame(rows).set_index("strategy")

    def save_trade_logs(self, out_dir: str = "results"):
        """Save each strategy's trade log to CSV inside out_dir."""
        import os
        os.makedirs(out_dir, exist_ok=True)
        for name, trades in self.results.items():
            safe = name.replace(" ", "_").replace("-", "").lower()
            path = f"{out_dir}/{safe}.csv"
            trades.to_csv(path, index=False)
            print(f"[Backtester] Saved → {path}")

    def plot_results(self, save_path: str = "results/backtest_comparison.png"):
        """Plot equity curves + win rate + signal distribution for all strategies."""
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        n = len(self.results)
        fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
        gs  = gridspec.GridSpec(3, n, figure=fig, hspace=0.45, wspace=0.35)
        colors = ["#00d4aa", "#f7931e", "#e05c7f", "#7eb8f7"]

        for idx, (name, trades) in enumerate(self.results.items()):
            color = colors[idx % len(colors)]
            m     = self._metrics(trades)

            # Equity curve
            ax1 = fig.add_subplot(gs[0, idx])
            ax1.set_facecolor("#161b22")
            cum = trades["cumulative_pnl"]
            ax1.plot(range(len(cum)), cum, color=color, linewidth=1.5)
            ax1.axhline(0, color="#555", linewidth=0.5, linestyle="--")
            ax1.fill_between(range(len(cum)), cum, 0,
                             where=(cum >= 0), alpha=0.15, color=color)
            ax1.fill_between(range(len(cum)), cum, 0,
                             where=(cum <  0), alpha=0.15, color="#e05c7f")
            ax1.set_title(name, color="white", fontsize=10, pad=6)
            ax1.set_ylabel("Cumulative P&L (₹)", color="#aaa", fontsize=8)
            ax1.tick_params(colors="#777", labelsize=7)
            for sp in ax1.spines.values(): sp.set_edgecolor("#333")

            # Metrics bar
            ax2 = fig.add_subplot(gs[1, idx])
            ax2.set_facecolor("#161b22")
            labels = ["Win %", "Profit\nFactor", "Sharpe\n×10"]
            vals   = [
                m["win_rate"],
                min(m["profit_factor"], 5) * 20,
                (m["sharpe"] + 3) * 10,
            ]
            bar_colors = [
                "#00d4aa" if m["win_rate"] > 50 else "#e05c7f",
                "#00d4aa" if m["profit_factor"] > 1 else "#e05c7f",
                "#00d4aa" if m["sharpe"] > 0 else "#e05c7f",
            ]
            bars = ax2.barh(labels, vals, color=bar_colors, height=0.5)
            ax2.set_xlim(0, 110)
            ax2.set_title("Key Metrics", color="white", fontsize=9, pad=4)
            ax2.tick_params(colors="#777", labelsize=8)
            for sp in ax2.spines.values(): sp.set_edgecolor("#333")
            for bar, v in zip(bars, [m["win_rate"], m["profit_factor"], m["sharpe"]]):
                ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                         f"{v:.2f}", va="center", color="white", fontsize=8)

            # Call/Put distribution
            ax3 = fig.add_subplot(gs[2, idx])
            ax3.set_facecolor("#161b22")
            calls = (trades["signal"] == "CALL").sum()
            puts  = (trades["signal"] == "PUT").sum()
            ax3.pie([calls, puts],
                    labels=["Call", "Put"],
                    colors=["#00d4aa", "#e05c7f"],
                    autopct="%1.0f%%", startangle=90,
                    textprops={"color": "white", "fontsize": 9})
            ax3.set_title("Signal Split", color="white", fontsize=9, pad=4)

        fig.suptitle(
            "Nifty 50 ATM Options — Monthly Expiry Backtest  |  SMA-9 vs EMA-9",
            color="white", fontsize=13, y=0.98, fontweight="bold"
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[Backtester] Chart saved → {save_path}")

    # ── private ─────────────────────────────────────────────
    def _add_cumulative(self, trades: pd.DataFrame) -> pd.DataFrame:
        trades = trades.copy().reset_index(drop=True)
        trades["cumulative_pnl"] = trades["pnl"].cumsum()
        running_max = trades["cumulative_pnl"].cummax()
        trades["drawdown"] = trades["cumulative_pnl"] - running_max
        return trades

    def _metrics(self, trades: pd.DataFrame) -> dict:
        pnl      = trades["pnl"]
        wins     = pnl[pnl > 0]
        losses   = pnl[pnl < 0]
        total    = len(trades)

        win_rate = len(wins) / total * 100 if total else 0
        gross_p  = wins.sum()
        gross_l  = losses.abs().sum()
        pf       = gross_p / gross_l if gross_l > 0 else float("inf")
        ret      = trades["pnl_pct"]
        sharpe   = (ret.mean() / ret.std() * np.sqrt(12)) if ret.std() > 0 else 0
        max_dd   = trades["drawdown"].min()
        total_pnl= trades["cumulative_pnl"].iloc[-1]
        roi      = total_pnl / self.INITIAL_CAPITAL * 100

        return {
            "total_trades"  : total,
            "win_rate"      : round(win_rate, 1),
            "total_pnl"     : round(total_pnl, 2),
            "roi_pct"       : round(roi, 1),
            "profit_factor" : round(pf, 2),
            "sharpe"        : round(sharpe, 2),
            "max_drawdown"  : round(max_dd, 2),
            "avg_pnl"       : round(pnl.mean(), 2),
        }

    @staticmethod
    def _print_strategy(name: str, m: dict):
        print(f"\n  {name}")
        print(f"  {'─'*40}")
        print(f"  Total Trades    : {m['total_trades']}")
        print(f"  Win Rate        : {m['win_rate']}%")
        print(f"  Total P&L       : ₹{m['total_pnl']:,.0f}")
        print(f"  ROI             : {m['roi_pct']}%")
        print(f"  Profit Factor   : {m['profit_factor']}")
        print(f"  Sharpe Ratio    : {m['sharpe']}")
        print(f"  Max Drawdown    : ₹{m['max_drawdown']:,.0f}")
        print(f"  Avg P&L/Trade   : ₹{m['avg_pnl']:,.0f}")
