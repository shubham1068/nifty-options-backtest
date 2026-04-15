# 📈 Nifty 50 ATM Options — Monthly Expiry Backtest

An end-to-end OOP backtesting framework for Nifty 50 monthly expiry ATM options.
Compares **SMA-9** vs **EMA-9** buy-side strategies across historical data fetched from Yahoo Finance.

---

## 🗂️ Project Structure

```
nifty-options-backtest/
├── src/
│   ├── __init__.py
│   └── backtest.py          # All 4 core classes
├── notebooks/
│   └── analysis.ipynb       # Colab-ready analysis notebook
├── results/                 # Auto-created: CSVs + charts (git-ignored)
├── run_backtest.py          # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A — Run locally

```bash
git clone https://github.com/YOUR_USERNAME/nifty-options-backtest.git
cd nifty-options-backtest
pip install -r requirements.txt

python run_backtest.py                    # default: 5 years
python run_backtest.py --years 3          # 3 years of data
python run_backtest.py --csv mydata.csv   # from local CSV
```

### Option B — Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/nifty-options-backtest/blob/main/notebooks/analysis.ipynb)

1. Click the badge above  
2. Update `REPO_URL` in Cell 1 with your repo URL  
3. Run all cells (`Runtime → Run all`)

---

## ⚙️ Strategy Logic

| Parameter | Detail |
|-----------|--------|
| Instrument | Nifty 50 ATM Options (CALL or PUT) |
| Entry | Every monthly expiry (last Thursday of month) |
| Signal | Close > MA → Buy CALL; Close < MA → Buy PUT |
| Exit | Hold until next monthly expiry |
| Lot size | 50 qty (1 lot) |
| Premium model | `0.4 × ATR_14 × √DTE` |

---

## 📊 Metrics Produced

- Win Rate (%)
- Total P&L (₹)
- ROI vs initial capital
- Profit Factor
- Sharpe Ratio (annualised, monthly)
- Max Drawdown (₹)
- Avg P&L per trade

---

## 🔧 Extending the Framework

### Add a custom strategy

```python
from src.backtest import _BaseStrategy
import pandas as pd

class RSI14Strategy(_BaseStrategy):
    name = "RSI-14 Strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        # ... your logic here
        return pd.DataFrame(trades)

# Then plug it in:
bt.add_strategy(RSI14Strategy())
```

### Use a local CSV

Your CSV must have these columns (case-insensitive):  
`date, open, high, low, close, volume`

```python
loader = NiftyDataLoader()
df_raw = loader.load_from_csv("path/to/nifty_data.csv")
```

---

## 📁 Output Files (in `results/`)

| File | Description |
|------|-------------|
| `sma9_strategy.csv` | Full SMA-9 trade log |
| `ema9_strategy.csv` | Full EMA-9 trade log |
| `metrics_summary.csv` | Side-by-side strategy metrics |
| `backtest_comparison.png` | Equity curves + metrics chart |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
The premium model is a simplified approximation — **not suitable for live trading** without significant enhancements (real IV data, slippage, brokerage, etc.).

---

## 📄 License

MIT
