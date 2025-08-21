# Assets, Benchmarks & Defaults

This doc explains why these tickers and the benchmark were chosen, what the risk-free rate and default weights mean, and how to tweak them for your use case.

---

## 📦 The Asset List

```yaml
tickers: ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "AMZN", "GOOGL", "TSLA", "SPY", "GLD"]
```

**What this mix gives you**

* **US large-caps across key sectors**

  * Tech: `AAPL`, `MSFT`, `AMZN`, `GOOGL`
  * Health Care: `JNJ`
  * Financials: `JPM`
  * Energy: `XOM`
  * High beta / growth: `TSLA`
* **Broad market proxy:** `SPY` (S\&P 500 ETF) stabilizes the single-name exposure.
* **Diversifier:** `GLD` (gold) adds a non-equity ballast that often behaves differently from stocks.

**Why single stocks *and* an ETF?**
Single names let you see specific contributors and risk concentrations. `SPY` smooths idiosyncratic risk and gives a clean benchmark-like component inside the portfolio itself.

> You can add or remove tickers. The app will re-compute weights, stats, and charts automatically.

---

## 🎯 Benchmark Choice

```yaml
benchmarks:
  "^GSPC": "S&P 500 Index (US)"
default_benchmark: "^GSPC"
```

**Why `^GSPC` (S\&P 500 index)?**
It’s the most common US equity yardstick. Using the headline index keeps your performance and risk metrics easy to interpret (beta, tracking error, information ratio).

**Tip for the Live (intraday) tab**
Index symbols (like `^GSPC`) may not always stream minute bars. If you want intraday benchmark lines reliably, consider using an ETF such as `SPY` for `default_benchmark`. Daily/EOD metrics work fine with either.

**When to pick a different benchmark**

* Global portfolio → MSCI ACWI / world ETF
* Non-US focus → regional index/ETF (e.g., Europe, India, Japan)
* Thematic book → a sector ETF (e.g., XLK for Tech)

Just update `default_benchmark` to a symbol you also fetch in your data jobs.

---

## 🧮 Risk-Free Rate

```yaml
risk_free_rate_annual: 0.04  # 4% per year
```

This feeds into **Sharpe**, **alpha**, and other excess-return metrics.
4% is a reasonable long-run placeholder. If you want it precise, set this to current short-term Treasury yield levels and redeploy.

---

## ⚖️ Default Weights

```yaml
default_weights:
  AAPL:  0.10
  MSFT:  0.10
  JNJ:   0.10
  JPM:   0.10
  XOM:   0.10
  AMZN:  0.10
  GOOGL: 0.10
  TSLA:  0.10
  SPY:   0.10
  GLD:   0.10
```

These sum to **1.00** and do three things:

1. **Diversify across sectors** (Tech, Health, Financials, Energy).
2. **Blend idiosyncratic + market exposure** (`SPY` anchors the portfolio).
3. **Add a defensive sleeve** (`GLD`) to dampen equity-only risk.

> In the app, you can switch to **Equal-Weight**, **Custom**, **Max-Sharpe**, or **Min-Variance**. Custom inputs are automatically re-normalized (excluding the benchmark when needed).

---

## 🗺️ Metadata (for charts)

```yaml
metadata:
  AAPL:  { sector: "Information Technology", region: "US" }
  MSFT:  { sector: "Information Technology", region: "US" }
  JNJ:   { sector: "Health Care",            region: "US" }
  JPM:   { sector: "Financials",             region: "US" }
  XOM:   { sector: "Energy",                 region: "US" }
  AMZN:  { sector: "Consumer Discretionary", region: "US" }
  GOOGL: { sector: "Communication Services", region: "US" }
  TSLA:  { sector: "Consumer Discretionary", region: "US" }
  SPY:   { sector: "ETF (US Broad Market)",  region: "US" }
  GLD:   { sector: "Commodity (Gold)",       region: "Global" }
```

Used for the **Diversification** section to build sector/region pies and the “By Asset” table.
Feel free to add your own groupings (e.g., “style: Growth/Value”)—the app will pick them up if you surface them.

---

## 🔄 Refresh Intervals

```yaml
refresh_interval:
  market_open: 60      # refresh every 60s when market is open
  market_closed: 900   # refresh every 15 min when market is closed
```

* During **US market hours** (approx. 09:30–16:00 New York), intraday updates every \~60s.
* Off hours, the app checks less frequently.
  End-of-day bars are appended after the close (with a small buffer).

---

## 🛠️ How to Customize

1. **Add/remove tickers**

   * Put the symbol in `tickers:` and (optionally) add `metadata:` for clean charts.
2. **Change the benchmark**

   * Set `default_benchmark:` to a symbol you also fetch. For intraday reliability, consider using an ETF (e.g., `SPY`).
3. **Tweak risk-free**

   * Update `risk_free_rate_annual:` to your preferred rate.
4. **Edit starting weights**

   * Update `default_weights:`. The app will re-normalize if needed.

> After changes, redeploy or restart your service so the app and workers read the new config.

---

This configuration aims to be sensible out of the box and easy to adapt. If you’re building a region-specific or multi-asset portfolio, swap in the right symbols/benchmarks and the rest of the dashboard will follow.
