# ============================================================
#  FACTOR CONSTRUCTION
#  Momentum, Value, and Quality signals for NYSE equities
#  Requires: data/nyse_ohlcv_daily.parquet (already built)
#            data/fundamentals.parquet     (pulled here)
# ============================================================

import lseg.data as ld
import pandas as pd
import numpy as np
import warnings
import glob
import os

warnings.filterwarnings("ignore")

APP_KEY = "1374e5ed54fd4c64a7fe3a65fe995f462fa0c764"   # same key as before

OHLCV_PATH       = "data/nyse_ohlcv_daily.parquet"
FUND_PATH        = "data/fundamentals.parquet"
OUTPUT_PATH      = "data/factors.parquet"
CHECKPOINT_DIR   = "data/checkpoints_fund"

# ── 1. LOAD PRICE DATA ───────────────────────────────────────

print("Loading OHLCV data...")
prices = pd.read_parquet(OHLCV_PATH)
prices["date"] = pd.to_datetime(prices["date"])

tickers = prices["ticker"].unique().tolist()
print(f"Loaded {len(prices):,} rows | {len(tickers)} tickers\n")

# ── 2. RESAMPLE TO MONTH-END ─────────────────────────────────
# Factors are typically computed monthly for rebalancing

print("Resampling to month-end...")
prices_monthly = (
    prices
    .set_index("date")
    .groupby("ticker")
    .resample("ME")
    .agg(
        open   = ("open",   "first"),
        high   = ("high",   "max"),
        low    = ("low",    "min"),
        close  = ("close",  "last"),
        volume = ("volume", "sum")
    )
    .reset_index()
)

prices_monthly = prices_monthly.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"Monthly panel: {len(prices_monthly):,} rows\n")

# ── 3. MOMENTUM FACTORS ──────────────────────────────────────
# Classic momentum: past returns over various lookback windows
# Skip the most recent month to avoid short-term reversal

print("Computing momentum factors...")

prices_monthly = prices_monthly.sort_values(["ticker", "date"])

def monthly_return(df, col="close"):
    return df.groupby("ticker")[col].pct_change()

def momentum(df, lookback, skip=1):
    """
    Compound return from t-lookback to t-skip months ago.
    Standard: lookback=12, skip=1 (12-1 momentum)
    """
    def calc(x):
        shifted = x.shift(skip)
        past    = x.shift(lookback)
        return (shifted / past) - 1
    return df.groupby("ticker")["close"].transform(calc)

prices_monthly["ret_1m"]  = monthly_return(prices_monthly)
prices_monthly["mom_12_1"] = momentum(prices_monthly, lookback=12, skip=1)  # 12-1 month momentum
prices_monthly["mom_6_1"]  = momentum(prices_monthly, lookback=6,  skip=1)  # 6-1 month momentum
prices_monthly["mom_3_1"]  = momentum(prices_monthly, lookback=3,  skip=1)  # 3-1 month momentum

# Short-term reversal (1 month)
prices_monthly["reversal_1m"] = prices_monthly.groupby("ticker")["close"].pct_change(1)

print("  Momentum signals: mom_12_1, mom_6_1, mom_3_1, reversal_1m")

# ── 4. PULL FUNDAMENTAL DATA FROM LSEG ──────────────────────
# For value and quality we need P/E, P/Book, ROE, Debt/Equity

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

FUND_FIELDS = [
    "TR.PriceToBVPerShare",   # Price-to-Book
    "TR.PERatio",             # P/E Ratio
    "TR.ROE",                 # Return on Equity
    "TR.TotalDebtToEquity",   # Debt-to-Equity (leverage)
    "TR.DividendYield",       # Dividend Yield
]

# Pull fundamentals at most recent date (point-in-time approximation)
# For a production model you'd want historical fundamental snapshots

def pull_fundamentals(tickers, fields, app_key):

    checkpoint_files = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "fund_checkpoint_*.parquet"))
    )

    BATCH_SIZE = 50
    batches    = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    n_batches  = len(batches)
    all_frames = []
    start_batch = 0

    if checkpoint_files:
        latest      = checkpoint_files[-1]
        latest_n    = int(latest.split("fund_checkpoint_")[1].replace(".parquet",""))
        print(f"  Checkpoint found at batch {latest_n} — resuming.\n")
        all_frames  = [pd.read_parquet(latest)]
        start_batch = latest_n

    ld.open_session(app_key=app_key)

    for i in range(start_batch, n_batches):
        print(f"  Fundamentals batch {i+1} / {n_batches}")
        try:
            df = ld.get_data(
                universe = batches[i],
                fields   = fields
            )
            if df is not None and not df.empty:
                all_frames.append(df)
        except Exception as e:
            print(f"    [!] Failed: {e}")

        if (i+1) % 20 == 0 and all_frames:
            cp = pd.concat(all_frames, ignore_index=True)
            cp.to_parquet(os.path.join(CHECKPOINT_DIR, f"fund_checkpoint_{i+1}.parquet"), index=False)
            print(f"    Checkpoint saved at batch {i+1}")

    ld.close_session()

    if not all_frames:
        return None

    result = pd.concat(all_frames, ignore_index=True)
    return result

if os.path.exists(FUND_PATH):
    print("Loading existing fundamentals data...")
    fund = pd.read_parquet(FUND_PATH)
else:
    print("Pulling fundamentals from LSEG...")
    fund = pull_fundamentals(tickers, FUND_FIELDS, APP_KEY)

    if fund is not None:
        # Standardise column names
        fund = fund.rename(columns={
            "Instrument"              : "ticker",
            "Price To Book Value Per Share (Daily Time Series Ratio)" : "pb_ratio",
            "PE (Excl. Extraordinary Items, TTM)"                    : "pe_ratio",
            "Return On Equity, Actual"                                : "roe",
            "Total Debt/Total Equity (Annual)"                        : "debt_equity",
            "Dividend Yield (Percent)"                                : "div_yield",
        })

        # Fallback: rename any column containing these keywords
        col_map = {}
        for col in fund.columns:
            low = col.lower()
            if "book" in low:      col_map[col] = "pb_ratio"
            elif "pe" in low or "p/e" in low: col_map[col] = "pe_ratio"
            elif "roe" in low or "return on equity" in low: col_map[col] = "roe"
            elif "debt" in low and "equity" in low:  col_map[col] = "debt_equity"
            elif "dividend" in low and "yield" in low: col_map[col] = "div_yield"
            elif "instrument" in low: col_map[col] = "ticker"
        fund = fund.rename(columns=col_map)

        fund.to_parquet(FUND_PATH, index=False)
        print(f"Saved fundamentals: {len(fund)} tickers\n")
    else:
        print("  [!] No fundamental data returned — value/quality factors will be skipped.")
        fund = None

# ── 5. VALUE & QUALITY FACTORS ───────────────────────────────

if fund is not None:
    print("Computing value and quality factors...")
    print(f"  Fundamentals columns: {fund.columns.tolist()}")

    # Merge fundamentals onto monthly price panel (as static snapshot)
    panel = prices_monthly.merge(fund, on="ticker", how="left")

    # ── Value factors ──────────────────────────────────────
    # Lower P/B = cheaper = higher value signal (invert so higher = better)
    if "pb_ratio" in panel.columns:
        panel["pb_ratio"] = pd.to_numeric(panel["pb_ratio"], errors="coerce")
        panel["value_pb"] = -panel["pb_ratio"]  # invert: lower P/B = better value

    if "pe_ratio" in panel.columns:
        panel["pe_ratio"] = pd.to_numeric(panel["pe_ratio"], errors="coerce")
        panel["value_pe"] = -panel["pe_ratio"]  # invert: lower P/E = better value

    if "div_yield" in panel.columns:
        panel["div_yield"] = pd.to_numeric(panel["div_yield"], errors="coerce")
        panel["value_dy"]  = panel["div_yield"]  # higher yield = better value

    # ── Quality factors ────────────────────────────────────
    if "roe" in panel.columns:
        panel["roe"] = pd.to_numeric(panel["roe"], errors="coerce")
        panel["quality_roe"] = panel["roe"]   # higher ROE = better quality

    if "debt_equity" in panel.columns:
        panel["debt_equity"] = pd.to_numeric(panel["debt_equity"], errors="coerce")
        panel["quality_lev"] = -panel["debt_equity"]  # lower leverage = better quality

    print("  Value signals   : value_pb, value_pe, value_dy")
    print("  Quality signals : quality_roe, quality_lev")

else:
    panel = prices_monthly.copy()
    print("  Skipping value/quality (no fundamental data)")

# ── 6. CROSS-SECTIONAL STANDARDISATION ───────────────────────
# Z-score each factor within each month (cross-sectionally)
# This makes factors comparable across time

print("\nStandardising factors cross-sectionally...")

factor_cols = [c for c in panel.columns if c.startswith(("mom_", "reversal_", "value_", "quality_"))]

def cross_sectional_zscore(df, factor_cols):
    def zscore(x):
        m, s = x.mean(), x.std()
        if pd.isna(s) or s == 0:
            return x * 0
        return (x - m) / s
    for col in factor_cols:
        if col in df.columns:
            df[col + "_z"] = df.groupby("date")[col].transform(zscore)
    return df

panel = cross_sectional_zscore(panel, factor_cols)

z_cols = [c for c in panel.columns if c.endswith("_z")]
print(f"  Standardised factors: {z_cols}")

# ── 7. WINSORISE OUTLIERS ─────────────────────────────────────
# Cap extreme values at ±3 standard deviations

print("Winsorising outliers at ±3 SD...")

for col in z_cols:
    panel[col] = panel[col].clip(-3, 3)

# ── 8. COMPOSITE ALPHA SCORE ─────────────────────────────────
# Simple equal-weighted composite of available z-scored factors
# In a production model you'd use IC-weighted combination

print("\nBuilding composite alpha score...")

mom_z_cols     = [c for c in z_cols if c.startswith("mom_")]
value_z_cols   = [c for c in z_cols if c.startswith("value_")]
quality_z_cols = [c for c in z_cols if c.startswith("quality_")]

if mom_z_cols:
    panel["factor_momentum"] = panel[mom_z_cols].mean(axis=1)

if value_z_cols:
    panel["factor_value"]    = panel[value_z_cols].mean(axis=1)

if quality_z_cols:
    panel["factor_quality"]  = panel[quality_z_cols].mean(axis=1)

composite_cols = [c for c in ["factor_momentum", "factor_value", "factor_quality"] if c in panel.columns]
panel["alpha_score"] = panel[composite_cols].mean(axis=1)

print(f"  Composite from: {composite_cols}")

# ── 9. SAVE ───────────────────────────────────────────────────

keep_cols = (
    ["date", "ticker", "open", "high", "low", "close", "volume", "ret_1m"] +
    factor_cols + z_cols +
    [c for c in ["factor_momentum", "factor_value", "factor_quality", "alpha_score"] if c in panel.columns]
)
keep_cols = [c for c in keep_cols if c in panel.columns]

output = panel[keep_cols].sort_values(["date", "ticker"]).reset_index(drop=True)
output.to_parquet(OUTPUT_PATH, index=False)

print(f"\n── Factor panel summary ─────────────────────────────────")
print(f"  Rows        : {len(output):,}")
print(f"  Tickers     : {output['ticker'].nunique():,}")
print(f"  Date range  : {output['date'].min().date()} → {output['date'].max().date()}")
print(f"  Factors     : {[c for c in output.columns if c.startswith('factor_')]}")
print(f"  Saved to    : {OUTPUT_PATH}")
print(f"────────────────────────────────────────────────────────")

print("\nSample alpha scores (latest month):")
latest = output[output["date"] == output["date"].max()].copy()
latest = latest.dropna(subset=["alpha_score"]).sort_values("alpha_score", ascending=False)
print("  Top 10:")
print(latest[["ticker", "alpha_score"]].head(10).to_string(index=False))
print("  Bottom 10:")
print(latest[["ticker", "alpha_score"]].tail(10).to_string(index=False))
