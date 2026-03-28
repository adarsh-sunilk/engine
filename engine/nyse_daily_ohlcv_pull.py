# ============================================================
#  NYSE DAILY OHLCV DATA PULL
#  LSEG Data Library for Python
#  ~2,500 tickers | Daily | Checkpointed
# ============================================================
#
#  SETUP BEFORE RUNNING:
#  1. Install library:
#       pip install lseg-data pandas
#
#  2. Paste your app key into APP_KEY below
#
#  3. Make sure LSEG Workspace is open and logged in
#     on your desktop — the library connects through it
#
#  EXPECTED RUNTIME : 60–120 minutes
#  EXPECTED OUTPUT  : ~50–60M rows as .parquet / .csv
# ============================================================

import lseg.data as ld
import pandas as pd
import time
import os
import glob
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────

APP_KEY         = ""   # paste from APPKEY app in Workspace

START_DATE      = "2015-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")
BATCH_SIZE      = 10
SLEEP_SECS      = 2
CHECKPOINT_EVERY = 10

FIELDS = ["TR.PriceOpen", "TR.PriceHigh", "TR.PriceLow", "TR.PriceClose", "TR.Volume"]

OUTPUT_DIR      = "data"
CHECKPOINT_DIR  = "data/checkpoints"
OUTPUT_PARQUET  = "data/nyse_ohlcv_daily.parquet"
OUTPUT_CSV      = "data/nyse_ohlcv_daily.csv"

# ── 1. CONNECT ───────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Connecting to LSEG Workspace...")
ld.open_session(app_key=APP_KEY)
print("Connected.\n")

# ── 2. FETCH NYSE CONSTITUENT LIST ───────────────────────────

print("Fetching NYSE constituent list...")

# Pull all constituents of the NYSE composite index
universe_df = ld.get_data(
    universe  = "0#.NYA",       # NYSE Composite index chain
    fields    = ["TR.RIC", "TR.CompanyName"]
)

tickers = universe_df["RIC"].dropna().tolist()
tickers = [t for t in tickers if isinstance(t, str) and len(t) > 0]

print(f"Universe: {len(tickers)} tickers\n")

# ── 3. BATCH PULL FUNCTION ───────────────────────────────────

def pull_batch(batch_tickers, fields, start, end):
    try:
        df = ld.get_history(
            universe   = batch_tickers,
            fields     = fields,
            start      = start,
            end        = end,
            interval   = "1D"
        )

        if df is None or df.empty:
            return None

        # Reset index — date and instrument are in the MultiIndex
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
         df = df.stack(level=0).reset_index()
         df.columns.name = None
        else:
         df = df.reset_index()

        # Rename whatever the date/instrument columns are called
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if low == "date":
                col_map[col] = "date"
            elif low == "instrument":
                col_map[col] = "ticker"
            elif "priceopen" in low or low == "open":
                col_map[col] = "open"
            elif "pricehigh" in low or low == "high":
                col_map[col] = "high"
            elif "pricelow" in low or low == "low":
                col_map[col] = "low"
            elif "priceclose" in low or low == "close":
                col_map[col] = "close"
            elif "volume" in low:
                col_map[col] = "volume"

        df = df.rename(columns=col_map)

        if "date" not in df.columns:
            print(f"  [!] Unexpected columns: {df.columns.tolist()}")
            return None

        df["date"] = pd.to_datetime(df["date"])
        return df

    except Exception as e:
        print(f"  [!] Batch failed: {e}")
        return None

# ── 4. RESUME FROM CHECKPOINT IF ONE EXISTS ──────────────────

checkpoint_files = sorted(
    glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_batch_*.parquet"))
)

start_batch   = 0
prior_frames  = []

if checkpoint_files:
    latest_file   = checkpoint_files[-1]
    latest_batch  = int(latest_file.split("checkpoint_batch_")[1].replace(".parquet", ""))
    print(f"Checkpoint found at batch {latest_batch} — resuming from batch {latest_batch + 1}.\n")
    prior_frames  = [pd.read_parquet(latest_file)]
    start_batch   = latest_batch + 1
else:
    print("No checkpoint found — starting fresh.\n")

# ── 5. BATCH LOOP ─────────────────────────────────────────────

batches   = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
n_batches = len(batches)
all_frames = prior_frames.copy()

for i in range(start_batch, n_batches):

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}]  Batch {i + 1} / {n_batches}  ({len(batches[i])} tickers)")

    result = pull_batch(
        batch_tickers = batches[i],
        fields        = FIELDS,
        start         = START_DATE,
        end           = END_DATE
    )

    if result is not None:
        all_frames.append(result)
    else:
        print(f"  [!] Batch {i + 1} returned no data — skipping.")

    time.sleep(SLEEP_SECS)

    # Save checkpoint every N batches
    if (i + 1) % CHECKPOINT_EVERY == 0:
        checkpoint_df   = pd.concat(all_frames, ignore_index=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_batch_{i + 1}.parquet")
        checkpoint_df.to_parquet(checkpoint_path, index=False)
        print(f"  Checkpoint saved → {checkpoint_path}")

# ── 6. ASSEMBLE FULL PANEL ───────────────────────────────────

print("\nAssembling full panel...")
nyse_ohlcv = pd.concat(all_frames, ignore_index=True)

# ── 7. CLEANING ───────────────────────────────────────────────

print("Cleaning data...")

# Enforce types
nyse_ohlcv["date"]   = pd.to_datetime(nyse_ohlcv["date"])
nyse_ohlcv["ticker"] = nyse_ohlcv["ticker"].astype(str)

for col in ["open", "high", "low", "close", "volume"]:
    if col in nyse_ohlcv.columns:
        nyse_ohlcv[col] = pd.to_numeric(nyse_ohlcv[col], errors="coerce")

# Drop rows with no close price
nyse_ohlcv = nyse_ohlcv.dropna(subset=["close"])

# Remove data errors
nyse_ohlcv = nyse_ohlcv[nyse_ohlcv["close"] > 0]
nyse_ohlcv = nyse_ohlcv[
    nyse_ohlcv["high"].isna() |
    nyse_ohlcv["low"].isna()  |
    (nyse_ohlcv["high"] >= nyse_ohlcv["low"])
]

# Deduplicate
nyse_ohlcv = nyse_ohlcv.drop_duplicates(subset=["ticker", "date"])

# Sort
nyse_ohlcv = nyse_ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)

# ── 8. SANITY CHECK ───────────────────────────────────────────

print("\n── Panel summary ───────────────────────────────────────")
print(f"  Rows        : {len(nyse_ohlcv):,}")
print(f"  Tickers     : {nyse_ohlcv['ticker'].nunique():,}")
print(f"  Date range  : {nyse_ohlcv['date'].min().date()}  →  {nyse_ohlcv['date'].max().date()}")
print(f"  Close pct   : {100 * nyse_ohlcv['close'].notna().mean():.1f}% non-null")
print(f"  Memory      : {nyse_ohlcv.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print("────────────────────────────────────────────────────────\n")

# ── 9. SAVE ───────────────────────────────────────────────────

print(f"Saving .parquet → {OUTPUT_PARQUET}")
nyse_ohlcv.to_parquet(OUTPUT_PARQUET, index=False)

print(f"Saving .csv     → {OUTPUT_CSV}")
nyse_ohlcv.to_csv(OUTPUT_CSV, index=False)

# Clean up checkpoints
old_checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_batch_*.parquet"))
for f in old_checkpoints:
    os.remove(f)
if old_checkpoints:
    print(f"Removed {len(old_checkpoints)} checkpoint file(s).")

# ── 10. CLOSE SESSION ────────────────────────────────────────

ld.close_session()
print("\nDone.")
