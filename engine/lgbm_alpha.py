# ============================================================
#  LIGHTGBM ALPHA MODEL
#  Replaces equal-weighted z-score composite with a trained
#  gradient boosted model using time-series cross-validation
#
#  Requires: data/factors.parquet (already built)
#  Outputs : data/lgbm_factors.parquet  (factor panel + ML alpha)
#            data/models/lgbm_model.txt (saved model)
# ============================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

FACTORS_PATH = "data/factors.parquet"
OUTPUT_PATH  = "data/lgbm_factors.parquet"
MODEL_DIR    = "data/models"
MODEL_PATH   = os.path.join(MODEL_DIR, "lgbm_model.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. LOAD FACTOR PANEL ─────────────────────────────────────

print("Loading factor panel...")
panel = pd.read_parquet(FACTORS_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
print(f"  {len(panel):,} rows | {panel['ticker'].nunique():,} tickers")
print(f"  Date range: {panel['date'].min().date()} → {panel['date'].max().date()}\n")

# ── 2. DEFINE FEATURES AND TARGET ────────────────────────────
# Features: all z-scored factor signals
# Target  : forward 1-month return (what we're trying to predict)

FEATURE_COLS = [c for c in panel.columns if c.endswith("_z")]
print(f"Features: {FEATURE_COLS}")

# Forward return: next month's return
panel = panel.sort_values(["ticker", "date"])
panel["fwd_ret_1m"] = panel.groupby("ticker")["ret_1m"].shift(-1)

# Drop rows missing features or target
panel_clean = panel.dropna(subset=FEATURE_COLS + ["fwd_ret_1m"]).copy()
print(f"Clean rows: {len(panel_clean):,} (after dropping NAs)\n")

# ── 3. TIME-SERIES CROSS-VALIDATION SETUP ────────────────────
# We MUST NOT use future data to train — use TimeSeriesSplit
# Each fold trains on past months and predicts the next period

print("Setting up time-series cross-validation...")

# Get unique dates in order
dates = sorted(panel_clean["date"].unique())
n_dates = len(dates)
print(f"  Total months: {n_dates}")

# Use 5 folds — each fold adds ~20% more training data
N_SPLITS     = 5
TRAIN_WINDOW = 24   # minimum months of training data required
cv_results   = []

# Map dates to integer indices for splitting
date_to_idx = {d: i for i, d in enumerate(dates)}
panel_clean["date_idx"] = panel_clean["date"].map(date_to_idx)

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
date_indices = np.arange(n_dates)

print(f"  Folds: {N_SPLITS}\n")

# ── 4. ROLLING TRAINING + PREDICTION ─────────────────────────

all_predictions = []

for fold, (train_date_idx, test_date_idx) in enumerate(tscv.split(date_indices)):

    # Skip folds with insufficient training data
    if len(train_date_idx) < TRAIN_WINDOW:
        print(f"  Fold {fold+1}: skipping (only {len(train_date_idx)} train months)")
        continue

    train_dates = [dates[i] for i in train_date_idx]
    test_dates  = [dates[i] for i in test_date_idx]

    train_df = panel_clean[panel_clean["date"].isin(train_dates)]
    test_df  = panel_clean[panel_clean["date"].isin(test_dates)]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["fwd_ret_1m"].to_numpy(dtype=float)
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["fwd_ret_1m"].to_numpy(dtype=float)

    # ── Train LightGBM ────────────────────────────────────────
    train_set = lgb.Dataset(X_train, label=y_train,
                            feature_name=FEATURE_COLS)

    params = {
        "objective"        : "regression",
        "metric"           : "rmse",
        "n_estimators"     : 400,
        "learning_rate"    : 0.05,
        "num_leaves"       : 31,
        "min_child_samples": 50,   # prevent overfitting on small monthly groups
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "reg_alpha"        : 0.1,
        "reg_lambda"       : 0.1,
        "verbose"          : -1,
        "n_jobs"           : -1,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X_test, label=y_test,
                                feature_name=FEATURE_COLS)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    # ── Predict on test set ───────────────────────────────────
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))

    # IC for this fold
    fold_df = test_df[["date", "ticker", "fwd_ret_1m"]].copy()
    fold_df["lgbm_alpha"] = preds
    ic = fold_df.groupby("date").apply(
        lambda g: g["lgbm_alpha"].corr(g["fwd_ret_1m"], method="spearman")
    ).mean()

    print(f"  Fold {fold+1}: train={len(train_dates)}m "
          f"test={len(test_dates)}m | RMSE={rmse:.4f} | IC={ic:.4f}")

    all_predictions.append(fold_df)

print()

# ── 5. TRAIN FINAL MODEL ON ALL DATA ─────────────────────────
# After CV, train on the full dataset for production scoring

print("Training final model on full dataset...")

X_all = panel_clean[FEATURE_COLS].values
y_all = panel_clean["fwd_ret_1m"].to_numpy(dtype=float)

train_set_full = lgb.Dataset(X_all, label=y_all,
                              feature_name=FEATURE_COLS)

final_model = lgb.train(
    params,
    train_set_full,
    num_boost_round=model.best_iteration or 400,
    callbacks=[lgb.log_evaluation(period=-1)]
)

# Save model
final_model.save_model(MODEL_PATH)
print(f"  Model saved → {MODEL_PATH}")

# ── 6. SCORE FULL PANEL ───────────────────────────────────────
# Generate alpha scores for every row using the final model

print("Scoring full panel with final model...")

panel_scored = panel.copy()
valid_mask   = panel_scored[FEATURE_COLS].notna().all(axis=1)

panel_scored["lgbm_alpha"] = np.nan
panel_scored.loc[valid_mask, "lgbm_alpha"] = final_model.predict(
    panel_scored.loc[valid_mask, FEATURE_COLS].values
)

# Cross-sectional z-score the LGBM alpha (standardise within each month)
def cs_zscore(x):
    m, s = x.mean(), x.std()
    if pd.isna(s) or s == 0:
        return x * 0
    return (x - m) / s

panel_scored["lgbm_alpha_z"] = (
    panel_scored.groupby("date")["lgbm_alpha"]
    .transform(cs_zscore)
    .clip(-3, 3)
)

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────

print("\n── Feature Importance ──────────────────────────────────")
importance = pd.Series(
    final_model.feature_importance(importance_type="gain"),
    index=FEATURE_COLS
).sort_values(ascending=False)

for feat, imp in importance.items():
    bar = "█" * int(imp / importance.max() * 20)
    print(f"  {feat:20s} {bar} {imp:.1f}")

# ── 8. OUT-OF-SAMPLE IC SUMMARY ──────────────────────────────

if all_predictions:
    oos_df = pd.concat(all_predictions, ignore_index=True)
    oos_ic = oos_df.groupby("date").apply(
        lambda g: g["lgbm_alpha"].corr(g["fwd_ret_1m"], method="spearman")
    )
    mean_ic = oos_ic.mean()
    icir    = mean_ic / oos_ic.std() if oos_ic.std() > 0 else 0
    ic_pos  = (oos_ic > 0).mean()

    print(f"\n── Out-of-Sample Performance ───────────────────────────")
    print(f"  Mean IC  : {mean_ic:.4f}")
    print(f"  ICIR     : {icir:.3f}")
    print(f"  IC > 0   : {ic_pos:.1%} of months")
    print(f"  Months   : {len(oos_ic)}")

    # Compare to old equal-weighted composite
    old_ic = oos_df.groupby("date").apply(
        lambda g: g.get("alpha_score", pd.Series([np.nan])).corr(
            g["fwd_ret_1m"], method="spearman")
    ).mean() if "alpha_score" in oos_df.columns else None

    print(f"\n── vs. Equal-Weighted Composite ─────────────────────────")
    if old_ic and not np.isnan(old_ic):
        print(f"  Old IC   : {old_ic:.4f}")
        print(f"  New IC   : {mean_ic:.4f}")
        improvement = (mean_ic - old_ic) / abs(old_ic) * 100 if old_ic != 0 else 0
        print(f"  Change   : {improvement:+.1f}%")
    else:
        print(f"  New IC   : {mean_ic:.4f}  (no old IC available for comparison)")

# ── 9. SAVE ───────────────────────────────────────────────────

keep_cols = (
    ["date", "ticker", "open", "high", "low", "close", "volume", "ret_1m"] +
    FEATURE_COLS +
    [c for c in ["factor_momentum", "factor_value", "alpha_score",
                 "lgbm_alpha", "lgbm_alpha_z"] if c in panel_scored.columns]
)
keep_cols = [c for c in keep_cols if c in panel_scored.columns]

output = panel_scored[keep_cols].sort_values(["date", "ticker"]).reset_index(drop=True)
output.to_parquet(OUTPUT_PATH, index=False)

print(f"\n── Output ───────────────────────────────────────────────")
print(f"  Rows        : {len(output):,}")
print(f"  Tickers     : {output['ticker'].nunique():,}")
print(f"  Date range  : {output['date'].min().date()} → {output['date'].max().date()}")
print(f"  Saved to    : {OUTPUT_PATH}")

# Latest month top/bottom stocks by LGBM alpha
latest = output[output["date"] == output["date"].max()].copy()
latest = latest.dropna(subset=["lgbm_alpha_z"]).sort_values("lgbm_alpha_z", ascending=False)

print(f"\n  Top 10 stocks (latest month, LGBM alpha):")
print(latest[["ticker", "lgbm_alpha_z"]].head(10).to_string(index=False))
print(f"\n  Bottom 10 stocks (latest month, LGBM alpha):")
print(latest[["ticker", "lgbm_alpha_z"]].tail(10).to_string(index=False))

print("\nDone.")
