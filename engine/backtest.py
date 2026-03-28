# ============================================================
#  BACKTESTING
#  Quintile portfolios, IC, ICIR, and cumulative returns
#  Requires: data/factors.parquet (already built)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

FACTORS_PATH = "data/lgbm_factors.parquet"
OUTPUT_DIR   = "data/backtest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD FACTOR PANEL ─────────────────────────────────────

print("Loading factor panel...")
panel = pd.read_parquet(FACTORS_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

print(f"  {len(panel):,} rows | {panel['ticker'].nunique():,} tickers")
print(f"  Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
print(f"  Factors: {[c for c in panel.columns if c.startswith('factor_')]}\n")

# ── 2. COMPUTE FORWARD RETURNS ───────────────────────────────
# Forward 1-month return: what the stock does NEXT month
# This is what the alpha score is trying to predict

print("Computing forward returns...")
panel = panel.sort_values(["ticker", "date"])
panel["fwd_ret_1m"] = panel.groupby("ticker")["ret_1m"].shift(-1)

# Drop the last month (no forward return available)
panel = panel.dropna(subset=["fwd_ret_1m"])
print(f"  Panel after forward return: {len(panel):,} rows\n")

# ── 3. INFORMATION COEFFICIENT (IC) ──────────────────────────
# IC = rank correlation between alpha score and forward return
# Measures how well the signal predicts next month's return
# IC > 0.05 is generally considered good; > 0.10 is strong

print("Computing Information Coefficient (IC)...")

factor_cols = [c for c in panel.columns if c.startswith("factor_")] + ["alpha_score", "lgbm_alpha_z"]

ic_results = {}

for factor in factor_cols:
    monthly_ic = (
        panel.dropna(subset=[factor, "fwd_ret_1m"])
        .groupby("date")
        .apply(lambda g: stats.spearmanr(g[factor], g["fwd_ret_1m"])[0])
        .reset_index()
    )
    monthly_ic.columns = ["date", "ic"]

    mean_ic  = monthly_ic["ic"].mean()
    std_ic   = monthly_ic["ic"].std()
    icir     = mean_ic / std_ic if std_ic > 0 else 0
    ic_pos   = (monthly_ic["ic"] > 0).mean()
    t_stat   = mean_ic / (std_ic / np.sqrt(len(monthly_ic)))

    ic_results[factor] = {
        "mean_ic"  : round(mean_ic, 4),
        "std_ic"   : round(std_ic, 4),
        "icir"     : round(icir, 4),
        "ic_pct_positive" : round(ic_pos, 3),
        "t_stat"   : round(t_stat, 3),
        "monthly_ic": monthly_ic
    }

    print(f"  {factor:20s} | IC={mean_ic:.4f} | ICIR={icir:.3f} | "
          f"IC>0: {ic_pos:.1%} | t={t_stat:.2f}")

print()

# ── 4. QUINTILE PORTFOLIO ANALYSIS ───────────────────────────
# Each month: rank stocks by alpha score into 5 buckets
# Q1 = top 20% (best alpha), Q5 = bottom 20% (worst alpha)
# Long-Short = Q1 - Q5 spread

print("Building quintile portfolios...")

def quintile_returns(panel, factor, n_quantiles=5):
    results = []

    for date, group in panel.dropna(subset=[factor, "fwd_ret_1m"]).groupby("date"):
        if len(group) < n_quantiles * 5:
            continue

        group = group.copy()
        group["quintile"] = pd.qcut(
            group[factor],
            q=n_quantiles,
            labels=[f"Q{i+1}" for i in range(n_quantiles)],
            duplicates="drop"
        )

        for q in [f"Q{i+1}" for i in range(n_quantiles)]:
            q_stocks = group[group["quintile"] == q]
            if len(q_stocks) > 0:
                results.append({
                    "date"     : date,
                    "quintile" : q,
                    "ret"      : q_stocks["fwd_ret_1m"].mean(),
                    "n_stocks" : len(q_stocks)
                })

    df = pd.DataFrame(results)

    # Add long-short
    ls_rows = []
    for date, g in df.groupby("date"):
        q1 = g[g["quintile"] == "Q1"]["ret"].values
        q5 = g[g["quintile"] == "Q5"]["ret"].values
        if len(q1) > 0 and len(q5) > 0:
            ls_rows.append({
                "date"    : date,
                "quintile": "LS",
                "ret"     : q1[0] - q5[0],
                "n_stocks": 0
            })

    df = pd.concat([df, pd.DataFrame(ls_rows)], ignore_index=True)
    return df

quint_df = quintile_returns(panel, "alpha_score")

# Summary stats by quintile
print("\n  Quintile monthly return stats (alpha_score):")
print(f"  {'Quintile':10s} {'Mean Ret':>10s} {'Ann. Ret':>10s} {'Sharpe':>8s}")
print(f"  {'-'*42}")

quint_summary = {}
for q in ["Q1","Q2","Q3","Q4","Q5","LS"]:
    q_data  = quint_df[quint_df["quintile"] == q]["ret"]
    mean_m  = q_data.mean()
    ann_ret = (1 + mean_m) ** 12 - 1
    sharpe  = mean_m / q_data.std() * np.sqrt(12) if q_data.std() > 0 else 0
    quint_summary[q] = {"mean_monthly": mean_m, "ann_ret": ann_ret, "sharpe": sharpe}
    print(f"  {q:10s} {mean_m:>10.4f} {ann_ret:>10.2%} {sharpe:>8.3f}")

print()

# ── 5. CUMULATIVE RETURNS CHART ───────────────────────────────

print("Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Alpha Model Backtest Results", fontsize=14, fontweight="bold")

# Plot 1: Cumulative returns by quintile
ax1 = axes[0, 0]
colors = {"Q1": "#1a7c4a", "Q3": "#888", "Q5": "#b22222", "LS": "#003f88"}

for q, color in colors.items():
    q_data = quint_df[quint_df["quintile"] == q].set_index("date")["ret"].sort_index()
    cum_ret = (1 + q_data).cumprod() - 1
    lw = 2.5 if q == "LS" else 1.5
    ax1.plot(cum_ret.index, cum_ret * 100, label=q, color=color, linewidth=lw)

ax1.set_title("Cumulative Returns by Quintile")
ax1.set_ylabel("Cumulative Return (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 2: Monthly IC over time
ax2 = axes[0, 1]
ic_data = ic_results["alpha_score"]["monthly_ic"].set_index("date")["ic"]
ax2.bar(ic_data.index, ic_data, width=20, color=["#1a7c4a" if v > 0 else "#b22222" for v in ic_data], alpha=0.7)
ax2.axhline(y=ic_data.mean(), color="navy", linestyle="--", linewidth=1.5, label=f"Mean IC = {ic_data.mean():.4f}")
ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_title("Monthly Information Coefficient (IC)")
ax2.set_ylabel("Spearman IC")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Plot 3: Mean return by quintile (bar chart)
ax3 = axes[1, 0]
qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ann_rets = [quint_summary[q]["ann_ret"] * 100 for q in qs]
bar_colors = ["#1a7c4a", "#4a9e6a", "#888", "#d4756a", "#b22222"]
bars = ax3.bar(qs, ann_rets, color=bar_colors, alpha=0.85)
ax3.axhline(y=0, color="black", linewidth=0.8)
ax3.set_title("Annualised Return by Quintile")
ax3.set_ylabel("Annualised Return (%)")
for bar, val in zip(bars, ann_rets):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Long-Short cumulative return
ax4 = axes[1, 1]
ls_data = quint_df[quint_df["quintile"] == "LS"].set_index("date")["ret"].sort_index()
ls_cum  = (1 + ls_data).cumprod() - 1
ax4.plot(ls_cum.index, ls_cum * 100, color="#003f88", linewidth=2)
ax4.fill_between(ls_cum.index, ls_cum * 100, 0,
                 where=(ls_cum >= 0), alpha=0.15, color="#1a7c4a")
ax4.fill_between(ls_cum.index, ls_cum * 100, 0,
                 where=(ls_cum < 0), alpha=0.15, color="#b22222")
ax4.axhline(y=0, color="black", linewidth=0.8)
ax4.set_title(f"Long-Short (Q1 - Q5) Cumulative Return\nSharpe: {quint_summary['LS']['sharpe']:.2f}")
ax4.set_ylabel("Cumulative Return (%)")
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "backtest_results.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved → {chart_path}")

# ── 6. SUMMARY REPORT ────────────────────────────────────────

print("\n══════════════════════════════════════════════════════")
print("  BACKTEST SUMMARY")
print("══════════════════════════════════════════════════════")

ls = quint_summary["LS"]
ic = ic_results["alpha_score"]

# Max drawdown for long-short
ls_data_sorted = quint_df[quint_df["quintile"] == "LS"].set_index("date")["ret"].sort_index()
cum = (1 + ls_data_sorted).cumprod()
rolling_max = cum.cummax()
drawdown = (cum - rolling_max) / rolling_max
max_dd = drawdown.min()

print(f"  Factor         : alpha_score (momentum + value)")
print(f"  Period         : {panel['date'].min().date()} → {panel['date'].max().date()}")
print(f"  Universe       : {panel['ticker'].nunique():,} NYSE tickers")
print(f"")
print(f"  ── Signal Quality ──────────────────────────────")
print(f"  Mean IC        : {ic['mean_ic']:.4f}")
print(f"  IC Std Dev     : {ic['std_ic']:.4f}")
print(f"  ICIR           : {ic['icir']:.3f}")
print(f"  IC > 0         : {ic['ic_pct_positive']:.1%} of months")
print(f"  t-statistic    : {ic['t_stat']:.2f}")
print(f"")
print(f"  ── Long-Short Portfolio ────────────────────────")
print(f"  Ann. Return    : {ls['ann_ret']:.2%}")
print(f"  Monthly Return : {ls['mean_monthly']:.4f}")
print(f"  Sharpe Ratio   : {ls['sharpe']:.3f}")
print(f"  Max Drawdown   : {max_dd:.2%}")
print(f"")
print(f"  ── Interpretation ──────────────────────────────")
ic_val = ic['mean_ic']
if abs(ic_val) > 0.05:
    print(f"  IC of {ic_val:.4f} is MEANINGFUL — signal has predictive power")
elif abs(ic_val) > 0.02:
    print(f"  IC of {ic_val:.4f} is WEAK but present — signal needs refinement")
else:
    print(f"  IC of {ic_val:.4f} is VERY WEAK — consider adding more factors")

icir_val = ic['icir']
if abs(icir_val) > 0.5:
    print(f"  ICIR of {icir_val:.3f} is STRONG — signal is consistent over time")
elif abs(icir_val) > 0.3:
    print(f"  ICIR of {icir_val:.3f} is MODERATE — some consistency")
else:
    print(f"  ICIR of {icir_val:.3f} is LOW — signal is noisy month-to-month")

print("══════════════════════════════════════════════════════\n")

# Save results
results_df = pd.DataFrame([{
    "mean_ic": ic["mean_ic"], "icir": ic["icir"],
    "ic_pct_positive": ic["ic_pct_positive"], "t_stat": ic["t_stat"],
    "ls_ann_ret": ls["ann_ret"], "ls_sharpe": ls["sharpe"],
    "ls_max_drawdown": max_dd
}])
results_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_summary.csv"), index=False)
quint_df.to_parquet(os.path.join(OUTPUT_DIR, "quintile_returns.parquet"), index=False)

print(f"Results saved to {OUTPUT_DIR}/")
print("Done.")
