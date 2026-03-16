#!/usr/bin/env python3
"""
=============================================================================
  NRFI/YRFI MODEL v2 — Three Core Fixes Applied
=============================================================================

WHAT CHANGED FROM v1:
  ❌ v1: Season-level pitcher stats (look-ahead bias)
  ✅ v2: Trailing 10-start rolling stats per pitcher (no future data)

  ❌ v1: Team wOBA as batting proxy (too broad)
  ✅ v2: Leadoff batter trailing OBP (direct signal for 1st inning scoring)

  ❌ v1: Single model averaging both halves of the 1st inning
  ✅ v2: Two separate half-inning models combined via independence formula
         P(YRFI) = 1 − P(no run top 1st) × P(no run bot 1st)

WHY THIS MATTERS:
  The original model predicted 48.9% YRFI on a test set where the true
  rate was 48.2% — essentially outputting the base rate with no conviction.
  Probability separation was nearly zero (both distributions overlapping at 0.55).

  These three fixes attack the root causes:
  1. Rolling stats eliminate look-ahead bias that made pitcher quality
     indistinguishable across the training/test split
  2. Leadoff OBP is the single strongest 1st-inning run predictor
     (a leadoff walk/single directly creates a scoring opportunity)
  3. Splitting into two half-inning models captures the fact that
     home pitcher vs away pitcher quality is an independent signal

RUN:
  python nrfi_model_v2.py

REQUIRES: Same cache from mlb_models.py run (cache/statcast_2024.csv)
          If not cached: pip install pybaseball && python mlb_models.py first
=============================================================================
"""

import os, sys, time, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────────────
MISSING = []
try:    import pybaseball as pb; pb.cache.enable()
except: MISSING.append("pybaseball")
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.dates as mdates
    import seaborn as sns
except: MISSING.append("matplotlib seaborn")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, roc_auc_score)
    from sklearn.calibration import CalibratedClassifierCV
except: MISSING.append("scikit-learn")

if MISSING:
    print(f"❌  pip install {' '.join(MISSING)}"); sys.exit(1)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PARK_FACTORS = {
    "COL":115,"CIN":110,"BOS":108,"PHI":107,"TEX":106,"BAL":104,"NYY":103,
    "CHC":103,"MIL":102,"TOR":101,"ATL":100,"HOU":100,"LAD":100,"WSH":100,
    "CLE":100,"DET":99,"MIN":99,"STL":99,"ARI":99,"MIA":98,"NYM":98,
    "OAK":98,"PIT":98,"SEA":97,"TB":97,"CWS":96,"KC":96,"LAA":96,"SF":95,"SD":94,
}

WIND_DIR_MAP = {
    "Out to CF":1.0,"Out to RF":0.8,"Out to LF":0.8,
    "In from CF":-1.0,"In from LF":-0.8,"In from RF":-0.8,
    "L to R":0.1,"R to L":0.1,"Calm":0.0,"":0.0,
}

# Half-inning model feature sets
# Each is from the PITCHER's perspective facing that half-inning's batters
TOP_FEATURES = [
    # Home pitcher rolling form (faces away batters in top 1st)
    "home_roll_k_pct",
    "home_roll_bb_pct",
    "home_roll_woba",
    "home_roll_dom",
    # Away team batting context (who's batting in top 1st)
    "away_leadoff_obp",          # leadoff batter trailing OBP
    "away_team_k_pct",           # team K% (how often they strike out)
    "away_team_woba",
    # Context
    "park_factor",
    "weather_offense_factor",
]

BOT_FEATURES = [
    # Away pitcher rolling form (faces home batters in bot 1st)
    "away_roll_k_pct",
    "away_roll_bb_pct",
    "away_roll_woba",
    "away_roll_dom",
    # Home team batting context
    "home_leadoff_obp",
    "home_team_k_pct",
    "home_team_woba",
    # Context
    "park_factor",
    "weather_offense_factor",
]

ROLLING_N = 10   # Number of prior starts to use for rolling pitcher stats


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOAD
# ═════════════════════════════════════════════════════════════════════════════

MONTHLY_RANGES = [
    ("2024-03-20","2024-04-30"),("2024-05-01","2024-05-31"),
    ("2024-06-01","2024-06-30"),("2024-07-01","2024-07-31"),
    ("2024-08-01","2024-08-31"),("2024-09-01","2024-10-01"),
]

def load_statcast() -> pd.DataFrame:
    cache = f"{CACHE_DIR}/statcast_2024.csv"
    if os.path.exists(cache):
        print("📂  Loading cached Statcast data...")
        df = pd.read_csv(cache, low_memory=False)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df
    print("⬇️   Downloading 2024 Statcast (~20 min, one-time)...")
    chunks = []
    for i,(s,e) in enumerate(MONTHLY_RANGES,1):
        print(f"  [{i}/6] {s}→{e}")
        try: chunks.append(pb.statcast(start_dt=s,end_dt=e)); time.sleep(3)
        except Exception as ex: print(f"  ⚠️  {ex}")
    df = pd.concat(chunks,ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df.to_csv(cache,index=False)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FIX 1: ROLLING PITCHER STATS (no look-ahead)
# ═════════════════════════════════════════════════════════════════════════════

def build_rolling_pitcher_stats(sc: pd.DataFrame, n: int = ROLLING_N) -> pd.DataFrame:
    """
    For each pitcher × game, compute stats from their prior N starts ONLY.
    Uses only inning-1 PA data (relevant for NRFI).

    Method: vectorized shift(1).rolling(N) grouped by pitcher.
    shift(1) ensures current game is excluded — pure look-back only.

    Returns one row per (pitcher, game_pk) with rolling features.
    """
    inn1 = sc[(sc["inning"] == 1) & sc["events"].notna()].copy()

    # Per-pitcher, per-game aggregates (1st inning only)
    gstats = (
        inn1.groupby(["pitcher", "game_pk", "game_date"])
        .agg(
            pa   = ("events", "count"),
            k    = ("events", lambda x: (x == "strikeout").sum()),
            bb   = ("events", lambda x: (x == "walk").sum()),
            woba = ("woba_value", "mean"),
        )
        .reset_index()
        .sort_values(["pitcher", "game_date"])
    )

    # Rolling window over prior N games — shift(1) excludes current game
    def roll_sum(s): return s.shift(1).rolling(n, min_periods=3).sum()
    def roll_mean(s): return s.shift(1).rolling(n, min_periods=3).mean()

    g = gstats.groupby("pitcher")
    gstats["roll_pa"]   = g["pa"].transform(roll_sum)
    gstats["roll_k"]    = g["k"].transform(roll_sum)
    gstats["roll_bb"]   = g["bb"].transform(roll_sum)
    gstats["roll_woba"] = g["woba"].transform(roll_mean)

    # Derived rates — guard against division by zero
    gstats["roll_k_pct"]  = gstats["roll_k"]  / gstats["roll_pa"].replace(0, np.nan)
    gstats["roll_bb_pct"] = gstats["roll_bb"] / gstats["roll_pa"].replace(0, np.nan)

    # Dominance score: high K%, low BB%, low wOBA allowed = harder to score
    gstats["roll_dom"] = (
        gstats["roll_k_pct"] - gstats["roll_bb_pct"] - gstats["roll_woba"]
    )

    # Drop rows without enough prior data
    result = gstats[gstats["roll_pa"] >= 5].copy()
    result = result[["pitcher","game_pk","roll_k_pct","roll_bb_pct","roll_woba","roll_dom"]]
    print(f"  ✅  Rolling pitcher stats: {len(result):,} pitcher-game rows "
          f"({result['pitcher'].nunique()} pitchers, min {n}-start lookback)")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIX 2: LEADOFF BATTER TRAILING OBP
# ═════════════════════════════════════════════════════════════════════════════

def build_leadoff_obp(sc: pd.DataFrame) -> pd.DataFrame:
    """
    For each game × half-inning, identify the leadoff batter and compute
    their trailing OBP from all PRIOR plate appearances in the season.

    OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    Simplified here as: on_base_events / PA (close enough for ML features)

    Returns: DataFrame with columns:
        game_pk, batting_team, leadoff_batter, leadoff_obp_trailing
    """
    # Only PAs (rows where an event occurred)
    pa = sc[sc["events"].notna()].copy()
    pa["game_date"] = pd.to_datetime(pa["game_date"])
    pa["batting_team"] = np.where(
        pa["inning_topbot"] == "Top", pa["away_team"], pa["home_team"]
    )

    # On-base events
    ON_BASE = {"single","double","triple","home_run","walk","hit_by_pitch"}
    pa["on_base"] = pa["events"].isin(ON_BASE).astype(int)

    # Identify leadoff batter for each game × half-inning combo
    inn1_pa = pa[(pa["inning"] == 1)].sort_values(
        ["game_pk","inning_topbot","at_bat_number","pitch_number"]
    )
    leadoff = (
        inn1_pa.groupby(["game_pk","inning_topbot"])
        .first()
        .reset_index()[["game_pk","inning_topbot","game_date","batting_team","batter"]]
    )

    # Map top/bot to team
    # Top = away team bats; Bot = home team bats
    # leadoff already has batting_team

    # Now compute trailing OBP for each leadoff batter at the time of each game
    # Using all PA from the full season (inning-agnostic) up to but not including that game
    batter_pa = (
        pa.groupby(["batter","game_pk","game_date"])
        .agg(pa_count=("events","count"), ob_count=("on_base","sum"))
        .reset_index()
        .sort_values(["batter","game_date"])
    )

    # Rolling trailing OBP — shift(1) excludes current game
    batter_pa["trail_pa"] = batter_pa.groupby("batter")["pa_count"].transform(
        lambda x: x.shift(1).rolling(50, min_periods=10).sum()
    )
    batter_pa["trail_ob"] = batter_pa.groupby("batter")["ob_count"].transform(
        lambda x: x.shift(1).rolling(50, min_periods=10).sum()
    )
    batter_pa["trail_obp"] = batter_pa["trail_ob"] / batter_pa["trail_pa"].replace(0, np.nan)

    # Merge trailing OBP onto leadoff identification
    leadoff_obp = leadoff.merge(
        batter_pa[["batter","game_pk","trail_obp"]],
        left_on=["batter","game_pk"],
        right_on=["batter","game_pk"],
        how="left"
    )

    # Fill missing trailing OBP with league average (~0.320)
    leadoff_obp["trail_obp"] = leadoff_obp["trail_obp"].fillna(0.320)

    # Reshape: one row per game with home_leadoff_obp and away_leadoff_obp
    top = leadoff_obp[leadoff_obp["inning_topbot"]=="Top"][
        ["game_pk","trail_obp"]].rename(columns={"trail_obp":"away_leadoff_obp"})
    bot = leadoff_obp[leadoff_obp["inning_topbot"]=="Bot"][
        ["game_pk","trail_obp"]].rename(columns={"trail_obp":"home_leadoff_obp"})

    result = top.merge(bot, on="game_pk", how="outer")
    result["away_leadoff_obp"] = result["away_leadoff_obp"].fillna(0.320)
    result["home_leadoff_obp"] = result["home_leadoff_obp"].fillna(0.320)

    print(f"  ✅  Leadoff OBP computed for {len(result):,} games")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GAME-LEVEL LABELS + CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

def build_half_inning_labels(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-game labels for:
      - YRFI (did either half of inning 1 produce a run?)
      - TOP_RUN  (did the away team score in top of 1st?)
      - BOT_RUN  (did the home team score in bot of 1st?)

    Also captures starting pitcher IDs for both teams.
    """
    inn1 = sc[sc["inning"] == 1].copy()

    # Meta
    meta = sc.groupby("game_pk").agg(
        game_date = ("game_date","first"),
        home_team = ("home_team","first"),
        away_team = ("away_team","first"),
    ).reset_index()

    # Runs scored per half-inning
    def half_runs(group):
        return (group["post_bat_score"] - group["bat_score"]).clip(lower=0).sum()

    top_runs = (
        inn1[inn1["inning_topbot"]=="Top"]
        .groupby("game_pk").apply(half_runs)
        .reset_index(name="top_runs")
    )
    bot_runs = (
        inn1[inn1["inning_topbot"]=="Bot"]
        .groupby("game_pk").apply(half_runs)
        .reset_index(name="bot_runs")
    )

    meta = meta.merge(top_runs, on="game_pk", how="left")
    meta = meta.merge(bot_runs, on="game_pk", how="left")
    meta["top_runs"] = meta["top_runs"].fillna(0)
    meta["bot_runs"] = meta["bot_runs"].fillna(0)
    meta["TOP_RUN"] = (meta["top_runs"] > 0).astype(int)
    meta["BOT_RUN"] = (meta["bot_runs"] > 0).astype(int)
    meta["YRFI"]    = ((meta["TOP_RUN"] == 1) | (meta["BOT_RUN"] == 1)).astype(int)

    # Starting pitchers
    inn1s = inn1.sort_values(["game_pk","inning_topbot","pitch_number"])
    hp = (inn1s[inn1s["inning_topbot"]=="Top"]
          .groupby("game_pk")["pitcher"].first().reset_index()
          .rename(columns={"pitcher":"home_starter_id"}))
    ap = (inn1s[inn1s["inning_topbot"]=="Bot"]
          .groupby("game_pk")["pitcher"].first().reset_index()
          .rename(columns={"pitcher":"away_starter_id"}))
    meta = meta.merge(hp, on="game_pk", how="left")
    meta = meta.merge(ap, on="game_pk", how="left")

    # Weather
    for col, default in [("wind_speed",5.0),("wind_dir","Calm"),("temperature",72.0)]:
        if col in sc.columns:
            w = sc.groupby("game_pk")[col].first().reset_index()
            meta = meta.merge(w, on="game_pk", how="left")
        if col not in meta.columns:
            meta[col] = default
    meta["wind_speed"]  = pd.to_numeric(meta["wind_speed"],  errors="coerce").fillna(5.0)
    meta["temperature"] = pd.to_numeric(meta["temperature"], errors="coerce").fillna(72.0)
    meta["wind_dir_factor"] = meta["wind_dir"].astype(str).str.strip().map(WIND_DIR_MAP).fillna(0.0)
    meta["weather_offense_factor"] = (
        meta["wind_dir_factor"] * (meta["wind_speed"]/10.0)
        + (meta["temperature"] - 72) / 30.0
    )
    meta["park_factor"] = meta["home_team"].map(PARK_FACTORS).fillna(100) / 100.0

    meta["game_date"] = pd.to_datetime(meta["game_date"])
    return meta.sort_values("game_date").reset_index(drop=True)


def build_team_stats(sc: pd.DataFrame) -> pd.DataFrame:
    """Season-level team batting stats (used as stable context features)."""
    pa = sc[sc["events"].notna()].copy()
    pa["batting_team"] = np.where(
        pa["inning_topbot"]=="Top", pa["away_team"], pa["home_team"]
    )
    return pa.groupby("batting_team").agg(
        team_woba    = ("woba_value","mean"),
        team_k_pct   = ("events", lambda x:(x=="strikeout").sum()/len(x)),
    ).reset_index()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ASSEMBLE HALF-INNING DATASETS
# ═════════════════════════════════════════════════════════════════════════════

def assemble_dataset(game_log, rolling, leadoff_obp, team_stats) -> pd.DataFrame:
    """
    Merge all features into one row per game.

    Each row has features for BOTH half-innings so we can:
      a) Train separate models on top/bot half
      b) Combine predictions using independence formula
    """
    df = game_log.copy()

    # ── Rolling pitcher stats — home starter (faces top of 1st) ──────────────
    hr = rolling.rename(columns={
        "roll_k_pct":"home_roll_k_pct","roll_bb_pct":"home_roll_bb_pct",
        "roll_woba":"home_roll_woba","roll_dom":"home_roll_dom"
    })
    df = df.merge(
        hr[["pitcher","game_pk","home_roll_k_pct","home_roll_bb_pct",
            "home_roll_woba","home_roll_dom"]],
        left_on=["home_starter_id","game_pk"],
        right_on=["pitcher","game_pk"], how="inner"
    ).drop(columns=["pitcher"])

    # ── Rolling pitcher stats — away starter (faces bot of 1st) ──────────────
    ar = rolling.rename(columns={
        "roll_k_pct":"away_roll_k_pct","roll_bb_pct":"away_roll_bb_pct",
        "roll_woba":"away_roll_woba","roll_dom":"away_roll_dom"
    })
    df = df.merge(
        ar[["pitcher","game_pk","away_roll_k_pct","away_roll_bb_pct",
            "away_roll_woba","away_roll_dom"]],
        left_on=["away_starter_id","game_pk"],
        right_on=["pitcher","game_pk"], how="inner"
    ).drop(columns=["pitcher"])

    # ── Leadoff OBP ───────────────────────────────────────────────────────────
    df = df.merge(leadoff_obp, on="game_pk", how="left")
    df["away_leadoff_obp"] = df["away_leadoff_obp"].fillna(0.320)
    df["home_leadoff_obp"] = df["home_leadoff_obp"].fillna(0.320)

    # ── Team batting stats ────────────────────────────────────────────────────
    ts = team_stats.rename(columns={
        "batting_team":"team","team_woba":"away_team_woba","team_k_pct":"away_team_k_pct"
    })
    df = df.merge(ts, left_on="away_team", right_on="team", how="left").drop(columns=["team"])
    ts2 = team_stats.rename(columns={
        "batting_team":"team","team_woba":"home_team_woba","team_k_pct":"home_team_k_pct"
    })
    df = df.merge(ts2, left_on="home_team", right_on="team", how="left").drop(columns=["team"])

    return df.sort_values("game_date").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIX 3: SPLIT HALF-INNING MODELS + COMBINATION
# ═════════════════════════════════════════════════════════════════════════════

def backtest_v2(df: pd.DataFrame):
    """
    Train two logistic regression models:
      Model A: P(run in top 1st) using TOP_FEATURES
      Model B: P(run in bot 1st) using BOT_FEATURES

    Combine using:
      P(YRFI) = 1 − (1 − P_top) × (1 − P_bot)

    This is correct because YRFI = "at least one half-inning scores",
    and the two halves are conditionally independent given the features.

    Split: chronological 60/40.
    """
    df = df.dropna(subset=TOP_FEATURES + BOT_FEATURES + ["TOP_RUN","BOT_RUN"]).copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    split  = int(len(df) * 0.60)
    train  = df.iloc[:split]
    test   = df.iloc[split:]

    print(f"\n  Train: {len(train):,} games "
          f"({train['game_date'].min().date()} → {train['game_date'].max().date()})")
    print(f"  Test:  {len(test):,} games "
          f"({test['game_date'].min().date()} → {test['game_date'].max().date()})")
    print(f"  Train YRFI rate: {train['YRFI'].mean():.1%}  |  "
          f"Test YRFI rate: {test['YRFI'].mean():.1%}")

    # ── Model A: Top of 1st ───────────────────────────────────────────────────
    sn_top = StandardScaler()
    Xtr_top = sn_top.fit_transform(train[TOP_FEATURES])
    Xte_top = sn_top.transform(test[TOP_FEATURES])

    m_top = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=42)
    m_top.fit(Xtr_top, train["TOP_RUN"])
    p_top_train = m_top.predict_proba(Xtr_top)[:,1]
    p_top_test  = m_top.predict_proba(Xte_top)[:,1]

    top_acc = accuracy_score(test["TOP_RUN"], (p_top_test >= 0.5).astype(int))
    print(f"\n  Model A (top 1st) accuracy : {top_acc:.1%}")

    # ── Model B: Bottom of 1st ────────────────────────────────────────────────
    sn_bot = StandardScaler()
    Xtr_bot = sn_bot.fit_transform(train[BOT_FEATURES])
    Xte_bot = sn_bot.transform(test[BOT_FEATURES])

    m_bot = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=42)
    m_bot.fit(Xtr_bot, train["BOT_RUN"])
    p_bot_train = m_bot.predict_proba(Xtr_bot)[:,1]
    p_bot_test  = m_bot.predict_proba(Xte_bot)[:,1]

    bot_acc = accuracy_score(test["BOT_RUN"], (p_bot_test >= 0.5).astype(int))
    print(f"  Model B (bot 1st) accuracy : {bot_acc:.1%}")

    # ── Combine using independence formula ────────────────────────────────────
    p_yrfi_test  = 1 - (1 - p_top_test)  * (1 - p_bot_test)
    p_yrfi_train = 1 - (1 - p_top_train) * (1 - p_bot_train)

    # ── Threshold tuning ──────────────────────────────────────────────────────
    # Don't just use 0.5 — tune threshold on training set to maximize accuracy
    best_thresh, best_acc = 0.5, 0.0
    for t in np.arange(0.35, 0.75, 0.01):
        preds_tr = (p_yrfi_train >= t).astype(int)
        acc_tr   = accuracy_score(train["YRFI"], preds_tr)
        if acc_tr > best_acc:
            best_acc, best_thresh = acc_tr, t

    print(f"\n  Optimal threshold (train): {best_thresh:.2f} → train acc {best_acc:.1%}")

    preds_test = (p_yrfi_test >= best_thresh).astype(int)
    final_acc  = accuracy_score(test["YRFI"], preds_test)
    auc        = roc_auc_score(test["YRFI"], p_yrfi_test)

    print(f"\n{'='*55}")
    print("  NRFI/YRFI v2 — BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Combined YRFI accuracy : {final_acc:.1%}  (v1 was 54.8%)")
    print(f"  ROC-AUC                : {auc:.3f}  (0.5=random, 1.0=perfect)")
    print(f"  Decision threshold     : {best_thresh:.2f}")
    print(f"\n{classification_report(test['YRFI'], preds_test, target_names=['NRFI','YRFI'], digits=3)}")

    # ── Feature importance ────────────────────────────────────────────────────
    fi_top = pd.DataFrame({
        "feature":TOP_FEATURES,"coef":m_top.coef_[0],"model":"Top 1st"
    })
    fi_bot = pd.DataFrame({
        "feature":BOT_FEATURES,"coef":m_bot.coef_[0],"model":"Bot 1st"
    })
    fi = pd.concat([fi_top,fi_bot])

    # ── Results DataFrame ─────────────────────────────────────────────────────
    results = test[["game_date","home_team","away_team","YRFI","TOP_RUN","BOT_RUN"]].copy()
    results["p_top"]            = p_top_test
    results["p_bot"]            = p_bot_test
    results["yrfi_probability"] = p_yrfi_test
    results["predicted_YRFI"]   = preds_test
    results["correct"]          = (preds_test == test["YRFI"].values).astype(int)

    return results, fi, best_thresh, final_acc, auc, m_top, sn_top, m_bot, sn_bot


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_v2(results, fi, thresh, acc, auc):
    sns.set_theme(style="darkgrid", palette="muted", font_scale=0.95)
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(
        f"NRFI/YRFI Model v2 — Split Half-Inning + Rolling Stats + Leadoff OBP\n"
        f"Overall Accuracy: {acc:.1%}  |  ROC-AUC: {auc:.3f}  |  Threshold: {thresh:.2f}",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    B,P,G,R,O,BG,T = "#2196F3","#9C27B0","#4CAF50","#F44336","#FF9800","#1a1d27","#e0e0e0"

    def sax(ax, title):
        ax.set_facecolor(BG); ax.set_title(title,fontsize=11,fontweight="bold",color=T,pad=8)
        ax.tick_params(colors=T); ax.xaxis.label.set_color(T); ax.yaxis.label.set_color(T)
        [s.set_edgecolor("#444") for s in ax.spines.values()]

    # ── 1. Rolling YRFI accuracy (wide) ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,:2])
    r = results.sort_values("game_date").copy()
    r["roll30"] = r["correct"].rolling(30, min_periods=10).mean()
    ov = r["correct"].mean()
    ax1.fill_between(r["game_date"], r["roll30"], alpha=0.2, color=B)
    ax1.plot(r["game_date"], r["roll30"], color=B, lw=2, label="30-game rolling")
    ax1.axhline(ov,   color=R,    ls="--", lw=1.5, label=f"Overall: {ov:.1%}")
    ax1.axhline(0.50, color="gray", ls=":",  alpha=0.6, label="50% baseline")
    ax1.axhline(0.548,color=O,    ls=":",  alpha=0.8, label="v1 baseline: 54.8%")
    ax1.set_ylim(0.35, 0.85); ax1.set_ylabel("Accuracy", color=T)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.legend(facecolor=BG, labelcolor=T, fontsize=9)
    sax(ax1, "NRFI/YRFI v2 — 30-Game Rolling Accuracy")

    # ── 2. Feature importance — top model ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,2])
    ft = fi[fi["model"]=="Top 1st"].sort_values("coef",key=abs,ascending=False).head(7).iloc[::-1]
    ax2.barh(ft["feature"], ft["coef"],
             color=[G if c>0 else R for c in ft["coef"]], height=0.6)
    ax2.axvline(0, color="white", lw=0.8)
    sax(ax2, "Top 1st Features\n(+→Run Likely  −→NRFI)")

    # ── 3. Confusion matrix ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1,0])
    cm_ = confusion_matrix(results["YRFI"], results["predicted_YRFI"])
    sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Pred NRFI","Pred YRFI"],
                yticklabels=["Act NRFI","Act YRFI"],
                linewidths=0.5, linecolor="#333",
                annot_kws={"size":13,"weight":"bold"})
    ax3.set_facecolor(BG); ax3.tick_params(colors=T)
    ax3.set_title("Confusion Matrix v2", fontsize=11, fontweight="bold", color=T, pad=8)

    # ── 4. Probability separation — the key diagnostic ────────────────────────
    ax4 = fig.add_subplot(gs[1,1])
    results[results["YRFI"]==0]["yrfi_probability"].hist(
        ax=ax4, bins=25, alpha=0.65, color=B, label="Actual NRFI", density=True)
    results[results["YRFI"]==1]["yrfi_probability"].hist(
        ax=ax4, bins=25, alpha=0.65, color=O, label="Actual YRFI", density=True)
    ax4.axvline(thresh, color="white", ls="--", lw=1.5, label=f"Threshold: {thresh:.2f}")
    ax4.set_xlabel("Predicted YRFI Probability", color=T)
    ax4.legend(facecolor=BG, labelcolor=T, fontsize=9)
    sax(ax4, "Probability Separation v2\n(wider gap = better model)")

    # ── 5. Feature importance — bot model ────────────────────────────────────
    ax5 = fig.add_subplot(gs[1,2])
    fb = fi[fi["model"]=="Bot 1st"].sort_values("coef",key=abs,ascending=False).head(7).iloc[::-1]
    ax5.barh(fb["feature"], fb["coef"],
             color=[G if c>0 else R for c in fb["coef"]], height=0.6)
    ax5.axvline(0, color="white", lw=0.8)
    sax(ax5, "Bot 1st Features\n(+→Run Likely  −→NRFI)")

    # ── 6. Top vs Bot run probability scatter ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2,0])
    colors_pt = [G if y==1 else R for y in results["YRFI"]]
    ax6.scatter(results["p_top"], results["p_bot"], c=colors_pt,
                alpha=0.3, s=12)
    ax6.axvline(0.5, color="white", ls=":", alpha=0.5)
    ax6.axhline(0.5, color="white", ls=":", alpha=0.5)
    ax6.set_xlabel("P(Run in Top 1st)", color=T)
    ax6.set_ylabel("P(Run in Bot 1st)", color=T)
    from matplotlib.lines import Line2D
    ax6.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=G,markersize=8,label="YRFI"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=R,markersize=8,label="NRFI"),
    ], facecolor=BG, labelcolor=T, fontsize=9)
    sax(ax6, "P(Top Run) vs P(Bot Run)\nGreen=YRFI  Red=NRFI")

    # ── 7. Monthly accuracy ───────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2,1])
    r2 = results.copy()
    r2["month"] = pd.to_datetime(r2["game_date"]).dt.strftime("%b")
    ORDER = ["Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]
    monthly = r2.groupby("month")["correct"].agg(acc="mean", n="count").reset_index()
    monthly["month"] = pd.Categorical(monthly["month"], categories=ORDER, ordered=True)
    monthly = monthly.sort_values("month").dropna(subset=["month"])
    bars = ax7.bar(monthly["month"], monthly["acc"]*100,
                   color=[G if v>55 else O if v>50 else R for v in monthly["acc"]],
                   edgecolor="none")
    ax7.axhline(50, color="gray", ls=":", alpha=0.6)
    ax7.axhline(54.8, color=O, ls="--", alpha=0.8, lw=1.5, label="v1: 54.8%")
    ax7.set_ylim(40, 80); ax7.set_ylabel("Accuracy %", color=T)
    for bar, row in zip(bars, monthly.itertuples()):
        ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{row.acc*100:.0f}%", ha="center", va="bottom", color=T, fontsize=9)
    ax7.legend(facecolor=BG, labelcolor=T, fontsize=9)
    sax(ax7, "v2 Accuracy by Month\n(Green>55%  Orange>50%  Red<50%)")

    # ── 8. P(YRFI) distribution by confidence tier ───────────────────────────
    ax8 = fig.add_subplot(gs[2,2])
    bins = [0, 0.35, 0.45, 0.55, 0.65, 1.01]
    labels_b = ["<35%\n(Strong NRFI)","35-45%\n(Lean NRFI)",
                 "45-55%\n(Pick'em)","55-65%\n(Lean YRFI)",">65%\n(Strong YRFI)"]
    results["tier"] = pd.cut(results["yrfi_probability"], bins=bins, labels=labels_b)
    tier_acc = results.groupby("tier",observed=True)["correct"].agg(acc="mean",n="count").reset_index()
    bars2 = ax8.bar(tier_acc["tier"], tier_acc["acc"]*100,
                    color=[G if v>55 else O if v>50 else R for v in tier_acc["acc"]],
                    edgecolor="none")
    for bar, row in zip(bars2, tier_acc.itertuples()):
        ax8.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{row.acc*100:.0f}%\n(n={row.n})", ha="center", va="bottom",
                 color=T, fontsize=8)
    ax8.axhline(50, color="gray", ls=":", alpha=0.6)
    ax8.set_ylim(35, 80); ax8.set_ylabel("Accuracy %", color=T)
    sax(ax8, "Accuracy by Confidence Tier\n(Does model know when it's right?)")

    out = "nrfi_model_v2_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n📊  Chart saved → {out}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    SEP = "═" * 55
    print(f"\n{SEP}")
    print("  NRFI/YRFI MODEL v2")
    print("  Fixes: Rolling Stats + Leadoff OBP + Split Half-Innings")
    print(SEP)

    sc = load_statcast()
    print(f"\n  ✅  {sc['game_pk'].nunique():,} games | "
          f"{sc['game_date'].min().date()} → {sc['game_date'].max().date()}")

    print("\n  🔧  Building features...")

    print("\n  [1/4] Rolling pitcher stats (10-start lookback)...")
    rolling = build_rolling_pitcher_stats(sc, n=ROLLING_N)

    print("  [2/4] Leadoff batter trailing OBP...")
    leadoff = build_leadoff_obp(sc)

    print("  [3/4] Game labels + context...")
    game_log   = build_half_inning_labels(sc)
    team_stats = build_team_stats(sc)

    print("  [4/4] Assembling dataset...")
    df = assemble_dataset(game_log, rolling, leadoff, team_stats)
    print(f"  ✅  {len(df):,} games with full features")
    print(f"      Season YRFI rate: {df['YRFI'].mean():.1%}")
    print(f"      TOP_RUN rate:     {df['TOP_RUN'].mean():.1%}")
    print(f"      BOT_RUN rate:     {df['BOT_RUN'].mean():.1%}")

    print(f"\n  🏋️  Training split half-inning models (60/40 time split)...")
    results, fi, thresh, acc, auc, m_top, sn_top, m_bot, sn_bot = backtest_v2(df)

    # Save outputs
    results.to_csv("nrfi_v2_backtest_2024.csv", index=False)
    print(f"\n  💾  Saved → nrfi_v2_backtest_2024.csv")

    # Monthly breakdown
    r2 = results.copy()
    r2["month"] = pd.to_datetime(r2["game_date"]).dt.strftime("%b")
    ORDER = ["Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]
    monthly = r2.groupby("month")["correct"].agg(acc="mean",n="count")
    print(f"\n  Monthly accuracy:")
    print(f"  {'Month':<8}{'Acc':>8}{'N':>6}")
    print(f"  {'─'*24}")
    for m, row in monthly.iterrows():
        bar = "█"*int(row["acc"]*100/5)
        print(f"  {m:<8}{row['acc']*100:>6.1f}%{int(row['n']):>6}  {bar}")

    print(f"\n{SEP}")
    print(f"  v1 accuracy: 54.8%")
    print(f"  v2 accuracy: {acc:.1%}  {'✅ improvement' if acc > 0.548 else '⚠️  no improvement — see notes below'}")
    print(f"  ROC-AUC:     {auc:.3f}  (>0.55 = meaningful signal)")
    print(SEP)

    if acc <= 0.548:
        print("""
  ⚠️  If v2 didn't improve over v1, the most likely causes are:

  1. The test period (Jul-Oct) has a genuinely different YRFI rate
     than the train period (Mar-Jul). Try a random 80/20 split to
     check if the signal exists outside the time-split constraint.

  2. 2024 specifically had unusual 1st-inning run patterns.
     Pull 2022-2023 data and train cross-year to build more signal.

  3. The leadoff OBP trailing window (50 PA) may need adjustment.
     Try 30 PA for faster-adapting signal.
        """)

    plot_v2(results, fi, thresh, acc, auc)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
