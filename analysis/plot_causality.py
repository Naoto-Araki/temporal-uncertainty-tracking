"""A/B 因果判断と遅延 τ の関係を可視化するスクリプト。"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str | os.PathLike):
    os.makedirs(path, exist_ok=True)


def _load_tracking_csv(path: str | os.PathLike) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    if "choice" not in df.columns:
        print(f"[skip] {path}: choice 列が無いため A/B 解析は実行できません。")
        return None
    return df


def _to_trial_level(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    required_cols = ["trial", "tau", "choice"]
    for col in required_cols:
        if col not in working.columns:
            raise KeyError(f"必要な列 {col} が見つかりません。")

    group_cols = [col for col in ["participant", "condition", "trial"] if col in working.columns]
    if not group_cols:
        group_cols = ["trial"]

    agg_dict = {"tau": "first", "choice": "first"}
    if "choice_rt" in working.columns:
        agg_dict["choice_rt"] = "first"

    trial_df = working.groupby(group_cols, as_index=False).agg(agg_dict)
    trial_df = trial_df.dropna(subset=["tau"]).reset_index(drop=True)
    trial_df["choice"] = trial_df["choice"].astype(str).str.upper()
    return trial_df


def plot_choice_probability_by_tau(trial_df: pd.DataFrame, outdir: str, suffix: str = "") -> bool:
    if "choice" not in trial_df.columns or "tau" not in trial_df.columns:
        return False

    df = trial_df.dropna(subset=["choice", "tau"]).copy()
    if df.empty:
        return False

    df["is_A"] = (df["choice"].str.upper() == "A").astype(int)
    df = df.sort_values("tau")

    unique_tau = df["tau"].nunique()
    num_bins = max(4, min(12, unique_tau))
    tau_min = df["tau"].min()
    tau_max = df["tau"].max()
    if np.isclose(tau_min, tau_max):
        tau_max = tau_min + 1e-6  # avoid zero-width range so pd.cut works
    bins = np.linspace(tau_min, tau_max, num_bins + 1)
    df["tau_bin"] = pd.cut(df["tau"], bins=bins, include_lowest=True)

    bin_means = df.groupby("tau_bin", observed=True)["is_A"].mean()
    bin_centers = df.groupby("tau_bin", observed=True)["tau"].mean()

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.015, 0.015, size=len(df))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(df["tau"], df["is_A"] + jitter, s=20, alpha=0.4, color="#1f77b4", label="Trial")
    ax.plot(bin_centers, bin_means, color="#d62728", linewidth=2.0, label="Bin mean")
    ax.set_xlabel("Delay τ [s]")
    ax.set_ylabel("P(A judged)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    ax.set_title("Causality judgement vs delay")
    ax.legend(loc="best")

    _ensure_dir(outdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "choice_vs_tau"
    if suffix:
        fname += f"_{suffix}"
    fig.savefig(Path(outdir) / f"{fname}_{timestamp}.png", dpi=150)
    plt.close(fig)
    return True


def plot_choice_distribution_by_condition(trial_df: pd.DataFrame, outdir: str, suffix: str = "") -> bool:
    if "condition" not in trial_df.columns:
        return False

    df = trial_df.dropna(subset=["choice", "tau"]).copy()
    if df.empty:
        return False

    df["choice"] = df["choice"].str.upper()
    counts = df.groupby(["condition", "choice"]).size().unstack(fill_value=0)
    total = counts.sum(axis=1)
    proportions = counts.divide(total, axis=0)

    conditions = list(proportions.index)
    heights_A = proportions.get("A", pd.Series(0, index=conditions))
    heights_B = proportions.get("B", pd.Series(0, index=conditions))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax0 = axes[0]
    x = np.arange(len(conditions))
    ax0.bar(x, heights_A.reindex(conditions), label="A", color="#2ca02c")
    ax0.bar(x, heights_B.reindex(conditions), bottom=heights_A.reindex(conditions), label="B", color="#ff7f0e")
    ax0.set_xticks(x)
    ax0.set_xticklabels(conditions, rotation=20)
    ax0.set_ylabel("Proportion")
    ax0.set_title("Choice proportion by condition")
    ax0.set_ylim(0, 1)
    ax0.legend()

    ax1 = axes[1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(conditions)))
    for cond, color in zip(conditions, colors):
        ax1.hist(
            df.loc[df["condition"] == cond, "tau"],
            bins=20,
            alpha=0.45,
            density=True,
            label=cond,
            color=color,
        )
    ax1.set_xlabel("Delay τ [s]")
    ax1.set_ylabel("Density")
    ax1.set_title("τ distribution by condition")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    plt.tight_layout()
    _ensure_dir(outdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = "choice_condition_summary"
    if suffix:
        fname += f"_{suffix}"
    fig.savefig(Path(outdir) / f"{fname}_{timestamp}.png", dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="τ と A/B 因果判断の関係を可視化します。")
    parser.add_argument("sources", nargs="+", help="tracking_*.csv へのパス")
    parser.add_argument("--outdir", default="analysis/figures", help="図の保存先ディレクトリ")
    parser.add_argument("--tag", default="", help="ファイル名に付ける任意タグ")
    args = parser.parse_args()

    any_generated = False
    for source in args.sources:
        df = _load_tracking_csv(source)
        if df is None:
            continue
        trial_df = _to_trial_level(df)
        suffix = args.tag or Path(source).stem
        made_tau = plot_choice_probability_by_tau(trial_df, args.outdir, suffix)
        made_cond = plot_choice_distribution_by_condition(trial_df, args.outdir, suffix)
        any_generated = made_tau or made_cond or any_generated

    if not any_generated:
        print("選択列を含むデータが無く、図を作成できませんでした。")


if __name__ == "__main__":
    main()
