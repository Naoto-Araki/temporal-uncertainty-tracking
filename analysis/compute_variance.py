import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

# どのディレクトリから実行しても config を参照できるようルートを検索パスに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ANALYSIS, L as L_PIX

# =====================
# 基本ユーティリティ
# =====================

def load_session(csv_path: str) -> pd.DataFrame:
    """
    実験 CSV を読み込み、必要な列の型を整える。

    Args
    ----
    csv_path : str
        `participant, condition, trial, tau, t, y_t, x_p, y_p` を含む CSV のパス。

    Returns
    -------
    pandas.DataFrame
        型変換済みで欠損行を除外したデータフレーム。
    """
    df = pd.read_csv(csv_path)
    df["participant"] = df["participant"].astype(str)
    df["condition"]   = df["condition"].astype(str)
    for c in ["trial"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["tau","t","y_t","x_p","y_p"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["trial","tau","t","y_t","x_p","y_p"])
    return df

def compute_velocity(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    サンプル値と時刻系列から速度列を推定する。
    中心差分で推定し、端点は片側差分。

    Args
    ----
    values : numpy.ndarray
        位置などのスカラー値配列。
    times : numpy.ndarray
        各サンプルの計測時刻（秒）。`values` と同じ長さを期待。

    Returns
    -------
    numpy.ndarray
        入力と同じ長さの速度配列。
    """
    v = np.zeros_like(values, dtype=float)
    n = len(values)
    if n == 0:
        return v
    if n == 1:
        v[0] = 0.0
        return v
    # 端点
    v[0]  = (values[1] - values[0]) / max(times[1] - times[0], 1e-12)
    v[-1] = (values[-1] - values[-2]) / max(times[-1] - times[-2], 1e-12)
    # 中心差分
    if n > 2:
        v[1:-1] = (values[2:] - values[:-2]) / np.clip(times[2:] - times[:-2], 1e-12, None)
    return v

def first_sustain_time(t: np.ndarray, cond: np.ndarray, min_duration_s: float) -> Optional[float]:
    """
    ブール条件が一定時間持続した箇所の開始時刻を検出する。

    Args
    ----
    t : numpy.ndarray
        各サンプルの時刻（秒）。
    cond : numpy.ndarray
        条件を示す True/False の配列。`t` と同じ長さ。
    min_duration_s : float
        条件が継続したと見なすための最小持続時間（秒）。

    Returns
    -------
    float | None
        条件が満たされた区間の開始時刻。該当が無ければ None。
    """
    if t.size == 0 or cond.size == 0 or t.size != cond.size:
        return None
    start_idx = None
    for i in range(cond.size):
        if cond[i]:
            if start_idx is None:
                start_idx = i
            # 連続時間がしきい値を超えたか？
            if (t[i] - t[start_idx]) >= min_duration_s:
                return float(t[start_idx])
        else:
            start_idx = None
    return None

def normalize_trial_0_L(g: pd.DataFrame, L: float) -> pd.DataFrame:
    """
    試行データの座標を 0→L に平行移動して正規化する。
    ここでは単純に [+L/2] 平行移動のみを行う（スケールは変更しない）。
    - y_p_norm = y_p + L/2
    - y_t_norm = y_t + L/2
    """
    g2 = g.copy()
    if "y_p" in g2.columns:
        g2["y_p_norm"] = g2["y_p"] + (L / 2.0)
    return g2

# =====================
#  trial レベルの指標
# =====================

def per_trial_metrics_triallevel(
    g: pd.DataFrame,
    T_for_truth: float,
    v_start_thresh: float,
    v_stop_thresh: float,
    hold_start_ms: float,
    hold_stop_ms: float,
    end_guard_s: Optional[float] = None,
) -> dict:
    """
    単一トライアルに対して、試行代表値のみを算出する。
      - 開始時刻 t_start（速度が一定時間上回った最初の時刻）
      - 終了時刻 t_end（開始後＋ガード時間以降で、速度が一定時間下回った最初の時刻）
      - 到達位置 y_end_final（t_end に最も近いサンプルの y 値。t_end が無ければ最終サンプル）

    ※ 0→L 正規化済み列 (y_p_norm) があればそれを使用。
    """
    t   = g["t"].to_numpy()
    y_col = "y_p_norm" if "y_p_norm" in g.columns else "y_p"
    y_p = g[y_col].to_numpy()
    tau = float(g["tau"].iloc[0])

    v_p = compute_velocity(y_p, t)

    # 開始検出（上方向に動く想定：v >= しきい値 が hold_start_ms 継続）
    t_start = first_sustain_time(t, v_p >= v_start_thresh, hold_start_ms / 1000.0)

    # 終了検出は「開始後＋ガード時間」以降に限定
    guard_s = ANALYSIS.get("end_guard_s", T_for_truth / 2.0) if end_guard_s is None else end_guard_s
    guard_s = max(float(guard_s), 0.0)
    if t_start is not None:
        time_mask = (t >= (t_start + guard_s))
    else:
        time_mask = np.ones_like(t, dtype=bool)

    stop_mask = (np.abs(v_p) <= v_stop_thresh)
    end_mask  = stop_mask & time_mask
    t_end = first_sustain_time(t, end_mask, hold_stop_ms / 1000.0)

    # y_end_final（試行の代表到達位置）
    if t_end is not None:
        # t_end に最も近いサンプルを採用
        idx = int(np.argmin(np.abs(t - t_end)))
        y_end_final = float(y_p[idx])
    else:
        # 終了が検出できなかった場合は最後のサンプルを代表値とする
        y_end_final = float(y_p[-1])

    # 相対時間ベースの指標（開始検出を0秒とみなす）
    if (t_start is not None) and (t_end is not None):
        t_end_rel = float(t_end - t_start)
        t_end_rel_offset = float(t_end_rel - T_for_truth)
    else:
        t_end_rel = np.nan
        t_end_rel_offset = np.nan

    # 参考：理想到達時刻（τ + T）
    t_end_ideal = tau + T_for_truth
    t_end_offset = (t_end - t_end_ideal) if t_end is not None else np.nan

    return {
        "trial": int(g["trial"].iloc[0]),
        "tau": tau,
        "t_start": t_start,
        "t_end":   t_end,
        "t_end_offset": t_end_offset,
        "t_end_rel": t_end_rel,
        "t_end_rel_offset": t_end_rel_offset,
        "y_end_final": y_end_final,
    }

# =====================
# 条件別集計
# =====================

def summarize_by_condition(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    参加者×条件ごとに、試行代表値（t_start, t_end, y_end_final）を集計する。
    """
    agg = df_trials.groupby(["participant","condition"]).agg(
        n_trials            = ("trial","count"),
        t_start_mean        = ("t_start","mean"),
        t_start_std         = ("t_start","std"),
        t_start_var         = ("t_start","var"),
        t_end_mean          = ("t_end","mean"),
        t_end_std           = ("t_end","std"),
        t_end_var           = ("t_end","var"),
        t_end_offset_mean   = ("t_end_offset","mean"),
        t_end_offset_std    = ("t_end_offset","std"),
        t_end_offset_var    = ("t_end_offset","var"),
        t_end_rel_mean      = ("t_end_rel","mean"),
        t_end_rel_std       = ("t_end_rel","std"),
        t_end_rel_var       = ("t_end_rel","var"),
        t_end_rel_offset_mean = ("t_end_rel_offset","mean"),
        t_end_rel_offset_std  = ("t_end_rel_offset","std"),
        t_end_rel_offset_var  = ("t_end_rel_offset","var"),
        y_end_final_mean    = ("y_end_final","mean"),
        y_end_final_std     = ("y_end_final","std"),
        y_end_final_var     = ("y_end_final","var"),
    ).reset_index()
    return agg

# =====================
# メイン
# =====================

def main():
    """
    CSV を読み込み、試行代表値の計算と要約を保存するエントリーポイント。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "csv",
        nargs="+",
        help="実験CSVファイルへのパス（participant, condition, trial, tau, t, y_t, x_p, y_p）。複数指定可。",
    )
    args = ap.parse_args()

    # config.ANALYSIS から固定取得（CLIでの上書きはしない）
    T_truth       = ANALYSIS.get("T", 1.0)
    v_start       = ANALYSIS.get("v_start", 50.0)
    v_stop        = ANALYSIS.get("v_stop", 20.0)
    hold_start_ms = ANALYSIS.get("hold_start_ms", 80.0)
    hold_stop_ms  = ANALYSIS.get("hold_stop_ms", 100.0)
    end_guard_s   = ANALYSIS.get("end_guard_s", T_truth / 2.0)
    normalize     = ANALYSIS.get("normalize_to_0_L", True)

    base_outdir = Path("analysis")
    trials_outdir = base_outdir / "trials"
    summary_outdir = base_outdir / "summary"
    trials_outdir.mkdir(parents=True, exist_ok=True)
    summary_outdir.mkdir(parents=True, exist_ok=True)

    SKIP_INITIAL = 0  # 無視したい試行数

    for csv_path in args.csv:
        print(f"\n=== Processing: {csv_path} ===")
        df = load_session(csv_path)
        if SKIP_INITIAL:
            df = df[df["trial"] > SKIP_INITIAL]

        rows = []
        for (participant, condition, trial), g in df.groupby(["participant","condition","trial"], sort=True):
            g = g.sort_values("t")
            # 解析段階で 0→L に正規化
            if normalize:
                g = normalize_trial_0_L(g, L_PIX)
            m = per_trial_metrics_triallevel(
                g=g,
                T_for_truth=T_truth,
                v_start_thresh=v_start,
                v_stop_thresh=v_stop,
                hold_start_ms=hold_start_ms,
                hold_stop_ms=hold_stop_ms,
                end_guard_s=end_guard_s
            )
            m["participant"] = str(participant)
            m["condition"]   = str(condition)
            rows.append(m)

        if not rows:
            print("⚠️ No trials remained after filtering; skipping.")
            continue

        df_trials = pd.DataFrame(rows).sort_values(["participant","condition","trial"])
        df_summary = summarize_by_condition(df_trials)

        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_trials  = trials_outdir / f"{base}_triallevel_trials.csv"
        out_summary = summary_outdir / f"{base}_triallevel_by_condition.csv"

        df_trials.to_csv(out_trials, index=False)
        df_summary.to_csv(out_summary, index=False)

        print(f"✅ per-trial metrics saved: {out_trials}")
        print(f"✅ by-condition summary saved: {out_summary}")
        if normalize:
            print("Note: Positions were normalized to 0→L during analysis.")
        print("\nColumns (per-trial):", ", ".join(df_trials.columns))
        print("Columns (summary):   ", ", ".join(df_summary.columns))
if __name__ == "__main__":
    main()
