import argparse
import os
import numpy as np
import pandas as pd
from typing import Optional
from config import ANALYSIS

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
    中心差分で速度を推定する（端点は前/後進み差分）。
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
    cond が True の状態が min_duration_s 以上連続したとみなせる
    最初の時刻を返す（なければ None）。
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

def first_cross_time_pos(t: np.ndarray, y: np.ndarray, thresh_value: float, direction: str = "up"):
    """
    位置が閾値を初めて超える（または下回る）時刻を取得する。

    Args
    ----
    t : numpy.ndarray
        時刻列。
    y : numpy.ndarray
        位置列。
    thresh_value : float
        閾値。
    direction : str, optional
        "up" で y >= thresh を探し、"down" で y <= thresh を探す。

    Returns
    -------
    float | None
        条件を満たした最初の時刻。該当がなければ None。
    """
    if direction == "up":
        idx = np.where(y >= thresh_value)[0]
    else:
        idx = np.where(y <= thresh_value)[0]
    return float(t[idx[0]]) if idx.size > 0 else None

# ---------- メイン計算 ----------

def per_trial_metrics_position_based(g: pd.DataFrame,
                                     L: float,
                                     poswin_ms: float,
                                     start_margin_px: float,
                                     end_margin_px: float,
                                     T_for_truth: float,
                                     v_start_thresh: float,
                                     v_stop_thresh: float,
                                     hold_start_ms: float,
                                     hold_stop_ms: float,
                                     use_velocity_detection: bool = True):
    """
    単一トライアルに対して速度ベースの開始・終了検出と真値中心の各種指標を計算する。

    Args
    ----
    g : pandas.DataFrame
        1 トライアル分の時系列データ。
    L : float
        移動距離 [px]。スタート/ゴールの位置推定に使用。
    poswin_ms : float
        位置分散を算出する窓の半幅 [ms]。
    start_margin_px : float
        スタート位置からの閾値マージン [px]。
    end_margin_px : float
        ゴール位置からの閾値マージン [px]。
    T_for_truth : float
        理想タイミング（tau, tau+T）の中心として用いる運動時間 T [s]。
    v_start_thresh : float
        速度ベース開始検出の閾値 [px/s]。
    v_stop_thresh : float
        速度ベース終了検出の閾値 [px/s]。
    hold_start_ms : float
        開始検出の連続時間しきい値 [ms]。
    hold_stop_ms : float
        終了検出の連続時間しきい値 [ms]。
    use_velocity_detection : bool, optional
        True なら速度ベースで開始/終了検出を行う。

    Returns
    -------
    dict
        トライアル番号、tau、開始/終了時刻、位置分散、到達位置平均、追従誤差を含む辞書。
    """
    t   = g["t"].to_numpy()
    y_p = g["y_p"].to_numpy()
    tau = float(g["tau"].iloc[0])

    v_p = compute_velocity(y_p, t)

    # スタート/ゴールの物理座標（縦移動前提）
    y_start = -L/2.0
    y_goal  =  L/2.0

    # 閾値（位置）の定義：開始はスタート位置 + margin，終了はゴール位置 - margin
    start_threshold = y_start + start_margin_px   # これを上抜けしたら“動作開始”
    end_threshold   = y_goal  - end_margin_px     # これを上抜けしたら“動作終了”

    # --- 開始/終了の検出 ---
    if use_velocity_detection:
        # 上向き移動を想定：開始は v が v_start_thresh 以上、終了は |v| が v_stop_thresh 以下
        t_start = first_sustain_time(t, v_p >= v_start_thresh, hold_start_ms / 1000.0)
        t_end   = first_sustain_time(t, np.abs(v_p) <= v_stop_thresh, hold_stop_ms / 1000.0)
    else:
        # 旧（位置閾値）法
        t_start = first_cross_time_pos(t, y_p, y_start + start_margin_px, direction="up")
        t_end   = first_cross_time_pos(t, y_p, y_goal  - end_margin_px,   direction="up")

    # 位置分散をとる窓の中心（理想時刻を使用）
    center_start = tau
    center_end   = tau + T_for_truth

    # 分散窓幅（秒）
    half_w = (poswin_ms / 1000.0)

    if center_start is not None:
        idx_start = np.where((t >= center_start - half_w) & (t <= center_start + half_w))[0]
    else:
        idx_start = np.array([], dtype=int)

    if center_end is not None:
        idx_end = np.where((t >= center_end - half_w) & (t <= center_end + half_w))[0]
    else:
        idx_end = np.array([], dtype=int)

    # 位置分散（不十分なら NaN）
    pos_var_start = float(np.var(y_p[idx_start], ddof=1)) if idx_start.size > 1 else np.nan
    pos_var_end   = float(np.var(y_p[idx_end],   ddof=1)) if idx_end.size   > 1 else np.nan

    # 到達位置の平均（終了検出後の小窓平均：end の中心窓と同じ）
    y_end_mean = float(np.mean(y_p[idx_end])) if idx_end.size > 0 else np.nan

    # 真値ウィンドウにおける追従誤差（二乗平均; 分散ではなく MSE）
    in_truth = (t >= center_start) & (t <= center_end)
    if np.any(in_truth) and ("y_t" in g.columns):
        y_t = g["y_t"].to_numpy()
        mse_truth = float(np.mean((y_p[in_truth] - y_t[in_truth])**2))
    else:
        mse_truth = np.nan

    return {
        "trial": int(g["trial"].iloc[0]),
        "tau": tau,
        "t_start": t_start,
        "t_end":   t_end,
        "pos_var_start": pos_var_start,
        "pos_var_end":   pos_var_end,
        "y_end_mean":    y_end_mean,
        "mse_truth":     mse_truth,
    }

def summarize_by_condition(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    参加者×条件ごとにトライアル指標を集計する。

    Args
    ----
    df_trials : pandas.DataFrame
        `per_trial_metrics_position_based` の結果をまとめたデータフレーム。

    Returns
    -------
    pandas.DataFrame
        平均・標準偏差・件数を列に持つ集計結果。
    """
    agg = df_trials.groupby(["participant","condition"]).agg(
        n_trials           = ("trial","count"),
        t_start_mean       = ("t_start","mean"),
        t_start_std        = ("t_start","std"),
        t_end_mean         = ("t_end","mean"),
        t_end_std          = ("t_end","std"),
        pos_var_start_mean = ("pos_var_start","mean"),
        pos_var_start_std  = ("pos_var_start","std"),
        pos_var_end_mean   = ("pos_var_end","mean"),
        pos_var_end_std    = ("pos_var_end","std"),
        y_end_mean_mean    = ("y_end_mean","mean"),
        y_end_mean_std     = ("y_end_mean","std"),
        mse_truth_mean    = ("mse_truth","mean"),
        mse_truth_std     = ("mse_truth","std"),
    ).reset_index()
    return agg

def main():
    """
    コマンドライン引数を解析し、CSV を読み込んで真値中心の指標計算と結果保存を行うエントリーポイント。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="実験CSVファイルへのパス（participant, condition, trial, tau, t, y_t, x_p, y_p）")
    ap.add_argument("--L", type=float, default=ANALYSIS.get("L", 400.0), help="移動距離L[px]（縦: -L/2→+L/2）")
    ap.add_argument("--poswin-ms", type=float, default=ANALYSIS.get("poswin_ms", 100.0), help="位置分散をとる窓の半幅[ms]（開始/終了それぞれ）")
    ap.add_argument("--start-margin-px", type=float, default=ANALYSIS.get("start_margin_px", 20.0), help="開始検出の位置閾値（スタート位置からこのpxだけ内側）")
    ap.add_argument("--end-margin-px", type=float, default=ANALYSIS.get("end_margin_px", 20.0), help="終了検出の位置閾値（ゴール位置からこのpxだけ内側）")
    ap.add_argument("--T", type=float, default=ANALYSIS.get("T", 1.0), help="理想的な運動時間T[s]（窓中心 tau, tau+T に使用）")
    ap.add_argument("--v-start", type=float, default=ANALYSIS.get("v_start", 50.0), help="開始検出の速度閾値[px/s]（v >= しきい値）")
    ap.add_argument("--v-stop",  type=float, default=ANALYSIS.get("v_stop", 20.0), help="終了検出の速度閾値[px/s]（|v| <= しきい値）")
    ap.add_argument("--hold-start-ms", type=float, default=ANALYSIS.get("hold_start_ms", 80.0), help="開始検出の連続時間しきい値[ms]")
    ap.add_argument("--hold-stop-ms",  type=float, default=ANALYSIS.get("hold_stop_ms", 100.0), help="終了検出の連続時間しきい値[ms]")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--use-velocity",     dest="use_velocity", action="store_true",  help="速度ベースで開始/終了を検出する")
    group.add_argument("--no-use-velocity",  dest="use_velocity", action="store_false", help="位置しきい値ベースで開始/終了を検出する")
    ap.set_defaults(use_velocity=ANALYSIS.get("use_velocity", True))
    args = ap.parse_args()

    os.makedirs("analysis", exist_ok=True)

    df = load_session(args.csv)

    rows = []
    for (participant, condition, trial), g in df.groupby(["participant","condition","trial"], sort=True):
        g = g.sort_values("t")
        m = per_trial_metrics_position_based(
            g=g, L=args.L, poswin_ms=args.poswin_ms,
            start_margin_px=args.start_margin_px, end_margin_px=args.end_margin_px,
            T_for_truth=args.T,
            v_start_thresh=args.v_start,
            v_stop_thresh=args.v_stop,
            hold_start_ms=args.hold_start_ms,
            hold_stop_ms=args.hold_stop_ms,
            use_velocity_detection=args.use_velocity
        )
        m["participant"] = str(participant)
        m["condition"]   = str(condition)
        rows.append(m)

    df_trials = pd.DataFrame(rows).sort_values(["participant","condition","trial"])
    df_summary = summarize_by_condition(df_trials)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    out_trials  = os.path.join("analysis", f"{base}_pos_based_trials.csv")
    out_summary = os.path.join("analysis", f"{base}_pos_based_by_condition.csv")

    df_trials.to_csv(out_trials, index=False)
    df_summary.to_csv(out_summary, index=False)

    print(f"✅ per-trial metrics saved: {out_trials}")
    print(f"✅ by-condition summary saved: {out_summary}")
    print("\nColumns (per-trial):", ", ".join(df_trials.columns))
    print("Columns (summary):   ", ", ".join(df_summary.columns))

if __name__ == "__main__":
    main()
