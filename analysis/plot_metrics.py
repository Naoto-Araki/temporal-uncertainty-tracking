# analysis/plot_metrics.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= ヘルパ =========
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _per_condition_x(condition_labels):
    # 条件ごとに x 位置を返す（0,1,2,...）
    uniq = list(pd.unique(condition_labels))
    mapper = {c:i for i,c in enumerate(uniq)}
    return np.array([mapper[c] for c in condition_labels]), uniq


def _drop_initial_trials(df: pd.DataFrame, drop_count: int, group_cols=("participant", "condition"), order_col="trial") -> pd.DataFrame:
    """指定したグループごとに先頭の試行を除外した DataFrame を返す。"""
    if drop_count <= 0:
        return df.copy()

    working = df.copy()
    valid_group_cols = [c for c in group_cols if c in working.columns]

    def _trim(group):
        ordered = group.sort_values(order_col) if order_col in group.columns else group
        return ordered.iloc[drop_count:]

    if not valid_group_cols:
        ordered = working.sort_values(order_col) if order_col in working.columns else working
        trimmed = ordered.iloc[drop_count:]
    else:
        trimmed = working.groupby(valid_group_cols, group_keys=False).apply(_trim)

    return trimmed.reset_index(drop=True)


def _collect_condition_variability(df: pd.DataFrame):
    """時刻／位置のばらつきを条件ごとに計算して Series をまとめる。"""
    if "condition" not in df.columns:
        raise KeyError("condition 列が見つかりません")

    df = df.dropna(subset=["condition"]).copy()
    if df.empty:
        return {}

    df["condition"] = df["condition"].astype(str)
    grouped = df.groupby("condition")

    metrics = {}

    if "t_start" in df.columns:
        start_sd = grouped["t_start"].std()
        if not start_sd.dropna().empty:
            metrics["Start time SD [s]"] = start_sd

    if "t_end_rel" in df.columns:
        end_sd = grouped["t_end_rel"].std()
        if not end_sd.dropna().empty:
            metrics["End time SD (relative) [s]"] = end_sd
    elif "t_end" in df.columns:
        end_sd = grouped["t_end"].std()
        if not end_sd.dropna().empty:
            metrics["End time SD [s]"] = end_sd

    def _pick_column(candidates):
        for name in candidates:
            if name in df.columns:
                return name
        return None

    col_ideal = _pick_column(["y_end_final", "y_end_mean_norm", "y_end_mean"])
    col_detect = _pick_column(["y_end_mean_dynamic_norm", "y_end_mean_dynamic"])

    if col_ideal is not None:
        var = grouped[col_ideal].var(ddof=1)
        if not var.dropna().empty:
            label = "End position variance (detected timing) [px²]" if col_ideal == "y_end_final" else "End position variance (ideal timing) [px²]"
            metrics[label] = var

    if col_detect is not None:
        var = grouped[col_detect].var(ddof=1)
        if not var.dropna().empty:
            metrics["End position variance (alternate) [px²]"] = var

    return metrics

# ========= 1) 相対時間ベースのばらつき =========
def _extract_timestamp_from_filename(path: str) -> str:
    """ファイル名から YYYYMMDD_HHMMSS 形式のタイムスタンプを抽出する。"""
    name = os.path.basename(path)
    match = re.search(r"_(\d{8}_\d{6})(?:_|\.csv$)", name)
    if match:
        return match.group(1)
    match = re.search(r"(\d{8}_\d{6})", name)
    if match:
        return match.group(1)
    return ""


def plot_time_variability_bar(df_trials: pd.DataFrame, outdir="analysis", filename_suffix=""):
    """
    条件ごとの時刻分散（標準偏差）を棒グラフで可視化。
    - 開始：t_start の試行間 SD
    - 終了：t_end_rel があればその SD、無ければ t_end の SD
    """
    _ensure_dir(outdir)
    if "t_start" not in df_trials.columns:
        return

    df = df_trials.copy()
    df["condition"] = df["condition"].astype(str)
    g = df.groupby("condition")

    start_sd = g["t_start"].std()

    if "t_end_rel" in df.columns:
        end_sd = g["t_end_rel"].std()
        end_label = "End SD (relative)"
        y_label = "Std. Dev. of time [s] (relative end)"
    else:
        end_sd = g["t_end"].std()
        end_label = "End SD"
        y_label = "Std. Dev. of time [s]"

    # インデックス合わせ
    end_sd = end_sd.reindex(start_sd.index)

    x = np.arange(len(start_sd.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x - width/2, start_sd.values, width=width, alpha=0.85, label="Start SD")
    ax.bar(x + width/2, end_sd.values,   width=width, alpha=0.85, label=end_label, color="orange")

    ax.set_xticks(x)
    ax.set_xticklabels(list(start_sd.index))
    ax.set_ylabel(y_label)
    ax.set_title("Timing variability by condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "time_variability_bar"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)

def plot_time_variability_bar_multi(
    trial_sources,
    labels=None,
    outdir="analysis",
    filename_suffix="",
):
    """
    複数の試行レベル CSV をまとめて読み込み、条件ごとの開始／終了時刻分散を比較表示する。

    Parameters
    ----------
    trial_sources : Sequence[Union[str, os.PathLike, pd.DataFrame]]
        読み込む試行レベル CSV へのパス、または既存の DataFrame。
    labels : Optional[Sequence[str]]
        図中に表示するデータセット名。None の場合は自動で Dataset 1, 2, ... を使用。
    outdir : str
        図の出力先ディレクトリ。
    filename_suffix : str
        ファイル名に付与する任意のサフィックス。
    """
    if not trial_sources:
        raise ValueError("trial_sources must contain at least one dataset.")

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(trial_sources))]
    if len(labels) != len(trial_sources):
        raise ValueError("labels must have the same length as trial_sources.")

    _ensure_dir(outdir)

    prepared = []
    end_labels = set()
    for label, source in zip(labels, trial_sources):
        if isinstance(source, (str, os.PathLike)):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError("Each trial source must be a path or a pandas DataFrame.")

        if "condition" not in df.columns or "t_start" not in df.columns:
            raise KeyError(f"{label}: 必要な列（condition, t_start）が見つかりません。")

        df = df.dropna(subset=["t_start"])
        if df.empty:
            continue

        df["condition"] = df["condition"].astype(str)
        g = df.groupby("condition")
        start_sd = g["t_start"].std()

        end_sd = None
        end_label = None
        if "t_end_rel" in df.columns:
            end_sd = g["t_end_rel"].std()
            end_label = "End SD (relative)"
        elif "t_end" in df.columns:
            end_sd = g["t_end"].std()
            end_label = "End SD"

        if end_sd is not None:
            end_labels.add(end_label)

        prepared.append(
            {
                "label": label,
                "start_sd": start_sd,
                "end_sd": end_sd,
                "end_label": end_label,
            }
        )

    if not prepared:
        raise ValueError("No datasets with valid timing information were found.")

    conditions = set()
    for item in prepared:
        conditions.update(item["start_sd"].dropna().index.tolist())
        if item["end_sd"] is not None:
            conditions.update(item["end_sd"].dropna().index.tolist())

    if not conditions:
        raise ValueError("No conditions available after computing variability.")

    conditions = sorted(conditions)

    metrics = ["Start SD"]
    if any(item["end_sd"] is not None for item in prepared):
        metrics.append(next(iter(end_labels)) if len(end_labels) == 1 else "End SD")

    bars_per_condition = len(metrics) * len(prepared)
    bar_width = min(0.8 / max(bars_per_condition, 1), 0.22)
    base_x = np.arange(len(conditions))

    cmap = plt.get_cmap("tab10", len(prepared))
    metric_hatches = ["", "//", ".."]
    metric_alphas = [0.9, 0.65, 0.55]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    dataset_handles = {}
    for dataset_idx, item in enumerate(prepared):
        color = cmap(dataset_idx)

        value_lookup = {
            "Start SD": item["start_sd"].reindex(conditions).values,
        }
        if len(metrics) > 1:
            end_metric = metrics[-1]
            if item["end_sd"] is not None:
                value_lookup[end_metric] = item["end_sd"].reindex(conditions).values
            else:
                value_lookup[end_metric] = np.full(len(conditions), np.nan)

        for metric_idx, metric in enumerate(metrics):
            slot = dataset_idx * len(metrics) + metric_idx
            offset = (slot - (bars_per_condition - 1) / 2.0) * bar_width
            x = base_x + offset
            hatch = metric_hatches[metric_idx % len(metric_hatches)]
            alpha = metric_alphas[metric_idx % len(metric_alphas)]
            bars = ax.bar(
                x,
                value_lookup.get(metric, np.full(len(conditions), np.nan)),
                width=bar_width,
                color=color,
                alpha=alpha,
                hatch=hatch,
                label=None,
            )
            if metric_idx == 0 and item["label"] not in dataset_handles:
                dataset_handles[item["label"]] = bars[0] if len(bars) else None

    ax.set_xticks(base_x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Std. Dev. of time [s]")
    ax.set_title("Timing variability comparison across datasets")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    metric_handles = []
    for metric_idx, metric in enumerate(metrics):
        hatch = metric_hatches[metric_idx % len(metric_hatches)]
        alpha = metric_alphas[metric_idx % len(metric_alphas)]
        metric_handles.append(
            Patch(
                facecolor="lightgray",
                edgecolor="black",
                hatch=hatch,
                alpha=alpha,
                label=metric,
            )
        )

    dataset_order = [item["label"] for item in prepared if item["label"] in dataset_handles]
    dataset_handles_list = [dataset_handles[label] for label in dataset_order if dataset_handles[label] is not None]
    dataset_labels_list = [label for label in dataset_order if dataset_handles[label] is not None]

    handles = dataset_handles_list + metric_handles
    labels = dataset_labels_list + [m.get_label() for m in metric_handles]
    ax.legend(handles, labels, ncol=max(1, len(handles) // 3 + 1))

    plt.tight_layout()
    fname = "time_variability_comparison"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)

# ========= 2.5) 相対終了時間の散布図（平均付き） =========

# ========= 2) 相対終了時間のオフセット散布図 =========
# (removed) plot_end_time_offsets_scatter: consolidated into plot_end_time_relative_summary

# ========= 2.5) 相対終了時間の散布図（平均付き） =========
def plot_end_time_relative_summary(df_trials: pd.DataFrame, T: float, outdir="analysis", filename_suffix=""):
    """
    検出開始を 0 秒とした相対終了時間 t_end_rel を散布図で可視化し、
    条件平均の t_end_rel と理想時間 T を重ねて表示する。
    出力: analysis/figures/relative_end_time_with_means.png
    """
    _ensure_dir(outdir)

    if "t_end_rel" not in df_trials.columns:
        return

    df = df_trials.copy()
    df = df.dropna(subset=["t_end_rel"])
    if df.empty:
        return

    df["condition"] = df["condition"].astype(str)
    df["duration"] = df["t_end_rel"]
    df["ideal_time"] = T
    ideal_label = "Ideal duration (T)"

    cond = df["condition"].values
    x, order = _per_condition_x(cond)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    # 1) 各試行の duration をジッター散布
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(x)) - 0.5) * 0.20
    ax.scatter(x + jitter, df["duration"].values, s=20, alpha=0.7, label="Relative end time (per trial)")

    has_abs_end_mean = False  # 条件平均の絶対終了時刻を描くかどうか

    # 2) 条件ごとの平均 duration と 平均 ideal_time
    for i, c in enumerate(order):
        sub = df.loc[df["condition"] == c]
        if len(sub) == 0:
            continue
        mean_t_end = sub["duration"].mean()
        mean_ideal = sub["ideal_time"].mean()
        # 平均 t_end（丸）
        ax.scatter([i], [mean_t_end], marker="o", s=80, linewidths=1.2, edgecolors="k", label=None)
        # 平均 ideal（赤三角）
        ax.scatter([i], [mean_ideal], marker="^", s=80, color="red", label=None)
        # 平均開始オフセット（t_start - tau）と相対終了時間の平均から絶対終了時刻を算出
        if {"t_start", "tau"}.issubset(sub.columns):
            rel_start = (sub["t_start"] - sub["tau"]).dropna()
            if not rel_start.empty and not np.isnan(mean_t_end):
                mean_abs_end = rel_start.mean() + mean_t_end
                ax.scatter([i], [mean_abs_end], marker="s", s=90, color="green", edgecolors="k", linewidths=0.8, label=None)
                has_abs_end_mean = True

    # 凡例（重複防止のためハンドルをユニーク化）
    handles, labels = ax.get_legend_handles_labels()
    # 追加説明のためダミーラインを作る
    mean_t_handle = plt.Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='C0', markersize=8, label='Mean relative end time')
    mean_ideal_handle = plt.Line2D([0], [0], marker='^', linestyle='None', color='red', markersize=8, label=ideal_label)
    handles = [handles[0], mean_t_handle, mean_ideal_handle]
    labels = ["Relative end time (per trial)", "Mean relative end time", ideal_label]
    if has_abs_end_mean:
        mean_abs_end_handle = plt.Line2D([0], [0], marker='s', linestyle='None', markeredgecolor='k', markerfacecolor='green', markersize=8, label='Mean end time (actual-start aligned)')
        handles.append(mean_abs_end_handle)
        labels.append("Mean end time (actual-start aligned)")
    ax.legend(handles, labels)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Duration after detected start [s]")
    ax.set_title("Relative end times by condition (with means & ideal)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "relative_end_time_with_means"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)

def plot_mean_end_time_bar(df_trials: pd.DataFrame, T: float, outdir="analysis", filename_suffix=""):
    """
    条件ごとの相対終了時間（t_end_rel）の平均を棒グラフ表示。
    理想時間 T を赤破線で重ねる。
    """
    _ensure_dir(outdir)
    if "t_end_rel" not in df_trials.columns:
        return

    df = df_trials.copy()
    df["condition"] = df["condition"].astype(str)
    mean_rel = df.groupby("condition")["t_end_rel"].mean().sort_index()
    if mean_rel.empty:
        return

    x = np.arange(len(mean_rel.index))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x, mean_rel.values, width=0.45, alpha=0.85)
    ax.axhline(T, color="red", linestyle="--", linewidth=1.5, label="Ideal duration (T)")

    ax.set_xticks(x)
    ax.set_xticklabels(list(mean_rel.index))
    ax.set_ylabel("Mean relative end time [s]")
    ax.set_title("Mean relative end time by condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "mean_end_time_bar"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)
# (removed) plot_end_time_relative_scatter: simplified into plot_end_time_relative_summary

# ========= 4) 終了位置の散布図（ideal / detected 両方） =========
def plot_end_position_with_mean_scatter(df_trials: pd.DataFrame, L: float, outdir="analysis", filename_suffix=""):
    """
    条件ごとの終了位置（試行代表値 y_end_final）を散布し、条件平均も重ねて可視化する。
    - 青点：各試行の終了位置（検出タイミングベース y_end_final）
    - 黒縁丸：条件ごとの平均終了位置
    - 灰破線：目標位置（0→L 正規化済みなら L）
    出力: analysis/figures/end_position_with_mean_scatter.png
    """
    _ensure_dir(outdir)

    if "y_end_final" not in df_trials.columns:
        # 代替列の探索（将来拡張用）
        candidates = [c for c in ["y_end_mean_norm", "y_end_mean"] if c in df_trials.columns]
        if not candidates:
            return
        col = candidates[0]
    else:
        col = "y_end_final"

    df = df_trials.dropna(subset=[col]).copy()
    if df.empty:
        return

    df["condition"] = df["condition"].astype(str)
    cond = df["condition"].values
    x, order = _per_condition_x(cond)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    rng = np.random.default_rng(11)
    jitter = (rng.random(len(x)) - 0.5) * 0.20
    ax.scatter(x + jitter, df[col].values, s=22, alpha=0.75, label="Per-trial end position")

    # 条件ごとの平均を重ねる
    means = df.groupby("condition")[col].mean().reindex(order)
    ax.scatter(np.arange(len(order)), means.values, s=90, marker="o",
               facecolors="white", edgecolors="black", linewidths=1.2,
               label="Mean end position")

    # 目標位置の参照線（0→L 正規化を簡易判定）
    vmin, vmax = np.nanmin(df[col].values), np.nanmax(df[col].values)
    goal = L if (vmin >= -1e-6 and vmax <= L + 1e-6) else L/2.0
    ax.axhline(goal, color="gray", linestyle="--", linewidth=1, label="Goal")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("End position [px]")
    ax.set_title("End positions by condition (with means)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fname = "end_position_with_mean_scatter"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)


def plot_end_position_variability_bar(df_trials: pd.DataFrame, outdir="analysis", filename_suffix=""):
    """
    条件ごとの終了位置の試行間分散を棒グラフで可視化する。
    利用できる列に合わせて凡例の表記を切り替える。
    """
    _ensure_dir(outdir)

    def _pick_column(candidates):
        for name in candidates:
            if name in df_trials.columns:
                return name
        return None

    col_ideal = _pick_column(["y_end_final", "y_end_mean_norm", "y_end_mean"])
    if col_ideal is None:
        raise KeyError("終了位置を示す列が見つかりません (例: y_end_final)")
    col_detect = _pick_column(["y_end_mean_dynamic_norm", "y_end_mean_dynamic"])

    label_ideal = "Detected timing" if col_ideal == "y_end_final" else "Ideal timing"
    if col_detect in ("y_end_mean_dynamic_norm", "y_end_mean_dynamic"):
        label_detect = "Detected timing"
    elif col_detect is not None:
        label_detect = "Alternate metric"
    else:
        label_detect = None

    df_trials = df_trials.copy()
    df_trials["condition"] = df_trials["condition"].astype(str)
    g = df_trials.groupby("condition")

    var_ideal = g[col_ideal].var(ddof=1).sort_index()

    has_detect = col_detect is not None
    if has_detect:
        var_detect = g[col_detect].var(ddof=1).sort_index().reindex(var_ideal.index)

    x = np.arange(len(var_ideal.index))
    w = 0.12

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x - w / 2, var_ideal.values, width=w, label=label_ideal)
    if has_detect and label_detect is not None:
        ax.bar(x + w / 2, var_detect.values, width=w, label=label_detect)

    n = len(var_ideal.index)
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(list(var_ideal.index))
    ax.set_ylabel("Across-trial variance of end position [px²]")
    ax.set_title("End-position variability by condition")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fname = "end_position_variability_bar"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)

def plot_end_position_variability_bar_multi(
    trial_sources,
    labels=None,
    outdir="analysis",
    filename_suffix="",
):
    """
    複数の試行レベル CSV から終了位置の分散を計算し、条件ごとに比較表示する。

    Parameters
    ----------
    trial_sources : Sequence[Union[str, os.PathLike, pd.DataFrame]]
        読み込む試行レベル CSV へのパス、または既存の DataFrame。
    labels : Optional[Sequence[str]]
        図中に表示するデータセット名。None の場合は Dataset 1,2,... を自動付与。
    outdir : str
        図の出力先ディレクトリ。
    filename_suffix : str
        ファイル名に付与する任意のサフィックス。
    """
    if not trial_sources:
        raise ValueError("trial_sources must contain at least one dataset.")

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(trial_sources))]
    if len(labels) != len(trial_sources):
        raise ValueError("labels must have the same length as trial_sources.")

    _ensure_dir(outdir)

    prepared = []
    metric_labels_order = []

    def _register_metric(label):
        if label not in metric_labels_order:
            metric_labels_order.append(label)

    for label, source in zip(labels, trial_sources):
        if isinstance(source, (str, os.PathLike)):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError("Each trial source must be a path or a pandas DataFrame.")

        if "condition" not in df.columns:
            raise KeyError(f"{label}: 必要な列（condition）が見つかりません。")

        df = df.dropna(subset=["condition"])
        if df.empty:
            continue

        def _pick_column(candidates):
            for name in candidates:
                if name in df.columns:
                    return name
            return None

        col_ideal = _pick_column(["y_end_final", "y_end_mean_norm", "y_end_mean"])
        if col_ideal is None:
            continue
        col_detect = _pick_column(["y_end_mean_dynamic_norm", "y_end_mean_dynamic"])

        metrics_for_dataset = {}

        df_tmp = df.copy()
        df_tmp["condition"] = df_tmp["condition"].astype(str)
        grouped = df_tmp.groupby("condition")

        var_ideal = grouped[col_ideal].var(ddof=1)
        ideal_label = "Detected timing" if col_ideal == "y_end_final" else "Ideal timing"
        metrics_for_dataset[ideal_label] = var_ideal
        _register_metric(ideal_label)

        if col_detect is not None:
            detect_label = "Detected timing" if col_detect in ("y_end_mean_dynamic_norm", "y_end_mean_dynamic") else "Alternate metric"
            var_detect = grouped[col_detect].var(ddof=1)
            metrics_for_dataset[detect_label] = var_detect
            _register_metric(detect_label)

        prepared.append(
            {
                "label": label,
                "metrics": metrics_for_dataset,
            }
        )

    if not prepared:
        raise ValueError("No datasets with valid end-position information were found.")

    conditions = set()
    for item in prepared:
        for series in item["metrics"].values():
            conditions.update(series.dropna().index.tolist())

    if not conditions:
        raise ValueError("No conditions available after computing end-position variability.")

    conditions = sorted(conditions)
    metrics = metric_labels_order or ["End position variance"]

    bars_per_condition = len(metrics) * len(prepared)
    bar_width = min(0.8 / max(bars_per_condition, 1), 0.18)
    base_x = np.arange(len(conditions))

    cmap = plt.get_cmap("tab10", len(prepared))
    metric_hatches = ["", "//", "..", "xx"]
    metric_alphas = [0.9, 0.65, 0.55, 0.45]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    dataset_handles = {}
    for dataset_idx, item in enumerate(prepared):
        color = cmap(dataset_idx)
        metrics_map = item["metrics"]

        for metric_idx, metric in enumerate(metrics):
            slot = dataset_idx * len(metrics) + metric_idx
            offset = (slot - (bars_per_condition - 1) / 2.0) * bar_width
            x = base_x + offset
            series = metrics_map.get(metric, pd.Series(index=conditions, dtype=float))
            values = series.reindex(conditions).values

            hatch = metric_hatches[metric_idx % len(metric_hatches)]
            alpha = metric_alphas[metric_idx % len(metric_alphas)]
            bars = ax.bar(
                x,
                values,
                width=bar_width,
                color=color,
                alpha=alpha,
                hatch=hatch,
                label=None,
            )
            if metric_idx == 0 and item["label"] not in dataset_handles:
                dataset_handles[item["label"]] = bars[0] if len(bars) else None

    ax.set_xticks(base_x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Across-trial variance of end position [px²]")
    ax.set_title("End-position variability comparison across datasets")
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch

    metric_handles = []
    for metric_idx, metric in enumerate(metrics):
        hatch = metric_hatches[metric_idx % len(metric_hatches)]
        alpha = metric_alphas[metric_idx % len(metric_alphas)]
        metric_handles.append(
            Patch(
                facecolor="lightgray",
                edgecolor="black",
                hatch=hatch,
                alpha=alpha,
                label=metric,
            )
        )

    dataset_order = [item["label"] for item in prepared if item["label"] in dataset_handles]
    dataset_handles_list = [dataset_handles[label] for label in dataset_order if dataset_handles[label] is not None]
    dataset_labels_list = [label for label in dataset_order if dataset_handles[label] is not None]

    handles = dataset_handles_list + metric_handles
    labels = dataset_labels_list + [m.get_label() for m in metric_handles]
    ax.legend(handles, labels, ncol=max(1, len(handles) // 3 + 1))

    plt.tight_layout()
    fname = "end_position_variability_comparison"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)


def plot_variability_vs_exclusion(
    trial_sources,
    drop_counts=(0, 5, 10),
    labels=None,
    group_cols=("participant", "condition"),
    order_col="trial",
    outdir="analysis",
    filename_suffix="",
):
    """
    1 つ以上の試行レベル CSV から、各条件（またはデータセット）の時刻／位置ばらつきを
    「先頭 n 試行を除外」した複数バージョンで比較する折れ線グラフを描画する。
    """

    if isinstance(trial_sources, (str, os.PathLike, pd.DataFrame)):
        sources = [trial_sources]
    else:
        sources = list(trial_sources)

    if not sources:
        raise ValueError("trial_sources には 1 つ以上の CSV か DataFrame を指定してください。")

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(sources))]
    if len(labels) != len(sources):
        raise ValueError("labels の長さが trial_sources と一致しません。")

    if not isinstance(drop_counts, (list, tuple)):
        drop_counts = [drop_counts]

    normalized_counts = []
    for count in drop_counts:
        try:
            normalized_counts.append(int(count))
        except (TypeError, ValueError):
            raise ValueError(f"drop_counts に整数変換できない値が含まれています: {count}")

    metric_lines = {}

    for dataset_label, source in zip(labels, sources):
        if isinstance(source, (str, os.PathLike)):
            df_trials = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df_trials = source.copy()
        else:
            raise TypeError("trial_sources には CSV パスか DataFrame を指定してください")

        for drop_count in normalized_counts:
            variant_df = _drop_initial_trials(
                df_trials,
                drop_count,
                group_cols=group_cols,
                order_col=order_col,
            )
            if variant_df.empty:
                continue

            metrics = _collect_condition_variability(variant_df)
            if not metrics:
                continue

            for metric_name, series in metrics.items():
                metric_dict = metric_lines.setdefault(metric_name, {})
                for condition_name, value in series.dropna().items():
                    if dataset_label:
                        line_label = f"{dataset_label} / {condition_name}"
                    else:
                        line_label = str(condition_name)
                    line_data = metric_dict.setdefault(line_label, {})
                    line_data[drop_count] = float(value)

    if not metric_lines:
        raise ValueError("有効な指標を計算できませんでした。列や入力ファイルを確認してください。")

    unique_drop_counts = sorted(set(normalized_counts))
    if not unique_drop_counts:
        raise ValueError("drop_counts に有効な値がありません。")

    _ensure_dir(outdir)

    metric_names = list(metric_lines.keys())
    n_metrics = len(metric_names)
    fig_height = 3.0 * max(n_metrics, 1)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(6.0, max(fig_height, 3.2)), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    all_line_labels = sorted({label for metric_dict in metric_lines.values() for label in metric_dict.keys()})
    color_map = plt.get_cmap("tab10", max(len(all_line_labels), 1))
    color_lookup = {label: color_map(idx) for idx, label in enumerate(all_line_labels)}

    x = np.array(unique_drop_counts, dtype=float)

    for metric_idx, metric_name in enumerate(metric_names):
        ax = axes[metric_idx]
        metric_dict = metric_lines[metric_name]
        for line_label, drop_dict in metric_dict.items():
            y = [drop_dict.get(dc, np.nan) for dc in unique_drop_counts]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                label=line_label,
                color=color_lookup.get(line_label),
            )

        ax.set_ylabel(metric_name)
        ax.grid(axis="y", alpha=0.3)
        if metric_idx == 0:
            ax.legend()

    axes[-1].set_xticks(unique_drop_counts)
    axes[-1].set_xticklabels([str(dc) for dc in unique_drop_counts])
    axes[-1].set_xlabel("Trials dropped at start")

    fig.suptitle("Variability vs. trial exclusion")
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    fname = "variability_vs_exclusion"
    if filename_suffix:
        fname = f"{fname}_{filename_suffix}"
    fig.savefig(os.path.join(outdir, f"{fname}.png"), dpi=150)
    plt.close(fig)

# ========= 終了位置の平均バーグラフ =========
# (removed) plot_mean_end_position_bar: replaced by plot_end_position_with_mean_scatter

# ========= エントリ =========
def main(trials_csv: str, L: float, T: float, outdir="analysis"):
    df_trials  = pd.read_csv(trials_csv)

    figures_dir = os.path.join(outdir, "figures")

    timestamp = _extract_timestamp_from_filename(trials_csv)

    plot_time_variability_bar(df_trials, outdir=figures_dir, filename_suffix=timestamp)
    plot_end_time_relative_summary(df_trials, T=T, outdir=figures_dir, filename_suffix=timestamp)
    plot_end_position_with_mean_scatter(df_trials, L=L, outdir=figures_dir, filename_suffix=timestamp)
    plot_end_position_variability_bar(df_trials, outdir=figures_dir, filename_suffix=timestamp)
    print("✅ Figures saved to:", figures_dir)

if __name__ == "__main__":
    # python analysis/plot_metrics.py
    L = 400.0
    T = 1.0
    # main(
    #     trials_csv = f"analysis/trials/tracking_001_20251029_124625_triallevel_trials.csv",
    #     L=L, T=T, outdir="analysis"
    # )
    # 複数CSVの比較例（必要に応じてパスとラベルを差し替えてください）
    comparison_sources = [
        "analysis/trials/tracking_furukawa_20251107_152908_triallevel_trials.csv",
        "analysis/trials/tracking_furukawa_20251107_153335_triallevel_trials.csv",
        "analysis/trials/tracking_furukawa_20251107_153754_triallevel_trials.csv",
        "analysis/trials/tracking_furukawa_20251107_154255_triallevel_trials.csv",
    ]
    comparison_labels = ["Condition 1_1", "Condition 2_1", "Condition 1_2", "Condition 2_2"]
    plot_time_variability_bar_multi(
        trial_sources=comparison_sources,
        labels=comparison_labels,
        outdir=os.path.join("analysis", "figures"),
        filename_suffix="Comparison_of_Conditions_1&2_furukawa",
    )
    plot_end_position_variability_bar_multi(
        trial_sources=comparison_sources,
        labels=comparison_labels,
        outdir=os.path.join("analysis", "figures"),
        filename_suffix="Comparison_of_Conditions_1&2_variability_furukawa",
    )
    # plot_variability_vs_exclusion(
    #     trial_sources=[
    #         "analysis/trials/tracking_002_20251030_160615_triallevel_trials.csv",
    #         "analysis/trials/tracking_002_20251030_161115_triallevel_trials.csv",
    #         "analysis/trials/tracking_002_20251030_161602_triallevel_trials.csv",
    #         "analysis/trials/tracking_002_20251030_162022_triallevel_trials.csv",
    #     ],
    #     labels=["trialset1", "trialset2", "trialset3", "trialset4"],
    #     drop_counts=(0, 5, 10),
    #     outdir="analysis/figures",
    #     filename_suffix="cond1_vs_cond2_002",
    # )
