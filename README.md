### 仮想環境を有効化（activate）
```bash
source .venv310/bin/activate
```

### 仮想環境を終了（deactivate）
```bash
deactivate
```

### 実験を実行する (`main.py`)

PsychoPy 実験の本体を実行します。モニタ設定や実験パラメータは `config.py` にまとめています。

#### 使い方

```bash
python main.py
```

#### 実行時の流れ

- 起動時にダイアログが開き、参加者 ID と条件（固定 / 可変）を入力します。
- 各トライアルは SPACE キーで開始。ターゲット移動と同時にマウス位置をサンプリングし、`data/` に CSV と JSON（メタ情報）を保存します。
- ESC キーで途中終了すると、それまでの結果が安全に保存されてウィンドウが閉じます。

### 分析スクリプト `analysis/compute_variance.py`

Psychopy 実験のトラッキング CSV（`participant, condition, trial, tau, t, y_t, x_p, y_p`）から、
開始/終了時刻の検出や位置分散、時間分散、到達位置、到達時間の平均を算出して CSV に出力する。

#### 使い方

```bash
python analysis/compute_variance.py data/tracking_XXXX.csv
# 複数ファイルをまとめて処理する例
python analysis/compute_variance.py data/tracking_XXXX.csv data/tracking_YYYY.csv
```

#### 主な設定 (`config.py`)

- `ANALYSIS["T"]`: 運動時間 [s]。終了理想時刻として窓中心に使用。
- `ANALYSIS["poswin_ms"]`: 位置分散をとる窓の半幅 [ms]。
- `ANALYSIS["v_start"]`, `ANALYSIS["v_stop"]`: 速度しきい値 [px/s]。
- `ANALYSIS["hold_start_ms"]`, `ANALYSIS["hold_stop_ms"]`: 持続判定時間 [ms]。

実行後、`analysis/trials/` にトライアル別、`analysis/summary/` に条件別集計の CSV が生成されます。

### 解析結果の指標

`analysis/compute_variance.py` で出力される主なカラム:

- `t_start`, `t_end` : 速度しきい値に基づいて自動検出した開始・終了時刻。
- `t_end_offset` : 検出終了時刻から理想時刻 `tau + T` を引いたオフセット。
- `t_end_rel`, `t_end_rel_offset` : 検出開始を 0 秒とした相対終了時刻と、その理想時間 (`T`) との差分。
- `y_end_final` : 終了検出時刻に最も近いサンプルのカーソル位置（0→L スケール）。

条件別 CSV では上記の平均・標準偏差・分散を集計しています。

### 図示スクリプト `analysis/plot_metrics.py`

`compute_variance.py` で生成した CSV を使い、主要指標を図として保存します。

#### 使い方

```bash
python - <<'PY'
from analysis.plot_metrics import main
main(
    trials_csv="analysis/<name>_triallevel_trials.csv",
    L=400.0,
    T=1.0,
    outdir="analysis",
)
PY
```

`analysis/plot_metrics.py` では以下の図を `analysis/figures/` に保存します。

- 時刻ばらつき棒グラフ（開始時刻の SD と相対終了時間の SD）
- 相対終了時間の散布図（条件平均と理想時間 `T` を重ねたまとめ）
- 相対終了時間の散布図（試行ごとの点を中心にしたビュー）
- 相対終了時間オフセット散布図（`t_end_rel - T` のズレ）
- 終了位置の散布図（検出タイミングでのカーソル位置）
- 終了位置分散の棒グラフ（条件ごとの試行間分散）

`main()` の `outdir` 引数を変えると保存先フォルダを切り替えられます（既定は `analysis`）。
