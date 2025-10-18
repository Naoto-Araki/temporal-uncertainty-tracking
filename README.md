### 仮想環境を有効化（activate）
```bash
source .venv/bin/activate
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
```

#### 主な設定 (`config.py`)

- `ANALYSIS["T"]`: 運動時間 [s]。開始/終了の理想時刻として窓中心に使用。
- `ANALYSIS["poswin_ms"]`: 位置分散をとる窓の半幅 [ms]。
- `ANALYSIS["start_margin_px"]` / `ANALYSIS["end_margin_px"]`: 開始/終了検出の位置マージン [px]。
- `ANALYSIS["use_velocity"]`: True で速度＋持続時間ベース検出を使用。
- `ANALYSIS["v_start"]`, `ANALYSIS["v_stop"]`: 速度しきい値 [px/s]。
- `ANALYSIS["hold_start_ms"]`, `ANALYSIS["hold_stop_ms"]`: 持続判定時間 [ms]。

実行後、`analysis/` にトライアル別と条件別集計の CSV が生成されます。
