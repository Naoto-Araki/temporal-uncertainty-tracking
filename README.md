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

## 主指標 / 補助指標（Metrics）

本研究の解析指標は、**時刻分散（Temporal Variance）**を主指標、**位置分散（Spatial Variance）**を補助指標として用います。

### 主指標：時刻分散（運動タイミングの安定性）
- **目的**: 被験者が「動作を始める・止める」タイミングの安定性（再現性）を評価します。
- **検出方法（既定）**: 速度ベース＋一定持続条件。
  - 開始: `v >= v_start` が `hold_start_ms` 以上 **連続** した最初の時刻を開始とみなす。
  - 終了: `|v| <= v_stop` が `hold_stop_ms` 以上 **連続** した最初の時刻を終了とみなす。
  - 速度はサンプル位置と実測時刻から中央差分で推定（不等間隔サンプリング対応）。
- **設定キー（`config.py` の `ANALYSIS`）**:
  - `v_start`（px/s）, `v_stop`（px/s）, `hold_start_ms`, `hold_stop_ms`, `use_velocity=True`
- **集計**: 参加者×条件ごとに開始・終了時刻の平均/標準偏差を算出（標準偏差が実質の「時刻分散」）。

### 補助指標：位置分散（空間的安定性）
- **目的**: 理想タイミングに同期したときの **空間的安定性** を評価します。
- **定義（理想タイミング窓）**: ターゲットの理想タイミング `τ`（開始）と `τ+T`（終了）を中心に、各々 **±`poswin_ms`** の固定窓でカーソル位置の分散を算出。
  - 中心: `τ`（開始）と `τ+T`（終了）
  - 窓幅: `poswin_ms`（ミリ秒; 半幅）
- **定義（動的窓）**: 速度検出で得た開始・終了時刻 `t_start`, `t_end` を中心に、同じ **±`poswin_ms`** の窓でカーソル位置の分散を算出。
- **動的窓**: 速度検出で得た実際の開始/終了時刻 `t_start`, `t_end` を中心とした窓でも同じ分散を計算。解析結果では `_dynamic` が付いた列として出力されます。
- **設定キー（`config.py` の `ANALYSIS`）**:
  - `poswin_ms`, `T`, `L`, `start_margin_px`, `end_margin_px`
- **解釈**: 値が小さいほど位置が安定／大きいほど補正動作や揺れが多い可能性。
