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

- 起動時にダイアログが開き、参加者 ID と `config.py` の `CONDITIONS` に定義した条件（Cond1～Cond2-3）の中から選択します。`delay_profiles.py` が条件に応じた遅延分布（固定/片側正規/指数/片側コーシー）をサンプルします。
- 各トライアルは SPACE キーで開始。ターゲット移動と同時にマウス位置をサンプリングし、`data/` に CSV と JSON（メタ情報）を保存します。
- 提示後には A/B 因果判断の質問が表示されるので、`A`（自分が原因）または `B`（外的要因）キーを押して回答します。回答と反応時間 (`choice`, `choice_rt`) は各サンプル行にも追記されます。
- ESC キーで途中終了すると、それまでの結果が安全に保存されてウィンドウが閉じます。

#### 条件セットと遅延プロファイル

- `config.py` の `CONDITIONS` 辞書で Cond1（固定 0s）、Cond2-1（片側正規）、Cond2-2（指数）、Cond2-3（片側コーシー）などを一括管理します。`DEFAULT_CONDITION_ID` を変更するとダイアログの優先表示が切り替わります。
- Cond2-* の確率分布には `MIN_POSITIVE_DELAY (= 1/REFRESH_HZ)` を `min_delay` として設定し、0 s 遅延がサンプリングされないよう一律で 1 フレーム分だけ遅延を持たせています。また `max_delay = 1.0s` で統一し、レンジを揃えたうえで分布形状の違いだけを比較します（Cond2-1 も他条件と同じくしきい値未満をリジェクトして抽選し直す方式です）。
- 各条件で使用する遅延分布は `delay_profiles.py` に実装されたサンプラを通じて決定されます。`min_delay` や `max_delay` を設定すれば、負の値の除去や片側分布の制御が可能です。

#### ログ拡張

- `data/tracking_*.csv` には従来の列に加え `choice`（"A" or "B"）と `choice_rt`（回答までの時間 [s]）が追加されました。
- `data/*.json` のメタ情報には `condition_id` / `condition_label` / `condition_config` などが保存され、解析時にどの分布を使用したか追跡できます。

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

`analysis/plot_metrics.py` では以下の図を `analysis/figures/` に保存します（`L` と `T` は実験設定に合わせて指定してください。`config.py` の `L` と `ANALYSIS["T"]` を流用するのが簡単です）。

- 時刻ばらつき棒グラフ（開始時刻の SD と相対終了時間の SD）
- 相対終了時間の散布図（試行ごとの点と条件平均、理想時間 `T` を重ねたまとめ）
- 終了位置の散布図（検出タイミングでのカーソル位置）
- 終了位置分散の棒グラフ（条件ごとの試行間分散）

`main()` を直接呼ぶ形のみ用意しています（`python analysis/plot_metrics.py` 単体実行は、ハードコードされた比較用パスを前提としているためそのままだと失敗します）。`outdir` 引数を変えると保存先フォルダを切り替えられます（既定は `analysis`）。

### 因果判断可視化スクリプト `analysis/plot_causality.py`

`data/tracking_*.csv`（`choice` 列付き）を入力に、遅延 τ と A/B 判断の関係を図示します。`choice` 列が無い旧データは自動的にスキップします。

```bash
python analysis/plot_causality.py data/tracking_*.csv --tag subject01
```

出力内容（`analysis/figures/`）

- `choice_vs_tau_*.png`: τ を横軸に、A 選択率とビン平均を重ねた散布＋折れ線グラフ。
- `choice_condition_summary_*.png`: 条件別の A/B 比率スタックバーと、同条件の τ ヒストグラムを並べた図。

適宜 `--outdir` や `--tag` を指定すると保存先やファイル名サフィックスを制御できます。

### 条件別遅延分布の可視化 `delay_profiles.py`

`config.py` に定義された条件の遅延分布を、そのままヒストグラムとして描画・保存できます。Cond2 系列だけを一括で確認したい場合は以下を実行してください。

```bash
python delay_profiles.py
```

特定の条件 ID を指定したい場合は引数で渡します（複数可）。例:

```bash
python delay_profiles.py cond1_fixed cond2_cauchy --samples 100000
```

既定では各条件ごとのヒストグラム（`cond*_distribution.png`）に加えて、3 条件を重ねた折れ線図 `analysis/figures/delay_distributions_line.png` も自動生成されます。ヒストグラムを省いて折れ線図だけ出したい場合は `--line-only` を指定してください。`--show` を付けると図を画面表示に切り替えられます（GUI 環境のみ）。
