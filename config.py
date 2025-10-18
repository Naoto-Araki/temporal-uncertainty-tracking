from psychopy import monitors

# ==== Monitor設定 ====
MONITOR_NAME = "Wacom"
SCREEN_SIZE_CM = 30.4
VIEW_DIST_CM = 57.0
RESOLUTION = [1280, 800]
REFRESH_HZ = 60

# ==== PsychoPyモニタ登録 ====
def setup_monitor():
    mon = monitors.Monitor(MONITOR_NAME, width=SCREEN_SIZE_CM, distance=VIEW_DIST_CM)
    mon.setSizePix(RESOLUTION)
    mon.save()
    return mon

# ==== 実験パラメータ ====
L = 400        # 移動距離 [px]
T = 1.0        # 移動時間 [s]
MU = 0.5       # 平均遅延 [s]
SIGMA = 0.12   # 遅延標準偏差 [s](0 or 0.12 or 0.3)
N_TRIALS = 50   # 各条件の試行回数 (30 or 50 or 100)
DELTA = 0.3    # 停止後の記録時間 [s]

# ==== 解析設定 ====
ANALYSIS = {
    # 幾何・時間パラメータ
    "L": 400.0,            # [px] 移動距離（縦: -L/2 → +L/2）
    "T": 1.0,              # [s] 理想運動時間（tau .. tau+T）

    # 位置分散を算出する窓の設定
    "poswin_ms": 100.0,    # [ms] 分散算出窓の半幅（開始・終了それぞれ）

    # 閾値ベース検出用の設定
    "start_margin_px": 20.0,
    "end_margin_px": 20.0,

    # 速度ベース検出（推奨）
    "use_velocity": True,  # True: 速度＋持続時間ベースで開始/終了を検出
    "v_start": 50.0,       # [px/s] 開始時の速度しきい値
    "v_stop":  20.0,       # [px/s] 終了時の速度しきい値
    "hold_start_ms": 80.0, # [ms] 開始検出の持続条件
    "hold_stop_ms": 100.0, # [ms] 終了検出の持続条件
}