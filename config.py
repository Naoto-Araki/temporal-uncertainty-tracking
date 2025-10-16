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
CONDITION = 1  # 1: 分散なし（固定）, 2: 分散あり
MU = 0.5       # 平均遅延 [s]
SIGMA = 0.12   # 遅延標準偏差 [s]
N_TRIALS = 5   # 各条件の試行回数
DELTA = 0.3    # 停止後の記録時間 [s]
