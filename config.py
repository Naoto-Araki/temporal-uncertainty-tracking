from psychopy import monitors
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==== Monitor設定 ====
MONITOR_NAME = "Wacom"
SCREEN_SIZE_CM = 30.4
VIEW_DIST_CM = 57.0
RESOLUTION = [1280, 800]
REFRESH_HZ = 60
SCREEN_INDEX = 1  # PsychoPy screen index (0=メイン, 1=外部モニタ等)
FULLSCREEN = True  # 刺激ウィンドウを全画面表示するかどうか

# ==== PsychoPyモニタ登録 ====
def setup_monitor():
    mon = monitors.Monitor(MONITOR_NAME, width=SCREEN_SIZE_CM, distance=VIEW_DIST_CM)
    mon.setSizePix(RESOLUTION)
    mon.save()
    return mon

# =====================
# 理想ベルシェイプ運動から速度しきい値を推定
# =====================
def estimate_vstart_from_ideal(L: float, T: float, ratio: float = 0.2) -> float:
    """
    理想ベルシェイプ運動（最小ジャークモデル）の速度から、開始しきい値を推定する。
    ratio: 立ち上がり区間のピーク速度に対する割合 (例: 0.2 → 20%)

    Returns
    -------
    float
        速度しきい値 [px/s]
    """
    t = np.linspace(0, T, 1000)
    tau = t / T
    v = (L / T) * (30 * tau**2 - 60 * tau**3 + 30 * tau**4)
    v_peak = np.max(v)
    v_thresh = v_peak * ratio
    return v_thresh

# ==== 実験パラメータ ====
L = 400        # 移動距離 [px]
T = 1.0        # 移動時間 [s]
MU = 0.5       # 平均遅延 [s]
SIGMA = 0.20   # 遅延標準偏差 [s](0 or 0.12 or 0.3)
N_TRIALS = 50   # 各条件の試行回数 (30 or 50 or 100)
DELTA = 1.0    # 停止後の記録時間 [s]

# ==== 解析設定 ====
ANALYSIS = {
    # 幾何・時間パラメータ
    "L": 400.0,            # [px] 移動距離（縦: -L/2 → +L/2）
    "T": 1.0,              # [s] 理想運動時間（tau .. tau+T）
    "v_start" : estimate_vstart_from_ideal(L=400.0, T=1.0, ratio=0.1),       # [px/s] 開始時の速度しきい値
    "v_stop":  1.0,       # [px/s] 終了時の速度しきい値
    "hold_start_ms": 80.0, # [ms] 開始検出の持続条件
    "hold_stop_ms": 100.0, # [ms] 終了検出の持続条件
}
