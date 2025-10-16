from psychopy import visual, core, event
import numpy as np
import os, time, csv
from stimuli import generate_motion
from config import L, T, MU, SIGMA, N_TRIALS, DELTA

def run_experiment(mon):
    """
    PsychoPyのウィンドウ上でターゲット刺激を提示し、時系列位置データを取得・保存する。

    Args:
        mon (psychopy.monitors.Monitor):
            実験に使用するモニタ設定。
            config.py の setup_monitor() により生成された Monitor オブジェクトを受け取る。

    Returns:
        None
            取得データは CSV ファイルとして data/ フォルダ内に自動保存される。
            各行は以下のカラムを持つ:
            [trial, delay (τ), time [s], x_pos [px]]

    処理概要:
        1. PsychoPyウィンドウを作成。
        2. 各試行で開始遅延 τ を正規分布 (平均 MU, 分散 SIGMA^2) からサンプリング。
        3. 試行時間 (T + Δ) の間、ベルシェイプ軌道に従ってターゲットを描画。
        4. 各フレームの時刻・位置を記録。
        5. 試行終了後、データを CSV 形式で保存。
    """
    # 1. 画面・ウィンドウの設定
    win = visual.Window(size=mon.getSizePix(), monitor=mon, color="gray", units="pix")
    
    # 2. データ保存用フォルダ
    os.makedirs("data", exist_ok=True)
    
    # 3. 試行ループ
    for trial in range(N_TRIALS):
        tau = max(0, np.random.normal(MU, SIGMA)) # ランダムな開始遅延
        target = visual.Circle(win, radius=8, fillColor='red') # ターゲット刺激
        
        clock = core.Clock()
        data = []
        
        t = clock.getTime() # スタート開始
        
        # 4. 時間経過に合わせてターゲットを動かす
        while t < T + DELTA:
            t = clock.getTime() # 経過時間
            y = generate_motion(t, tau, L, T)
            target.pos = [0, y]
            target.draw()
            win.flip()
            
            # データを記録
            data.append([trial, tau, t, y])
        
        # 5. 試行データを保存
        save_data(data)
    
    win.close()

def save_data(records):
    """
    実験データを CSV ファイルとして保存する関数。

    Args:
        records (list[list]):
            各試行の記録を格納した2次元リスト。
            各行は [trial_id, delay (τ), time [s], x_pos [px]] の形式。

    Returns:
        None
            data/ ディレクトリ内に `exp_YYYYMMDD_HHMMSS.csv` として出力される。
    """
    filename = f"data/exp_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "delay", "time", "x_pos"])
        writer.writerows(records)
    print(f"✅ Data saved: {filename}")