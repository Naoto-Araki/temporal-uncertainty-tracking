from psychopy import visual, core, event
import os, csv, time
import numpy as np
from stimuli import generate_motion
from config import L, T, MU, SIGMA, N_TRIALS, DELTA

def run_experiment(mon):
    """
    実験を実行するメイン関数。
    スタート/ゴール位置を常時表示し、被験者のキー押下（SPACE）で試行を開始。
    ターゲットは遅延 τ ののちベルシェイプ軌道で移動。被験者カーソルを画面に可視化し、
    各フレームでターゲット位置と被験者カーソル位置を記録する。

    Args
    ----
    mon : psychopy.monitors.Monitor
        config.setup_monitor() が返す Monitor オブジェクト。

    Returns
    -------
    None
        記録データは ./data/ に CSV で自動保存。
        カラム: trial, tau, t, x_t, x_p, y_p
    """
    # ウィンドウの設定
    win = visual.Window(size=mon.getSizePix(), monitor=mon, color="black", units="pix", fullscr=True, waitBlanking=True)

    # スタート/ゴールの固定目印の設定
    start_dot = visual.Circle(win, radius=10, fillColor="white", lineColor=None, pos=[-L/2, 0])
    goal_dot  = visual.Circle(win, radius=10, fillColor="white", lineColor=None, pos=[ L/2, 0])

    # ターゲット（動く円）の設定
    target = visual.Circle(win, radius=8, fillColor="red", lineColor=None)

    # 被験者カーソル（システムカーソルを見せる＋描画カーソルも重ねる）
    mouse = event.Mouse(visible=True, win=win)
    cursor = visual.Circle(win, radius=6, fillColor="cyan", lineColor=None)

    # メッセージの設定
    txt = visual.TextStim(win, text="", color="white", height=24, pos=(0, -80))

    os.makedirs("data", exist_ok=True)
    all_rows = []
    clock = core.Clock()

    for trial in range(1, N_TRIALS + 1):
        # 各試行の遅延（負値を防ぐ）
        tau = max(0.0, float(np.random.normal(MU, SIGMA)))

        # 待機画面：スタート/ゴールを見せつつ案内
        txt.text = f"Trial {trial}/{N_TRIALS}\nPress SPACE to start"
        
        while True:
            render_wait_frame(start_dot, goal_dot, txt, mouse, cursor)
            win.flip()
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                _save_csv(all_rows)
                _safe_close(win)
                return
            if "space" in keys:
                win.callOnFlip(clock.reset)
                render_wait_frame(start_dot, goal_dot, txt, mouse, cursor)
                # ここでターゲットが“出現”する仕様なら、この瞬間にだけ描画
                target.pos = [-L/2, 0]; target.draw()
                win.flip() 
                break
        
        # 刺激提示ループ（移動 T + 停止延長 DELTA）
        while clock.getTime() < (T + DELTA):
            # 現在時刻と位置計算
            t = clock.getTime()
            x_t = generate_motion(t, tau, L, T)  # -L/2 ～ L/2 を返す
            target.pos = [x_t, 0]

            # カーソル座標の取得
            x_p, y_p = mouse.getPos()
            cursor.pos = [x_p, y_p]

            # 描画順：固定目印 → ターゲット → カーソル
            start_dot.draw(); goal_dot.draw()
            target.draw()
            cursor.draw()

            # 画面更新
            win.flip()

            # 記録
            all_rows.append([trial, tau, t, x_t, x_p, y_p])

            # ESC で緊急終了
            if "escape" in event.getKeys(keyList=["escape"]):
                _save_csv(all_rows)
                _safe_close(win)
                return


        # 試行間インターバル（短めに）
        core.wait(0.3)
        
    _save_csv(all_rows)
    _safe_close(win)

def render_wait_frame(start_dot, goal_dot, txt, mouse, cursor):
    start_dot.draw(); goal_dot.draw(); txt.draw()
    x_p, y_p = mouse.getPos(); cursor.pos = [x_p, y_p]; cursor.draw()

def _save_csv(rows):
    """記録をCSV保存。"""
    if not rows:
        return
    fn = f"data/tracking_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "tau", "t", "x_t", "x_p", "y_p"])
        w.writerows(rows)
    print(f"✅ Saved: {fn}")

def _safe_close(win):
    """ウィンドウを安全に閉じる。"""
    try:
        win.close()
    finally:
        core.quit()
