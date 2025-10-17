from psychopy import visual, core, event, gui
import os, csv, time, json
import numpy as np
from stimuli import generate_motion
from config import L, T, MU, SIGMA, N_TRIALS, DELTA, CONDITION

# RGB colors in -1..1 range to avoid deprecated fillRGB/lineRGB usage
COLOR_BLACK = [-1, -1, -1]
COLOR_WHITE = [1, 1, 1]
COLOR_RED = [1, -1, -1]
COLOR_CYAN = [-1, 1, 1]

def get_experiment_info():
    """
    PsychoPy のダイアログで参加者情報を取得する。

    Args
    ----
    None

    Returns
    -------
    dict
        `participant` (str): 参加者 ID。
        `condition` (str): 条件番号（"1" または "2"）。
        `timestamp` (str): 実験開始時刻（YYYYMMDD_HHMMSS 形式）。
    """
    dlg = gui.Dlg(title="Experiment Info")
    dlg.addText("Participant Information")
    dlg.addField("Participant ID:")
    dlg.addField("Condition:", choices=["1 (Fixed τ)", "2 (Variable τ)"])
    dlg.show()
    if dlg.OK:
        info = dlg.data
        return {
            "participant": info[0],
            "condition": info[1][0],
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
    else:
        core.quit()

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
        カラム: trial, tau, t, y_t, x_p, y_p
    """
    info = get_experiment_info()
    size_pix = mon.getSizePix()
    base_name = f"tracking_{info['participant']}_{info['timestamp']}"
    meta = {
        "participant": info["participant"],
        "condition": info["condition"],
        "timestamp": info["timestamp"],
        "file_base": base_name,
        "monitor": {
            "size_pix": list(size_pix) if size_pix is not None else None,
            "distance_cm": mon.getDistance(),
        },
        "params": {
            "L": L,
            "T": T,
            "MU": MU,
            "SIGMA": SIGMA,
            "DELTA": DELTA,
            "N_TRIALS": N_TRIALS,
        },
        "trials": [],
    }
    
    # ウィンドウの設定
    win = visual.Window(
        size=mon.getSizePix(),
        monitor=mon,
        color=COLOR_BLACK,
        colorSpace="rgb",
        units="pix",
        fullscr=True,
        waitBlanking=True,
    )

    # スタート/ゴールの固定目印の設定
    start_dot = visual.Circle(win, radius=10, fillColor=COLOR_WHITE, lineColor=None, colorSpace="rgb", pos=[0, -L/2])
    goal_dot  = visual.Circle(win, radius=10, fillColor=COLOR_WHITE, lineColor=None, colorSpace="rgb", pos=[0, L/2])

    # ターゲット（動く円）の設定
    target = visual.Circle(win, radius=8, fillColor=COLOR_RED, lineColor=None, colorSpace="rgb")

    # 被験者カーソル（システムカーソルを見せる＋描画カーソルも重ねる）
    mouse = event.Mouse(visible=True, win=win)
    cursor = visual.Circle(win, radius=6, fillColor=COLOR_CYAN, lineColor=None, colorSpace="rgb")

    # メッセージの設定
    txt = visual.TextStim(win, text="", color=COLOR_WHITE, colorSpace="rgb", height=24, pos=(0, -80))

    os.makedirs("data", exist_ok=True)
    all_rows = []
    clock = core.Clock()

    for trial in range(1, N_TRIALS + 1):
        if info["condition"].startswith("1"): # 1: 分散なし（固定）, 2: 分散あり]
            tau = MU
        else:
            tau = max(0.0, float(np.random.normal(MU, SIGMA)))
        meta["trials"].append({"trial": trial, "tau": float(tau)})
    
        # 待機画面：スタート/ゴールを見せつつ案内
        txt.text = f"Trial {trial}/{N_TRIALS}\nPress SPACE to start"
        
        while True:
            render_wait_frame(start_dot, goal_dot, txt, mouse, cursor)
            win.flip()
            keys = event.getKeys(keyList=["space", "escape"])
            if "escape" in keys:
                _finalize_meta(meta, all_rows)
                _save_csv(all_rows, meta, base_name)
                _save_meta(meta, base_name)
                _safe_close(win)
                return
            if "space" in keys:
                win.callOnFlip(clock.reset)
                render_wait_frame(start_dot, goal_dot, txt, mouse, cursor)
                # ここでターゲットが“出現”する仕様なら、この瞬間にだけ描画
                target.pos = [0, -L/2]; target.draw()
                win.flip() 
                break
        
        # 刺激提示ループ（移動 T + 停止延長 DELTA）
        while clock.getTime() < (T + tau + DELTA):
            # 現在時刻と位置計算
            t = clock.getTime()
            y_t = generate_motion(t, tau, L, T)  # -L/2 ～ L/2 を返す
            target.pos = [0, y_t]

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
            all_rows.append([trial, tau, t, y_t, x_p, y_p])

            # ESC で緊急終了
            if "escape" in event.getKeys(keyList=["escape"]):
                _finalize_meta(meta, all_rows)
                _save_csv(all_rows, meta, base_name)
                _save_meta(meta, base_name)
                _safe_close(win)
                return


        # 試行間インターバル（短めに）
        core.wait(0.3)
        
    _finalize_meta(meta, all_rows)
    _save_csv(all_rows, meta, base_name)
    _save_meta(meta, base_name)
    _safe_close(win)

def render_wait_frame(start_dot, goal_dot, txt, mouse, cursor):
    start_dot.draw(); goal_dot.draw(); txt.draw()
    x_p, y_p = mouse.getPos(); cursor.pos = [x_p, y_p]; cursor.draw()

def _save_csv(rows, meta, filename_base=None):
    """記録をCSV保存。"""
    base = filename_base or meta.get("file_base") or f"tracking_{meta['participant']}_{meta['timestamp']}"
    fn = f"data/{base}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "condition", "trial", "tau", "t", "y_t", "x_p", "y_p"])
        for r in rows:
            w.writerow([meta["participant"], meta["condition"]] + r)
    print(f"✅ Saved: {fn}")

def _save_meta(meta, filename_base=None):
    """メタデータをJSON保存。"""
    base = filename_base or meta.get("file_base") or f"tracking_{meta['participant']}_{meta['timestamp']}"
    fn = f"data/{base}.json"
    with open(fn, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved meta: {fn}")

def _finalize_meta(meta, rows):
    """書き出し前にメタ情報へ集計値を付加。"""
    meta["completed_trials"] = len(meta.get("trials", []))
    meta["samples_recorded"] = len(rows)
    return meta

def _safe_close(win):
    """ウィンドウを安全に閉じる。"""
    try:
        win.close()
    finally:
        core.quit()
