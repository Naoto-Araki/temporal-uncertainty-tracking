import numpy as np

def bell_shape(t, T):
    """
    ベルシェイプ関数（最小ジャーク軌道）
    ----------
    Parameters
    ----------
    t : float
        現在の時刻（秒）
    T : float
        運動全体の時間（秒）

    Returns
    -------
    float
        正規化された位置（0〜1の範囲）
    """
    s = np.clip(t / T, 0, 1)
    return 10*s**3 - 15*s**4 + 6*s**5

def generate_motion(t: float, tau: float, L: float, T: float) -> float:
    """
    ターゲットのx座標を計算する関数。
    ----------
    Parameters
    ----------
    t : float
        現在の時刻（秒）
    tau : float
        動作開始遅延（秒）
    L : float
        移動距離（ピクセル）
    T : float
        運動時間（秒）

    Returns
    -------
    float
        ターゲットのx座標（ピクセル単位）
    """
    if t < tau:
        # 動き出す前 → スタート位置
        return -L/2
    elif t < tau + T:
        # ベルシェイプ軌道に沿って移動
        return -L/2 + L * bell_shape(t - tau, T)
    else:
        # 完全停止後 → ゴール位置
        return L/2
