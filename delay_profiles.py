"""遅延分布サンプラを管理するモジュール。"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional

import numpy as np

Sampler = Callable[[], float]

_GLOBAL_RNG = np.random.default_rng()


def _ensure_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng or _GLOBAL_RNG


def fixed_sampler(tau: float) -> Sampler:
    return lambda: float(tau)


def half_normal_sampler(
    *,
    mu: float = 0.0,
    sigma: float = 0.12,
    min_delay: float = 0.0,
    max_delay: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Sampler:
    rng = _ensure_rng(rng)

    def _sample() -> float:
        while True:
            val = abs(rng.normal(loc=mu, scale=sigma))
            if val < min_delay:
                continue
            if max_delay is not None and val > max_delay:
                continue
            return float(val)

    return _sample


def exponential_sampler(
    *,
    lam: float = 1.0,
    min_delay: float = 0.0,
    max_delay: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Sampler:
    rng = _ensure_rng(rng)
    scale = 1.0 / max(lam, 1e-6)

    def _sample() -> float:
        while True:
            val = rng.exponential(scale=scale) + min_delay
            if max_delay is not None and val > max_delay:
                continue
            return float(val)

    return _sample


def cauchy_sampler(
    *,
    loc: float = 0.0,
    scale: float = 0.05,
    min_delay: float = 0.0,
    max_delay: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Sampler:
    rng = _ensure_rng(rng)

    def _sample() -> float:
        while True:
            val = rng.standard_cauchy() * scale + loc
            if val < min_delay:
                continue
            if max_delay is not None and val > max_delay:
                continue
            return float(val)

    return _sample


def get_sampler(config: Dict[str, Any], rng: Optional[np.random.Generator] = None) -> Sampler:
    """条件設定から適切な遅延サンプラを返す。"""
    distribution = config.get("distribution")
    params = dict(config.get("params", {}))
    params.setdefault("rng", rng)

    if distribution == "fixed":
        return fixed_sampler(params.get("tau", 0.0))
    if distribution == "half_normal":
        return half_normal_sampler(**params)
    if distribution == "exponential":
        lam = params.pop("lambda", params.pop("lam", None))
        if lam is not None:
            params["lam"] = lam
        return exponential_sampler(**params)
    if distribution == "cauchy":
        return cauchy_sampler(**params)

    raise ValueError(f"Unsupported distribution: {distribution}")


def _setup_psychopy_user_dir() -> Path:
    """Ensure PsychoPy writes config files to a writable directory."""
    default_dir = Path.cwd() / ".psychopy3"
    target = Path(os.environ.get("PSYCHOPY_USERCONFIGDIR", default_dir))
    target.mkdir(parents=True, exist_ok=True)
    os.environ["PSYCHOPY_USERCONFIGDIR"] = str(target)
    theme_dir = target / "themes"
    theme_dir.mkdir(parents=True, exist_ok=True)
    for theme_name in ["ClassicDark", "ClassicLight"]:
        theme_file = theme_dir / f"{theme_name}.json"
        if not theme_file.exists():
            theme_file.write_text("{}")
    return target


def _install_psychopy_stub():
    """Provide a lightweight stub if PsychoPy cannot be imported."""
    if "psychopy" in sys.modules:
        return

    stub = ModuleType("psychopy")
    monitors_mod = ModuleType("psychopy.monitors")

    class _StubMonitor:
        def __init__(self, *_, **__):
            pass

        def setSizePix(self, *_):
            pass

        def save(self):
            pass

    monitors_mod.Monitor = _StubMonitor
    stub.monitors = monitors_mod
    sys.modules["psychopy"] = stub
    sys.modules["psychopy.monitors"] = monitors_mod


def _prepare_matplotlib(show: bool):
    """Configure Matplotlib backend and return pyplot."""
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", Path.cwd() / ".mplconfig"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _sample_values(sampler: Sampler, n_samples: int) -> np.ndarray:
    return np.array([sampler() for _ in range(n_samples)], dtype=float)


def _plot_distribution(name: str, samples: np.ndarray, plt, outdir: Path, bins: int, show: bool):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(
        samples,
        bins=bins,
        range=(float(samples.min()), float(samples.max())),
        density=True,
        color="#1f77b4",
        alpha=0.75,
    )
    ax.set_xlabel("Delay τ [s]")
    ax.set_ylabel("Density")
    ax.set_title(f"{name} delay distribution")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}_distribution.png", dpi=150)
        plt.close(fig)


def _plot_line_summary(
    distributions: list[tuple[str, np.ndarray]],
    plt,
    outdir: Path,
    bins: int,
    show: bool,
    fname: str = "delay_distributions_line.png",
):
    if not distributions:
        return

    global_min = min(float(samples.min()) for _, samples in distributions)
    global_max = max(float(samples.max()) for _, samples in distributions)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cid, samples in distributions:
        density, edges = np.histogram(
            samples,
            bins=bins,
            range=(global_min, global_max),
            density=True,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, density, label=cid)
    ax.set_xlabel("Delay τ [s]")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="条件で定義された遅延分布をサンプルし、ヒストグラムとして可視化します。"
    )
    parser.add_argument(
        "conditions",
        nargs="*",
        help="プロットする条件 ID。未指定なら cond2-* を自動選択。",
    )
    parser.add_argument("--samples", type=int, default=50000, help="各条件から引くサンプル数。")
    parser.add_argument("--bins", type=int, default=60, help="ヒストグラムのビン数。")
    parser.add_argument("--outdir", default="analysis/figures", help="保存先ディレクトリ。")
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存ではなく画面表示を行います（GUI 利用時のみ）。",
    )
    parser.add_argument(
        "--line-only",
        action="store_true",
        help="条件別ヒストグラムをスキップし、重ね線プロットのみ生成します。",
    )
    args = parser.parse_args()

    _setup_psychopy_user_dir()

    try:
        import config as config_module  # type: ignore
    except Exception:
        sys.modules.pop("config", None)
        _install_psychopy_stub()
        try:
            config_module = importlib.import_module("config")  # type: ignore
        except Exception as exc:
            parser.error(f"config.py の読み込みに失敗しました: {exc}")

    config = config_module

    condition_ids = args.conditions or [cid for cid in config.CONDITIONS if cid.startswith("cond2_")]
    missing = [cid for cid in condition_ids if cid not in config.CONDITIONS]
    if missing:
        parser.error(f"未知の条件 ID: {', '.join(missing)}")

    plt = _prepare_matplotlib(args.show)
    outdir = Path(args.outdir)

    distributions = []
    for cid in condition_ids:
        sampler = get_sampler(config.CONDITIONS[cid])
        samples = _sample_values(sampler, args.samples)
        distributions.append((cid, samples))

    if not args.line_only:
        for cid, samples in distributions:
            _plot_distribution(cid, samples, plt, outdir, args.bins, args.show)

    _plot_line_summary(distributions, plt, outdir, args.bins, args.show)


if __name__ == "__main__":
    main()
