from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _default_abs_adv_bins() -> np.ndarray:
    # abs(adv_meaned) 的分桶（可按需要调整）
    return np.asarray([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 5.0, np.inf], dtype=np.float64)


@dataclass
class AdvFlipClipHistConfig:
    out_dir: str
    plot_every_steps: int = 100_000
    abs_adv_bins: Optional[np.ndarray] = None
    file_prefix: str = "adv_flip_clip_hist"


class AdvFlipClipHistogramLogger:
    """
    记录并定期输出一个直方图：
    - 样本满足：adv 做“减均值”后符号翻转（sign flip），且 PPO clip 机制生效（clipped objective active）
    - 对这些样本的 |adv_meaned| 做分桶直方图

    设计目标：
    - 与 PPO 主训练逻辑解耦（PPO 只负责喂数据）
    - 不依赖 wandb，输出 png + json 到指定目录
    - 调用者需要传入当前 global_step，用于控制每隔 N 步输出一次
    """

    def __init__(self, config: AdvFlipClipHistConfig):
        self.config = config
        self.abs_adv_bins = _default_abs_adv_bins() if config.abs_adv_bins is None else config.abs_adv_bins
        assert self.abs_adv_bins.ndim == 1 and self.abs_adv_bins.size >= 2
        assert np.isfinite(self.abs_adv_bins[0]) and self.abs_adv_bins[0] == 0.0
        assert self.config.plot_every_steps > 0

        self._counts = np.zeros((self.abs_adv_bins.size - 1,), dtype=np.int64)
        self._total_seen = 0
        self._total_selected = 0
        self._last_dump_step: Optional[int] = None

        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

    def add_batch(self, abs_adv_meaned: np.ndarray, selected_mask: np.ndarray, global_step: int) -> None:
        """
        :param abs_adv_meaned: shape [B], |adv_meaned|
        :param selected_mask: shape [B], bool mask for (sign_flip_after_mean & clipped)
        :param global_step: 当前全局步数（用于控制输出频率）
        """
        if abs_adv_meaned.size == 0:
            return
        assert abs_adv_meaned.shape == selected_mask.shape

        self._total_seen += int(abs_adv_meaned.size)
        sel = selected_mask.astype(bool)
        if sel.any():
            vals = abs_adv_meaned[sel]
            # np.histogram: bins include left edge, exclude right edge (except last)
            hist, _ = np.histogram(vals, bins=self.abs_adv_bins)
            self._counts += hist.astype(np.int64)
            self._total_selected += int(sel.sum())

        if self._should_dump(global_step):
            self.dump(global_step)

    def _should_dump(self, global_step: int) -> bool:
        if self._last_dump_step is None:
            # 让第一次也能在达到阈值时输出
            return global_step >= self.config.plot_every_steps
        return (global_step - self._last_dump_step) >= self.config.plot_every_steps

    def dump(self, global_step: int) -> None:
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "global_step": int(global_step),
            "plot_every_steps": int(self.config.plot_every_steps),
            "abs_adv_bins": self.abs_adv_bins.tolist(),
            "counts": self._counts.tolist(),
            "total_seen": int(self._total_seen),
            "total_selected": int(self._total_selected),
            "selected_rate": float(self._total_selected) / float(self._total_seen) if self._total_seen > 0 else 0.0,
        }

        json_path = out_dir / f"{self.config.file_prefix}_step_{global_step}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # 可选画图：没有 matplotlib 时也能落盘 json
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self._last_dump_step = int(global_step)
            return

        # 画条形图（用 log scale 更容易看长尾）
        left = self.abs_adv_bins[:-1]
        right = self.abs_adv_bins[1:]
        # label 形如 [0.1,0.2)
        labels = []
        for l, r in zip(left, right):
            if np.isinf(r):
                labels.append(f"[{l:g}, inf)")
            else:
                labels.append(f"[{l:g}, {r:g})")

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(np.arange(len(self._counts)), self._counts, width=0.9)
        ax.set_xticks(np.arange(len(self._counts)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("count")
        ax.set_title(
            f"Sign-flip-after-mean & clipped histogram @ step={global_step} | "
            f"selected={meta['total_selected']}/{meta['total_seen']} ({meta['selected_rate']:.4f})"
        )
        ax.set_yscale("symlog")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        png_path = out_dir / f"{self.config.file_prefix}_step_{global_step}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        self._last_dump_step = int(global_step)

    def reset(self) -> None:
        self._counts[:] = 0
        self._total_seen = 0
        self._total_selected = 0
        self._last_dump_step = None


