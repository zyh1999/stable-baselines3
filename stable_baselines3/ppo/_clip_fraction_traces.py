from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ClipFractionTraceConfig:
    out_dir: str
    plot_every_steps: int = 100_000
    file_prefix: str = "clip_fraction_traces"


class ClipFractionTraceLogger:
    """
    每次 PPO 的 train()（对应一次 rollout 的多 epoch 更新）记录 3 条曲线，横轴为 mini-batch 序号：
    - overall_clip_fraction：全样本 clip fraction（|ratio-1| > clip_range）
    - flip_clip_fraction：仅“adv-mean 后符号翻转”子集的 clip fraction
    - nonflip_clip_fraction：去掉符号翻转样本后的 clip fraction

    然后按 global_step 每隔 N 步输出一次 png + json（不依赖 wandb）。
    """

    def __init__(self, config: ClipFractionTraceConfig):
        self.config = config
        assert self.config.plot_every_steps > 0
        Path(self.config.out_dir).mkdir(parents=True, exist_ok=True)

        self._last_dump_step: Optional[int] = None
        self._enabled_this_rollout = False
        self._global_step_for_rollout: Optional[int] = None

        self._overall: list[float] = []
        self._flip: list[float] = []
        self._nonflip: list[float] = []
        self._flip_rate: list[float] = []
        # 新增：只在 flip 子集上统计“clip 决策相对 raw-adv 的对照是否出错”
        self._flip_wrong_clip: list[float] = []
        self._flip_missed_clip: list[float] = []
        self._flip_mismatch: list[float] = []

    def start_rollout(self, global_step: int) -> None:
        """
        在 train() 开始时调用，重置本轮 rollout 的曲线缓存。
        """
        self._overall.clear()
        self._flip.clear()
        self._nonflip.clear()
        self._flip_rate.clear()
        self._flip_wrong_clip.clear()
        self._flip_missed_clip.clear()
        self._flip_mismatch.clear()
        # 第一次输出强制使用 step=0，满足“0 时刻也要画一张”
        self._global_step_for_rollout = 0 if self._last_dump_step is None else int(global_step)

        if self._last_dump_step is None:
            # 首次无条件输出一张（step=0）
            self._enabled_this_rollout = True
        else:
            self._enabled_this_rollout = (global_step - self._last_dump_step) >= self.config.plot_every_steps

    def add_minibatch_from_losses(
        self,
        *,
        clip_range: float,
        ratio: "th.Tensor",
        adv_raw: "th.Tensor",
        adv_meaned: "th.Tensor",
        loss1_used: "th.Tensor",
        loss2_used: "th.Tensor",
        loss1_raw: "th.Tensor",
        loss2_raw: "th.Tensor",
    ) -> None:
        """
        PPO 侧把必要张量喂进来，这里负责计算并追加本 minibatch 的各条曲线点。

        约定：
        - loss*_used 对应训练时实际使用的 advantage（包含 mean/std/scale 后的那个）
        - loss*_raw  对应用 raw advantage（未减均值）构造的 PPO per-sample objective
        """
        if not self._enabled_this_rollout:
            return

        import torch as th  # local import to avoid hard dependency when logger is unused

        with th.no_grad():
            clipped_mask = (th.abs(ratio - 1) > clip_range)
            overall_cf = clipped_mask.float().mean().item()

            # sign flip: raw >0 -> meaned <0 或 raw <0 -> meaned >0（排除 0）
            sign_flip = ((adv_raw > 0) & (adv_meaned < 0)) | ((adv_raw < 0) & (adv_meaned > 0))
            flip_count = int(sign_flip.sum().item())
            nonflip_count = int((~sign_flip).sum().item())
            total_count = flip_count + nonflip_count
            flip_rate = (flip_count / total_count) if total_count > 0 else float("nan")

            if flip_count > 0:
                flip_cf = clipped_mask[sign_flip].float().mean().item()
            else:
                flip_cf = float("nan")

            if nonflip_count > 0:
                nonflip_cf = clipped_mask[~sign_flip].float().mean().item()
            else:
                nonflip_cf = float("nan")

            # “clip 决策是否出错”：比较 used(当前训练的 adv) 与 raw-adv 的 clip 分支选择是否一致
            # PPO 的 per-sample clip 分支：loss2 < loss1 表示 clipped objective 被选中
            used_clip = (loss2_used < loss1_used)
            raw_clip = (loss2_raw < loss1_raw)

            if flip_count > 0:
                wrong_clip = (used_clip & ~raw_clip & sign_flip).float().sum().item() / flip_count
                missed_clip = ((~used_clip) & raw_clip & sign_flip).float().sum().item() / flip_count
                mismatch = wrong_clip + missed_clip
            else:
                wrong_clip = float("nan")
                missed_clip = float("nan")
                mismatch = float("nan")

        self._overall.append(float(overall_cf))
        self._flip.append(float(flip_cf))
        self._nonflip.append(float(nonflip_cf))
        self._flip_rate.append(float(flip_rate))
        self._flip_wrong_clip.append(float(wrong_clip))
        self._flip_missed_clip.append(float(missed_clip))
        self._flip_mismatch.append(float(mismatch))

    def finish_rollout(self) -> None:
        if not self._enabled_this_rollout:
            return
        assert self._global_step_for_rollout is not None
        self.dump(self._global_step_for_rollout)

    def dump(self, global_step: int) -> None:
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "global_step": int(global_step),
            "plot_every_steps": int(self.config.plot_every_steps),
            "n_minibatches": int(len(self._overall)),
            "overall_clip_fraction": self._overall,
            "flip_clip_fraction": self._flip,
            "nonflip_clip_fraction": self._nonflip,
            "flip_sample_rate": self._flip_rate,
            "flip_wrong_clip_fraction": self._flip_wrong_clip,
            "flip_missed_clip_fraction": self._flip_missed_clip,
            "flip_mismatch_clip_fraction": self._flip_mismatch,
        }

        json_path = out_dir / f"{self.config.file_prefix}_step_{global_step}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            self._last_dump_step = int(global_step)
            return

        x = np.arange(len(self._overall))
        y0 = np.asarray(self._overall, dtype=np.float64)
        y1 = np.asarray(self._flip, dtype=np.float64)
        y2 = np.asarray(self._nonflip, dtype=np.float64)
        y3 = np.asarray(self._flip_rate, dtype=np.float64)
        y4 = np.asarray(self._flip_wrong_clip, dtype=np.float64)
        y5 = np.asarray(self._flip_missed_clip, dtype=np.float64)
        y6 = np.asarray(self._flip_mismatch, dtype=np.float64)

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y0, label="overall clip fraction", linewidth=1.5)
        ax.plot(x, y1, label="flip-only clip fraction", linewidth=1.5)
        ax.plot(x, y2, label="nonflip-only clip fraction", linewidth=1.5)
        ax.plot(x, y4, label="flip: wrong-clip fraction", linewidth=1.2, linestyle="--")
        ax.plot(x, y5, label="flip: missed-clip fraction", linewidth=1.2, linestyle="--")
        ax.plot(x, y6, label="flip: mismatch (wrong+missed)", linewidth=1.2, linestyle=":")
        ax.set_xlabel("minibatch index (within this rollout update)")
        ax.set_ylabel("clip fraction")
        flip_rate_mean = float(np.nanmean(y3)) if y3.size > 0 else float("nan")
        ax.set_title(f"Clip fraction traces @ step={global_step} | flip_rate_mean={flip_rate_mean:.4f}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        png_path = out_dir / f"{self.config.file_prefix}_step_{global_step}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        self._last_dump_step = int(global_step)


