"""Plot S4b training curves from train_history.jsonl (no tensorboard needed).

Reads the per-step JSONL the trainer writes and renders loss / train IoU /
val IoU / LR curves to a PNG so training convergence can be inspected without
TensorBoard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True, help="path to train_history.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    args = parser.parse_args(argv)

    steps, loss, iou, bce, dice, objectness, swap, lr, gnorm, val_image, val_video = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    with args.history.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record["step"])
            steps.append(step)
            loss.append(record.get("loss"))
            iou.append(record.get("anchor_iou"))
            bce.append(record.get("bce"))
            dice.append(record.get("dice"))
            objectness.append(record.get("objectness"))
            swap.append(record.get("swap"))
            lr.append(record.get("lr"))
            gnorm.append(record.get("gnorm"))
            eval_ = record.get("eval") or {}
            val_image.append(eval_.get("image"))
            val_video.append(eval_.get("video"))

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    def plot(ax, values, title, color="tab:blue"):
        points = [(s, v) for s, v in zip(steps, values) if v is not None]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    plot(axes[0, 0], loss, "total loss", "tab:red")
    plot(axes[0, 1], iou, "train anchor IoU", "tab:green")
    axes[1, 0].plot(steps, lr, color="tab:purple")
    axes[1, 0].set_title("learning rate")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(
        [s for s, v in zip(steps, val_image) if v is not None],
        [v for v in val_image if v is not None],
        label="val image IoU",
        color="tab:orange",
    )
    axes[1, 1].plot(
        [s for s, v in zip(steps, val_video) if v is not None],
        [v for v in val_video if v is not None],
        label="val video IoU",
        color="tab:blue",
    )
    axes[1, 1].set_title("validation IoU")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=120)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
