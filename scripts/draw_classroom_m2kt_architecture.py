"""Draw the retained classroom M2KT adapter architecture as a paper-style PNG.

Run from any directory::

    python scripts/draw_classroom_m2kt_architecture.py

The figure mirrors classroom_m2kt_adapter.py and the training/prediction path.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "m2kt-mpl-cache"))

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "classroom_m2kt_architecture.png"

INK = "#17233A"
MUTED = "#536174"
LINE = "#67788F"
BLUE = "#EAF3FC"
BLUE_EDGE = "#6391BE"
AMBER = "#FFF1DA"
AMBER_EDGE = "#D19A41"
GREEN = "#E9F5EB"
GREEN_EDGE = "#69A77D"
NEUTRAL = "#F3F6FA"
NEUTRAL_EDGE = "#AAB8C8"


def setup_font() -> None:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ]
    matches = next((path for path in candidates if path.is_file()), None)
    if matches is None:
        raise RuntimeError("A CJK font is required; install Noto Sans CJK")
    font_manager.fontManager.addfont(str(matches))
    family = font_manager.FontProperties(fname=str(matches)).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10,
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })


def box(ax, x, y, w, h, title, detail="", *, face=NEUTRAL, edge=NEUTRAL_EDGE,
        title_size=13, detail_size=10.5, rounding=0.16, linewidth=1.45):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={rounding}",
        linewidth=linewidth, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    center = x + w / 2
    if detail:
        ax.text(center, y + h * 0.65, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=3)
        ax.text(center, y + h * 0.31, detail, ha="center", va="center",
                fontsize=detail_size, color=MUTED, linespacing=1.3, zorder=3)
    else:
        ax.text(center, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    return patch


def arrow(ax, start, end, *, color=LINE, width=1.7, style="-|>",
          dashed=False, mutation=14, zorder=4):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=width, color=color, linestyle="--" if dashed else "-",
        connectionstyle="arc3,rad=0", zorder=zorder,
    ))


def panel(ax, x, y, w, h, title, *, edge=AMBER_EDGE, face="#FFFCF7"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.2",
        linewidth=1.4, edgecolor=edge, facecolor=face, zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.3, y + h - 0.32, title, ha="left", va="center",
            fontsize=14, fontweight="bold", color=INK, zorder=3)


def draw(output: Path, dpi: int) -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(18, 10.2), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.text(0.65, 9.67, "课堂音视频报告生成模型", fontsize=23, fontweight="bold",
            color=INK, va="center")
    ax.text(0.67, 9.22, "Qwen2.5-Omni-3B  +  M2KT 风格可学习记忆适配器",
            fontsize=12.5, color=MUTED, va="center")
    for x, fill, border, label in [
        (12.35, BLUE, BLUE_EDGE, "冻结"),
        (14.0, AMBER, AMBER_EDGE, "可训练"),
        (15.95, GREEN, GREEN_EDGE, "输出"),
    ]:
        box(ax, x, 9.31, 0.35, 0.28, "", face=fill, edge=border,
            linewidth=1.0, rounding=0.05)
        ax.text(x + 0.48, 9.45, label, fontsize=10.5, color=MUTED, va="center")

    ax.text(0.68, 8.91, "A  推理主路径", color=INK, fontsize=12, fontweight="bold")
    top_y, top_h = 7.46, 1.15
    main = [
        (0.65, 1.95, "课堂片段", "2 × 约 8 秒 MP4", NEUTRAL, NEUTRAL_EDGE),
        (2.95, 2.40, "Omni 音视频编码器", "提取同步声音与画面", BLUE, BLUE_EDGE),
        (5.70, 1.95, "音视频 token", "2048 维", NEUTRAL, NEUTRAL_EDGE),
        (8.00, 2.25, "M2KT 记忆适配器", "仅更新音视频 token", AMBER, AMBER_EDGE),
        (10.60, 2.40, "Omni Thinker + LoRA", "生成中文报告", BLUE, BLUE_EDGE),
        (13.35, 1.85, "单段整理", "去栏目标题与换行", NEUTRAL, NEUTRAL_EDGE),
        (15.55, 1.80, "中文报告", "待核查草稿", GREEN, GREEN_EDGE),
    ]
    for x, w, title, detail, face, edge in main:
        box(ax, x, top_y, w, top_h, title, detail, face=face, edge=edge,
            title_size=12.3, detail_size=10)
    for left, right in zip(main[:-1], main[1:]):
        arrow(ax, (left[0] + left[1] + 0.06, top_y + top_h / 2),
              (right[0] - 0.06, top_y + top_h / 2), mutation=13)

    box(ax, 10.91, 6.66, 1.8, 0.40, "原训练提示词", face=NEUTRAL,
        edge=NEUTRAL_EDGE, title_size=10.4, rounding=0.08, linewidth=1.0)
    arrow(ax, (11.81, 7.07), (11.81, 7.40), width=1.2, mutation=11)
    arrow(ax, (9.125, 7.41), (9.125, 6.23), color=AMBER_EDGE,
          dashed=True, width=1.35, mutation=12)

    panel(ax, 0.65, 3.22, 16.70, 2.95,
          "B  记忆适配器内部结构   ·   1,585,409 个可训练参数")
    parts = [
        (1.00, 3.35, "特征投影", "2048 → 256 维\n+ 模态 / 片段嵌入"),
        (4.88, 3.35, "记忆读取", "16 个可学习记忆槽\n从音视频 token 读取证据"),
        (8.76, 3.35, "Token 读取", "音视频 token\n从更新后的记忆读取"),
        (12.64, 3.35, "门控残差写回", "256 → 2048 维\n仅写回音视频位置"),
    ]
    for x, w, title, detail in parts:
        box(ax, x, 4.04, w, 1.28, title, detail, face=AMBER,
            edge=AMBER_EDGE, title_size=12.5, detail_size=10)
    for left, right in zip(parts[:-1], parts[1:]):
        arrow(ax, (left[0] + left[1] + 0.10, 4.68),
              (right[0] - 0.10, 4.68), color="#A97730", mutation=13)
    ax.text(1.10, 3.59, "残差旁路：原始 2048 维 token 保留；文本 token 原样通过",
            fontsize=10.5, color=MUTED, va="center")
    arrow(ax, (8.05, 3.57), (14.20, 3.57), color=AMBER_EDGE,
          width=1.1, mutation=10)

    panel(ax, 0.65, 0.57, 16.70, 2.05,
          "C  训练监督   ·   仅训练阶段", edge="#A5B5C7", face="#FAFCFE")
    box(ax, 1.02, 0.90, 3.15, 0.90, "原始中文报告", "作为目标文本",
        face=NEUTRAL, edge=NEUTRAL_EDGE, title_size=11.8, detail_size=9.8)
    box(ax, 5.16, 0.90, 3.45, 0.90, "冻结 Thinker 的预测", "报告 token logits",
        face=BLUE, edge=BLUE_EDGE, title_size=11.8, detail_size=9.8)
    box(ax, 9.63, 0.90, 2.90, 0.90, "Token 交叉熵", "对齐原始报告",
        face=NEUTRAL, edge=NEUTRAL_EDGE, title_size=11.8, detail_size=9.8)
    box(ax, 13.53, 0.90, 3.42, 0.90, "仅更新记忆适配器", "Omni 与 LoRA 均冻结",
        face=AMBER, edge=AMBER_EDGE, title_size=11.8, detail_size=9.8)
    ax.plot([2.60, 2.60, 11.08], [1.84, 2.03, 2.03],
            color=LINE, linewidth=1.3, zorder=3)
    arrow(ax, (11.08, 2.03), (11.08, 1.84), width=1.3, mutation=10)
    arrow(ax, (8.66, 1.35), (9.55, 1.35), width=1.35, mutation=11)
    arrow(ax, (12.60, 1.35), (13.46, 1.35), color=AMBER_EDGE,
          dashed=True, width=1.6, mutation=12)

    ax.text(0.68, 0.20,
            "说明：适配器借鉴 M2KT 的可学习记忆思路；记忆槽由课堂报告监督学习，不是手工标注的课堂知识库。",
            fontsize=9.8, color=MUTED, va="center")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("--dpi must be at least 100")
    draw(args.output, args.dpi)
    print(args.output)


if __name__ == "__main__":
    main()
