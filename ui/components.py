"""UI plotting components for season analysis."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_season_map(
    x: float,
    y: float,
    x_label: str,
    y_label: str,
    x_unc: float | None = None,
    y_unc: float | None = None,
    point_color: str = "#1f77b4",
):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_facecolor("#fbfcfe")
    ax.grid(True, color="#e8edf2", linewidth=0.8, alpha=0.9)

    ax.axvline(0.0, color="#8d98a5", linewidth=1.1)
    ax.axhline(0.0, color="#8d98a5", linewidth=1.1)

    ax.scatter([x], [y], s=95, color=point_color, zorder=4, edgecolors="white", linewidths=0.9)
    ax.annotate(
        "You",
        (x, y),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=9,
        color=point_color,
        fontweight="semibold",
    )

    if x_unc is not None and y_unc is not None and x_unc > 0 and y_unc > 0:
        e = Ellipse(
            (x, y),
            width=2.0 * x_unc,
            height=2.0 * y_unc,
            facecolor=point_color,
            edgecolor=point_color,
            linewidth=1.0,
            alpha=0.14,
        )
        ax.add_patch(e)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8.5, colors="#516272")
    for spine in ax.spines.values():
        spine.set_color("#d2dce6")
        spine.set_linewidth(1.0)

    y_is_value = "Depth" in y_label or "Light" in y_label or "Deep" in y_label
    y_is_chroma = "Saturation" in y_label or "Bright" in y_label or "Soft" in y_label
    if y_is_value:
        top_word, bot_word = "Deep", "Light"
    elif y_is_chroma:
        top_word, bot_word = "Bright", "Soft"
    else:
        top_word, bot_word = "High", "Low"

    ax.text(-0.98, 0.98, f"Q2: Cool + {top_word}", fontsize=8, color="#667686", va="top")
    ax.text(0.98, 0.98, f"Q1: Warm + {top_word}", fontsize=8, color="#667686", va="top", ha="right")
    ax.text(-0.98, -0.98, f"Q3: Cool + {bot_word}", fontsize=8, color="#667686", va="bottom")
    ax.text(0.98, -0.98, f"Q4: Warm + {bot_word}", fontsize=8, color="#667686", va="bottom", ha="right")

    ax.text(-0.98, 0.02, "Cool", fontsize=8, color="#728394", va="bottom")
    ax.text(0.98, 0.02, "Warm", fontsize=8, color="#728394", va="bottom", ha="right")
    ax.text(0.02, 0.98, top_word, fontsize=8, color="#728394", va="top")
    ax.text(0.02, -0.98, bot_word, fontsize=8, color="#728394", va="bottom")

    fig.tight_layout()
    return fig


def plot_season_scores(scores: Sequence[float], labels: Sequence[str]):
    pairs = list(zip(labels, scores))
    pairs.sort(key=lambda p: p[1], reverse=True)
    labels_sorted = [p[0] for p in pairs]
    scores_sorted = [p[1] for p in pairs]

    max_idx = 0
    second_idx = 1 if len(scores_sorted) > 1 else 0
    colors = ["#1f77b4" if i in {max_idx, second_idx} else "#d4dce3" for i in range(len(scores_sorted))]

    fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=110)
    y = list(range(len(labels_sorted)))
    bars = ax.barh(y, scores_sorted, color=colors, edgecolor="#9aa7b3")
    ax.set_yticks(y, labels_sorted)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color="#e8edf2", linewidth=0.8, alpha=0.9)
    ax.set_facecolor("#fbfcfe")
    for spine in ax.spines.values():
        spine.set_color("#d2dce6")
        spine.set_linewidth(1.0)

    xmin = min(0.0, min(scores_sorted) - 0.1)
    xmax = max(scores_sorted) + 0.1
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Calibrated Season Score", fontsize=10)
    ax.tick_params(labelsize=8.5, colors="#516272")
    ax.axvline(0.0, color="#8d98a5", linewidth=1.0, alpha=0.8)

    for b, v in zip(bars, scores_sorted):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2.0, f"{v:.3f}", va="center", fontsize=8, color="#334455")

    if len(labels_sorted) >= 2:
        ax.set_title(f"Top: {labels_sorted[0]}  •  Runner-up: {labels_sorted[1]}", fontsize=10, color="#334455")

    fig.tight_layout()
    return fig
