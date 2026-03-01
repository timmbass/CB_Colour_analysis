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
):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_facecolor("#fafafa")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)

    ax.axvline(0.0, color="#8c8c8c", linewidth=1.0)
    ax.axhline(0.0, color="#8c8c8c", linewidth=1.0)

    ax.scatter([x], [y], s=85, color="#1f77b4", zorder=4)
    ax.annotate("You", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9, color="#1f77b4")

    if x_unc is not None and y_unc is not None and x_unc > 0 and y_unc > 0:
        e = Ellipse((x, y), width=2.0 * x_unc, height=2.0 * y_unc, facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.12)
        ax.add_patch(e)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    y_is_value = "Depth" in y_label or "Light" in y_label or "Deep" in y_label
    y_is_chroma = "Saturation" in y_label or "Bright" in y_label or "Soft" in y_label
    if y_is_value:
        top_word, bot_word = "Deep", "Light"
    elif y_is_chroma:
        top_word, bot_word = "Bright", "Soft"
    else:
        top_word, bot_word = "High", "Low"

    ax.text(-0.98, 0.98, f"Q2: Cool + {top_word}", fontsize=8, color="#666", va="top")
    ax.text(0.98, 0.98, f"Q1: Warm + {top_word}", fontsize=8, color="#666", va="top", ha="right")
    ax.text(-0.98, -0.98, f"Q3: Cool + {bot_word}", fontsize=8, color="#666", va="bottom")
    ax.text(0.98, -0.98, f"Q4: Warm + {bot_word}", fontsize=8, color="#666", va="bottom", ha="right")

    ax.text(-0.98, 0.02, "Cool", fontsize=8, color="#777", va="bottom")
    ax.text(0.98, 0.02, "Warm", fontsize=8, color="#777", va="bottom", ha="right")

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

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = list(range(len(labels_sorted)))
    bars = ax.barh(y, scores_sorted, color=colors, edgecolor="#9aa7b3")
    ax.set_yticks(y, labels_sorted)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.7)

    xmin = min(0.0, min(scores_sorted) - 0.1)
    xmax = max(scores_sorted) + 0.1
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Calibrated score")

    for b, v in zip(bars, scores_sorted):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2.0, f"{v:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig
