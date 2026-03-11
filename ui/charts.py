"""Chart helpers isolated from core analysis logic."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Ellipse


def plot_season_map(
    x: float,
    y: float,
    x_label: str,
    y_label: str,
    x_unc: float | None = None,
    y_unc: float | None = None,
    point_color: str = "#1f77b4",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_facecolor("#fbfcfe")
    ax.grid(True, color="#e8edf2", linewidth=0.8, alpha=0.9)
    ax.axvline(0.0, color="#8d98a5", linewidth=1.1)
    ax.axhline(0.0, color="#8d98a5", linewidth=1.1)
    ax.scatter([x], [y], s=95, color=point_color, zorder=4, edgecolors="white", linewidths=0.9)
    ax.annotate("You", (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9, color=point_color, fontweight="semibold")
    if x_unc is not None and y_unc is not None and x_unc > 0 and y_unc > 0:
        ax.add_patch(Ellipse((x, y), width=2.0 * x_unc, height=2.0 * y_unc, facecolor=point_color, edgecolor=point_color, linewidth=1.0, alpha=0.14))
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
    top_word, bot_word = ("Deep", "Light") if y_is_value else ("Bright", "Soft") if y_is_chroma else ("High", "Low")
    ax.text(-0.98, 0.98, f"Q2: Cool + {top_word}", fontsize=8, color="#667686", va="top")
    ax.text(0.98, 0.98, f"Q1: Warm + {top_word}", fontsize=8, color="#667686", va="top", ha="right")
    ax.text(-0.98, -0.98, f"Q3: Cool + {bot_word}", fontsize=8, color="#667686", va="bottom")
    ax.text(0.98, -0.98, f"Q4: Warm + {bot_word}", fontsize=8, color="#667686", va="bottom", ha="right")
    fig.tight_layout()
    return fig


def plot_season_scores(scores: Sequence[float], labels: Sequence[str]) -> plt.Figure:
    pairs = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    labels_sorted = [item[0] for item in pairs]
    scores_sorted = [item[1] for item in pairs]
    colors = ["#1f77b4" if idx in {0, 1} else "#d4dce3" for idx in range(len(scores_sorted))]
    fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=110)
    bars = ax.barh(list(range(len(labels_sorted))), scores_sorted, color=colors, edgecolor="#9aa7b3")
    ax.set_yticks(list(range(len(labels_sorted))), labels_sorted)
    ax.invert_yaxis()
    ax.grid(True, axis="x", color="#e8edf2", linewidth=0.8, alpha=0.9)
    ax.set_facecolor("#fbfcfe")
    for spine in ax.spines.values():
        spine.set_color("#d2dce6")
        spine.set_linewidth(1.0)
    ax.set_xlim(min(0.0, min(scores_sorted) - 0.1), max(scores_sorted) + 0.1)
    ax.set_xlabel("Calibrated Season Score", fontsize=10)
    ax.tick_params(labelsize=8.5, colors="#516272")
    ax.axvline(0.0, color="#8d98a5", linewidth=1.0, alpha=0.8)
    for bar, score in zip(bars, scores_sorted):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{score:.3f}", va="center", fontsize=8, color="#334455")
    if len(labels_sorted) >= 2:
        ax.set_title(f"Top: {labels_sorted[0]}  •  Runner-up: {labels_sorted[1]}", fontsize=10, color="#334455")
    fig.tight_layout()
    return fig


def render_hist_chart(values: list[float], title: str, x_label: str, threshold: float | None = None) -> None:
    if not values:
        st.write(f"{title}: no data")
        return
    counts, edges = np.histogram(np.array(values, dtype=np.float32), bins=min(10, max(4, len(values))))
    chart_values = [{"x": float(x), "count": int(c)} for x, c in zip(((edges[:-1] + edges[1:]) / 2.0).tolist(), counts.tolist())]
    layer: list[dict[str, object]] = [{
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": x_label},
            "y": {"field": "count", "type": "quantitative", "title": "Image count"},
        },
    }]
    if threshold is not None:
        layer.append({"mark": {"type": "rule", "color": "#c0392b", "strokeWidth": 2}, "encoding": {"x": {"datum": float(threshold)}}})
    st.write(f"**{title}**")
    st.vega_lite_chart({"values": chart_values}, {"layer": layer}, use_container_width=True)
