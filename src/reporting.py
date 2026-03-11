"""PDF scorecard generation."""

from __future__ import annotations

import io
import textwrap
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def build_scorecard_pdf_bytes(
    subject_name: str,
    season_display: str,
    confidence: float,
    axes: dict[str, float],
    palette_hex: list[str],
    dynamic_suggestions: dict[str, Any],
    text_suggestions: dict[str, Any],
    input_rgb: np.ndarray | None,
) -> bytes:
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor("white")
    try:
        ax_header = fig.add_axes([0.07, 0.91, 0.86, 0.07])
        ax_header.axis("off")
        ax_header.text(0.0, 0.62, "Color Analysis Scorecard", fontsize=18, fontweight="bold", ha="left", va="center")
        ax_header.text(0.0, 0.18, subject_name, fontsize=10, color="#4b5563", ha="left", va="center")

        ax_summary = fig.add_axes([0.07, 0.80, 0.42, 0.10])
        ax_summary.axis("off")
        ax_summary.text(0.0, 1.0, "Summary Metrics", fontsize=12, fontweight="bold", ha="left", va="top")
        for index, row in enumerate((
            f"Season: {season_display}",
            f"Confidence: {confidence:.1%}",
            f"Temperature: {axes.get('temp', 0.0):.3f}",
            f"Value: {axes.get('value', 0.0):.3f}",
            f"Chroma: {axes.get('chroma', 0.0):.3f}",
            f"Contrast: {axes.get('contrast', 0.0):.3f}",
        )):
            ax_summary.text(0.0, 0.82 - index * 0.15, row, fontsize=9.5, ha="left", va="top")

        ax_palette = fig.add_axes([0.53, 0.80, 0.40, 0.10])
        ax_palette.set_xlim(0, 1)
        ax_palette.set_ylim(0, 1)
        ax_palette.axis("off")
        ax_palette.text(0.0, 1.0, "Palette", fontsize=12, fontweight="bold", ha="left", va="top")
        if palette_hex:
            swatch_width = min(0.12, 0.86 / max(1, len(palette_hex)))
            for index, hex_code in enumerate(palette_hex[:7]):
                x_pos = 0.01 + index * (swatch_width + 0.01)
                ax_palette.add_patch(Rectangle((x_pos, 0.42), swatch_width, 0.32, facecolor=hex_code, edgecolor="#cfcfcf", linewidth=0.8))
                ax_palette.text(x_pos + swatch_width / 2.0, 0.35, hex_code, fontsize=6.5, ha="center", va="top")
        else:
            ax_palette.text(0.0, 0.62, "n/a", fontsize=9, ha="left", va="center")

        ax_photo = fig.add_axes([0.07, 0.56, 0.28, 0.21])
        ax_photo.axis("off")
        ax_photo.text(0.0, 1.03, "Input Photo", fontsize=12, fontweight="bold", ha="left", va="bottom")
        if input_rgb is not None:
            src_h, src_w = input_rgb.shape[:2]
            thumb = cv2.resize(input_rgb, (max(1, int(round((src_w / src_h) * 240))), 240), interpolation=cv2.INTER_AREA) if src_h > 0 and src_w > 0 else input_rgb
            ax_photo.imshow(thumb)
            ax_photo.set_aspect("equal")
        else:
            ax_photo.text(0.0, 0.5, "n/a", fontsize=9, ha="left", va="center")

        ax_dynamic = fig.add_axes([0.07, 0.33, 0.86, 0.17])
        ax_dynamic.set_xlim(0, 1)
        ax_dynamic.set_ylim(0, 1)
        ax_dynamic.axis("off")
        ax_dynamic.text(0.0, 1.02, "Dynamic Best Colours", fontsize=12, fontweight="bold", ha="left", va="top")
        colors = dynamic_suggestions.get("colors", [])
        for index, item in enumerate(colors[:8]):
            row = index // 4
            col = index % 4
            x0 = 0.01 + col * (0.23 + 0.015)
            y0 = [0.54, 0.08][row]
            ax_dynamic.add_patch(Rectangle((x0, y0), 0.23, 0.42, facecolor="#fafafa", edgecolor="#e2e2e2", linewidth=0.8))
            hex_code = item.get("hex", "#999999")
            ax_dynamic.add_patch(Rectangle((x0 + 0.012, y0 + 0.20), 0.07, 0.16, facecolor=hex_code, edgecolor="#cfcfcf", linewidth=0.8))
            ax_dynamic.text(x0 + 0.09, y0 + 0.33, str(item.get("label", item.get("name", ""))), fontsize=7.6, fontweight="bold", ha="left", va="center")
            ax_dynamic.text(x0 + 0.09, y0 + 0.23, hex_code, fontsize=7.2, family="monospace", ha="left", va="center")
            ax_dynamic.text(x0 + 0.012, y0 + 0.17, textwrap.fill(str(item.get("reason", "")), width=30), fontsize=6.8, ha="left", va="top")
        ax_dynamic.text(0.0, 0.0, "These shades are computed from your measured warmth, depth, saturation, and contrast.", fontsize=7.0, color="#444444", ha="left", va="bottom")

        ax_notes = fig.add_axes([0.07, 0.06, 0.86, 0.24])
        ax_notes.axis("off")
        ax_notes.text(0.0, 1.0, "Text Suggestions", fontsize=12, fontweight="bold", ha="left", va="top")
        y_pos = 0.9
        headline = str(text_suggestions.get("headline", "")).strip()
        if headline:
            for seg in textwrap.wrap(headline, width=108)[:2]:
                ax_notes.text(0.0, y_pos, seg, fontsize=8.7, fontweight="bold", ha="left", va="top")
                y_pos -= 0.06
        for block in text_suggestions.get("blocks", []):
            title = str(block.get("title", "")).strip()
            if title and y_pos >= 0.06:
                ax_notes.text(0.0, y_pos, title, fontsize=8.3, fontweight="bold", ha="left", va="top")
                y_pos -= 0.055
            text = str(block.get("text", "")).strip()
            if text and y_pos >= 0.06:
                for seg in textwrap.wrap(text, width=110)[:2]:
                    ax_notes.text(0.0, y_pos, seg, fontsize=8.0, ha="left", va="top")
                    y_pos -= 0.05
            for bullet in block.get("bullets", [])[:6]:
                for idx, seg in enumerate(textwrap.wrap(str(bullet), width=106)[:2]):
                    ax_notes.text(0.0, y_pos, f"{'- ' if idx == 0 else '  '}{seg}", fontsize=7.9, ha="left", va="top")
                    y_pos -= 0.048
                    if y_pos < 0.03:
                        break
                if y_pos < 0.03:
                    break
            if y_pos < 0.03:
                break

        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()
    finally:
        plt.close(fig)
