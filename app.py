from __future__ import annotations

import hashlib
import io
import json
import time
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

import numpy as np

from src.calibration import apply_calibration, margin_confidence, predict_topk
from src.color_features import ColorFeatures, compute_robust_lab_features, compute_skin_chroma_variance
from src.copy_renderer import render_text_suggestions
from src.drape_scoring import apply_drape_color, evaluate_drape_scores
from src.dynamic_colors import config_hash, load_dynamic_colors_config, suggest_dynamic_colors
from src.diagnostics_regions import compute_definition_score, compute_region_diagnostics, render_diagnostics_overlay
try:
    from src.drape import render_color_strip
except ImportError:
    # Fallback for stale environments where render_color_strip is unavailable.
    def render_color_strip(colors: list[str], width: int, height: int, style: str = "blocks") -> np.ndarray | None:
        if not colors or width <= 0 or height <= 0:
            return None
        if style != "blocks":
            raise ValueError("Only 'blocks' style is supported")

        def _hex_to_rgb_local(hex_color: str) -> tuple[int, int, int]:
            c = hex_color.strip().lstrip("#")
            if len(c) != 6:
                return (128, 128, 128)
            return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))

        strip = np.zeros((height, width, 3), dtype=np.uint8)
        block_width = max(1, width // len(colors))
        border = np.array([228, 228, 228], dtype=np.uint8)
        outer = np.array([180, 180, 180], dtype=np.uint8)

        for i, color_hex in enumerate(colors):
            start_x = i * block_width
            end_x = width if i == len(colors) - 1 else min(width, (i + 1) * block_width)
            strip[:, start_x:end_x] = _hex_to_rgb_local(color_hex)
            if i > 0 and start_x < width:
                strip[:, max(0, start_x - 1) : start_x] = border

        strip[0:1, :] = outer
        strip[-1:, :] = outer
        strip[:, 0:1] = outer
        strip[:, -1:] = outer
        return strip
from src.face_regions import FaceMeshDetector, build_region_masks, render_debug_overlay
from src.image_io import decode_uploaded_image
from src.image_quality import evaluate_image_quality
from src.palettes import (
    load_palette_metadata,
    load_palettes,
    palette_for_season,
)
from src.season_index import IDX_TO_SEASON, SEASON_TO_IDX, SEASONS
from src.season_rules import classify_season
from src.stress_features import cool_stress_delta, summer_winter_nudge
from src.variant_rules import choose_variant
from ui.components import plot_season_map, plot_season_scores
from ui.copy import AXIS_HELP, AXIS_LABELS
from ui.styles import apply_base_styles


st.set_page_config(page_title="Personal Color Analysis", layout="wide")
apply_base_styles()
st.title("🎨 Personal Color Analysis")
st.write(
    "Upload one or more photos. The app samples cheek/hair/iris regions, computes 4 continuous color axes "
    "(temperature, value, chroma, contrast), predicts your season, and shows a digital drape palette."
)
st.write(
    "Lab and chroma are a way to describe color numerically: L* is lightness, a* runs from green to red, "
    "and b* runs from blue to yellow. Chroma is how saturated the color is, calculated from a* and b* "
    "and shown here as an easy read on how vivid or muted your cheek tones appear."
)

show_debug = st.checkbox("Show debug overlays (cheek polylines + masks)", value=False)
show_per_image = st.checkbox("Show per-image results", value=True)
show_diagnostics = st.checkbox("Show diagnostics panel", value=True)
show_drape_previews = st.checkbox("Show top drape previews (winning season)", value=False)
use_advanced_dynamic = st.checkbox(
    "Optimise drape colours (slower)",
    value=False,
    help="Tries multiple candidate colours and selects the one that produces the best drape score.",
)
season_map_y_axis = st.selectbox("Season Map Y-axis", options=["value", "chroma", "contrast"], index=0)
use_quality_weighted_aggregation = st.checkbox(
    "Use quality-weighted multi-image aggregation",
    value=False,
    help="Uses weighted median across per-image season score vectors. Current quality proxy uses usable skin sample count.",
)
quality_handling = st.selectbox(
    "Low-quality handling",
    options=["exclude", "down-weight", "off"],
    index=0,
    help="Exclude low-quality images or keep them with reduced aggregation weight.",
)
quality_threshold = st.slider(
    "Quality threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.01,
)
wb_method = st.selectbox(
    "White-balance method",
    options=["none", "grayworld"],
    index=1,
    help="Applied before face detection and color sampling.",
)


@st.cache_resource
def get_face_detector() -> FaceMeshDetector:
    return FaceMeshDetector(static_image_mode=True, max_num_faces=1)


@st.cache_data(show_spinner=False)
def analyze_image(file_bytes: bytes, wb_method_choice: str) -> dict:
    image_hash = hashlib.sha256(file_bytes).hexdigest()
    image_rgb = decode_uploaded_image(file_bytes)
    if image_rgb is None:
        return {"status": "decode_failed", "image_hash": image_hash}
    image_rgb_pre_wb = image_rgb.copy()

    wb_method_used = wb_method_choice
    wb_applied = wb_method_choice == "grayworld"
    if wb_method_choice == "grayworld":
        used_xphoto = False
        try:
            if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "createGrayworldWB"):
                wb = cv2.xphoto.createGrayworldWB()
                bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                corrected_bgr = wb.balanceWhite(bgr)
                if corrected_bgr is not None:
                    image_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
                    used_xphoto = True
        except Exception:
            used_xphoto = False

        if not used_xphoto:
            rgb_f = image_rgb.astype(np.float32)
            channel_means = np.mean(rgb_f.reshape(-1, 3), axis=0)
            gray_target = float(np.mean(channel_means))
            scale = gray_target / np.maximum(channel_means, 1e-6)
            image_rgb = np.clip(rgb_f * scale[None, None, :], 0, 255).astype(np.uint8)
            wb_method_used = "grayworld (fallback)"

    detector = get_face_detector()
    landmarks = detector.detect_single_face(image_rgb)
    if landmarks is None:
        return {
            "status": "no_face",
            "image_rgb": image_rgb,
            "image_hash": image_hash,
            "wb_method": wb_method_used,
            "wb_applied": wb_applied,
        }

    regions = build_region_masks(image_rgb.shape, landmarks)
    features = compute_robust_lab_features(image_rgb, regions.left_mask, regions.right_mask)
    pre_wb_features = compute_robust_lab_features(image_rgb_pre_wb, regions.left_mask, regions.right_mask)
    quality = evaluate_image_quality(
        image_rgb=image_rgb,
        image_rgb_pre_wb=image_rgb_pre_wb,
        landmarks_px=regions.landmarks_px.astype(np.float32),
        left_mask=regions.left_mask,
        right_mask=regions.right_mask,
        cheek_kept_count=features.sample_count if features is not None else None,
    )
    skin_chroma_var = compute_skin_chroma_variance(image_rgb, regions.left_mask, regions.right_mask)
    if features is None:
        return {
            "status": "weak_sample",
            "image_rgb": image_rgb,
            "image_hash": image_hash,
            "regions": regions,
            "quality": quality,
            "wb_method": wb_method_used,
            "wb_applied": wb_applied,
        }

    return {
        "status": "ok",
        "image_rgb": image_rgb,
        "image_hash": image_hash,
        "regions": regions,
        "features": features,
        "pre_wb_skin_b": None if pre_wb_features is None else float(pre_wb_features.b),
        "post_wb_skin_b": float(features.b),
        "skin_chroma_var": 0.0 if skin_chroma_var is None else float(skin_chroma_var),
        "quality": quality,
        "wb_method": wb_method_used,
        "wb_applied": wb_applied,
    }


def aggregate_features(feature_list: list[ColorFeatures]) -> ColorFeatures:
    weights = np.array([f.sample_count for f in feature_list], dtype=np.float32)
    l_vals = np.array([f.l for f in feature_list], dtype=np.float32)
    a_vals = np.array([f.a for f in feature_list], dtype=np.float32)
    b_vals = np.array([f.b for f in feature_list], dtype=np.float32)

    l_avg = float(np.average(l_vals, weights=weights))
    a_avg = float(np.average(a_vals, weights=weights))
    b_avg = float(np.average(b_vals, weights=weights))
    chroma = float(np.sqrt(a_avg**2 + b_avg**2))
    sample_count = int(np.sum(weights))

    return ColorFeatures(l=l_avg, a=a_avg, b=b_avg, chroma=chroma, sample_count=sample_count)


def _fmt_feature(label: str, feat: ColorFeatures | None) -> str:
    if feat is None:
        return f"- {label}: n/a"
    return (
        f"- {label}: L*={feat.l:.2f}, a*={feat.a:.2f}, b*={feat.b:.2f}, "
        f"chroma={feat.chroma:.2f} (n={feat.sample_count})"
    )


def _fmt_delta(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: n/a"
    return f"- {label}: {value:.2f}"


def _hue_proxy(feat: ColorFeatures | None) -> float | None:
    if feat is None:
        return None
    return float(np.arctan2(feat.b, feat.a))


def _mutedness_proxy(feat: ColorFeatures | None) -> float | None:
    if feat is None:
        return None
    return float(feat.chroma / (feat.l + 1e-6))


def _temp_band_from_b(b_value: float) -> str:
    if b_value < 10.0:
        return "cool"
    if b_value > 18.0:
        return "warm"
    return "neutral"


def _palette_season_from_variant(base_season: str, variant_key: str, palette_code: str) -> str:
    code_to_base = {"A": "Autumn", "B": "Summer", "C": "Winter", "D": "Spring"}
    code_base = code_to_base.get(palette_code)
    variant_lower = variant_key.lower()
    if "winter" in variant_lower:
        return "Winter"
    if "spring" in variant_lower:
        return "Spring"
    if "summer" in variant_lower:
        return "Summer"
    if "autumn" in variant_lower:
        return "Autumn"
    if base_season in {"Spring", "Summer", "Autumn", "Winter"}:
        return base_season
    if code_base is not None:
        return code_base
    return base_season


def _normalize_axis(raw: float, axis_name: str) -> float:
    # Stable fallback normalization constants if raw axis drifts beyond [-1,1].
    constants = {
        "temperature": (0.0, 0.75),
        "value": (0.0, 0.75),
        "chroma": (0.0, 0.75),
        "contrast": (0.0, 0.75),
    }
    if -1.0 <= raw <= 1.0:
        return float(raw)
    mean, std = constants.get(axis_name, (0.0, 1.0))
    z = (float(raw) - mean) / max(std, 1e-6)
    return float(np.clip(z, -2.0, 2.0) / 2.0)


def _season_accent_color(season: str) -> str:
    palette = {
        "Spring": "#f39c12",
        "Summer": "#4a90c2",
        "Autumn": "#b5653c",
        "Winter": "#2f4f8f",
    }
    return palette.get(season, "#1f77b4")


def _hist_chart(values: list[float], title: str, x_label: str, threshold: float | None = None) -> None:
    if not values:
        st.write(f"{title}: no data")
        return

    counts, edges = np.histogram(np.array(values, dtype=np.float32), bins=min(10, max(4, len(values))))
    mids = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    chart_values = [{"x": float(x), "count": int(c)} for x, c in zip(mids, counts.tolist())]
    layer = [
        {
            "mark": {"type": "bar", "tooltip": True},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": x_label},
                "y": {"field": "count", "type": "quantitative", "title": "Image count"},
            },
        }
    ]
    if threshold is not None:
        layer.append(
            {
                "mark": {"type": "rule", "color": "#c0392b", "strokeWidth": 2},
                "encoding": {"x": {"datum": float(threshold)}},
            }
        )

    st.write(f"**{title}**")
    st.vega_lite_chart({"values": chart_values}, {"layer": layer}, use_container_width=True)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")
    arr = np.array(values, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0.0:
        return float(np.median(arr))
    order = np.argsort(arr)
    sorted_vals = arr[order]
    sorted_w = w[order]
    cum_w = np.cumsum(sorted_w)
    cutoff = 0.5 * float(np.sum(sorted_w))
    idx = int(np.searchsorted(cum_w, cutoff, side="left"))
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _aggregate_season_scores(
    per_image_scores: list[dict[str, float]],
    per_image_weights: list[float] | None = None,
    use_weights: bool = False,
) -> dict[str, float]:
    seasons = SEASONS
    if not per_image_scores:
        return {s: 0.0 for s in seasons}

    if use_weights and per_image_weights is not None and len(per_image_weights) == len(per_image_scores):
        return {
            s: _weighted_median([scores[s] for scores in per_image_scores], per_image_weights)
            for s in seasons
        }
    return {s: float(np.median(np.array([scores[s] for scores in per_image_scores], dtype=np.float64))) for s in seasons}


@st.cache_data(show_spinner=False)
def load_calibration_params(path: str = "calibration_params.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {"alpha": 0.5, "bias": [0.0, 0.0, 0.0, 0.0], "gamma": 3.0, "source": "default"}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        alpha = float(payload.get("alpha", 0.5))
        bias = payload.get("bias", [0.0, 0.0, 0.0, 0.0])
        gamma = float(payload.get("gamma", 3.0))
        if not isinstance(bias, list) or len(bias) != 4:
            bias = [0.0, 0.0, 0.0, 0.0]
        return {"alpha": alpha, "bias": [float(x) for x in bias], "gamma": gamma, "source": "file"}
    except Exception:
        return {"alpha": 0.5, "bias": [0.0, 0.0, 0.0, 0.0], "gamma": 3.0, "source": "default"}


@st.cache_data(show_spinner=False)
def load_dynamic_color_config(path: str = "dynamic_colors_config.json") -> dict:
    return load_dynamic_colors_config(path)


@st.cache_data(show_spinner=False)
def compute_dynamic_color_suggestions(
    image_hash: str,
    axes_payload: tuple[float, float, float, float],
    mode: str,
    cfg_hash: str,
    return_set: bool,
    cfg_json: str,
    season_hint: str | None,
    image_rgb: np.ndarray | None,
    regions,
    baseline_skin: ColorFeatures | None,
    diagnostics: bool,
) -> dict:
    _ = image_hash, cfg_hash
    config = json.loads(cfg_json)
    axes = {
        "temp": float(axes_payload[0]),
        "value": float(axes_payload[1]),
        "chroma": float(axes_payload[2]),
        "contrast": float(axes_payload[3]),
    }
    face_context = None
    if mode == "advanced":
        face_context = {
            "image_rgb": image_rgb,
            "regions": regions,
            "baseline_skin": baseline_skin,
            "season_hint": season_hint,
        }
    return suggest_dynamic_colors(
        face_context=face_context,
        axes=axes,
        mode=mode,
        config=config,
        diagnostics=diagnostics,
        return_set=return_set,
    )


def _compose_personalised_drape_preview(image_rgb: np.ndarray, dynamic_suggestions: dict, strip_height: int = 92) -> np.ndarray:
    colors = [c.get("hex", "#888888") for c in dynamic_suggestions.get("colors", []) if c.get("hex")]
    strip = render_color_strip(colors=colors, width=image_rgb.shape[1], height=strip_height, style="blocks")
    if strip is None:
        return image_rgb
    return np.vstack([image_rgb, strip])


def _render_color_swatch(hex_code: str) -> None:
    st.markdown(
        f"<div style='height:72px;border-radius:10px;border:1px solid #d9d9d9;background:{hex_code};'></div>",
        unsafe_allow_html=True,
    )


def _render_dynamic_best_colours(title: str, suggestions: dict, axes_used: dict[str, float], show_diag: bool) -> None:
    st.write(f"**{title}**")
    swatches = suggestions.get("colors", [])
    n_cols = 4 if len(swatches) >= 8 else 3
    for i in range(0, len(swatches), n_cols):
        row = swatches[i : i + n_cols]
        cols = st.columns(n_cols)
        for col, color in zip(cols, row):
            with col:
                _render_color_swatch(color["hex"])
                st.caption(color.get("label", color.get("name", "")))
                st.code(color["hex"])
                st.caption(color.get("reason", ""))
    st.caption("These shades are computed from your measured warmth, depth, saturation, and contrast.")

    with st.expander("How this was chosen", expanded=False):
        st.write(
            f"- Axes used: temp={axes_used['temp']:.3f}, value={axes_used['value']:.3f}, "
            f"chroma={axes_used['chroma']:.3f}, contrast={axes_used['contrast']:.3f}"
        )
        st.write(f"- Mode: **{suggestions.get('mode', 'simple')}**")
        diag_payload = suggestions.get("diagnostics", {})
        fallback = diag_payload.get("advanced_fallback_reason")
        if fallback:
            st.write(f"- Advanced fallback: `{fallback}`")
        if show_diag and suggestions.get("mode") == "advanced":
            for color in suggestions.get("colors", []):
                adv = color.get("advanced", {})
                if not adv:
                    continue
                st.write(
                    f"- {color['name']}: penalty={adv.get('penalty', 0.0):.4f}, "
                    f"candidates_tested={int(adv.get('candidates_tested', 0))}"
                )
                top3 = adv.get("top3", [])
                if top3:
                    st.code(", ".join(f"{row['hex']} ({row['penalty']:.4f})" for row in top3))


COPY_RULES_PATH = Path("assets/copy_rules.v1.json")


def _to_copy_key(value: str | None) -> str:
    if not value:
        return ""
    return "_".join(value.strip().lower().replace("-", " ").split())


def _build_copy_payload(
    image_id: str,
    season_display: str,
    top1_season: str,
    top2_season: str | None,
    top1_variant: str | None,
    confidence: float,
    axes: dict[str, float],
    dynamic_suggestions: dict,
) -> dict:
    return {
        "image_id": image_id,
        "season_display": season_display,
        "top1_key": _to_copy_key(top1_season),
        "top2_key": _to_copy_key(top2_season),
        "top1_variant_key": _to_copy_key(top1_variant),
        "profile_key": _to_copy_key(top1_variant) or _to_copy_key(top1_season),
        "top1_display": top1_variant or top1_season,
        "top2_display": top2_season or "",
        "confidence": float(confidence),
        "axes": {
            "temp": float(axes.get("temp", 0.0)),
            "value": float(axes.get("value", 0.0)),
            "chroma": float(axes.get("chroma", 0.0)),
            "contrast": float(axes.get("contrast", 0.0)),
        },
        "dynamic_colors": dynamic_suggestions.get("colors", []),
    }


def _render_text_suggestions_section(suggestions: dict) -> None:
    headline = str(suggestions.get("headline", "")).strip()
    if headline:
        st.markdown(f"**{headline}**")

    for block in suggestions.get("blocks", []):
        title = str(block.get("title", "")).strip()
        if title:
            st.markdown(f"**{title}**")
        if block.get("id") == "close_call":
            st.warning(str(block.get("text", "")))
        elif block.get("text"):
            st.markdown(str(block.get("text")))
        bullets = block.get("bullets", [])
        if isinstance(bullets, list) and bullets:
            st.markdown("\n".join(f"- {item}" for item in bullets))


def _build_scorecard_pdf_bytes(
    subject_name: str,
    season_display: str,
    confidence: float,
    axes: dict[str, float],
    palette_hex: list[str],
    dynamic_suggestions: dict,
    text_suggestions: dict,
    input_rgb: np.ndarray | None,
) -> bytes:
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
    fig.patch.set_facecolor("white")

    ax_header = fig.add_axes([0.07, 0.91, 0.86, 0.07])
    ax_header.axis("off")
    ax_header.text(0.0, 0.62, "Color Analysis Scorecard", fontsize=18, fontweight="bold", ha="left", va="center")
    ax_header.text(0.0, 0.18, subject_name, fontsize=10, color="#4b5563", ha="left", va="center")

    ax_summary = fig.add_axes([0.07, 0.80, 0.42, 0.10])
    ax_summary.axis("off")
    summary_rows = [
        f"Season: {season_display}",
        f"Confidence: {confidence:.1%}",
        f"Temperature: {axes.get('temp', 0.0):.3f}",
        f"Value: {axes.get('value', 0.0):.3f}",
        f"Chroma: {axes.get('chroma', 0.0):.3f}",
        f"Contrast: {axes.get('contrast', 0.0):.3f}",
    ]
    ax_summary.text(0.0, 1.0, "Summary Metrics", fontsize=12, fontweight="bold", ha="left", va="top")
    for i, row in enumerate(summary_rows):
        ax_summary.text(0.0, 0.82 - i * 0.15, row, fontsize=9.5, ha="left", va="top")

    ax_palette = fig.add_axes([0.53, 0.80, 0.40, 0.10])
    ax_palette.set_xlim(0, 1)
    ax_palette.set_ylim(0, 1)
    ax_palette.axis("off")
    ax_palette.text(0.0, 1.0, "Palette", fontsize=12, fontweight="bold", ha="left", va="top")
    if palette_hex:
        sw = min(0.12, 0.86 / max(1, len(palette_hex)))
        for i, hx in enumerate(palette_hex[:7]):
            x = 0.01 + i * (sw + 0.01)
            ax_palette.add_patch(Rectangle((x, 0.42), sw, 0.32, facecolor=hx, edgecolor="#cfcfcf", linewidth=0.8))
            ax_palette.text(x + sw / 2.0, 0.35, hx, fontsize=6.5, ha="center", va="top")
    else:
        ax_palette.text(0.0, 0.62, "n/a", fontsize=9, ha="left", va="center")

    ax_photo = fig.add_axes([0.07, 0.56, 0.28, 0.21])
    ax_photo.axis("off")
    ax_photo.text(0.0, 1.03, "Input Photo", fontsize=12, fontweight="bold", ha="left", va="bottom")
    if input_rgb is not None:
        thumb_h = 240
        src_h, src_w = input_rgb.shape[:2]
        if src_h > 0 and src_w > 0:
            thumb_w = max(1, int(round((src_w / src_h) * thumb_h)))
            thumb = cv2.resize(input_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = input_rgb
        ax_photo.imshow(thumb)
        ax_photo.set_aspect("equal")
    else:
        ax_photo.text(0.0, 0.5, "n/a", fontsize=9, ha="left", va="center")

    ax_dynamic = fig.add_axes([0.07, 0.33, 0.86, 0.17])
    ax_dynamic.set_xlim(0, 1)
    ax_dynamic.set_ylim(0, 1)
    ax_dynamic.axis("off")
    ax_dynamic.text(0.0, 1.02, "Dynamic Best Colours", fontsize=12, fontweight="bold", ha="left", va="top")
    dyn = dynamic_suggestions.get("colors", [])
    cols = 4
    rows = 2
    card_w = 0.23
    card_h = 0.42
    x_gap = 0.015
    y_rows = [0.54, 0.08]
    for i, item in enumerate(dyn[: cols * rows]):
        r = i // cols
        c = i % cols
        x0 = 0.01 + c * (card_w + x_gap)
        y0 = y_rows[r]
        ax_dynamic.add_patch(Rectangle((x0, y0), card_w, card_h, facecolor="#fafafa", edgecolor="#e2e2e2", linewidth=0.8))
        hx = item.get("hex", "#999999")
        ax_dynamic.add_patch(Rectangle((x0 + 0.012, y0 + 0.20), 0.07, 0.16, facecolor=hx, edgecolor="#cfcfcf", linewidth=0.8))
        ax_dynamic.text(
            x0 + 0.09,
            y0 + 0.33,
            str(item.get("label", item.get("name", ""))),
            fontsize=7.6,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax_dynamic.text(x0 + 0.09, y0 + 0.23, hx, fontsize=7.2, family="monospace", ha="left", va="center")
        reason = textwrap.fill(str(item.get("reason", "")), width=30)
        ax_dynamic.text(x0 + 0.012, y0 + 0.17, reason, fontsize=6.8, ha="left", va="top")
    ax_dynamic.text(
        0.0,
        0.0,
        "These shades are computed from your measured warmth, depth, saturation, and contrast.",
        fontsize=7.0,
        color="#444444",
        ha="left",
        va="bottom",
    )

    ax_notes = fig.add_axes([0.07, 0.06, 0.86, 0.24])
    ax_notes.axis("off")
    ax_notes.text(0.0, 1.0, "Text Suggestions", fontsize=12, fontweight="bold", ha="left", va="top")
    y = 0.9
    headline = str(text_suggestions.get("headline", "")).strip()
    if headline:
        for seg in textwrap.wrap(headline, width=108)[:2]:
            ax_notes.text(0.0, y, seg, fontsize=8.7, fontweight="bold", ha="left", va="top")
            y -= 0.06

    blocks = text_suggestions.get("blocks", []) if isinstance(text_suggestions, dict) else []
    for block in blocks:
        title = str(block.get("title", "")).strip()
        if title and y >= 0.06:
            ax_notes.text(0.0, y, title, fontsize=8.3, fontweight="bold", ha="left", va="top")
            y -= 0.055
        text = str(block.get("text", "")).strip()
        if text and y >= 0.06:
            for seg in textwrap.wrap(text, width=110)[:2]:
                ax_notes.text(0.0, y, seg, fontsize=8.0, ha="left", va="top")
                y -= 0.05
                if y < 0.03:
                    break
        bullets = block.get("bullets", [])
        if isinstance(bullets, list):
            for bullet in bullets[:6]:
                for idx, seg in enumerate(textwrap.wrap(str(bullet), width=106)[:2]):
                    prefix = "- " if idx == 0 else "  "
                    ax_notes.text(0.0, y, f"{prefix}{seg}", fontsize=7.9, ha="left", va="top")
                    y -= 0.048
                    if y < 0.03:
                        break
                if y < 0.03:
                    break
        if y < 0.03:
            break

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

uploads = st.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

palettes = load_palettes(Path("assets/palettes/seasonal_palettes.json"))
palette_meta = load_palette_metadata(Path("assets/palettes/seasonal_palettes.json"))
calibration_params = load_calibration_params("calibration_params.json")
dynamic_color_config = load_dynamic_color_config("dynamic_colors_config.json")
dynamic_color_config_json = json.dumps(dynamic_color_config, sort_keys=True)
dynamic_color_cfg_hash = config_hash(dynamic_color_config)

if uploads:
    summary_slot = st.empty()
    valid_features: list[ColorFeatures] = []
    valid_skin_chroma_vars: list[float] = []
    valid_definition_scores: list[float] = []
    valid_hair_features: list[ColorFeatures] = []
    valid_iris_features: list[ColorFeatures] = []
    per_image_final_scores: list[dict[str, float]] = []
    per_image_baseline_scores: list[dict[str, float]] = []
    per_image_drape_scores: list[dict[str, float]] = []
    per_image_confidences: list[float] = []
    per_image_quality_weights: list[float] = []
    per_image_axes: list[dict[str, float]] = []
    per_image_payloads: list[dict] = []
    gallery_items: list[tuple[str, np.ndarray, bool]] = []
    diagnostics_records: list[dict] = []
    first_valid_image = None
    first_valid_regions = None
    first_valid_skin_features = None
    skipped = {"decode_failed": 0, "no_face": 0, "weak_sample": 0, "quality_excluded": 0}

    for upload in uploads:
        if show_per_image:
            st.divider()
            st.subheader(upload.name)

        result = analyze_image(upload.read(), wb_method)
        status = result["status"]

        if status == "decode_failed":
            if show_per_image:
                st.warning("Could not decode this image. Please try a different JPG/PNG file.")
            skipped["decode_failed"] += 1
            continue

        image_rgb = result["image_rgb"]
        image_hash = str(result.get("image_hash", ""))
        if status == "no_face":
            if show_per_image:
                st.info("No face was detected in this image. Try a clearer, front-facing photo with good lighting.")
                st.image(image_rgb, caption="Original", use_container_width=True)
            skipped["no_face"] += 1
            continue

        regions = result["regions"]
        if status == "weak_sample":
            if show_per_image:
                st.info(
                    "The cheek sample region was too small or too noisy. "
                    "Please upload a higher-resolution, well-lit image."
                )
                st.image(image_rgb, caption="Original", use_container_width=True)
            skipped["weak_sample"] += 1
            continue

        features = result["features"]
        skin_chroma_var = float(result.get("skin_chroma_var", 0.0))
        quality = result.get("quality", {})
        quality_score = float(quality.get("score", 0.0))
        quality_reasons = quality.get("reasons", [])
        if quality_handling == "exclude" and quality_score < quality_threshold:
            if show_per_image:
                reason_text = ", ".join(quality_reasons) if quality_reasons else "below_quality_threshold"
                st.warning(f"Image excluded: reason = {reason_text} (score={quality_score:.2f})")
                st.image(image_rgb, caption="Excluded image", use_container_width=True)
            skipped["quality_excluded"] += 1
            continue

        valid_features.append(features)
        valid_skin_chroma_vars.append(skin_chroma_var)
        if first_valid_image is None:
            first_valid_image = image_rgb
            first_valid_regions = regions
            first_valid_skin_features = features

        region_diag = compute_region_diagnostics(image_rgb, regions.landmarks_px, features)
        hair_features = region_diag.hair_features
        iris_features = region_diag.iris_features
        skin_mask = ((regions.left_mask > 0) | (regions.right_mask > 0)).astype(np.uint8)
        definition_score = compute_definition_score(image_rgb, region_diag.iris_mask, skin_mask)
        if hair_features is not None:
            valid_hair_features.append(hair_features)
        if iris_features is not None:
            valid_iris_features.append(iris_features)

        delta_l_hair_skin = abs(hair_features.l - features.l) if hair_features is not None else None
        delta_l_iris_skin = abs(iris_features.l - features.l) if iris_features is not None else None
        diagnostics_records.append(
            {
                "image": upload.name,
                "wb_method": result.get("wb_method", "none"),
                "skin": features,
                "hair": hair_features,
                "iris": iris_features,
                "delta_hair_skin": delta_l_hair_skin,
                "delta_iris_skin": delta_l_iris_skin,
            }
        )

        decision = classify_season(
            skin_features=features,
            hair_features=hair_features,
            iris_features=iris_features,
            delta_l_hair_skin=delta_l_hair_skin,
            delta_l_iris_skin=delta_l_iris_skin,
            skin_chroma_variance=skin_chroma_var,
            definition_score=definition_score,
        )
        drape_eval = evaluate_drape_scores(
            image_rgb=image_rgb,
            regions=regions,
            baseline_skin=features,
            baseline_scores=decision.season_scores,
            palettes=palettes,
            baseline_weight=0.6,
            drape_weight=0.4,
        )
        z_base_vec = np.array([decision.season_scores[s] for s in SEASONS], dtype=np.float64)
        z_drape_vec = np.array([drape_eval.drape_scores[s] for s in SEASONS], dtype=np.float64)
        stress_delta = cool_stress_delta(drape_eval.drape_metrics)
        stress_nudge = summer_winter_nudge(stress_delta, scale=10.0)
        z_base_vec[SEASON_TO_IDX["Winter"]] += stress_nudge
        z_base_vec[SEASON_TO_IDX["Summer"]] -= stress_nudge

        alpha = float(calibration_params.get("alpha", 0.5))
        bias = calibration_params.get("bias", [0.0, 0.0, 0.0, 0.0])
        gamma = float(calibration_params.get("gamma", 3.0))
        z_cal_vec = apply_calibration(z_base_vec, z_drape_vec, alpha=alpha, bias=bias)
        conf, top1_idx, top2_idx, margin = margin_confidence(z_cal_vec, gamma=gamma)
        topk = predict_topk(z_cal_vec, k=2)
        final_season = IDX_TO_SEASON[top1_idx]
        second_season = IDX_TO_SEASON[top2_idx]
        z_base_adj = {s: float(z_base_vec[i]) for i, s in enumerate(SEASONS)}
        z_drape_map = {s: float(z_drape_vec[i]) for i, s in enumerate(SEASONS)}
        z_cal_map = {s: float(z_cal_vec[i]) for i, s in enumerate(SEASONS)}

        variant_decision = choose_variant(
            base_season=final_season,
            temp_score=decision.temp_score,
            chroma_score=decision.chroma_score,
            contrast_score=decision.contrast_score,
        )
        season_display = f"{variant_decision.base_season} → {variant_decision.variant_key}"
        palette_season = _palette_season_from_variant(
            variant_decision.base_season,
            variant_decision.variant_key,
            variant_decision.palette_code,
        )
        per_image_final_scores.append(z_cal_map)
        per_image_baseline_scores.append(z_base_adj)
        per_image_drape_scores.append(z_drape_map)
        per_image_confidences.append(float(conf))
        valid_definition_scores.append(float(definition_score))
        per_image_axes.append(
            {
                "temperature": float(decision.temp_score),
                "value": float(decision.value_score),
                "chroma": float(decision.chroma_score),
                "contrast": float(decision.contrast_score),
            }
        )
        if quality_handling == "down-weight":
            per_image_quality_weights.append(float(features.sample_count) * max(0.05, quality_score))
        else:
            per_image_quality_weights.append(float(features.sample_count))
        axes_payload = (
            float(decision.temp_score),
            float(decision.value_score),
            float(decision.chroma_score),
            float(decision.contrast_score),
        )
        dynamic_mode = "advanced" if use_advanced_dynamic else "simple"
        _dyn_t0 = time.perf_counter()
        dynamic_suggestions = compute_dynamic_color_suggestions(
            image_hash=image_hash,
            axes_payload=axes_payload,
            mode=dynamic_mode,
            cfg_hash=dynamic_color_cfg_hash,
            return_set=True,
            cfg_json=dynamic_color_config_json,
            season_hint=final_season,
            image_rgb=image_rgb if dynamic_mode == "advanced" else None,
            regions=regions if dynamic_mode == "advanced" else None,
            baseline_skin=features if dynamic_mode == "advanced" else None,
            diagnostics=show_diagnostics,
        )
        dynamic_elapsed_s = time.perf_counter() - _dyn_t0
        analysis_payload = {
            "axes": {
                "temp": float(decision.temp_score),
                "value": float(decision.value_score),
                "chroma": float(decision.chroma_score),
                "contrast": float(decision.contrast_score),
            },
            "season_logits": {
                "baseline": z_base_adj,
                "drape": z_drape_map,
                "calibrated": z_cal_map,
            },
            "drape_penalties": drape_eval.drape_metrics,
            "dynamic_color_suggestions": dynamic_suggestions,
        }
        per_image_payloads.append(analysis_payload)
        gallery_items.append((upload.name, image_rgb, bool(result.get("wb_applied", False))))
        season_colors = palette_for_season(palettes, palette_season)
        draped = _compose_personalised_drape_preview(image_rgb, dynamic_suggestions, strip_height=92)
        text_suggestions = render_text_suggestions(
            _build_copy_payload(
                image_id=image_hash,
                season_display=season_display,
                top1_season=final_season,
                top2_season=second_season,
                top1_variant=variant_decision.variant_key,
                confidence=float(conf),
                axes=analysis_payload["axes"],
                dynamic_suggestions=dynamic_suggestions,
            ),
            COPY_RULES_PATH,
        )

        if show_per_image:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(draped, caption="Personalised drape strip", use_container_width=True)
                if show_debug:
                    dbg = render_debug_overlay(image_rgb, regions)
                    dbg = render_diagnostics_overlay(dbg, region_diag.hair_mask, region_diag.iris_mask)
                    st.image(dbg, caption="Debug overlay", use_container_width=True)
                    st.caption("Overlay legend: cheek masks + hair sample (green) + iris sample (yellow).")
                if show_drape_previews:
                    winner = drape_eval.best_season
                    st.write(f"**Top drape previews ({winner})**")
                    top_colors = [c.color_hex for c in drape_eval.season_details[winner].colors[:4]]
                    if top_colors:
                        cols = st.columns(len(top_colors))
                        for col, color_hex in zip(cols, top_colors):
                            with col:
                                preview = apply_drape_color(image_rgb, regions.landmarks_px, color_hex)
                                st.image(preview, caption=color_hex, use_container_width=True)

            with c2:
                st.markdown("<div class='app-card'>", unsafe_allow_html=True)
                st.subheader(season_display)
                st.caption(f"Confidence: {conf:.0%}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(f"Top season: {final_season}")

                st.pyplot(plot_season_scores([z_cal_map[s] for s in SEASONS], list(SEASONS)), use_container_width=True)

                y_raw = {
                    "value": decision.value_score,
                    "chroma": decision.chroma_score,
                    "contrast": decision.contrast_score,
                }[season_map_y_axis]
                x_map = _normalize_axis(decision.temp_score, "temperature")
                y_map = _normalize_axis(y_raw, season_map_y_axis)
                y_label_key = "value" if season_map_y_axis == "value" else season_map_y_axis
                st.write("**Season Map**")
                st.pyplot(
                    plot_season_map(
                        x_map,
                        y_map,
                        AXIS_LABELS["temperature"],
                        AXIS_LABELS[y_label_key],
                        point_color=_season_accent_color(final_season),
                    ),
                    use_container_width=True,
                )
                st.caption(
                    f"This coordinate sits closer to {('Cool' if x_map < 0 else 'Warm')} + "
                    f"{('Deep' if y_map > 0 else 'Light/Soft')} region; map is directional, final season comes from logits."
                )

                m1, m2 = st.columns(2)
                with m1:
                    st.metric(AXIS_LABELS["temperature"], f"{decision.temp_score:.2f}", help=AXIS_HELP["temperature"])
                    st.metric(AXIS_LABELS["chroma"], f"{decision.chroma_score:.2f}", help=AXIS_HELP["chroma"])
                with m2:
                    st.metric(AXIS_LABELS["value"], f"{decision.value_score:.2f}", help=AXIS_HELP["value"])
                    st.metric(AXIS_LABELS["contrast"], f"{decision.contrast_score:.2f}", help=AXIS_HELP["contrast"])

                st.write("**Text Suggestions**")
                _render_text_suggestions_section(text_suggestions)
                _render_dynamic_best_colours(
                    title="Your Dynamic Best Colours",
                    suggestions=dynamic_suggestions,
                    axes_used=analysis_payload["axes"],
                    show_diag=show_diagnostics,
                )
                if dynamic_mode == "advanced" and dynamic_elapsed_s > 0.5:
                    st.caption(f"Advanced optimisation time: {dynamic_elapsed_s:.2f}s")
                scorecard_pdf = _build_scorecard_pdf_bytes(
                    subject_name=upload.name,
                    season_display=season_display,
                    confidence=float(conf),
                    axes=analysis_payload["axes"],
                    palette_hex=season_colors,
                    dynamic_suggestions=dynamic_suggestions,
                    text_suggestions=text_suggestions,
                    input_rgb=image_rgb,
                )
                st.download_button(
                    "Download scorecard (PDF)",
                    data=scorecard_pdf,
                    file_name=f"scorecard_{Path(upload.name).stem}.pdf",
                    mime="application/pdf",
                    key=f"scorecard_{image_hash[:12]}",
                )

                if show_diagnostics:
                    with st.expander("Diagnostics", expanded=False):
                        st.write(f"- White balance: **{result.get('wb_method', 'none')}**")
                        st.write(f"- WB applied: **{'yes' if result.get('wb_applied', False) else 'no'}**")
                        st.write(f"- calibration source: `{calibration_params.get('source', 'default')}`")
                        st.write(f"- z_base: `{[round(v, 4) for v in z_base_vec.tolist()]}`")
                        st.write(f"- z_drape: `{[round(v, 4) for v in z_drape_vec.tolist()]}`")
                        st.write(f"- z_cal: `{[round(v, 4) for v in z_cal_vec.tolist()]}`")
                        st.write(f"- top2 idx/logit: `{[(i, round(v, 4)) for i, v in topk]}`")
                        st.write(f"- calibrated margin(top1-top2): `{margin:.4f}`")
                        st.write(f"- calibrated conf: `{conf:.4f}`")
                        st.write(f"- calibration alpha: `{alpha:.3f}`")
                        st.write(f"- calibration gamma: `{gamma:.3f}`")
                        st.write(f"- calibration bias: `{[round(float(x), 4) for x in bias]}`")
                        st.write(f"- quality_score: `{quality_score:.3f}`")
                        st.write(f"- definition_score: `{definition_score:.3f}`")
                        st.write(f"- cool_stress_delta: `{stress_delta:.3f}`")
                        st.write(f"- summer_winter_nudge: `{stress_nudge:.3f}`")
                        st.write(f"- skin_chroma_variance: `{skin_chroma_var:.3f}`")
                        st.write(_fmt_feature("Skin", features))
                        st.write(_fmt_feature("Hair", hair_features))
                        st.write(_fmt_feature("Iris", iris_features))
                        st.write("**Season palette (hex)**")
                        st.code(", ".join(season_colors) if season_colors else "No palette colors found.")

    if gallery_items:
        st.subheader("Input Photos")
        cols = st.columns(min(3, len(gallery_items)))
        for i, item in enumerate(gallery_items):
            name, img, wb_applied = item
            with cols[i % len(cols)]:
                st.image(img, caption=f"{name} {'(white-balanced)' if wb_applied else '(original)'}", use_container_width=True)

    if valid_features:
        if show_diagnostics:
            st.divider()
            st.subheader("Diagnostics Summary Across Uploaded Images")
            for record in diagnostics_records:
                skin_feat = record["skin"]
                st.write(
                    f"- `{record['image']}` | wb={record['wb_method']} | "
                    f"skin b*={skin_feat.b:.2f}, skin chroma={skin_feat.chroma:.2f}, "
                    f"temp_band={_temp_band_from_b(float(skin_feat.b))}"
                )

            skin_b_vals = [r["skin"].b for r in diagnostics_records]
            skin_chroma_vals = [r["skin"].chroma for r in diagnostics_records]
            b_arr = np.array(skin_b_vals, dtype=np.float32)
            c_arr = np.array(skin_chroma_vals, dtype=np.float32)
            cool_count = int(np.count_nonzero(b_arr < 10.0))
            neutral_count = int(np.count_nonzero((b_arr >= 10.0) & (b_arr <= 18.0)))
            warm_count = int(np.count_nonzero(b_arr > 18.0))
            bright_count = int(np.count_nonzero(c_arr >= 18.0))
            hair_count = int(np.count_nonzero([r["hair"] is not None for r in diagnostics_records]))
            iris_count = int(np.count_nonzero([r["iris"] is not None for r in diagnostics_records]))

            st.write("**Skin b* / chroma distribution summary**")
            st.write(
                f"- b*: min={float(np.min(b_arr)):.2f}, median={float(np.median(b_arr)):.2f}, "
                f"max={float(np.max(b_arr)):.2f}, mean={float(np.mean(b_arr)):.2f}"
            )
            st.write(
                f"- chroma: min={float(np.min(c_arr)):.2f}, median={float(np.median(c_arr)):.2f}, "
                f"max={float(np.max(c_arr)):.2f}, mean={float(np.mean(c_arr)):.2f}"
            )
            st.write(
                f"- Temperature bands by b*: cool<{10} => **{cool_count}/{len(skin_b_vals)}**, "
                f"neutral 10-18 => **{neutral_count}/{len(skin_b_vals)}**, "
                f"warm>18 => **{warm_count}/{len(skin_b_vals)}**"
            )
            st.write(f"- Bright threshold pass count (chroma >= 18.0): **{bright_count}/{len(skin_chroma_vals)}**")
            st.write(f"- Hair sample availability: **{hair_count}/{len(diagnostics_records)}**")
            st.write(f"- Iris sample availability: **{iris_count}/{len(diagnostics_records)}**")
            if per_image_confidences:
                conf_arr = np.array(per_image_confidences, dtype=np.float32)
                st.write(
                    f"- Per-image confidence: min={float(np.min(conf_arr)):.3f}, "
                    f"median={float(np.median(conf_arr)):.3f}, max={float(np.max(conf_arr)):.3f}"
                )
            st.write(f"- Low-quality handling mode: **{quality_handling}** (threshold={quality_threshold:.2f})")
            _hist_chart(skin_b_vals, "Distribution of skin b*", "skin b*", threshold=3.0)
            _hist_chart(skin_chroma_vals, "Distribution of skin chroma", "skin chroma", threshold=18.0)

    if len(valid_features) > 1:
        composite_features = aggregate_features(valid_features)
        composite_skin_chroma_var = float(np.mean(np.array(valid_skin_chroma_vars, dtype=np.float32)))
        composite_definition_score = (
            float(np.median(np.array(valid_definition_scores, dtype=np.float32))) if valid_definition_scores else 0.0
        )
        composite_hair = aggregate_features(valid_hair_features) if valid_hair_features else None
        composite_iris = aggregate_features(valid_iris_features) if valid_iris_features else None
        composite_delta_l_hair_skin = (
            abs(composite_hair.l - composite_features.l) if composite_hair is not None else None
        )
        composite_delta_l_iris_skin = (
            abs(composite_iris.l - composite_features.l) if composite_iris is not None else None
        )
        aggregated_final_scores = _aggregate_season_scores(
            per_image_final_scores,
            per_image_weights=per_image_quality_weights,
            use_weights=use_quality_weighted_aggregation,
        )
        aggregated_baseline_scores = _aggregate_season_scores(
            per_image_baseline_scores,
            per_image_weights=per_image_quality_weights,
            use_weights=use_quality_weighted_aggregation,
        )
        aggregated_drape_scores = _aggregate_season_scores(
            per_image_drape_scores,
            per_image_weights=per_image_quality_weights,
            use_weights=use_quality_weighted_aggregation,
        )
        z_comp = np.array([aggregated_final_scores[s] for s in SEASONS], dtype=np.float64)
        gamma = float(calibration_params.get("gamma", 3.0))
        composite_confidence, comp_top1_idx, comp_top2_idx, comp_margin = margin_confidence(z_comp, gamma=gamma)
        composite_season = IDX_TO_SEASON[comp_top1_idx]
        composite_second = IDX_TO_SEASON[comp_top2_idx]
        composite_ranked = [(IDX_TO_SEASON[i], float(z_comp[i])) for i in np.argsort(z_comp)[::-1]]
        composite_axis_decision = classify_season(
            skin_features=composite_features,
            hair_features=composite_hair,
            iris_features=composite_iris,
            delta_l_hair_skin=composite_delta_l_hair_skin,
            delta_l_iris_skin=composite_delta_l_iris_skin,
            skin_chroma_variance=composite_skin_chroma_var,
            definition_score=composite_definition_score,
        )
        composite_variant = choose_variant(
            base_season=composite_season,
            temp_score=composite_axis_decision.temp_score,
            chroma_score=composite_axis_decision.chroma_score,
            contrast_score=composite_axis_decision.contrast_score,
        )
        composite_season_display = f"{composite_variant.base_season} → {composite_variant.variant_key}"
        composite_palette_season = _palette_season_from_variant(
            composite_variant.base_season,
            composite_variant.variant_key,
            composite_variant.palette_code,
        )
        composite_colors = palette_for_season(palettes, composite_palette_season)
        composite_dynamic_mode = "advanced" if use_advanced_dynamic else "simple"
        composite_axes = {
            "temp": float(composite_axis_decision.temp_score),
            "value": float(composite_axis_decision.value_score),
            "chroma": float(composite_axis_decision.chroma_score),
            "contrast": float(composite_axis_decision.contrast_score),
        }
        _comp_dyn_t0 = time.perf_counter()
        composite_dynamic_suggestions = compute_dynamic_color_suggestions(
            image_hash="composite::" + "|".join(sorted(name for name, _, _ in gallery_items)),
            axes_payload=(
                composite_axes["temp"],
                composite_axes["value"],
                composite_axes["chroma"],
                composite_axes["contrast"],
            ),
            mode=composite_dynamic_mode,
            cfg_hash=dynamic_color_cfg_hash,
            return_set=True,
            cfg_json=dynamic_color_config_json,
            season_hint=composite_season,
            image_rgb=first_valid_image if composite_dynamic_mode == "advanced" else None,
            regions=first_valid_regions if composite_dynamic_mode == "advanced" else None,
            baseline_skin=first_valid_skin_features if composite_dynamic_mode == "advanced" else None,
            diagnostics=show_diagnostics,
        )
        composite_dynamic_elapsed_s = time.perf_counter() - _comp_dyn_t0

        with summary_slot.container():
            st.subheader("Composite across images")
            st.write(
                f"Based on {len(valid_features)} images with valid cheek samples. "
                "Each image is weighted by the number of usable pixels."
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                if first_valid_image is not None:
                    st.image(
                        _compose_personalised_drape_preview(first_valid_image, composite_dynamic_suggestions, strip_height=92),
                        caption="Personalised drape strip (composite)",
                        use_container_width=True,
                    )
            with c2:
                st.metric("Composite season", composite_season_display)
                st.metric("Composite confidence", f"{composite_confidence:.0%}")
                st.caption("Composite confidence = sigmoid(gamma * calibrated top1-top2 margin).")
                st.pyplot(plot_season_scores([aggregated_final_scores[s] for s in SEASONS], list(SEASONS)), use_container_width=True)

                comp_y_raw = {
                    "value": composite_axis_decision.value_score,
                    "chroma": composite_axis_decision.chroma_score,
                    "contrast": composite_axis_decision.contrast_score,
                }[season_map_y_axis]
                comp_x_map = _normalize_axis(composite_axis_decision.temp_score, "temperature")
                comp_y_map = _normalize_axis(comp_y_raw, season_map_y_axis)
                y_key = "value" if season_map_y_axis == "value" else season_map_y_axis
                x_unc = None
                y_unc = None
                if len(per_image_axes) > 1:
                    x_vals = [_normalize_axis(a["temperature"], "temperature") for a in per_image_axes]
                    y_vals = [_normalize_axis(a[season_map_y_axis], season_map_y_axis) for a in per_image_axes]
                    x_unc = float(np.std(np.array(x_vals, dtype=np.float32)))
                    y_unc = float(np.std(np.array(y_vals, dtype=np.float32)))
                st.write("**Season Map**")
                st.pyplot(
                    plot_season_map(
                        comp_x_map,
                        comp_y_map,
                        AXIS_LABELS["temperature"],
                        AXIS_LABELS[y_key],
                        x_unc=x_unc,
                        y_unc=y_unc,
                        point_color=_season_accent_color(composite_season),
                    ),
                    use_container_width=True,
                )
                st.write(f"**Base season**: {composite_variant.base_season}")
                st.write(f"**Variant**: {composite_variant.variant_key}")
                st.write(f"**Palette hex list**: {', '.join(composite_colors) if composite_colors else 'n/a'}")
                composite_text_suggestions = render_text_suggestions(
                    _build_copy_payload(
                        image_id="composite",
                        season_display=composite_season_display,
                        top1_season=composite_season,
                        top2_season=composite_second,
                        top1_variant=composite_variant.variant_key,
                        confidence=float(composite_confidence),
                        axes=composite_axes,
                        dynamic_suggestions=composite_dynamic_suggestions,
                    ),
                    COPY_RULES_PATH,
                )
                st.write("**Text Suggestions**")
                _render_text_suggestions_section(composite_text_suggestions)

                am1, am2 = st.columns(2)
                with am1:
                    st.metric(AXIS_LABELS["temperature"], f"{composite_axis_decision.temp_score:.2f}", help=AXIS_HELP["temperature"])
                    st.metric(AXIS_LABELS["chroma"], f"{composite_axis_decision.chroma_score:.2f}", help=AXIS_HELP["chroma"])
                with am2:
                    st.metric(AXIS_LABELS["value"], f"{composite_axis_decision.value_score:.2f}", help=AXIS_HELP["value"])
                    st.metric(AXIS_LABELS["contrast"], f"{composite_axis_decision.contrast_score:.2f}", help=AXIS_HELP["contrast"])

                if show_diagnostics:
                    with st.expander("Diagnostics", expanded=False):
                        st.write(f"- White balance: **{wb_method}**")
                        st.write(f"- WB applied: **{'yes' if wb_method == 'grayworld' else 'no'}**")
                        st.write(f"- calibration source: `{calibration_params.get('source', 'default')}`")
                        st.write(f"- z_base (aggregated): `{[round(float(aggregated_baseline_scores[s]), 4) for s in SEASONS]}`")
                        st.write(f"- z_drape (aggregated): `{[round(float(aggregated_drape_scores[s]), 4) for s in SEASONS]}`")
                        st.write(f"- z_cal (aggregated): `{[round(float(aggregated_final_scores[s]), 4) for s in SEASONS]}`")
                        st.write(f"- composite calibrated margin(top1-top2): `{comp_margin:.4f}`")
                        st.write(f"- composite calibrated conf: `{composite_confidence:.4f}`")
                        st.write(_fmt_feature("Skin", composite_features))
                        st.write(_fmt_feature("Hair", composite_hair))
                        st.write(_fmt_feature("Iris", composite_iris))
                _render_dynamic_best_colours(
                    title="Your Dynamic Best Colours",
                    suggestions=composite_dynamic_suggestions,
                    axes_used=composite_axes,
                    show_diag=show_diagnostics,
                )
                if composite_dynamic_mode == "advanced" and composite_dynamic_elapsed_s > 0.5:
                    st.caption(f"Advanced optimisation time: {composite_dynamic_elapsed_s:.2f}s")
                composite_scorecard_pdf = _build_scorecard_pdf_bytes(
                    subject_name="Composite across images",
                    season_display=composite_season_display,
                    confidence=float(composite_confidence),
                    axes=composite_axes,
                    palette_hex=composite_colors,
                    dynamic_suggestions=composite_dynamic_suggestions,
                    text_suggestions=composite_text_suggestions,
                    input_rgb=first_valid_image,
                )
                st.download_button(
                    "Download composite scorecard (PDF)",
                    data=composite_scorecard_pdf,
                    file_name="scorecard_composite.pdf",
                    mime="application/pdf",
                    key="scorecard_composite",
                )
                st.write("**Composite palette (hex)**")
                st.code(", ".join(composite_colors) if composite_colors else "No palette colors found.")
                composite_info = palette_meta.get(composite_season, {})
                description = composite_info.get("description")
                if description:
                    st.write("**Composite overview**")
                    st.markdown(description)

            if any(skipped.values()):
                st.caption(
                    "Skipped images: "
                    f"{skipped['decode_failed']} decode failures, "
                    f"{skipped['no_face']} no-face, "
                    f"{skipped['weak_sample']} weak samples, "
                    f"{skipped['quality_excluded']} quality-excluded."
                )
else:
    st.caption("Upload at least one image to start analysis.")
