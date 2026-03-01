from __future__ import annotations

import json
from pathlib import Path

import cv2
import streamlit as st

import numpy as np

from src.calibration import apply_calibration, margin_confidence, predict_topk
from src.color_features import ColorFeatures, compute_robust_lab_features, compute_skin_chroma_variance
from src.drape_scoring import apply_drape_color, evaluate_drape_scores
from src.diagnostics_regions import compute_definition_score, compute_region_diagnostics, render_diagnostics_overlay
from src.drape import render_drape_strip
from src.face_regions import FaceMeshDetector, build_region_masks, render_debug_overlay
from src.image_io import decode_uploaded_image
from src.image_quality import evaluate_image_quality
from src.palettes import (
    load_palette_metadata,
    load_palettes,
    load_recommendations,
    load_season_descriptions,
    palette_for_season,
)
from src.season_index import IDX_TO_SEASON, SEASON_TO_IDX, SEASONS
from src.season_rules import classify_season
from src.stress_features import cool_stress_delta, summer_winter_nudge
from src.variant_rules import choose_variant
from ui.components import plot_season_map, plot_season_scores
from ui.copy import AXIS_HELP, AXIS_LABELS, LOW_CONF_TIEBREAKER_BULLETS
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
    image_rgb = decode_uploaded_image(file_bytes)
    if image_rgb is None:
        return {"status": "decode_failed"}
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
            "regions": regions,
            "quality": quality,
            "wb_method": wb_method_used,
            "wb_applied": wb_applied,
        }

    return {
        "status": "ok",
        "image_rgb": image_rgb,
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

uploads = st.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

palettes = load_palettes(Path("assets/palettes/seasonal_palettes.json"))
palette_meta = load_palette_metadata(Path("assets/palettes/seasonal_palettes.json"))
recommendations = load_recommendations(Path("assets/palettes/seasonal_recommendations.json"))
season_descriptions = load_season_descriptions(Path("assets/palettes/season_descriptions.json"))
calibration_params = load_calibration_params("calibration_params.json")

season_to_reco_key = {
    "Autumn": "Warm Autumn (A)",
    "Summer": "Cool Summer (B)",
    "Winter": "Cool Winter (C)",
    "Spring": "Warm Spring (D)",
}
code_to_reco_key = {
    "A": "Warm Autumn (A)",
    "B": "Cool Summer (B)",
    "C": "Cool Winter (C)",
    "D": "Warm Spring (D)",
}


def _recommendation_key_for_variant(variant_key: str, palette_code: str, base_season: str) -> str | None:
    if variant_key in recommendations:
        return variant_key
    if palette_code in code_to_reco_key:
        return code_to_reco_key[palette_code]
    return season_to_reco_key.get(base_season)


def _render_recommendation_block(block: dict, as_note: bool = False) -> None:
    title = block.get("title")
    description = block.get("description")
    if title:
        st.markdown(f"**{title}**" if as_note else title)
    if description:
        st.markdown(description)

    def render_list(label: str, items: list | None) -> None:
        if not items:
            return
        bullets = "\n".join(f"- {item}" for item in items)
        st.markdown(f"**{label}**\n{bullets}")

    render_list("Best colors", block.get("best_colors"))
    render_list("Best neutrals", block.get("best_neutrals"))
    render_list("Metals", block.get("metals"))
    render_list("Avoid", block.get("avoid"))

    overall = block.get("overall_effect")
    if overall:
        st.markdown(f"**Overall effect**: {overall}")

    style_note = block.get("style_note")
    if style_note:
        st.markdown(f"**Style note**: {style_note}")

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
    gallery_items: list[tuple[str, np.ndarray, bool]] = []
    diagnostics_records: list[dict] = []
    first_valid_image = None
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
        variant_description = season_descriptions.get(variant_decision.variant_key, {})
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
        gallery_items.append((upload.name, image_rgb, bool(result.get("wb_applied", False))))
        season_colors = palette_for_season(palettes, palette_season)
        draped = render_drape_strip(image_rgb, season_colors)

        if show_per_image:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(draped, caption=f"Digital drape ({final_season})", use_container_width=True)
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

                top1, top2 = final_season, second_season
                if conf < 0.65:
                    st.warning(f"Close call between {top1} and {top2}.")
                    top1_variant = choose_variant(top1, decision.temp_score, decision.chroma_score, decision.contrast_score)
                    top2_variant = choose_variant(top2, decision.temp_score, decision.chroma_score, decision.contrast_score)
                    top1_palette = palette_for_season(
                        palettes,
                        _palette_season_from_variant(top1_variant.base_season, top1_variant.variant_key, top1_variant.palette_code),
                    )
                    top2_palette = palette_for_season(
                        palettes,
                        _palette_season_from_variant(top2_variant.base_season, top2_variant.variant_key, top2_variant.palette_code),
                    )
                    pa, pb = st.columns(2)
                    with pa:
                        st.caption(f"{top1_variant.base_season} → {top1_variant.variant_key}")
                        st.code(", ".join(top1_palette) if top1_palette else "No palette colors found.")
                    with pb:
                        st.caption(f"{top2_variant.base_season} → {top2_variant.variant_key}")
                        st.code(", ".join(top2_palette) if top2_palette else "No palette colors found.")
                    st.caption("Tie-breaker checks")
                    for bullet in LOW_CONF_TIEBREAKER_BULLETS:
                        st.write(f"- {bullet}")
                else:
                    st.caption(f"Top season: {top1}")

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

                if variant_description:
                    st.write("**Style Guidance**")
                    _render_recommendation_block(variant_description)

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
                        render_drape_strip(first_valid_image, composite_colors)
                        if composite_colors
                        else first_valid_image,
                        caption=f"Composite digital drape ({composite_season}, using first valid photo)",
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
                composite_variant_description = season_descriptions.get(composite_variant.variant_key, {})
                if composite_variant_description:
                    st.write("**Variant description**")
                    _render_recommendation_block(composite_variant_description)
                if composite_confidence < 0.65:
                    comp_top1 = composite_season
                    comp_top2 = composite_second
                    comp_top1_variant = choose_variant(
                        comp_top1,
                        composite_axis_decision.temp_score,
                        composite_axis_decision.chroma_score,
                        composite_axis_decision.contrast_score,
                    )
                    comp_top2_variant = choose_variant(
                        comp_top2,
                        composite_axis_decision.temp_score,
                        composite_axis_decision.chroma_score,
                        composite_axis_decision.contrast_score,
                    )
                    comp_top1_palette = palette_for_season(
                        palettes,
                        _palette_season_from_variant(
                            comp_top1_variant.base_season,
                            comp_top1_variant.variant_key,
                            comp_top1_variant.palette_code,
                        ),
                    )
                    comp_top2_palette = palette_for_season(
                        palettes,
                        _palette_season_from_variant(
                            comp_top2_variant.base_season,
                            comp_top2_variant.variant_key,
                            comp_top2_variant.palette_code,
                        ),
                    )
                    st.warning(f"Close call between {comp_top1} and {comp_top2}.")
                    st.write(f"- {comp_top1_variant.base_season} → {comp_top1_variant.variant_key}")
                    st.code(", ".join(comp_top1_palette) if comp_top1_palette else "No palette colors found.")
                    st.write(f"- {comp_top2_variant.base_season} → {comp_top2_variant.variant_key}")
                    st.code(", ".join(comp_top2_palette) if comp_top2_palette else "No palette colors found.")
                    for bullet in LOW_CONF_TIEBREAKER_BULLETS:
                        st.write(f"- {bullet}")

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
                st.write("**Composite palette (hex)**")
                st.code(", ".join(composite_colors) if composite_colors else "No palette colors found.")
                composite_info = palette_meta.get(composite_season, {})
                description = composite_info.get("description")
                if description:
                    st.write("**Composite overview**")
                    st.markdown(description)

                rec_key = _recommendation_key_for_variant(
                    composite_variant.variant_key,
                    composite_variant.palette_code,
                    composite_variant.base_season,
                )
                rec_payload = recommendations.get(rec_key, {}) if rec_key else {}
                primary = rec_payload.get("primary")
                strong = rec_payload.get("strong")
                borderline = rec_payload.get("borderline")
                if primary or strong or borderline:
                    st.write("**Composite recommendations**")
                    if primary:
                        _render_recommendation_block(primary)
                    if strong:
                        st.write("**Notes: strong**")
                        _render_recommendation_block(strong, as_note=True)
                    if borderline:
                        st.write("**Notes: borderline**")
                        _render_recommendation_block(borderline, as_note=True)

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
