from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import streamlit as st

from src.aggregation import (
    aggregate_features as _aggregate_features_impl,
)
from src.aggregation import (
    aggregate_season_scores as _aggregate_season_scores_impl,
)
from src.aggregation import (
    format_feature as _format_feature_impl,
)
from src.aggregation import (
    normalize_axis as _normalize_axis,
)
from src.aggregation import (
    palette_season_from_variant as _palette_season_from_variant,
)
from src.aggregation import (
    season_accent_color as _season_accent_color,
)
from src.aggregation import (
    temp_band_from_b as _temp_band_from_b,
)
from src.aggregation import (
    weighted_median as _weighted_median_impl,
)
from src.analysis import analyze_image_bytes
from src.calibration import apply_calibration, margin_confidence, predict_topk
from src.color_features import ColorFeatures
from src.config import load_calibration_config
from src.config import load_dynamic_color_config as _load_dynamic_color_config_impl
from src.copy_renderer import render_text_suggestions
from src.diagnostics_regions import compute_definition_score, compute_region_diagnostics, render_diagnostics_overlay
from src.drape import render_color_strip
from src.drape_scoring import apply_drape_color, evaluate_drape_scores
from src.dynamic_colors import config_hash, suggest_dynamic_colors
from src.face_regions import FaceMeshDetector, render_debug_overlay
from src.palettes import (
    load_palette_metadata,
    load_palettes,
    palette_for_season,
)
from src.paths import CALIBRATION_PARAMS_PATH, DYNAMIC_COLORS_CONFIG_PATH, PALETTES_PATH
from src.paths import COPY_RULES_PATH as APP_COPY_RULES_PATH
from src.reporting import build_scorecard_pdf_bytes as _build_scorecard_pdf_bytes_impl
from src.season_index import IDX_TO_SEASON, SEASON_TO_IDX, SEASONS
from src.season_rules import classify_season
from src.stress_features import cool_stress_delta, summer_winter_nudge
from src.variant_rules import choose_variant
from ui.charts import plot_season_map, plot_season_scores, render_hist_chart
from ui.copy import AXIS_HELP, AXIS_LABELS
from ui.sidebar import render_sidebar_controls
from ui.styles import apply_base_styles

logger = logging.getLogger(__name__)

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

_sidebar = render_sidebar_controls()
show_debug = _sidebar.show_debug
show_per_image = _sidebar.show_per_image
show_diagnostics = _sidebar.show_diagnostics
show_drape_previews = _sidebar.show_drape_previews
use_advanced_dynamic = _sidebar.use_advanced_dynamic
season_map_y_axis = _sidebar.season_map_y_axis
use_quality_weighted_aggregation = _sidebar.use_quality_weighted_aggregation
quality_handling = _sidebar.quality_handling
quality_threshold = _sidebar.quality_threshold
wb_method = _sidebar.wb_method


@st.cache_resource
def get_face_detector() -> FaceMeshDetector:
    return FaceMeshDetector(static_image_mode=True, max_num_faces=1)


@st.cache_data(show_spinner=False)
def analyze_image(file_bytes: bytes, wb_method_choice: str) -> dict:
    return analyze_image_bytes(file_bytes, wb_method_choice, get_face_detector()).as_payload()


def aggregate_features(feature_list: list[ColorFeatures]) -> ColorFeatures:
    return _aggregate_features_impl(feature_list)


def _fmt_feature(label: str, feat: ColorFeatures | None) -> str:
    return _format_feature_impl(label, feat)


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


def _hist_chart(values: list[float], title: str, x_label: str, threshold: float | None = None) -> None:
    render_hist_chart(values, title, x_label, threshold)


def _weighted_median(values: list[float], weights: list[float]) -> float:
    return _weighted_median_impl(values, weights)


def _aggregate_season_scores(
    per_image_scores: list[dict[str, float]],
    per_image_weights: list[float] | None = None,
    use_weights: bool = False,
) -> dict[str, float]:
    return _aggregate_season_scores_impl(per_image_scores, per_image_weights=per_image_weights, use_weights=use_weights)


@st.cache_data(show_spinner=False)
def load_calibration_params(path: str = "calibration_params.json") -> dict:
    cfg = load_calibration_config(path)
    return {"alpha": cfg.alpha, "bias": cfg.bias, "gamma": cfg.gamma, "source": cfg.source}


@st.cache_data(show_spinner=False)
def load_dynamic_color_config(path: str = "dynamic_colors_config.json") -> dict:
    return _load_dynamic_color_config_impl(path)


@st.cache_data(show_spinner=False)
def _compute_simple_dynamic_color_suggestions(
    image_hash: str,
    axes_payload: tuple[float, float, float, float],
    mode: str,
    cfg_hash: str,
    return_set: bool,
    cfg_json: str,
    season_hint: str | None,
    diagnostics: bool,
) -> dict:
    _ = image_hash, cfg_hash, season_hint
    config = json.loads(cfg_json)
    axes = {
        "temp": float(axes_payload[0]),
        "value": float(axes_payload[1]),
        "chroma": float(axes_payload[2]),
        "contrast": float(axes_payload[3]),
    }
    return suggest_dynamic_colors(
        face_context=None,
        axes=axes,
        mode=mode,
        config=config,
        diagnostics=diagnostics,
        return_set=return_set,
    )


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
    if mode != "advanced":
        return _compute_simple_dynamic_color_suggestions(
            image_hash=image_hash,
            axes_payload=axes_payload,
            mode=mode,
            cfg_hash=cfg_hash,
            return_set=return_set,
            cfg_json=cfg_json,
            season_hint=season_hint,
            diagnostics=diagnostics,
        )

    fallback = _compute_simple_dynamic_color_suggestions(
        image_hash=image_hash,
        axes_payload=axes_payload,
        mode="simple",
        cfg_hash=cfg_hash,
        return_set=return_set,
        cfg_json=cfg_json,
        season_hint=season_hint,
        diagnostics=diagnostics,
    )
    fallback.setdefault("diagnostics", {})
    fallback["diagnostics"]["advanced_fallback_reason"] = "Advanced colour draping temporarily disabled during white-screen investigation."
    fallback["mode"] = "simple"
    logger.warning("advanced dynamic colors bypassed image_hash=%s", image_hash[:12])
    return fallback

    config = json.loads(cfg_json)
    axes = {
        "temp": float(axes_payload[0]),
        "value": float(axes_payload[1]),
        "chroma": float(axes_payload[2]),
        "contrast": float(axes_payload[3]),
    }
    started_at = time.perf_counter()
    logger.warning(
        "advanced dynamic colors start image_hash=%s season_hint=%s shape=%s",
        image_hash[:12],
        season_hint,
        None if image_rgb is None else tuple(int(v) for v in image_rgb.shape),
    )
    try:
        out = suggest_dynamic_colors(
            face_context={
                "image_rgb": image_rgb,
                "regions": regions,
                "baseline_skin": baseline_skin,
                "season_hint": season_hint,
            },
            axes=axes,
            mode="advanced",
            config=config,
            diagnostics=diagnostics,
            return_set=return_set,
        )
        logger.warning(
            "advanced dynamic colors ok image_hash=%s elapsed_s=%.3f colors=%s",
            image_hash[:12],
            time.perf_counter() - started_at,
            len(out.get("colors", [])),
        )
        return out
    except Exception as exc:
        logger.exception("advanced dynamic colors failed image_hash=%s", image_hash[:12])
        fallback = suggest_dynamic_colors(
            face_context=None,
            axes=axes,
            mode="simple",
            config=config,
            diagnostics=diagnostics,
            return_set=return_set,
        )
        fallback.setdefault("diagnostics", {})
        fallback["diagnostics"]["advanced_fallback_reason"] = f"{type(exc).__name__}: {exc}"
        fallback["mode"] = "simple"
        logger.warning(
            "advanced dynamic colors fallback image_hash=%s elapsed_s=%.3f reason=%s",
            image_hash[:12],
            time.perf_counter() - started_at,
            type(exc).__name__,
        )
        return fallback


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


COPY_RULES_PATH = APP_COPY_RULES_PATH


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
    return _build_scorecard_pdf_bytes_impl(
        subject_name=subject_name,
        season_display=season_display,
        confidence=confidence,
        axes=axes,
        palette_hex=palette_hex,
        dynamic_suggestions=dynamic_suggestions,
        text_suggestions=text_suggestions,
        input_rgb=input_rgb,
    )

uploads = st.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

palettes = load_palettes(PALETTES_PATH)
palette_meta = load_palette_metadata(PALETTES_PATH)
calibration_params = load_calibration_params(CALIBRATION_PARAMS_PATH)
dynamic_color_config = load_dynamic_color_config(DYNAMIC_COLORS_CONFIG_PATH)
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
