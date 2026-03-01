"""Drape simulation and season scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.color_features import ColorFeatures, compute_robust_lab_features
from src.face_regions import FaceRegionResult
from src.season_index import SEASONS


@dataclass
class DrapeColorResult:
    color_hex: str
    color_score: float
    total_penalty: float
    delta_l_skin: float
    delta_chroma_skin: float
    delta_chroma_skin_signed: float
    delta_hue_skin: float
    grey_cast_penalty: float
    var_l_increase: float
    var_ab_increase: float
    edge_harshness_increase: float


@dataclass
class DrapeSeasonResult:
    season: str
    drape_score: float
    avg_delta_l_skin: float
    avg_delta_chroma_skin: float
    avg_delta_chroma_skin_signed: float
    avg_delta_hue_skin: float
    avg_grey_cast_penalty: float
    avg_var_l_increase: float
    avg_var_ab_increase: float
    avg_edge_harshness_increase: float
    colors: List[DrapeColorResult]


@dataclass
class DrapeEvaluation:
    final_scores: Dict[str, float]
    drape_scores: Dict[str, float]
    baseline_scores: Dict[str, float]
    best_season: str
    top2: List[str]
    confidence: float
    season_details: Dict[str, DrapeSeasonResult]
    drape_metrics: Dict[str, float]
    drape_broken: bool
    effective_drape_weight: float


@dataclass
class DrapeRenderContext:
    drape_mask: np.ndarray
    face_ys: np.ndarray
    face_xs: np.ndarray
    face_alpha: np.ndarray


@dataclass(frozen=True)
class ColorDrapeContext:
    image_rgb: np.ndarray
    regions: FaceRegionResult
    baseline_skin: ColorFeatures
    season_hint: Optional[str]
    baseline_hue: float
    baseline_stats: dict[str, float]
    render_context: DrapeRenderContext


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        return 128, 128, 128
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _select_representative_colors(colors: List[str], min_colors: int = 6, max_colors: int = 10) -> List[str]:
    if not colors:
        return []
    if len(colors) <= max_colors:
        return colors
    k = max(min_colors, min(max_colors, len(colors)))
    idx = np.linspace(0, len(colors) - 1, num=k)
    return [colors[int(round(i))] for i in idx]


def _build_drape_render_context(image_shape: tuple[int, int, int], landmarks_px: np.ndarray) -> DrapeRenderContext:
    """Precompute reusable drape mask and reflection indices/weights."""
    h, w = image_shape[:2]
    drape_mask = np.zeros((h, w), dtype=np.uint8)

    y0 = int(0.56 * h)
    drape_mask[y0:h, :] = 255
    side_y0 = int(0.25 * h)
    side_w = int(0.16 * w)
    drape_mask[side_y0:h, :side_w] = 255
    drape_mask[side_y0:h, w - side_w :] = 255

    face_hull = np.zeros((h, w), dtype=np.uint8)
    face_ys = np.array([], dtype=np.int32)
    face_xs = np.array([], dtype=np.int32)
    face_alpha = np.array([], dtype=np.float32)
    if landmarks_px.size:
        hull = cv2.convexHull(np.round(landmarks_px).astype(np.int32))
        cv2.fillConvexPoly(face_hull, hull, 255)
        face_hull = cv2.dilate(face_hull, np.ones((31, 31), dtype=np.uint8), iterations=1)
        drape_mask[face_hull > 0] = 0
        ys, xs = np.where(face_hull > 0)
        if ys.size > 0:
            y_min = float(np.min(ys))
            y_max = float(np.max(ys))
            denom = max(1.0, y_max - y_min)
            y_norm = np.clip((ys.astype(np.float32) - y_min) / denom, 0.0, 1.0)
            reflection = np.clip((y_norm - 0.25) / 0.75, 0.0, 1.0)
            alpha = (0.02 + 0.12 * reflection).astype(np.float32)
            face_ys = ys.astype(np.int32)
            face_xs = xs.astype(np.int32)
            face_alpha = alpha

    return DrapeRenderContext(
        drape_mask=drape_mask,
        face_ys=face_ys,
        face_xs=face_xs,
        face_alpha=face_alpha,
    )


def apply_drape_color(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    color_hex: str,
    render_context: Optional[DrapeRenderContext] = None,
) -> np.ndarray:
    """Render a large drape region around/under the face while keeping face area untouched."""
    h, w = image_rgb.shape[:2]
    out = image_rgb.copy()
    ctx = render_context if render_context is not None else _build_drape_render_context(image_rgb.shape, landmarks_px)

    color = np.array(_hex_to_rgb(color_hex), dtype=np.uint8)
    out[ctx.drape_mask > 0] = color

    # Simulate reflected color cast from drape onto lower face so cheek samples respond.
    if ctx.face_ys.size > 0:
        face_px = out[ctx.face_ys, ctx.face_xs].astype(np.float32)
        tint = color.astype(np.float32)[None, :]
        out[ctx.face_ys, ctx.face_xs] = np.clip(
            (1.0 - ctx.face_alpha[:, None]) * face_px + ctx.face_alpha[:, None] * tint,
            0,
            255,
        ).astype(np.uint8)
    return out


def _hue_proxy(features: ColorFeatures) -> float:
    return float(np.arctan2(features.b, features.a))


def _angle_delta(a1: float, a2: float) -> float:
    d = abs(a1 - a2)
    return float(min(d, 2.0 * np.pi - d))


def _season_margin_confidence(scores: Dict[str, float]) -> float:
    raw = np.array([scores[s] for s in SEASONS], dtype=np.float64)
    shifted = raw - np.max(raw)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)
    sorted_probs = sorted([float(v) for v in probs], reverse=True)
    return float(sorted_probs[0] - sorted_probs[1])


def _hsv_from_hex(hex_color: str) -> tuple[float, float]:
    rgb = np.uint8([[list(_hex_to_rgb(hex_color))]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0]
    hue_deg = float(hsv[0]) * 2.0
    sat = float(hsv[1]) / 255.0
    return hue_deg, sat


def _extract_cheek_lab_stats(image_rgb: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> dict:
    cheek_mask = (left_mask > 0) | (right_mask > 0)
    pixels = image_rgb[cheek_mask]
    if pixels.size == 0:
        return {"var_l": 0.0, "var_ab": 0.0, "edge_var": 0.0}

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l_vals = lab[:, 0] * (100.0 / 255.0)
    a_vals = lab[:, 1] - 128.0
    b_vals = lab[:, 2] - 128.0
    var_l = float(np.var(l_vals))
    var_ab = float(np.var(a_vals) + np.var(b_vals))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_vals = lap[cheek_mask]
    edge_var = float(np.var(edge_vals)) if edge_vals.size else 0.0
    return {"var_l": var_l, "var_ab": var_ab, "edge_var": edge_var}


def _compute_color_result(
    image_rgb: np.ndarray,
    regions: FaceRegionResult,
    baseline_skin: ColorFeatures,
    baseline_hue: float,
    baseline_stats: dict[str, float],
    color_hex: str,
    season_hint: Optional[str] = None,
    render_context: Optional[DrapeRenderContext] = None,
) -> Optional[DrapeColorResult]:
    draped = apply_drape_color(image_rgb, regions.landmarks_px, color_hex, render_context=render_context)
    skin_after = compute_robust_lab_features(draped, regions.left_mask, regions.right_mask, min_samples=300)
    if skin_after is None:
        return None

    draped_stats = _extract_cheek_lab_stats(draped, regions.left_mask, regions.right_mask)
    delta_l_skin = abs(skin_after.l - baseline_skin.l)
    delta_chroma_skin_signed = skin_after.chroma - baseline_skin.chroma
    delta_chroma_skin = abs(delta_chroma_skin_signed)
    delta_hue_skin = _angle_delta(_hue_proxy(skin_after), baseline_hue)
    base_sat = abs(baseline_skin.a) + abs(baseline_skin.b)
    draped_sat = abs(skin_after.a) + abs(skin_after.b)
    # Penalize washout toward grey (lower a/b magnitude than baseline).
    grey_cast_penalty = max(0.0, base_sat - draped_sat) / max(base_sat, 1e-6)
    grey_cast_penalty = float(min(grey_cast_penalty, 1.0))
    var_l_increase = max(0.0, draped_stats["var_l"] - baseline_stats["var_l"])
    var_ab_increase = max(0.0, draped_stats["var_ab"] - baseline_stats["var_ab"])
    edge_harshness_increase = max(0.0, draped_stats["edge_var"] - baseline_stats["edge_var"])

    p_l = min(delta_l_skin / 12.0, 1.0)
    p_c = min(delta_chroma_skin / 10.0, 1.0)
    p_h = min(delta_hue_skin / 0.9, 1.0)
    p_g = grey_cast_penalty
    p_var_l = min(var_l_increase / 20.0, 1.0)
    p_var_ab = min(var_ab_increase / 50.0, 1.0)
    p_edge = min(edge_harshness_increase / max(20.0, 0.45 * baseline_stats["edge_var"] + 1e-6), 1.0)

    p_sat_pos = min(max(0.0, delta_chroma_skin_signed) / 8.0, 1.0)
    p_sat_neg = min(max(0.0, -delta_chroma_skin_signed) / 8.0, 1.0)
    # Default saturation stress.
    p_sat = 0.5 * p_sat_pos + 0.5 * p_sat_neg
    if season_hint == "Summer":
        # Penalize over-saturation stress more strongly in Summer drapes.
        p_sat = 0.75 * p_sat_pos + 0.25 * p_sat_neg
    elif season_hint == "Winter":
        # Winter can handle saturation; less penalty on positive shifts by default.
        p_sat = 0.35 * p_sat_pos + 0.65 * p_sat_neg

    clarity_norm = float(np.clip((baseline_skin.chroma - 18.0) / 12.0, -1.0, 1.0))
    p_clarity_mismatch = 0.0
    if season_hint == "Winter":
        # Clear winter colors penalize muted faces more.
        muted_factor = max(0.0, -clarity_norm)
        p_clarity_mismatch = muted_factor * (0.45 * p_sat_pos + 0.30 * p_var_ab + 0.25 * p_edge)
    elif season_hint == "Summer":
        # Muted summer colors penalize high-clarity faces more.
        clear_factor = max(0.0, clarity_norm)
        p_clarity_mismatch = clear_factor * (0.55 * p_sat_neg + 0.25 * p_var_ab + 0.20 * p_edge)

    total_penalty = (
        0.20 * p_l
        + 0.15 * p_c
        + 0.15 * p_h
        + 0.10 * p_g
        + 0.12 * p_var_l
        + 0.10 * p_var_ab
        + 0.08 * p_edge
        + 0.06 * p_sat
        + 0.04 * min(p_clarity_mismatch, 1.0)
    )
    color_score = float(1.0 - total_penalty)

    return DrapeColorResult(
        color_hex=color_hex,
        color_score=color_score,
        total_penalty=float(total_penalty),
        delta_l_skin=float(delta_l_skin),
        delta_chroma_skin=float(delta_chroma_skin),
        delta_chroma_skin_signed=float(delta_chroma_skin_signed),
        delta_hue_skin=float(delta_hue_skin),
        grey_cast_penalty=float(grey_cast_penalty),
        var_l_increase=float(var_l_increase),
        var_ab_increase=float(var_ab_increase),
        edge_harshness_increase=float(edge_harshness_increase),
    )


def _validate_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(rgb, tuple) or len(rgb) != 3:
        raise ValueError("rgb must be a tuple of length 3")
    vals = []
    for ch in rgb:
        if not isinstance(ch, (int, np.integer)):
            raise ValueError("rgb channels must be integers in [0,255]")
        v = int(ch)
        if v < 0 or v > 255:
            raise ValueError("rgb channels must be integers in [0,255]")
        vals.append(v)
    return int(vals[0]), int(vals[1]), int(vals[2])


def prepare_color_drape_context(face_context: dict[str, Any]) -> ColorDrapeContext:
    """Prepare immutable scoring context for repeated single-colour drape scoring."""
    image_rgb = face_context.get("image_rgb")
    regions = face_context.get("regions")
    baseline_skin = face_context.get("baseline_skin")
    season_hint = face_context.get("season_hint")
    if image_rgb is None or regions is None or baseline_skin is None:
        raise ValueError("face_context must include image_rgb, regions, and baseline_skin")
    if not isinstance(image_rgb, np.ndarray):
        raise ValueError("face_context.image_rgb must be a numpy array")
    if not isinstance(regions, FaceRegionResult):
        raise ValueError("face_context.regions must be FaceRegionResult")
    if not isinstance(baseline_skin, ColorFeatures):
        raise ValueError("face_context.baseline_skin must be ColorFeatures")
    return ColorDrapeContext(
        image_rgb=image_rgb,
        regions=regions,
        baseline_skin=baseline_skin,
        season_hint=None if season_hint is None else str(season_hint),
        baseline_hue=_hue_proxy(baseline_skin),
        baseline_stats=_extract_cheek_lab_stats(image_rgb, regions.left_mask, regions.right_mask),
        render_context=_build_drape_render_context(image_rgb.shape, regions.landmarks_px),
    )


def score_color_drape(
    ctx: ColorDrapeContext,
    rgb: tuple[int, int, int],
    diagnostics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Score one candidate drape color with the same penalty terms as season drape scoring."""
    rr, gg, bb = _validate_rgb(rgb)
    color_hex = f"#{rr:02X}{gg:02X}{bb:02X}"
    result = _compute_color_result(
        image_rgb=ctx.image_rgb,
        regions=ctx.regions,
        baseline_skin=ctx.baseline_skin,
        baseline_hue=ctx.baseline_hue,
        baseline_stats=ctx.baseline_stats,
        color_hex=color_hex,
        season_hint=ctx.season_hint,
        render_context=ctx.render_context,
    )
    penalty = 1.0 if result is None else float(result.total_penalty)
    if not diagnostics:
        return penalty
    return penalty, {
        "baseline_hue": ctx.baseline_hue,
        "color_hex": color_hex,
        "result": result,
    }


def evaluate_drape_scores(
    image_rgb: np.ndarray,
    regions: FaceRegionResult,
    baseline_skin: ColorFeatures,
    baseline_scores: Dict[str, float],
    palettes: Dict[str, List[str]],
    baseline_weight: float = 0.6,
    drape_weight: float = 0.4,
) -> DrapeEvaluation:
    baseline_hue = _hue_proxy(baseline_skin)
    baseline_stats = _extract_cheek_lab_stats(image_rgb, regions.left_mask, regions.right_mask)
    render_context = _build_drape_render_context(image_rgb.shape, regions.landmarks_px)
    season_details: Dict[str, DrapeSeasonResult] = {}
    drape_scores: Dict[str, float] = {}
    all_penalties: List[float] = []
    cool_hi_penalties: List[float] = []
    cool_lo_penalties: List[float] = []

    for season in SEASONS:
        season_colors = _select_representative_colors(palettes.get(season, []), min_colors=6, max_colors=10)
        color_results: List[DrapeColorResult] = []
        for color_hex in season_colors:
            color_result = _compute_color_result(
                image_rgb=image_rgb,
                regions=regions,
                baseline_skin=baseline_skin,
                baseline_hue=baseline_hue,
                baseline_stats=baseline_stats,
                color_hex=color_hex,
                season_hint=season,
                render_context=render_context,
            )
            if color_result is None:
                continue
            total_penalty = float(color_result.total_penalty)
            p_sat_pos = min(max(0.0, color_result.delta_chroma_skin_signed) / 8.0, 1.0)
            p_sat_neg = min(max(0.0, -color_result.delta_chroma_skin_signed) / 8.0, 1.0)
            p_sat = 0.5 * p_sat_pos + 0.5 * p_sat_neg
            if season == "Summer":
                p_sat = 0.75 * p_sat_pos + 0.25 * p_sat_neg
            elif season == "Winter":
                p_sat = 0.35 * p_sat_pos + 0.65 * p_sat_neg
            clarity_norm = float(np.clip((baseline_skin.chroma - 18.0) / 12.0, -1.0, 1.0))
            p_clarity_mismatch = 0.0
            if season == "Winter":
                muted_factor = max(0.0, -clarity_norm)
                p_clarity_mismatch = muted_factor * (
                    0.45 * p_sat_pos
                    + 0.30 * min(color_result.var_ab_increase / 50.0, 1.0)
                    + 0.25 * min(
                        color_result.edge_harshness_increase / max(20.0, 0.45 * baseline_stats["edge_var"] + 1e-6),
                        1.0,
                    )
                )
            elif season == "Summer":
                clear_factor = max(0.0, clarity_norm)
                p_clarity_mismatch = clear_factor * (
                    0.55 * p_sat_neg
                    + 0.25 * min(color_result.var_ab_increase / 50.0, 1.0)
                    + 0.20 * min(
                        color_result.edge_harshness_increase / max(20.0, 0.45 * baseline_stats["edge_var"] + 1e-6),
                        1.0,
                    )
                )
            all_penalties.extend(
                [
                    float(min(color_result.delta_l_skin / 12.0, 1.0)),
                    float(min(color_result.delta_chroma_skin / 10.0, 1.0)),
                    float(min(color_result.delta_hue_skin / 0.9, 1.0)),
                    float(color_result.grey_cast_penalty),
                    float(min(color_result.var_l_increase / 20.0, 1.0)),
                    float(min(color_result.var_ab_increase / 50.0, 1.0)),
                    float(
                        min(
                            color_result.edge_harshness_increase / max(20.0, 0.45 * baseline_stats["edge_var"] + 1e-6),
                            1.0,
                        )
                    ),
                    float(p_sat),
                    float(min(p_clarity_mismatch, 1.0)),
                ]
            )
            color_results.append(color_result)

            hue_deg, sat = _hsv_from_hex(color_hex)
            is_cool = 80.0 <= hue_deg <= 320.0
            if is_cool and sat >= 0.55:
                cool_hi_penalties.append(float(total_penalty))
            elif is_cool and sat <= 0.35:
                cool_lo_penalties.append(float(total_penalty))

        if color_results:
            drape_score = float(np.mean([r.color_score for r in color_results]))
            avg_delta_l = float(np.mean([r.delta_l_skin for r in color_results]))
            avg_delta_c = float(np.mean([r.delta_chroma_skin for r in color_results]))
            avg_delta_c_signed = float(np.mean([r.delta_chroma_skin_signed for r in color_results]))
            avg_delta_h = float(np.mean([r.delta_hue_skin for r in color_results]))
            avg_grey = float(np.mean([r.grey_cast_penalty for r in color_results]))
            avg_var_l = float(np.mean([r.var_l_increase for r in color_results]))
            avg_var_ab = float(np.mean([r.var_ab_increase for r in color_results]))
            avg_edge = float(np.mean([r.edge_harshness_increase for r in color_results]))
        else:
            drape_score = 0.5
            avg_delta_l = 0.0
            avg_delta_c = 0.0
            avg_delta_c_signed = 0.0
            avg_delta_h = 0.0
            avg_grey = 0.0
            avg_var_l = 0.0
            avg_var_ab = 0.0
            avg_edge = 0.0

        drape_scores[season] = drape_score
        season_details[season] = DrapeSeasonResult(
            season=season,
            drape_score=drape_score,
            avg_delta_l_skin=avg_delta_l,
            avg_delta_chroma_skin=avg_delta_c,
            avg_delta_chroma_skin_signed=avg_delta_c_signed,
            avg_delta_hue_skin=avg_delta_h,
            avg_grey_cast_penalty=avg_grey,
            avg_var_l_increase=avg_var_l,
            avg_var_ab_increase=avg_var_ab,
            avg_edge_harshness_increase=avg_edge,
            colors=sorted(color_results, key=lambda r: r.color_score, reverse=True),
        )

    score_vals = np.array([drape_scores[s] for s in SEASONS], dtype=np.float64)
    identical_scores = bool(np.max(score_vals) - np.min(score_vals) <= 1e-6)
    penalties_all_zero = (len(all_penalties) == 0) or bool(np.max(np.array(all_penalties, dtype=np.float64)) <= 1e-12)
    drape_broken = identical_scores or penalties_all_zero
    if drape_broken:
        print("DRAPE_BROKEN")
        effective_baseline_weight = 1.0
        effective_drape_weight = 0.0
    else:
        total_w = max(1e-6, baseline_weight + drape_weight)
        effective_baseline_weight = baseline_weight / total_w
        effective_drape_weight = drape_weight / total_w

    final_scores = {
        season: effective_baseline_weight * baseline_scores.get(season, 0.0)
        + effective_drape_weight * drape_scores.get(season, 0.0)
        for season in SEASONS
    }
    ranked = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)
    top2 = [ranked[0][0], ranked[1][0]]
    drape_metrics = {
        "cool_high_chroma_penalty": float(np.mean(cool_hi_penalties)) if cool_hi_penalties else 0.0,
        "cool_low_chroma_penalty": float(np.mean(cool_lo_penalties)) if cool_lo_penalties else 0.0,
    }
    return DrapeEvaluation(
        final_scores=final_scores,
        drape_scores=drape_scores,
        baseline_scores=baseline_scores,
        best_season=top2[0],
        top2=top2,
        confidence=_season_margin_confidence(final_scores),
        season_details=season_details,
        drape_metrics=drape_metrics,
        drape_broken=drape_broken,
        effective_drape_weight=effective_drape_weight,
    )
