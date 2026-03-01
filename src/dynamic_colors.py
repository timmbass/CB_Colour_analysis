"""Dynamic colour suggestions derived from continuous axes.

This module provides two deterministic modes:
- simple: direct axis->HSL mapping
- advanced: candidate search + drape-penalty optimisation
"""

from __future__ import annotations

import colorsys
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.drape_scoring import prepare_color_drape_context, score_color_drape

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "dynamic_colors_config.json"


@dataclass(frozen=True)
class Candidate:
    hex: str
    h: float
    s: float
    l: float
    meta: dict[str, Any]


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def clip11(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


def load_dynamic_colors_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_axis(value: float, name: str, config: dict[str, Any]) -> float:
    axis_name = "temp" if name in {"temperature", "temp"} else name
    raw = float(value)
    if -1.0 <= raw <= 1.0:
        return raw
    axis_cfg = config.get("axis_normalization", {}).get(axis_name, {})
    mean = float(axis_cfg.get("mean", 0.0))
    std = max(1e-6, float(axis_cfg.get("std", 1.0)))
    z_clip = max(1e-6, float(axis_cfg.get("z_clip", 2.0)))
    z = (raw - mean) / std
    return clip11(z / z_clip)


def _lerp(a: float, b: float, t01: float) -> float:
    return float(a + (b - a) * clip01(t01))


def _hue_lerp_deg(cool_deg: float, warm_deg: float, warmness01: float) -> float:
    """Shortest-path hue interpolation on a circular degree scale."""
    a = cool_deg % 360.0
    b = warm_deg % 360.0
    delta = (b - a + 540.0) % 360.0 - 180.0
    return float((a + clip01(warmness01) * delta) % 360.0)


def _resolve_family_name(
    family: str,
    config: dict[str, Any],
    notes: list[str] | None = None,
) -> str:
    families = config.get("families", {})
    if family in families:
        return family
    if "accent" in families:
        fallback = "accent"
    elif families:
        fallback = next(iter(families.keys()))
    else:
        raise KeyError("No colour families configured. Expected at least one entry under config['families'].")
    if notes is not None:
        notes.append(
            f"Family '{family}' not found; using fallback '{fallback}'. "
            f"Available families: {', '.join(sorted(families.keys()))}"
        )
    return fallback


def axes_to_hsl(
    temp: float,
    value: float,
    chroma: float,
    contrast: float,
    family: str,
    config: dict[str, Any],
    diagnostics_notes: list[str] | None = None,
) -> tuple[float, float, float]:
    resolved_family = _resolve_family_name(family, config, diagnostics_notes)
    fam_cfg = config["families"][resolved_family]
    map_cfg = config["mapping"]

    t = clip11(temp)
    v = clip11(value)
    c = clip11(chroma)
    k = clip11(contrast)

    warm01 = (t + 1.0) * 0.5
    value01 = (v + 1.0) * 0.5
    chroma01 = (c + 1.0) * 0.5

    h = _hue_lerp_deg(float(fam_cfg["hue_cool"]), float(fam_cfg["hue_warm"]), warm01)
    s = _lerp(float(map_cfg["saturation"]["min"]), float(map_cfg["saturation"]["max"]), chroma01)
    l = _lerp(float(map_cfg["lightness"]["deep"]), float(map_cfg["lightness"]["light"]), value01)

    s = s * float(fam_cfg.get("sat_scale", 1.0)) + float(fam_cfg.get("sat_offset", 0.0))
    l = l + float(fam_cfg.get("light_offset", 0.0))

    s = s + float(map_cfg.get("contrast_sat_boost", 0.0)) * k
    l = l - float(map_cfg.get("contrast_light_drop", 0.0)) * k

    sat_min = float(map_cfg.get("sat_min", 0.0))
    sat_max = float(map_cfg.get("sat_max", 100.0))
    light_min = float(map_cfg.get("light_min", 0.0))
    light_max = float(map_cfg.get("light_max", 100.0))

    return float(h % 360.0), float(max(sat_min, min(sat_max, s))), float(max(light_min, min(light_max, l)))


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL(0..360,0..100,0..100) to hex using colorsys HLS order.

    colorsys expects HLS as (h, l, s) in [0,1], not HSL.
    """
    h01 = (float(h) % 360.0) / 360.0
    s01 = clip01(float(s) / 100.0)
    l01 = clip01(float(l) / 100.0)
    r, g, b = colorsys.hls_to_rgb(h01, l01, s01)
    return f"#{int(round(r * 255)):02X}{int(round(g * 255)):02X}{int(round(b * 255)):02X}"


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    s = hex_code.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Expected #RRGGBB hex")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def generate_candidates(
    temp: float,
    value: float,
    chroma: float,
    contrast: float,
    family: str,
    config: dict[str, Any],
    n: int = 12,
) -> list[Candidate]:
    base_h, base_s, base_l = axes_to_hsl(temp, value, chroma, contrast, family, config)
    search_cfg = config.get("candidate_search", {})
    hue_steps = list(search_cfg.get("hue_steps_deg", [0]))
    sat_steps = list(search_cfg.get("sat_steps", [0]))
    light_steps = list(search_cfg.get("light_steps", [0]))

    sat_min = float(config["mapping"].get("sat_min", 0.0))
    sat_max = float(config["mapping"].get("sat_max", 100.0))
    light_min = float(config["mapping"].get("light_min", 0.0))
    light_max = float(config["mapping"].get("light_max", 100.0))

    out: list[Candidate] = []
    seen_hex: set[str] = set()
    for h_step in hue_steps:
        for s_step in sat_steps:
            for l_step in light_steps:
                h = (base_h + float(h_step)) % 360.0
                s = float(max(sat_min, min(sat_max, base_s + float(s_step))))
                l = float(max(light_min, min(light_max, base_l + float(l_step))))
                hex_code = hsl_to_hex(h, s, l)
                if hex_code in seen_hex:
                    continue
                seen_hex.add(hex_code)
                out.append(
                    Candidate(
                        hex=hex_code,
                        h=h,
                        s=s,
                        l=l,
                        meta={
                            "hue_offset_index": int(h_step),
                            "saturation_variant": int(s_step),
                            "lightness_variant": int(l_step),
                        },
                    )
                )

    out.sort(
        key=lambda c: (
            abs(float(c.meta["hue_offset_index"])),
            abs(float(c.meta["saturation_variant"])),
            abs(float(c.meta["lightness_variant"])),
        )
    )
    max_n = max(1, int(n))
    max_cfg = int(search_cfg.get("max_per_family", max_n))
    return out[: min(max_n, max_cfg)]


def _reason_text(family_name: str, temp: float, value: float, chroma: float, contrast: float) -> str:
    temp_txt = "cool-leaning" if temp < -0.15 else "warm-leaning" if temp > 0.15 else "balanced temperature"
    chroma_txt = "clearer chroma" if chroma > 0.2 else "softer chroma" if chroma < -0.2 else "moderate chroma"
    value_txt = "a lighter value" if value > 0.2 else "a deeper value" if value < -0.2 else "a mid value"
    contrast_txt = "higher contrast" if contrast > 0.2 else "lower contrast" if contrast < -0.2 else "balanced contrast"

    if family_name == "signature_blue":
        return f"{temp_txt} + {chroma_txt} points to a cleaner blue direction with {value_txt}."
    if family_name == "neutral":
        return f"{value_txt} and {contrast_txt} support a restrained neutral that stays harmonious on skin."
    return f"{temp_txt} and {chroma_txt} suggest this accent direction while respecting {contrast_txt}."


def _slot_reason(slot_key: str, temp: float, value: float, chroma: float, contrast: float) -> str:
    warm = temp > 0.15
    cool = temp < -0.15
    high_chroma = chroma > 0.2
    low_chroma = chroma < -0.2
    deep = value < -0.2
    light = value > 0.2
    high_contrast = contrast > 0.2

    if slot_key == "signature_blue":
        return "Cool/warm balance steers this blue so it stays clear without looking harsh."
    if slot_key == "best_neutral":
        return "Depth and contrast point to a balanced neutral that sits quietly beside your skin."
    if slot_key == "best_accent":
        return "Your saturation profile supports an accent with clarity but controlled intensity."
    if slot_key == "soft_neutral":
        return "Lower visual pressure neutral for everyday wear and softer transitions."
    if slot_key == "deep_neutral":
        return "A deeper anchor neutral helps structure outfits without flattening the face."
    if slot_key == "secondary_accent":
        return "Alternate accent direction adds variety while staying within your axis tolerance."
    if slot_key == "everyday_green":
        if warm:
            return "Warm-leaning tone shifts this green toward olive for an easy daily option."
        if cool:
            return "Cool-leaning tone shifts this green toward pine to keep it crisp."
        return "Balanced temperature supports a versatile mid-green."
    if slot_key == "pop_colour":
        if high_chroma:
            return "Higher chroma allows a brighter pop without overpowering your features."
        if low_chroma:
            return "Muted chroma keeps the pop colour lively but still wearable."
        if deep and high_contrast:
            return "Depth + contrast support a vivid pop with slightly lower lightness."
        if light:
            return "Lighter value supports a bright pop with a lifted finish."
        return "Controlled bright tone chosen to add energy while staying harmonious."
    return _reason_text("accent", temp, value, chroma, contrast)


def _slot_specs(return_set: bool) -> list[dict[str, Any]]:
    base = [
        {"key": "signature_blue", "label": "Signature Blue", "family": "signature_blue", "h": 0.0, "s": 0.0, "l": 0.0},
        {"key": "best_neutral", "label": "Best Neutral", "family": "neutral", "h": 0.0, "s": 0.0, "l": 0.0},
        {"key": "best_accent", "label": "Best Accent", "family": "accent", "h": 0.0, "s": 0.0, "l": 0.0},
    ]
    if not return_set:
        return base
    return base + [
        {"key": "soft_neutral", "label": "Soft Neutral", "family": "neutral", "h": 0.0, "s": -6.0, "l": 10.0},
        {"key": "deep_neutral", "label": "Deep Neutral", "family": "neutral", "h": 0.0, "s": -2.0, "l": -10.0},
        {"key": "secondary_accent", "label": "Secondary Accent", "family": "accent", "h": 0.0, "s": -5.0, "l": 2.0},
        {"key": "everyday_green", "label": "Everyday Green", "family": "everyday_green", "h": 0.0, "s": -2.0, "l": 0.0},
        {"key": "pop_colour", "label": "Pop Colour", "family": "pop_colour", "h": 0.0, "s": 6.0, "l": 2.0},
    ]


def _apply_slot_adjustments(
    slot: dict[str, Any],
    temp: float,
    base_h: float,
    base_s: float,
    base_l: float,
    cfg: dict[str, Any],
) -> tuple[float, float, float]:
    map_cfg = cfg["mapping"]
    sat_min = float(map_cfg.get("sat_min", 0.0))
    sat_max = float(map_cfg.get("sat_max", 100.0))
    light_min = float(map_cfg.get("light_min", 0.0))
    light_max = float(map_cfg.get("light_max", 100.0))

    h = float(base_h + float(slot.get("h", 0.0)))
    if slot.get("key") == "secondary_accent":
        h += -14.0 if temp > 0.0 else 14.0
    s = float(base_s + float(slot.get("s", 0.0)))
    l = float(base_l + float(slot.get("l", 0.0)))
    return float(h % 360.0), float(max(sat_min, min(sat_max, s))), float(max(light_min, min(light_max, l)))


def _generate_candidates_around(
    center_h: float,
    center_s: float,
    center_l: float,
    config: dict[str, Any],
    n: int,
) -> list[Candidate]:
    search_cfg = config.get("candidate_search", {})
    hue_steps = list(search_cfg.get("hue_steps_deg", [0]))
    sat_steps = list(search_cfg.get("sat_steps", [0]))
    light_steps = list(search_cfg.get("light_steps", [0]))
    sat_min = float(config["mapping"].get("sat_min", 0.0))
    sat_max = float(config["mapping"].get("sat_max", 100.0))
    light_min = float(config["mapping"].get("light_min", 0.0))
    light_max = float(config["mapping"].get("light_max", 100.0))

    out: list[Candidate] = []
    seen: set[str] = set()
    for hs in hue_steps:
        for ss in sat_steps:
            for ls in light_steps:
                h = (center_h + float(hs)) % 360.0
                s = float(max(sat_min, min(sat_max, center_s + float(ss))))
                l = float(max(light_min, min(light_max, center_l + float(ls))))
                hx = hsl_to_hex(h, s, l)
                if hx in seen:
                    continue
                seen.add(hx)
                out.append(
                    Candidate(
                        hex=hx,
                        h=h,
                        s=s,
                        l=l,
                        meta={"hue_offset_index": int(hs), "saturation_variant": int(ss), "lightness_variant": int(ls)},
                    )
                )
    out.sort(
        key=lambda c: (
            abs(float(c.meta["hue_offset_index"])),
            abs(float(c.meta["saturation_variant"])),
            abs(float(c.meta["lightness_variant"])),
        )
    )
    return out[: max(1, int(n))]


def select_best_candidate(
    face_context: dict[str, Any],
    candidates: list[Candidate],
    scorer: Callable[[Candidate], tuple[float, dict[str, Any]]],
) -> tuple[Candidate, float, list[dict[str, float]], int]:
    best: Candidate | None = None
    best_penalty = float("inf")
    ranked: list[tuple[Candidate, float]] = []

    _ = face_context
    for cand in candidates:
        penalty, _diag = scorer(cand)
        ranked.append((cand, float(penalty)))
        if penalty < best_penalty:
            best_penalty = float(penalty)
            best = cand

    if best is None:
        raise ValueError("No candidates to score")

    ranked.sort(key=lambda x: x[1])
    top3 = [{"hex": c.hex, "penalty": float(p)} for c, p in ranked[:3]]
    return best, best_penalty, top3, len(ranked)


def _build_scorer(face_context: dict[str, Any]) -> Callable[[Candidate], tuple[float, dict[str, Any]]]:
    custom_scorer = face_context.get("scorer")
    if callable(custom_scorer):
        return custom_scorer

    image_rgb = face_context.get("image_rgb")
    regions = face_context.get("regions")
    baseline_skin = face_context.get("baseline_skin")
    season_hint = face_context.get("season_hint")

    if image_rgb is None or regions is None or baseline_skin is None:
        raise ValueError("Advanced mode requires image_rgb, regions, and baseline_skin in face_context")

    ctx = prepare_color_drape_context(
        {
            "image_rgb": image_rgb,
            "regions": regions,
            "baseline_skin": baseline_skin,
            "season_hint": season_hint,
        }
    )

    def _scorer(cand: Candidate) -> tuple[float, dict[str, Any]]:
        rgb = _hex_to_rgb(cand.hex)
        penalty, diag = score_color_drape(ctx, rgb, diagnostics=True)
        return float(penalty), diag

    return _scorer


def suggest_dynamic_colors(
    face_context: dict[str, Any] | None,
    axes: dict[str, float],
    mode: str = "simple",
    config: dict[str, Any] | None = None,
    diagnostics: bool = False,
    return_set: bool = False,
) -> dict[str, Any]:
    cfg = config if config is not None else load_dynamic_colors_config()
    req_mode = mode.lower().strip()
    if req_mode not in {"simple", "advanced"}:
        req_mode = "simple"

    temp = normalize_axis(float(axes.get("temp", axes.get("temperature", 0.0))), "temp", cfg)
    value = normalize_axis(float(axes.get("value", 0.0)), "value", cfg)
    chroma = normalize_axis(float(axes.get("chroma", 0.0)), "chroma", cfg)
    contrast = normalize_axis(float(axes.get("contrast", 0.0)), "contrast", cfg)

    slots = _slot_specs(return_set=return_set)
    runtime_notes: list[str] = []
    available_families = set(cfg.get("families", {}).keys())
    requested_families = [str(s["family"]) for s in slots]
    missing_families = sorted({f for f in requested_families if f not in available_families})
    if missing_families:
        runtime_notes.append(
            "Missing configured families for requested slots: "
            + ", ".join(missing_families)
            + ". Fallback mapping will be used where needed."
        )

    effective_mode = req_mode
    fallback_reason = None
    scorer: Callable[[Candidate], tuple[float, dict[str, Any]]] | None = None
    if req_mode == "advanced":
        try:
            scorer = _build_scorer(face_context or {})
        except ValueError as exc:
            effective_mode = "simple"
            fallback_reason = str(exc)

    colors: list[dict[str, Any]] = []
    for slot in slots:
        family = str(slot["family"])
        center_h, center_s, center_l = axes_to_hsl(
            temp,
            value,
            chroma,
            contrast,
            family,
            cfg,
            diagnostics_notes=runtime_notes,
        )
        center_h, center_s, center_l = _apply_slot_adjustments(slot, temp, center_h, center_s, center_l, cfg)
        chosen = Candidate(
            hex=hsl_to_hex(center_h, center_s, center_l),
            h=center_h,
            s=center_s,
            l=center_l,
            meta={"source": "simple_center"},
        )
        adv_payload: dict[str, Any] = {}
        if effective_mode == "advanced" and scorer is not None:
            center_penalty, _ = scorer(chosen)
            candidates = _generate_candidates_around(
                center_h=center_h,
                center_s=center_s,
                center_l=center_l,
                config=cfg,
                n=int(cfg.get("candidate_search", {}).get("max_per_family", 18)),
            )
            # Early pruning: if center is already strong, evaluate a smaller neighborhood.
            prune_threshold = float(cfg.get("candidate_search", {}).get("center_prune_threshold", 0.18))
            if center_penalty <= prune_threshold:
                candidates = candidates[: max(6, min(10, len(candidates)))]
            chosen, best_penalty, top3, tested = select_best_candidate(face_context or {}, candidates, scorer)
            adv_payload = {
                "penalty": float(best_penalty),
                "candidates_tested": int(tested),
                "center_penalty": float(center_penalty),
            }
            if diagnostics:
                adv_payload["top3"] = top3

        color_payload = {
            "key": str(slot["key"]),
            "label": str(slot["label"]),
            "name": str(slot["label"]),
            "family": family,
            "hex": chosen.hex,
            "hsl": {"h": round(float(chosen.h), 2), "s": round(float(chosen.s), 2), "l": round(float(chosen.l), 2)},
            "reason": _slot_reason(str(slot["key"]), temp, value, chroma, contrast),
            "advanced": adv_payload,
        }
        colors.append(color_payload)

    out: dict[str, Any] = {
        "mode": effective_mode,
        "colors": colors,
        "meta": {
            "count": len(colors),
            "return_set": bool(return_set),
            "requested_families": requested_families,
            "available_families": sorted(available_families),
            "missing_families": missing_families,
            "notes": runtime_notes,
        },
    }
    if diagnostics:
        out["diagnostics"] = {
            "axes_used": {
                "temp": temp,
                "value": value,
                "chroma": chroma,
                "contrast": contrast,
            },
            "config_hash": config_hash(cfg),
        }
        if fallback_reason:
            out["diagnostics"]["advanced_fallback_reason"] = fallback_reason
        if runtime_notes:
            out["diagnostics"]["family_notes"] = runtime_notes
    return out


__all__ = [
    "Candidate",
    "clip01",
    "clip11",
    "normalize_axis",
    "axes_to_hsl",
    "hsl_to_hex",
    "generate_candidates",
    "score_color_drape",
    "select_best_candidate",
    "suggest_dynamic_colors",
    "load_dynamic_colors_config",
    "config_hash",
]
