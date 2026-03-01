"""Data-driven renderer for dynamic text suggestions."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_REQUIRED_RULE_KEYS = ("lexicon", "bands", "templates", "profiles")
_RULES_CACHE: dict[str, dict[str, Any]] = {}


def _normalize_key(value: str | None) -> str:
    if not value:
        return ""
    token = str(value).strip().lower()
    for ch in (" ", "-", "/"):
        token = token.replace(ch, "_")
    return "_".join(part for part in token.split("_") if part)


def _base_season_key(value: str | None) -> str:
    norm = _normalize_key(value)
    for season in ("spring", "summer", "autumn", "winter"):
        if season in norm:
            return season
    return norm


def _flatten_context(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_context(child_prefix, child, out)
    else:
        out[prefix] = value


def _safe_format(template: str, ctx: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    flat: dict[str, Any] = {}
    _flatten_context("", ctx, flat)

    missing = diagnostics.setdefault("missing_placeholders", [])
    output = template
    for part in _extract_placeholders(template):
        value = flat.get(part)
        if value is None:
            value = _resolve_path(ctx, part)
        if value is None:
            if part not in missing:
                missing.append(part)
            value = ""
        output = output.replace("{" + part + "}", str(value))
    return output


def _extract_placeholders(template: str) -> set[str]:
    return set(re.findall(r"{([^{}]+)}", template))


def _deterministic_choice(options: list[str], seed_text: str) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(options)
    return options[idx]


def load_copy_rules(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    cache_key = str(path_obj.resolve())
    cached = _RULES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rules = json.loads(path_obj.read_text(encoding="utf-8"))
    missing = [key for key in _REQUIRED_RULE_KEYS if key not in rules]
    if missing:
        raise ValueError(f"copy rules missing required keys: {', '.join(missing)}")

    _RULES_CACHE[cache_key] = rules
    return rules


def band_for(value: float, band_defs: list[dict[str, Any]]) -> dict[str, Any]:
    if not band_defs:
        return {"id": "unknown", "label": "unknown"}

    value_f = float(value)
    for band in band_defs:
        lower = band.get("min")
        upper = band.get("max")
        if lower is not None and value_f < float(lower):
            continue
        if upper is not None and value_f >= float(upper):
            continue
        return band

    best = band_defs[0]
    best_dist = float("inf")
    for band in band_defs:
        lower = band.get("min")
        upper = band.get("max")
        if lower is None and upper is None:
            dist = 0.0
        elif lower is None:
            dist = abs(value_f - float(upper))
        elif upper is None:
            dist = abs(value_f - float(lower))
        else:
            center = (float(lower) + float(upper)) / 2.0
            dist = abs(value_f - center)
        if dist < best_dist:
            best_dist = dist
            best = band
    return best


def _resolve_profile(ctx: dict[str, Any], rules: dict[str, Any]) -> tuple[dict[str, Any], str]:
    profiles = rules.get("profiles", {})
    candidates = [
        ctx.get("profile_key", ""),
        ctx.get("top1_variant_key", ""),
        ctx.get("top1_key", ""),
        ctx.get("top1_base", ""),
    ]
    for candidate in candidates:
        key = _normalize_key(str(candidate))
        if key and key in profiles:
            return profiles[key], key
        base = _base_season_key(key)
        if base and base in profiles:
            return profiles[base], base
    fallback = profiles.get("fallback_profile")
    if isinstance(fallback, dict):
        return fallback, "fallback_profile"
    return {}, ""


def _resolve_tie_breaker_tests(ctx: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    tie = rules.get("tie_breaker_tests", {})
    pairs = tie.get("pairs", []) if isinstance(tie, dict) else []
    top1 = _base_season_key(ctx.get("top1_key"))
    top2 = _base_season_key(ctx.get("top2_key"))
    target = {top1, top2}
    for row in pairs:
        pair = row.get("pair", [])
        norm_pair = {_base_season_key(str(x)) for x in pair}
        if norm_pair == target:
            bullets = row.get("bullets", [])
            if isinstance(bullets, list):
                return [str(x) for x in bullets]
    default = tie.get("default", []) if isinstance(tie, dict) else []
    return [str(x) for x in default] if isinstance(default, list) else []


def build_copy_context(inference_payload: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    payload = dict(inference_payload or {})
    confidence = float(payload.get("confidence", 0.0))
    axes = payload.get("axes", {})
    if not isinstance(axes, dict):
        axes = {}

    bands = rules.get("bands", {})
    confidence_band = band_for(confidence, list(bands.get("confidence", [])))

    axis_ctx: dict[str, dict[str, Any]] = {}
    for axis_name in ("temp", "value", "chroma", "contrast"):
        raw = float(axes.get(axis_name, 0.0))
        axis_band = band_for(raw, list(bands.get(axis_name, [])))
        axis_ctx[axis_name] = {
            "value": raw,
            "value_rounded": f"{raw:.3f}",
            "band_id": str(axis_band.get("id", "")),
            "band_label": str(axis_band.get("label", axis_band.get("id", ""))),
        }

    top1_key = _normalize_key(payload.get("top1_key"))
    top2_key_raw = payload.get("top2_key")
    top2_key = _normalize_key(top2_key_raw)
    top1_display = str(payload.get("top1_display") or rules.get("lexicon", {}).get("season_names", {}).get(top1_key) or top1_key or "Season")
    top2_display = str(payload.get("top2_display") or rules.get("lexicon", {}).get("season_names", {}).get(top2_key) or top2_key)

    ctx: dict[str, Any] = {
        "top1_key": top1_key,
        "top2_key": top2_key,
        "top2_present": bool(top2_key),
        "top1_base": _base_season_key(top1_key),
        "top2_base": _base_season_key(top2_key),
        "top1_variant_key": _normalize_key(payload.get("top1_variant_key")),
        "profile_key": _normalize_key(payload.get("profile_key")),
        "top1_display": top1_display,
        "top2_display": top2_display,
        "season_display": str(payload.get("season_display") or top1_display),
        "confidence": confidence,
        "confidence_band": str(confidence_band.get("id", "")),
        "confidence_label": str(confidence_band.get("label", confidence_band.get("id", ""))),
        "dynamic_colors": payload.get("dynamic_colors") if isinstance(payload.get("dynamic_colors"), list) else [],
        "image_id": str(payload.get("image_id", "")),
    }
    ctx.update(axis_ctx)

    temp_family = rules.get("lexicon", {}).get("season_temperature_family", {})
    t1 = temp_family.get(ctx["top1_key"]) or temp_family.get(ctx["top1_base"])
    t2 = temp_family.get(ctx["top2_key"]) or temp_family.get(ctx["top2_base"])
    ctx["top2_is_same_temperature"] = bool(t1 and t2 and t1 == t2)

    profile, profile_key = _resolve_profile(ctx, rules)
    ctx["profile"] = profile
    ctx["resolved_profile_key"] = profile_key

    descriptors = rules.get("templates", {}).get("descriptors", {})
    ctx["chroma_descriptor"] = str(descriptors.get("chroma_descriptor", {}).get(ctx["chroma"]["band_id"], "balanced"))
    ctx["temp_descriptor"] = str(descriptors.get("temp_descriptor", {}).get(ctx["temp"]["band_id"], "balanced warmth"))
    ctx["contrast_descriptor"] = str(descriptors.get("contrast_descriptor", {}).get(ctx["contrast"]["band_id"], "clean and cohesive"))

    season_key = profile.get("tagline_key") if isinstance(profile, dict) else None
    if not season_key:
        season_key = ctx["top1_variant_key"] or ctx["top1_key"] or ctx["top1_base"]
    season_key = _normalize_key(str(season_key))
    taglines = rules.get("templates", {}).get("taglines", {})
    ctx["tagline"] = str(taglines.get(season_key) or taglines.get(ctx["top1_base"]) or "Personal palette guidance")

    tone_words = rules.get("lexicon", {}).get("tone_words", {})
    seed_base = f"{ctx.get('image_id', '')}:{ctx.get('top1_key', '')}:{ctx.get('confidence_band', '')}"
    for tone_id, options in tone_words.items():
        if isinstance(options, list):
            ctx[f"tone_word_{tone_id}"] = _deterministic_choice([str(x) for x in options], seed_base + tone_id)

    ctx["tie_breaker_tests"] = _resolve_tie_breaker_tests(ctx, rules)
    return ctx


def _matches_when(when: dict[str, Any] | None, ctx: dict[str, Any]) -> bool:
    if not when:
        return True

    if "confidence_band" in when:
        allowed = when.get("confidence_band")
        if isinstance(allowed, list) and ctx.get("confidence_band") not in [str(x) for x in allowed]:
            return False

    if "top2_present" in when:
        expected = bool(when.get("top2_present"))
        if bool(ctx.get("top2_present")) != expected:
            return False

    if "top2_is_same_temperature" in when:
        expected = bool(when.get("top2_is_same_temperature"))
        if bool(ctx.get("top2_is_same_temperature")) != expected:
            return False

    for key in ("top1_key", "top2_key"):
        if key in when:
            allowed = when.get(key)
            if isinstance(allowed, list) and _normalize_key(str(ctx.get(key, ""))) not in [_normalize_key(str(x)) for x in allowed]:
                return False

    axis_bands = when.get("axis_bands")
    if isinstance(axis_bands, dict):
        for axis_name, allowed in axis_bands.items():
            if not isinstance(allowed, list):
                continue
            band_id = str(ctx.get(axis_name, {}).get("band_id", ""))
            if band_id not in [str(x) for x in allowed]:
                return False

    return True


def select_blocks(rules: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = rules.get("templates", {}).get("blocks", [])
    selected: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, dict) and _matches_when(block.get("when"), ctx):
            selected.append(block)
    return selected


def _resolve_path(ctx: dict[str, Any], dotted: str) -> Any:
    current: Any = ctx
    for token in dotted.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
        else:
            return None
    return current


def render_block(block_def: dict[str, Any], ctx: dict[str, Any], rules: dict[str, Any], diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    _ = rules
    diag = diagnostics if diagnostics is not None else {}
    block: dict[str, Any] = {
        "id": str(block_def.get("id", "")),
        "title": str(block_def.get("title", "")),
    }

    if isinstance(block_def.get("text"), str):
        block["text"] = _safe_format(str(block_def["text"]), ctx, diag).strip()

    bullets: list[str] = []
    raw_bullets = block_def.get("bullets")
    if isinstance(raw_bullets, list):
        for item in raw_bullets:
            bullets.append(_safe_format(str(item), ctx, diag).strip())

    bullets_from = block_def.get("bullets_from")
    if isinstance(bullets_from, str):
        source = _resolve_path(ctx, bullets_from)
        if isinstance(source, list):
            bullets.extend(str(x) for x in source)

    if bullets:
        block["bullets"] = [b for b in bullets if b]

    if not block.get("text") and not block.get("bullets"):
        block["skip"] = True
    return block


def render_text_suggestions(payload: dict[str, Any], rules_path: str | Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        rules = load_copy_rules(rules_path)
    except Exception as exc:
        return {
            "headline": "Personal palette guidance",
            "blocks": [
                {
                    "id": "fallback",
                    "title": "How to use this",
                    "text": "Use medium-contrast outfits and compare drapes in daylight to confirm your best harmony.",
                }
            ],
            "diagnostics": {"error": f"rules_load_failed: {exc}"},
        }

    ctx = build_copy_context(payload, rules)
    headline_template = str(rules.get("templates", {}).get("headline", "{season_display}"))
    headline = _safe_format(headline_template, ctx, diagnostics).strip()

    rendered_blocks: list[dict[str, Any]] = []
    for block_def in select_blocks(rules, ctx):
        rendered = render_block(block_def, ctx, rules, diagnostics)
        if not rendered.get("skip"):
            rendered_blocks.append({k: v for k, v in rendered.items() if k != "skip"})

    if not rendered_blocks:
        rendered_blocks = [
            {
                "id": "fallback",
                "title": "How to use this",
                "text": "Use the suggested palette in daylight and choose combinations that keep your skin clear and balanced.",
            }
        ]

    diagnostics["resolved_profile_key"] = ctx.get("resolved_profile_key", "")
    diagnostics["confidence_band"] = ctx.get("confidence_band", "")
    return {
        "headline": headline or "Personal palette guidance",
        "blocks": rendered_blocks,
        "diagnostics": diagnostics,
    }
