"""User-facing copy constants."""

from __future__ import annotations

AXIS_LABELS = {
    "temperature": "Warmth (Cool ↔ Warm)",
    "value": "Depth (Light ↔ Deep)",
    "chroma": "Saturation (Soft ↔ Bright)",
    "contrast": "Contrast (Low ↔ High)",
}

AXIS_HELP = {
    "temperature": "Higher values indicate warmer/yellower direction.",
    "value": "Higher values indicate deeper/darker appearance.",
    "chroma": "Higher values indicate clearer/brighter saturation.",
    "contrast": "Higher values indicate stronger hair/iris vs skin separation.",
}

LOW_CONF_TIEBREAKER_BULLETS = [
    "Compare the top-2 palettes in neutral daylight and keep the one with less skin hue shift.",
    "Favor the option that keeps under-eye/cheek shadows softer (less edge harshness).",
    "Prefer the palette where skin looks clearer without looking over-saturated.",
]
