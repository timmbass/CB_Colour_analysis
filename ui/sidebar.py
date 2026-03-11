"""Sidebar and control definitions."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class SidebarSettings:
    show_debug: bool
    show_per_image: bool
    show_diagnostics: bool
    show_drape_previews: bool
    use_advanced_dynamic: bool
    season_map_y_axis: str
    use_quality_weighted_aggregation: bool
    quality_handling: str
    quality_threshold: float
    wb_method: str


def render_sidebar_controls() -> SidebarSettings:
    with st.sidebar:
        st.header("Controls")
        return SidebarSettings(
            show_debug=st.checkbox("Show debug overlays (cheek polylines + masks)", value=False),
            show_per_image=st.checkbox("Show per-image results", value=True),
            show_diagnostics=st.checkbox("Show diagnostics panel", value=True),
            show_drape_previews=st.checkbox("Show top drape previews (winning season)", value=False),
            use_advanced_dynamic=st.checkbox(
                "Optimise drape colours (slower)",
                value=False,
                help="Tries multiple candidate colours and selects the one that produces the best drape score.",
            ),
            season_map_y_axis=st.selectbox("Season Map Y-axis", options=["value", "chroma", "contrast"], index=0),
            use_quality_weighted_aggregation=st.checkbox(
                "Use quality-weighted multi-image aggregation",
                value=False,
                help="Uses weighted median across per-image season score vectors. Current quality proxy uses usable skin sample count.",
            ),
            quality_handling=st.selectbox(
                "Low-quality handling",
                options=["exclude", "down-weight", "off"],
                index=0,
                help="Exclude low-quality images or keep them with reduced aggregation weight.",
            ),
            quality_threshold=st.slider("Quality threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01),
            wb_method=st.selectbox(
                "White-balance method",
                options=["none", "grayworld"],
                index=1,
                help="Applied before face detection and color sampling.",
            ),
        )
