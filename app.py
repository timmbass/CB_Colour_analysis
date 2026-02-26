from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.color_features import compute_robust_lab_features
from src.drape import render_drape_strip
from src.face_regions import FaceMeshDetector, build_region_masks, render_debug_overlay
from src.image_io import decode_uploaded_image
from src.palettes import load_palettes, palette_for_season
from src.season_rules import classify_season


st.set_page_config(page_title="Personal Color Analysis", layout="wide")
st.title("🎨 Personal Color Analysis")
st.write(
    "Upload one or more photos. The app samples cheek regions, estimates undertone/clarity/depth, "
    "predicts your season, and shows a digital drape palette."
)

show_debug = st.checkbox("Show debug overlays (cheek polylines + masks)", value=False)

uploads = st.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

palettes = load_palettes(Path("assets/palettes/seasonal_palettes.json"))

if uploads:
    detector = FaceMeshDetector(static_image_mode=True, max_num_faces=1)

    for upload in uploads:
        st.divider()
        st.subheader(upload.name)

        image_rgb = decode_uploaded_image(upload.read())
        if image_rgb is None:
            st.warning("Could not decode this image. Please try a different JPG/PNG file.")
            continue

        landmarks = detector.detect_single_face(image_rgb)
        if landmarks is None:
            st.info("No face was detected in this image. Try a clearer, front-facing photo with good lighting.")
            st.image(image_rgb, caption="Original", use_container_width=True)
            continue

        regions = build_region_masks(image_rgb.shape, landmarks)
        features = compute_robust_lab_features(image_rgb, regions.left_mask, regions.right_mask)

        if features is None:
            st.info(
                "The cheek sample region was too small or too noisy. "
                "Please upload a higher-resolution, well-lit image."
            )
            st.image(image_rgb, caption="Original", use_container_width=True)
            continue

        decision = classify_season(features)
        season_colors = palette_for_season(palettes, decision.season)
        draped = render_drape_strip(image_rgb, season_colors)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(draped, caption=f"Digital drape ({decision.season})", use_container_width=True)
            if show_debug:
                dbg = render_debug_overlay(image_rgb, regions)
                st.image(dbg, caption="Debug overlay", use_container_width=True)

        with c2:
            st.metric("Season", decision.season)
            st.metric("Confidence", f"{decision.confidence:.0%}")

            st.write("**Derived attributes**")
            st.write(f"- Undertone: **{decision.undertone}**")
            st.write(f"- Clarity: **{decision.clarity}**")
            st.write(f"- Depth: **{decision.depth}**")

            st.write("**Lab features (robust medians)**")
            st.write(f"- L*: `{features.l:.2f}`")
            st.write(f"- a*: `{features.a:.2f}`")
            st.write(f"- b*: `{features.b:.2f}`")
            st.write(f"- Chroma: `{features.chroma:.2f}`")
            st.write(f"- Samples kept: `{features.sample_count}`")

            st.write("**Season palette (hex)**")
            st.code(", ".join(season_colors) if season_colors else "No palette colors found.")

    detector.close()
else:
    st.caption("Upload at least one image to start analysis.")
