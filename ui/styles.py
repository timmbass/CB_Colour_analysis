"""Small Streamlit style helpers."""

from __future__ import annotations

import streamlit as st


def apply_base_styles() -> None:
    st.markdown(
        """
        <style>
          .app-card {
            background: #f8fafc;
            border: 1px solid #e7edf3;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
          }
          .muted-text { color: #4f6272; font-size: 0.93rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
