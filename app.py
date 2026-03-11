from __future__ import annotations

import traceback

import streamlit as st

try:
    from ui.app_runner import run
except Exception as exc:
    st.set_page_config(page_title="Personal Color Analysis", layout="wide")
    st.title("Personal Color Analysis")
    st.error("The app failed during startup.")
    st.exception(exc)
    st.code(traceback.format_exc())
else:
    run()
