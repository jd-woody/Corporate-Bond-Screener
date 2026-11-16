"""ui_helpers.py - small Streamlit guards."""
import streamlit as st


def fix_state(key, options, multi=False):
    """Drop remembered widget values that are no longer valid options (e.g. a bond filtered out by the sidebar) so Streamlit does not error."""
    if key not in st.session_state:
        return
    opts = set(options)
    if multi:
        st.session_state[key] = [v for v in st.session_state[key] if v in opts]
    elif st.session_state[key] not in opts:
        del st.session_state[key]


def nav(labels, key, default=None):
    """Lazy page switcher: only the chosen page is rendered (much faster than st.tabs, which runs every tab on every click)."""
    seg = getattr(st, 'segmented_control', None)
    if seg is not None:
        v = seg('Navigate', labels, default=default or labels[0], key=key, label_visibility='collapsed')
        return v or (default or labels[0])
    return st.radio('Navigate', labels, horizontal=True, key=key, label_visibility='collapsed')
