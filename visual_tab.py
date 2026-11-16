"""visual_tab.py - Visual Analytics: yield / price / maturity / rating pictures (never plots negative values)."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import theme as T


def render_visual_tab(f, notch):
    """f = sidebar-filtered bonds; notch = {rating: notch} (kept for call compatibility)."""
    if f is None or f.empty:
        st.info('No bonds match the current sidebar filters.')
        return
    cap = st.number_input('Hide yields above (%)', min_value=1.0, value=30.0, step=5.0, key='va_cap',
                          help='A few bonds carry stale or distressed yields (e.g. above 50%) that squash every chart. Raise this to show them.')
    z = f.copy()
    for c in ['YTM', 'Years to Maturity', 'Ask Price']:
        z[c] = pd.to_numeric(z[c], errors='coerce')
    total = len(z)
    ok_y = z['YTM'].notna() & (z['YTM'] > 0)
    over = ok_y & (z['YTM'] > cap)
    z = z[ok_y & ~over].copy()
    z['Rating Bucket'] = z['Rating Bucket'].replace('', np.nan).fillna('NR') if 'Rating Bucket' in z else z['Rating'].fillna('NR')
    kw = dict(color='Rating Bucket', color_discrete_map=T.RATING_COLORS, category_orders={'Rating Bucket': T.RATING_ORDER})
    mat = z[z['Years to Maturity'].notna() & (z['Years to Maturity'] >= 0)].copy()
    mat['Years to Maturity'] = mat['Years to Maturity'].abs()
    prc = z[z['Ask Price'].notna() & (z['Ask Price'] > 0)]
    hover = [c for c in ['Ask Price', 'Years to Maturity', 'Rating'] if c in z]

    def nn(fig, xt, yt, title, h=440, legend=True):
        fig.update_xaxes(rangemode='nonnegative', title_text=xt)
        fig.update_yaxes(rangemode='nonnegative', title_text=yt)
        return T.style(fig, h=h, title=title, legend=legend)

    a, b = st.columns(2)
    fig = px.scatter(mat, x='Years to Maturity', y='YTM', hover_name='Label', hover_data=hover, opacity=0.55, **kw)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    a.plotly_chart(nn(fig, 'Years to maturity', 'Yield to maturity (%)', 'Yield x maturity'), use_container_width=True)
    fig = px.scatter(prc, x='Ask Price', y='YTM', hover_name='Label', hover_data=hover, opacity=0.55, **kw)
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    b.plotly_chart(nn(fig, 'Ask price', 'Yield to maturity (%)', 'Yield x price'), use_container_width=True)
    c, d = st.columns(2)
    fig = px.box(z, x='Rating Bucket', y='YTM', points=False, **kw)
    c.plotly_chart(nn(fig, '', 'Yield to maturity (%)', 'Yield dispersion by rating bucket', legend=False), use_container_width=True)
    fig = px.histogram(mat, x='Years to Maturity', nbins=20, **kw)
    fig.update_traces(xbins=dict(start=0, size=5))
    fig.update_layout(barmode='stack')
    d.plotly_chart(nn(fig, 'Years to maturity', 'Bonds', 'Maturity distribution (bond count)'), use_container_width=True)
    notes = []
    if total - len(z) - int(over.sum()):
        notes.append(f'{total - len(z) - int(over.sum())} bond(s) with a missing or non-positive yield')
    if int(over.sum()):
        notes.append(f'{int(over.sum())} bond(s) with yield above {cap:g}%')
    st.caption(f'Showing {len(z):,} of {total:,} bonds. ' + ('Left out: ' + '; '.join(notes) + '. ' if notes else '') + 'All axes start at zero; no negative values are plotted.')
