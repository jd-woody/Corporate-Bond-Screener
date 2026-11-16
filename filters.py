"""filters.py - advanced left-hand screener. render_sidebar(d, notch) -> (filtered frame, sort column, ascending, summary lines)."""
import re

import numpy as np
import pandas as pd
import streamlit as st

SORT_OPTIONS = {'Ask yield to maturity': 'YTM', 'Ask price': 'Ask Price', 'Years to maturity': 'Years to Maturity',
                'Outstanding amount': 'Outstanding Amount', 'Coupon': 'Coupon', 'Rating (best to worst)': 'Notch', 'Issuer name': 'Name'}

# name -> (description, function(frame) -> boolean mask). Presets stack ON TOP of the widgets above them.
PRESETS = {
    'Custom': ('No preset - only the filters above apply.', None),
    'IG liquid 3-10y': ('Investment grade, $500mm+ outstanding, 3-10 years to maturity.',
                        lambda x: (x['Universe'] == 'IG') & (x['Outstanding Amount'] >= 500e6) & x['Years to Maturity'].between(3, 10)),
    'HY liquid 2-7y': ('High yield, $300mm+ outstanding, 2-7 years to maturity.',
                       lambda x: (x['Universe'] == 'HY') & (x['Outstanding Amount'] >= 300e6) & x['Years to Maturity'].between(2, 7)),
    'Crossover (BBB / BB)': ('Ratings BBB, BBB-, BB+ and BB - the investment-grade / high-yield border.',
                             lambda x: x['Rating'].isin(['BBB', 'BBB-', 'BB+', 'BB'])),
    'Short-dated (<3y)': ('Less than 3 years to maturity.', lambda x: x['Years to Maturity'] < 3),
    'Long-dated (15y+)': ('15 or more years to maturity.', lambda x: x['Years to Maturity'] >= 15),
    'Discount bonds (<95)': ('Ask price below 95.', lambda x: x['Ask Price'] < 95),
    'Premium bonds (>105)': ('Ask price above 105.', lambda x: x['Ask Price'] > 105),
    'Benchmark size ($1bn+)': ('At least $1bn outstanding.', lambda x: x['Outstanding Amount'] >= 1e9),
    'Senior unsecured only': ('Seniority rank 3 (senior unsecured / company guaranteed).', lambda x: x['Seniority Rank'] == 3),
}


def _reset():
    for k in [k for k in st.session_state if str(k).startswith('flt_')]:
        del st.session_state[k]


def _opts(s):
    return sorted(v for v in s.dropna().astype(str).unique() if v.strip())


def render_sidebar(d, notch):
    """Draw the sidebar and return (filtered frame, sort column, ascending flag, list of active-filter descriptions)."""
    x = d
    active = []
    with st.sidebar:
        st.header('Screener')
        q = st.text_input('Name / ticker / ISIN / CUSIP', key='flt_q', help='Separate several with commas, e.g. AAPL, ORCL, US68389XBU84. Matches any term.')
        uni = st.multiselect('Universe', ['IG', 'HY'], key='flt_uni', placeholder='All')
        rating = st.multiselect('Normalized rating', sorted(x['Rating'].dropna().unique(), key=lambda r: notch.get(r, 99)), key='flt_rating', placeholder='All')
        coll = st.multiselect('Collateral type', _opts(x['Collateral Type']), key='flt_coll', placeholder='All')
        sector = st.multiselect('Sector', _opts(x['Sector']), key='flt_sector', placeholder='All')
        country = st.multiselect('Country (from ISIN prefix)', _opts(x['Country']), key='flt_country', placeholder='All',
                                 help='ISIN prefix: US = US-registered, XS = international (Euroclear/Clearstream), EU = supranational, etc. The CSVs have no issuer-country field.')
        currency = st.multiselect('Currency', _opts(x['Currency']), key='flt_ccy', placeholder='All') if 'Currency' in x else []
        with st.expander('More filters'):
            mtype = st.multiselect('Maturity type', _opts(x['Maturity Type']), key='flt_mtype', placeholder='All') if 'Maturity Type' in x else []
            industry = st.multiselect('Industry', _opts(x['Industry']), key='flt_industry', placeholder='All') if 'Industry' in x else []
            psrc = st.multiselect('Pricing source', _opts(x['Pricing Source']), key='flt_psrc', placeholder='All') if 'Pricing Source' in x else []
            hide_flag = st.checkbox('Hide data-flagged bonds (stale / perpetual)', value=False, key='flt_flag') if 'Data Flag' in x else False
        preset = st.selectbox('Preset', list(PRESETS), key='flt_preset', help='Presets are applied on top of the filters above.')
        st.caption(PRESETS[preset][0])
        excl = st.checkbox('Exclude matured / missing maturity', value=True, key='flt_excl')
        apply_rng = st.checkbox('Apply numeric ranges', value=False, key='flt_apply', help='Tick to activate the sliders below.')
        ytm_hi = float(np.ceil(x['YTM'].max())) if x['YTM'].notna().any() else 30.0
        p_lo, p_hi = float(np.floor(x['Ask Price'].min())), float(np.ceil(x['Ask Price'].max()))
        y_hi = float(np.ceil(x['Years to Maturity'].max()))
        c_hi = float(np.ceil(x['Coupon'].max()))
        r_ytm = st.slider('Ask YTM (%)', 0.0, ytm_hi, (0.0, min(30.0, ytm_hi)), step=0.25, key='flt_ytm', disabled=not apply_rng)
        r_px = st.slider('Ask price (% of par)', p_lo, p_hi, (max(p_lo, 40.0), min(p_hi, 130.0)), step=0.5, key='flt_px', disabled=not apply_rng)
        r_yrs = st.slider('Years to maturity', 0.0, y_hi, (0.0, y_hi), step=0.25, key='flt_yrs', disabled=not apply_rng)
        r_cpn = st.slider('Coupon (%)', 0.0, c_hi, (0.0, c_hi), step=0.125, key='flt_cpn', disabled=not apply_rng)
        min_size = st.number_input('Min outstanding ($mm)', min_value=0.0, value=0.0, step=50.0, key='flt_size')
        sort_label = st.selectbox('Sort by', list(SORT_OPTIONS), key='flt_sort')
        desc = st.checkbox('Descending', value=True, key='flt_desc')

        # ---- apply
        m = pd.Series(True, index=x.index)

        def need(mask, text):
            nonlocal m
            m &= mask
            active.append(text)

        terms = [t.strip().lower() for t in re.split(r'[,;\n]+', q or '') if t.strip()]
        if terms:
            need(x['_search'].apply(lambda s: any(t in s for t in terms)), f'search: {", ".join(terms)}')
        for col, sel, nm in [('Universe', uni, 'universe'), ('Rating', rating, 'rating'), ('Collateral Type', coll, 'collateral'),
                             ('Sector', sector, 'sector'), ('Country', country, 'country'), ('Currency', currency, 'currency'),
                             ('Maturity Type', mtype, 'maturity type'), ('Industry', industry, 'industry'), ('Pricing Source', psrc, 'pricing source')]:
            if sel:
                need(x[col].isin(sel), f'{nm}: {len(sel)} selected')
        if hide_flag:
            need(x['Data Flag'].fillna('').astype(str).str.strip() == '', 'data-flagged bonds hidden')
        if excl:
            need(x['Years to Maturity'].notna() & (x['Years to Maturity'] > 0), 'matured / missing maturity excluded')
        if apply_rng:
            need(x['YTM'].between(*r_ytm), f'YTM {r_ytm[0]:g}-{r_ytm[1]:g}%')
            need(x['Ask Price'].between(*r_px), f'price {r_px[0]:g}-{r_px[1]:g}')
            need(x['Years to Maturity'].between(*r_yrs) | (x['Years to Maturity'].isna() & (not excl) & (r_yrs[0] == 0)), f'years {r_yrs[0]:g}-{r_yrs[1]:g}')
            need(x['Coupon'].between(*r_cpn) | x['Coupon'].isna(), f'coupon {r_cpn[0]:g}-{r_cpn[1]:g}%')
        if min_size > 0:
            need(x['Outstanding Amount'].fillna(0) >= min_size * 1e6, f'outstanding >= ${min_size:g}mm')
        fn = PRESETS[preset][1]
        if fn is not None:
            need(fn(x).fillna(False), f'preset: {preset}')
        out = x[m].copy()
        st.markdown('---')
        st.metric('Matching bonds', f'{len(out):,} / {len(x):,}')
        st.button('Reset filters', on_click=_reset, key='flt_reset', use_container_width=True)
    return out, SORT_OPTIONS[sort_label], (not desc), active
