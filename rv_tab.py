"""rv_tab.py - Relative Value tab: bond comparison, cheap/rich vs peers, curve comparison, radar, position sizing (DV01 / convexity neutral)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import theme as T
from context import MET
from ui_helpers import fix_state, nav

PALETTE = T.CAT
LETTERS = 'ABCDEF'


def _w(df, col, w='Outstanding Amount'):
    z = df[[col, w]].dropna()
    z = z[z[w] > 0]
    return float(np.average(z[col], weights=z[w])) if len(z) else np.nan


def _style(fig, xt='', yt='', h=440, nonneg=True):
    if nonneg:
        fig.update_xaxes(rangemode='nonnegative')
        fig.update_yaxes(rangemode='nonnegative')
    fig.update_xaxes(title_text=xt)
    fig.update_yaxes(title_text=yt)
    return T.style(fig, h=h)


def _short(r):
    cpn = f'{r["Coupon"]:g}%' if pd.notna(r.get('Coupon')) else 'FRN'
    return f'{str(r["Name"])[:24]} {cpn} {r.get("Maturity", "")}'


# ------------------------------------------------------------------------------------------------ 1. bond comparison
SPEC = [('Issuer', 'Name', None), ('Ticker', 'Ticker', None), ('Sector', 'Sector', None), ('Rating', 'Rating', None),
        ('Collateral', 'Collateral Type', None), ('Maturity type', 'Maturity Type', None), ('Coupon (%)', 'Coupon', '{:.3f}'),
        ('Ask price', 'Ask Price', '{:.2f}'), ('Ask YTM (%)', 'YTM', '{:.2f}'), ('Years to maturity', 'Years to Maturity', '{:.2f}'),
        ('Modified duration (model)', 'Mod Duration', '{:.2f}'), ('Convexity (model)', 'Convexity', '{:.1f}'),
        ('DV01 ($ per bp, per $1mm face)', 'DV01 per $1mm', '{:.0f}'), ('Yield per year of duration (bp)', '_ypd', '{:.0f}'),
        ('Outstanding ($mm)', '_osmm', '{:,.0f}'), ('Yield vs peers (bp)', 'vs peers (bp)', '{:+.0f}'), ('ISIN', 'ISIN', None)]


def _tape(s, names):
    """Tale of the tape: one small bar chart per metric, one bar per selected bond."""
    metrics = [('Yield (%)', 'YTM'), (f'{MET}', MET), ('Mod. duration', 'Mod Duration'), ('Convexity', 'Convexity'), ('DV01 ($/bp per $1mm)', 'DV01 per $1mm'),
               ('Size ($mm)', '_osmm'), ('Yield vs peers (bp)', 'vs peers (bp)'), ('Years to maturity', 'Years to Maturity')]
    long = [{'Metric': lab, 'Bond': names[k], 'Value': float(r[col])} for k, (_, r) in enumerate(s.iterrows()) for lab, col in metrics if col in s and pd.notna(r[col])]
    if not long:
        return
    fig = px.bar(pd.DataFrame(long), x='Bond', y='Value', color='Bond', facet_col='Metric', facet_col_wrap=4, color_discrete_sequence=PALETTE, category_orders={'Bond': names})
    fig.update_yaxes(matches=None, showticklabels=True, title_text='')
    fig.update_xaxes(showticklabels=False, title_text='')
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    st.plotly_chart(T.style(fig, h=470, title='Tale of the tape'), use_container_width=True)


def _tab_compare(pool, peers, lbl):
    opts = pool.index.tolist()
    fix_state('rv_cmp', opts, multi=True)
    sel = st.multiselect('Bonds to compare (2-6; A is the anchor)', opts, default=pool.nlargest(2, 'Outstanding Amount').index.tolist(),
                         max_selections=6, format_func=lambda i: lbl[i], key='rv_cmp')
    if len(sel) < 2:
        st.info('Pick at least two bonds.')
        return
    s = pool.loc[sel].copy()
    s['_ypd'] = s['YTM'] * 100 / s['Mod Duration']
    s['_osmm'] = s['Outstanding Amount'] / 1e6
    s['vs peers (bp)'] = peers['vs peers (bp)'].reindex(s.index)
    cols = [f'{LETTERS[k]}: {_short(s.iloc[k])}' for k in range(len(s))]
    rows, delta = [], []
    for name, col, fmt in SPEC:
        vals = s[col].tolist()
        rows.append([name] + [('-' if (v is None or (isinstance(v, float) and pd.isna(v))) else (fmt.format(v) if fmt else str(v))) for v in vals])
        if fmt:
            a = vals[0]
            delta.append([name] + ['-' if (pd.isna(a) or pd.isna(v)) else ('{:+,.2f}'.format(v - a) if k else 'anchor') for k, v in enumerate(vals)])
    _tape(s, cols)
    with st.expander('Full side-by-side table and differences vs. A'):
        st.dataframe(pd.DataFrame(rows, columns=['Metric'] + cols), hide_index=True, use_container_width=True, height=38 + 35 * len(rows))
        st.dataframe(pd.DataFrame(delta, columns=['Metric'] + cols), hide_index=True, use_container_width=True)
    a, b = s.iloc[0], s.iloc[1]
    c = st.columns(5)
    c[0].metric('Yield pickup A vs B (bp)', f'{(a.YTM - b.YTM) * 100:+.0f}')
    c[1].metric('Maturity gap (yrs)', f'{a["Years to Maturity"] - b["Years to Maturity"]:+.2f}')
    c[2].metric('Rating-notch gap', f'{a.Notch - b.Notch:+.0f}' if pd.notna(a.Notch) and pd.notna(b.Notch) else '-')
    c[3].metric('Price gap', f'{a["Ask Price"] - b["Ask Price"]:+.2f}')
    dd = a['Mod Duration'] - b['Mod Duration']
    c[4].metric('Duration gap (yrs)', f'{dd:+.2f}')
    if pd.notna(dd) and abs(dd) > 0.05:
        st.caption(f'Yield pickup per extra year of duration (A vs B): {(a.YTM - b.YTM) * 100 / dd:+.0f} bp. Model duration ignores call features.')
    left, right = st.columns(2)
    for col, xcol, xt, ttl in [(left, 'Years to Maturity', 'Years to maturity', 'Yield x maturity'), (right, 'Mod Duration', 'Modified duration (model)', 'Yield x duration')]:
        base = pool.dropna(subset=[xcol])
        fig = px.scatter(base, x=xcol, y='YTM', opacity=0.25, hover_name='Label', color_discrete_sequence=[T.GREY], title=ttl)
        ss = s.dropna(subset=[xcol])
        fig.add_scatter(x=ss[xcol], y=ss['YTM'], mode='markers+text', text=[LETTERS[sel.index(i)] for i in ss.index], textposition='top center',
                        marker=dict(size=14, color=[PALETTE[sel.index(i)] for i in ss.index]), name='Selected')
        col.plotly_chart(_style(fig, xt, 'Yield to maturity (%)'), use_container_width=True)


# ------------------------------------------------------------------------------------------------ 2. cheap / rich vs peers
def _tab_peers(pool):
    c1, c2, c3 = st.columns([3, 1, 1])
    grp = c1.multiselect('Peer group = maturity bucket plus', ['Rating Bucket', 'Sector', 'Collateral Type', 'Universe', 'Maturity Type'],
                         default=['Rating Bucket'], key='rv_grp')
    minp = c2.number_input('Min peers per group', min_value=3, max_value=100, value=8, key='rv_minp')
    zmin = c3.number_input('Min |z-score|', min_value=0.0, max_value=5.0, value=0.0, step=0.25, key='rv_zmin')
    pr = an.peer_relative(pool, grp, int(minp))
    pr = pr[pr['vs peers (bp)'].notna()]
    if pr.empty:
        st.warning('No peer group has enough bonds. Lower "Min peers" or use a broader peer group.')
        return
    m = st.columns(4)
    m[0].metric('Bonds ranked', f'{len(pr):,}')
    m[1].metric('Peer groups', f'{pr.groupby(list(dict.fromkeys(list(grp) + ["Maturity Bucket"])), observed=True).ngroups:,}')
    m[2].metric('Median |vs peers|', f'{pr["vs peers (bp)"].abs().median():.0f} bp')
    m[3].metric('Cheapest vs peers', f'{pr["vs peers (bp)"].max():+.0f} bp')
    top = pr['Sector'].value_counts().head(14).index
    hm = pr[pr['Sector'].isin(top)].groupby(['Sector', 'Maturity Bucket'], observed=True)['vs peers (bp)'].median().unstack('Maturity Bucket')
    hm = hm.reindex(columns=[l for l in an.MAT_LABELS if l in hm.columns])
    fig = px.imshow(hm, text_auto='.0f', aspect='auto', color_continuous_scale=T.DIVERGING, color_continuous_midpoint=0)
    st.plotly_chart(T.style(fig, h=470, title='Cheapness map: median yield vs. peers (bp) by sector and maturity (red = cheap, teal = rich)', legend=False), use_container_width=True)
    lim = float(pr['vs peers (bp)'].abs().quantile(0.99))
    sc = pr[pr['vs peers (bp)'].abs() <= lim]
    fig = px.scatter(sc, x='Years to Maturity', y='vs peers (bp)', size=sc['Outstanding Amount'] / 1e6, color='Rating Bucket', hover_name='Label', size_max=28, opacity=0.65,
                     color_discrete_map=T.RATING_COLORS, category_orders={'Rating Bucket': T.RATING_ORDER})
    fig.add_hline(y=0, line_color=T.INK)
    fig.update_xaxes(rangemode='nonnegative', title_text='Years to maturity')
    fig.update_yaxes(title_text='Yield vs. peer median (bp)')
    st.plotly_chart(T.style(fig, h=470, title='Where the dislocations sit (bubble = size; top 1% outliers hidden)'), use_container_width=True)
    pr = pr.assign(**{'Outstanding ($mm)': pr['Outstanding Amount'] / 1e6})
    show = [c for c in ['Label', 'Rating', 'Sector', 'Years to Maturity', 'YTM', 'Peer median YTM', 'vs peers (bp)', 'Peer z', 'Peer count', 'Outstanding ($mm)'] if c in pr]
    cfg = {'Years to Maturity': st.column_config.NumberColumn(format='%.2f'), 'YTM': st.column_config.NumberColumn(format='%.2f'),
           'Peer median YTM': st.column_config.NumberColumn(format='%.2f'), 'vs peers (bp)': st.column_config.NumberColumn(format='%.0f'),
           'Peer z': st.column_config.NumberColumn(format='%.2f'), 'Outstanding ($mm)': st.column_config.NumberColumn(format='%.0f')}
    base = pr[pr['Peer z'].abs() >= zmin] if zmin > 0 else pr
    pick = pd.concat([base.nsmallest(15, 'vs peers (bp)'), base.nlargest(15, 'vs peers (bp)')]).drop_duplicates('Label').sort_values('vs peers (bp)')
    if len(pick):
        fig = go.Figure(go.Bar(y=pick['Label'].str.slice(0, 48), x=pick['vs peers (bp)'], orientation='h', marker_color=[T.RED if v > 0 else T.TEAL for v in pick['vs peers (bp)']],
                               text=[f'{v:+.0f}' for v in pick['vs peers (bp)']], textposition='outside'))
        fig.add_vline(x=0, line_color=T.INK)
        fig.update_xaxes(title_text='bp vs. peer median')
        st.plotly_chart(T.style(fig, h=760, title='15 richest and 15 cheapest bonds versus peers', legend=False), use_container_width=True)
    with st.expander('Tables: cheapest, richest and issuers'):
        t1, t2, t3 = st.tabs(['Cheapest 25', 'Richest 25', 'Issuers'])
        with t1:
            st.dataframe(base.nlargest(25, 'vs peers (bp)')[show], hide_index=True, use_container_width=True, column_config=cfg)
        with t2:
            st.dataframe(base.nsmallest(25, 'vs peers (bp)')[show], hide_index=True, use_container_width=True, column_config=cfg)
        with t3:
            rows = []
            for tk, g in pr.groupby('Ticker'):
                if len(g) >= 2:
                    rows.append({'Ticker': tk, 'Issuer': g['Name'].iloc[0], 'Bonds': len(g), 'Outstanding ($bn)': g['Outstanding Amount'].sum() / 1e9, 'Wtd vs peers (bp)': _w(g, 'vs peers (bp)')})
            if rows:
                it = pd.DataFrame(rows).dropna().sort_values('Wtd vs peers (bp)', ascending=False)
                st.dataframe(pd.concat([it.head(15), it.tail(15)]).drop_duplicates('Ticker'), hide_index=True, use_container_width=True)
    st.download_button('Export all ranked bonds', pr[show].to_csv(index=False).encode(), 'relative_value_vs_peers.csv', 'text/csv', key='rv_dl')
    st.caption('Peer median includes the bond itself and is not adjusted for call features, seniority (unless chosen above), liquidity or stale quotes. '
               'A bond that is cheap vs peers may simply carry risk the peer group does not capture - a prompt to investigate, not a signal.')


# ------------------------------------------------------------------------------------------------ 3. curve comparison
DIMS = {'Issuer (ticker)': 'Ticker', 'Sector': 'Sector', 'Rating bucket': 'Rating Bucket', 'Rating': 'Rating', 'Collateral type': 'Collateral Type',
        'Universe': 'Universe', 'Country': 'Country'}
TENORS = [2, 5, 10, 30]


def _tab_curves(pool):
    c1, c2 = st.columns([1, 3])
    dim = c1.selectbox('Compare by', list(DIMS), key='rv_dim')
    col = DIMS[dim]
    size = pool.groupby(col)['Outstanding Amount'].sum().sort_values(ascending=False)
    fix_state(f'rv_mem_{col}', size.index.tolist(), multi=True)
    members = c2.multiselect(f'{dim} (2-8)', size.index.tolist(), default=size.index.tolist()[:4], max_selections=8, key=f'rv_mem_{col}')
    if len(members) < 1:
        st.info('Pick at least one group.')
        return
    cmap = {m: PALETTE[k % len(PALETTE)] for k, m in enumerate(members)}
    sub = pool[pool[col].isin(members)]
    fig = px.scatter(sub, x='Years to Maturity', y='YTM', color=col, color_discrete_map=cmap, hover_name='Label', opacity=0.45,
                     category_orders={col: members}, title=f'Yield curves by {dim.lower()} (dotted line = fitted curve)')
    rows, tenor_rows = [], []
    for m in members:
        g = sub[sub[col] == m]
        fit = an.curve_coef(g['Years to Maturity'], g['YTM'])
        if fit is not None:
            tt = np.linspace(fit[1], fit[2], 60)
            fig.add_scatter(x=tt, y=np.polyval(fit[0], tt), mode='lines', name=f'{m} fit', line=dict(dash='dot', color=cmap[m]), showlegend=False)
        ys = an.curve_at(fit, TENORS)
        for t, y in zip(TENORS, ys):
            if np.isfinite(y):
                tenor_rows.append({col: m, 'Tenor': f'{t}y', 'YTM (%)': y})
        notch = _w(g, 'Notch')
        rows.append({dim: m, 'Bonds': len(g), 'Outstanding ($bn)': g['Outstanding Amount'].sum() / 1e9, 'Wtd YTM (%)': _w(g, 'YTM'),
                     'Wtd maturity (yrs)': _w(g, 'Years to Maturity'), 'Wtd duration': _w(g, 'Mod Duration'), 'Avg notch': notch,
                     **{f'{t}y fit (%)': y for t, y in zip(TENORS, ys)},
                     '2s10s slope (bp)': (ys[2] - ys[0]) * 100 if np.isfinite(ys[0]) and np.isfinite(ys[2]) else np.nan})
    st.plotly_chart(_style(fig, 'Years to maturity', 'Yield to maturity (%)', 480), use_container_width=True)
    if tenor_rows:
        tf = pd.DataFrame(tenor_rows)
        fig2 = px.bar(tf, x='Tenor', y='YTM (%)', color=col, barmode='group', color_discrete_map=cmap, category_orders={'Tenor': [f'{t}y' for t in TENORS], col: members},
                      title='Fitted yield at standard tenors (blank where the group has no bonds near that tenor)')
        st.plotly_chart(_style(fig2, '', 'Yield (%)', 380), use_container_width=True)
    num = {c: st.column_config.NumberColumn(format='%.2f') for c in ['Outstanding ($bn)', 'Wtd YTM (%)', 'Wtd maturity (yrs)', 'Wtd duration', 'Avg notch'] + [f'{t}y fit (%)' for t in TENORS]}
    num['2s10s slope (bp)'] = st.column_config.NumberColumn(format='%.0f')
    with st.expander('Curve table'):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config=num)
    st.caption('Fitted curves are quadratic (6+ bonds) or linear (3-5) least squares in maturity, never extrapolated beyond a group\'s own bonds. '
               'A slope of a few bp from a thin group is noise; compare groups with plenty of bonds. Avg notch: 1 = AAA ... 22 = D.')


# ------------------------------------------------------------------------------------------------ 4. radar
AXES = [('Yield', 'YTM'), ('Yield per duration', '_ypd'), ('Duration', 'Mod Duration'), ('Convexity', 'Convexity'),
        ('Credit quality', '_cq'), ('Liquidity (size)', 'Outstanding Amount'), ('Cheap vs peers', 'vs peers (bp)')]


def _entities(pool, peers, mode):
    z = pool.copy()
    z['_ypd'] = z['YTM'] * 100 / z['Mod Duration']
    z['_cq'] = -z['Notch']
    z['vs peers (bp)'] = peers['vs peers (bp)'].reindex(z.index)
    if mode == 'Bonds':
        return z
    key = 'Ticker' if mode == 'Issuers' else 'Sector'
    rows = []
    for k, g in z.groupby(key):
        r = {c: _w(g, c) for _, c in AXES if c != 'Outstanding Amount'}
        r['Outstanding Amount'] = g['Outstanding Amount'].sum()
        r['Name'] = f'{g["Name"].iloc[0]} ({k})' if key == 'Ticker' else k
        r['Bonds'] = len(g)
        rows.append(pd.Series(r, name=k))
    return pd.DataFrame(rows)


def _tab_radar(pool, peers, lbl):
    mode = st.radio('Compare', ['Bonds', 'Issuers', 'Sectors'], horizontal=True, key='rv_rmode')
    ent = _entities(pool, peers, mode)
    names = lbl if mode == 'Bonds' else ent['Name'].to_dict()
    opts = ent.sort_values('Outstanding Amount', ascending=False).index.tolist()
    fix_state(f'rv_rsel_{mode}', opts, multi=True)
    sel = st.multiselect(f'{mode} (2-6)', opts, default=opts[:3], max_selections=6, format_func=lambda i: names[i], key=f'rv_rsel_{mode}')
    if len(sel) < 1:
        st.info('Pick at least one.')
        return
    cols = [c for _, c in AXES]
    pct = ent[cols].rank(pct=True) * 100
    long = []
    for i in sel:
        for ax, c in AXES:
            v = pct.at[i, c]
            long.append({'Entity': f'{LETTERS[sel.index(i)]}: {names[i][:34]}', 'Axis': ax, 'Score (percentile)': 0.0 if pd.isna(v) else float(v)})
    fig = px.line_polar(pd.DataFrame(long), r='Score (percentile)', theta='Axis', color='Entity', line_close=True, range_r=[0, 100],
                        color_discrete_sequence=PALETTE, title=f'Percentile rank within {len(ent):,} {mode.lower()} in the current screen (100 = highest)')
    fig.update_traces(fill='toself', opacity=0.35)
    fig.update_layout(height=520, margin=dict(l=40, r=40, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)
    raw = ent.loc[sel, cols].copy()
    raw.insert(0, 'Entity', [f'{LETTERS[k]}: {names[i][:40]}' for k, i in enumerate(sel)])
    raw = raw.rename(columns={'YTM': 'Yield (%)', '_ypd': 'Yield per duration (bp)', 'Mod Duration': 'Duration', '_cq': 'Credit quality (-notch)',
                              'Outstanding Amount': 'Outstanding ($mm)', 'vs peers (bp)': 'vs peers (bp)'})
    raw['Outstanding ($mm)'] = raw['Outstanding ($mm)'] / 1e6
    with st.expander('Raw values behind the radar'):
        st.dataframe(raw, hide_index=True, use_container_width=True, column_config={c: st.column_config.NumberColumn(format='%.2f') for c in raw.columns if c != 'Entity'})
    st.caption('Every axis is a percentile rank (0-100) so different units are comparable. Credit quality = better rating scores higher; "Cheap vs peers" uses a '
               'rating-bucket x maturity-bucket peer median (aggregated by outstanding weight for issuers / sectors). A missing value plots as 0. '
               'Duration and convexity are model values and ignore call features.')


# ------------------------------------------------------------------------------------------------ 5. position sizing
def _risk(r):
    px_ = float(r['Ask Price'])
    return float(r['DV01 per $1mm']), float(an.cvx_per_mm(px_, r['Convexity']))


def _legs_table(legs):
    """legs = [(side sign, face $mm, label, row)] -> numeric DataFrame with net row."""
    rows = []
    for sign, face, label, r in legs:
        dv, cv = _risk(r)
        mv = face * r['Ask Price'] / 100.0
        rows.append({'Side': 'Long' if sign > 0 else 'Short', 'Bond': label, 'Face ($mm)': face, 'Market value ($mm)': sign * mv,
                     'Yield (%)': r['YTM'], 'Mod duration': r['Mod Duration'], 'DV01 ($/bp)': sign * face * dv,
                     'Dollar convexity ($/bp\u00b2)': sign * face * cv, 'Annual yield income ($)': sign * mv * 1e6 * r['YTM'] / 100.0})
    df = pd.DataFrame(rows)
    net = {'Side': 'NET', 'Bond': '', 'Face ($mm)': np.nan, 'Market value ($mm)': df['Market value ($mm)'].sum(), 'Yield (%)': np.nan, 'Mod duration': np.nan,
           'DV01 ($/bp)': df['DV01 ($/bp)'].sum(), 'Dollar convexity ($/bp\u00b2)': df['Dollar convexity ($/bp\u00b2)'].sum(),
           'Annual yield income ($)': df['Annual yield income ($)'].sum()}
    return pd.concat([df, pd.DataFrame([net])], ignore_index=True)


def _exposure(df):
    """Signed DV01 and dollar convexity by leg - the picture of what the structure is (and is not) hedged against."""
    d = df.copy()
    d['Leg'] = d['Side'].where(d['Side'] == 'NET', d['Side'] + ': ' + d['Bond'].str.slice(0, 26))
    a, b = st.columns(2)
    for col, ttl, ax in [(a, 'DV01 by leg ($/bp)', 'DV01 ($/bp)'), (b, 'Dollar convexity by leg ($/bp\u00b2)', 'Dollar convexity ($/bp\u00b2)')]:
        fig = go.Figure(go.Bar(x=d['Leg'], y=d[ax], marker_color=[T.INK if s_ == 'NET' else (T.TEAL if v >= 0 else T.ORANGE) for s_, v in zip(d['Side'], d[ax])]))
        col.plotly_chart(T.style(fig, h=320, title=ttl, legend=False), use_container_width=True)


def _show_legs(df):
    _exposure(df)
    with st.expander('Position table', expanded=True):
      st.dataframe(df, hide_index=True, use_container_width=True, column_config={
        'Face ($mm)': st.column_config.NumberColumn(format='%.2f'), 'Market value ($mm)': st.column_config.NumberColumn(format='%.2f'),
        'Yield (%)': st.column_config.NumberColumn(format='%.2f'), 'Mod duration': st.column_config.NumberColumn(format='%.2f'),
        'DV01 ($/bp)': st.column_config.NumberColumn(format='%.0f'), 'Dollar convexity ($/bp\u00b2)': st.column_config.NumberColumn(format='%.2f'),
        'Annual yield income ($)': st.column_config.NumberColumn(format='%.0f')})


SHIFTS = [-100, -50, -25, -10, 0, 10, 25, 50, 100]


def _scenarios(legs, rel_idx, rel_name):
    """Parallel-shift P&L for the whole book, and a relative shock where only legs[rel_idx] moves."""
    out = []
    for s in SHIFTS:
        par = rel = 0.0
        for k, (sign, face, _, r) in enumerate(legs):
            dv, cv = _risk(r)
            par += sign * an.pnl_one_leg(face, dv, cv, s)
            if k == rel_idx:
                rel += sign * an.pnl_one_leg(face, dv, cv, s)
        out.append({'Yield shift (bp)': s, 'Parallel shift P&L ($)': par, f'{rel_name} P&L ($)': rel})
    df = pd.DataFrame(out)
    st.dataframe(df, hide_index=True, use_container_width=True, column_config={c: st.column_config.NumberColumn(format='%.0f') for c in df.columns[1:]})
    long = df.melt('Yield shift (bp)', var_name='Scenario', value_name='P&L ($)')
    fig = px.bar(long, x='Yield shift (bp)', y='P&L ($)', color='Scenario', barmode='group', color_discrete_sequence=PALETTE, title='Scenario P&L ($, duration + convexity approximation)')
    st.plotly_chart(_style(fig, 'Yield shift (bp)', 'P&L ($)', 380, nonneg=False), use_container_width=True)


def _spread_defaults(sp, targets=(7, 3, 15)):
    """Three distinct bonds near 7y (belly), 3y and 15y maturity, largest issue in each window - a sensible butterfly starting point."""
    chosen = []
    for t in targets:
        c = sp[((sp['Years to Maturity'] - t).abs() <= 2) & ~sp.index.isin(chosen)]
        if len(c):
            chosen.append(c['Outstanding Amount'].idxmax())
    for i in sp.nlargest(10, 'Outstanding Amount').index:
        if len(chosen) >= 3:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:3]


def _tab_sizing(pool, lbl):
    sp = pool[pool['Mod Duration'].notna() & pool['Convexity'].notna()]
    if len(sp) < 3:
        st.info('Not enough bonds with duration and convexity in the current screen.')
        return
    opts = sp.index.tolist()
    mode = st.radio('Structure', ['Two-leg pair - DV01-neutral', 'Three-leg butterfly - DV01 and convexity neutral'], horizontal=True, key='rv_smode')
    top = _spread_defaults(sp)
    for k_ in ['rv_s_long', 'rv_s_short', 'rv_b_a', 'rv_b_w1', 'rv_b_w2']:
        fix_state(k_, opts)
    if mode.startswith('Two'):
        c1, c2 = st.columns(2)
        long_id = c1.selectbox('Long bond', opts, index=opts.index(top[0]), format_func=lambda i: lbl[i], key='rv_s_long')
        short_id = c2.selectbox('Short bond', opts, index=opts.index(top[1]), format_func=lambda i: lbl[i], key='rv_s_short')
        L, S = sp.loc[long_id], sp.loc[short_id]
        basis = st.radio('Size the long leg by', ['Face amount ($mm)', 'DV01 budget ($ per bp)'], horizontal=True, key='rv_s_basis')
        if basis.startswith('Face'):
            face_l = st.number_input('Long face ($mm)', min_value=0.1, value=10.0, step=1.0, key='rv_s_face')
        else:
            budget = st.number_input('Long DV01 ($ per bp)', min_value=1.0, value=10000.0, step=500.0, key='rv_s_dv')
            face_l = budget / _risk(L)[0]
        face_s = an.dv01_neutral_short(face_l, _risk(L)[0], _risk(S)[0])
        legs = [(1, face_l, f'A: {_short(L)}', L), (-1, face_s, f'B: {_short(S)}', S)]
        tbl = _legs_table(legs)
        _show_legs(tbl)
        net = tbl.iloc[-1]
        m = st.columns(4)
        m[0].metric('Hedge ratio (short face / long face)', f'{face_s / face_l:.3f}')
        m[1].metric('Net DV01 ($/bp)', f'{round(net["DV01 ($/bp)"], 6) + 0.0:.0f}')
        m[2].metric('Net dollar convexity ($/bp\u00b2)', f'{net["Dollar convexity ($/bp\u00b2)"]:+.2f}')
        income = net['Annual yield income ($)']
        be = income / abs(tbl.iloc[0]['DV01 ($/bp)']) if tbl.iloc[0]['DV01 ($/bp)'] else np.nan
        m[3].metric('Breakeven spread move over 1y (bp)', f'{be:.1f}' if np.isfinite(be) else '-', help='Net annual yield income / long-leg DV01. Ignores funding, borrow, roll-down and default.')
        st.caption(f'Net annual yield income ${income:,.0f} on net market value ${net["Market value ($mm)"]:.2f}mm. A two-leg trade can neutralise DV01 only; '
                   'the leftover convexity sign shows who wins in a large move. Use the butterfly to neutralise both.')
        _scenarios(legs, 0, 'Long bond only (spread move)')
    else:
        c = st.columns(3)
        a_id = c[0].selectbox('Belly (anchor, long)', opts, index=opts.index(top[0]), format_func=lambda i: lbl[i], key='rv_b_a')
        w1_id = c[1].selectbox('Wing 1', opts, index=opts.index(top[1]), format_func=lambda i: lbl[i], key='rv_b_w1')
        w2_id = c[2].selectbox('Wing 2', opts, index=opts.index(top[2]), format_func=lambda i: lbl[i], key='rv_b_w2')
        face_a = st.number_input('Belly face ($mm)', min_value=0.1, value=10.0, step=1.0, key='rv_b_face')
        A_, W1, W2 = sp.loc[a_id], sp.loc[w1_id], sp.loc[w2_id]
        n1, n2, det = an.butterfly_weights(face_a, _risk(A_), _risk(W1), _risk(W2))
        if len({a_id, w1_id, w2_id}) < 3:
            st.warning('Choose three different bonds.')
            return
        if not np.isfinite(n1):
            st.error('These three bonds have almost identical DV01 / convexity profiles, so no unique convexity-neutral solution exists. Pick bonds with clearly different maturities.')
            return
        legs = [(1, face_a, f'Belly: {_short(A_)}', A_), (1 if n1 > 0 else -1, abs(n1), f'Wing 1: {_short(W1)}', W1), (1 if n2 > 0 else -1, abs(n2), f'Wing 2: {_short(W2)}', W2)]
        tbl = _legs_table(legs)
        _show_legs(tbl)
        net = tbl.iloc[-1]
        m = st.columns(3)
        m[0].metric('Net DV01 ($/bp)', f'{round(net["DV01 ($/bp)"], 6) + 0.0:.1f}')
        m[1].metric('Net dollar convexity ($/bp\u00b2)', f'{round(net["Dollar convexity ($/bp\u00b2)"], 6) + 0.0:.3f}')
        m[2].metric('Net market value ($mm)', f'{net["Market value ($mm)"]:.2f}')
        big = max(abs(n1), abs(n2)) / face_a
        if big > 10:
            st.warning(f'The wing sizes are {big:.0f}x the belly - the three bonds are nearly collinear in risk, so this hedge is fragile. Pick bonds spread further apart in maturity.')
        st.caption('Solves two equations (net DV01 = 0, net dollar convexity = 0) for the two wing face amounts given the belly. Signs can come out either way; '
                   'the table shows Long/Short. Net market value is the cash the structure ties up (or releases).')
        _scenarios(legs, 0, 'Belly only (butterfly shock)')
    st.caption('P&L uses -DV01 x shift + \u00bd x convexity x shift\u00b2 on model duration/convexity (plain fixed-rate bond priced from its yield; callable bonds will behave '
               'differently). Not an executable hedge - check call features, financing, liquidity and bid/ask first.')


# ------------------------------------------------------------------------------------------------ entry point
def render_rv_tab(f, notch):
    z = f.copy()
    for c in ['YTM', 'Years to Maturity', 'Ask Price', 'Outstanding Amount', 'Notch', 'Coupon']:
        z[c] = pd.to_numeric(z[c], errors='coerce')
    z['Rating Bucket'] = z['Rating Bucket'].replace('', np.nan).fillna('NR') if 'Rating Bucket' in z else z['Rating'].fillna('NR')
    z['Sector'] = z['Sector'].replace('', np.nan).fillna('Unclassified')
    cap = st.number_input('Hide yields above (%)', min_value=1.0, value=30.0, step=5.0, key='rv_cap',
                          help='Stale / distressed yields (e.g. above 50%) distort comparisons. Raise to include them.')
    pool = z[z['YTM'].notna() & (z['YTM'] > 0) & (z['YTM'] <= cap) & z['Years to Maturity'].notna() & (z['Years to Maturity'] > 0)
             & z['Ask Price'].notna() & (z['Ask Price'] > 0) & z['Outstanding Amount'].notna()].copy()
    if len(pool) < 5:
        st.info('Fewer than 5 usable bonds in the current screen. Loosen the sidebar filters.')
        return
    lbl = (pool['Label'] + ' | ' + pool['ISIN'].fillna('')).to_dict()
    peers = an.peer_relative(pool, ['Rating Bucket'], 8)
    st.caption(f'{len(pool):,} bonds in the screen (positive yields and maturities only). Duration and convexity are model values from the quoted yield.')
    page = nav(['Bond comparison', 'Cheap / rich vs peers', 'Curve comparison', 'Radar', 'Position sizing'], 'rv_page')
    if page == 'Bond comparison':
        _tab_compare(pool, peers, lbl)
    elif page == 'Cheap / rich vs peers':
        _tab_peers(pool)
    elif page == 'Curve comparison':
        _tab_curves(pool)
    elif page == 'Radar':
        _tab_radar(pool, peers, lbl)
    else:
        _tab_sizing(pool, lbl)
