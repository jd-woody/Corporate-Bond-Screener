"""sector_tab.py - Sector Analysis: landscape, credit-spread benchmarks over time, sector / industry story, issuers."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import benchmarks as bm
import story as S
import theme as T
from capiq_tab import get_companies
from context import MET
from ui_helpers import fix_state, nav

PAGES = ['Landscape', 'Benchmarks', 'Sector & industry story', 'Issuers']
DIMS = {'Sector': 'Sector', 'Industry': 'Industry', 'Rating bucket': 'Rating Bucket', 'Universe': 'Universe', 'Country': 'Country', 'Collateral type': 'Collateral Type'}
HIST_LEVEL = {'Sector': 'Sector', 'Industry': 'Industry', 'Rating Bucket': 'Rating Bucket', 'Universe': 'Universe'}


def _weights(g):
    w = g[g['Rating Bucket'].isin(bm.BUCKETS)].groupby('Rating Bucket')['Outstanding Amount'].sum()
    return (w / w.sum()).to_dict() if w.sum() > 0 else {}


def _bucket_bench(d_ok, ctx, source):
    """{bucket: benchmark level in bp}: latest ICE BofA OAS, or the screener-wide weighted average of that bucket."""
    out = {}
    for b in bm.BUCKETS:
        if source == 'index' and b in ctx.bench.oas.columns and ctx.bench.oas[b].notna().any():
            out[b] = float(ctx.bench.oas[b].dropna().iloc[-1])
        else:
            ref = d_ok[d_ok['Rating Bucket'] == b]
            if len(ref):
                out[b] = an.wavg(ref, MET)
    return out


def _blend(weights, bb):
    ws = {b: w for b, w in weights.items() if b in bb and pd.notna(bb[b])}
    return float(sum(bb[b] * w for b, w in ws.items()) / sum(ws.values())) if ws else np.nan


def _summ(z, key):
    rows = []
    for k, g in z.groupby(key):
        out = g['Outstanding Amount'].sum()
        rows.append({key: k, 'Bonds': len(g), 'Issuers': g['Ticker'].nunique(), 'Outstanding ($bn)': out / 1e9, 'Metric': an.wavg(g, MET), 'Years': an.wavg(g, 'Years to Maturity'),
                     'HY share (%)': g.loc[g['Universe'] == 'HY', 'Outstanding Amount'].sum() / out * 100 if out else np.nan, 'Notch': an.wavg(g, 'Notch')})
    return pd.DataFrame(rows).sort_values('Outstanding ($bn)', ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ landscape
def _page_landscape(z, ctx, summ):
    fig = px.treemap(z.assign(Industry=z['Industry'].replace('', 'Unclassified')), path=[px.Constant('Market'), 'Sector', 'Industry'], values='Outstanding Amount', color=MET,
                     color_continuous_scale=T.SEQUENTIAL)
    fig.update_layout(coloraxis_colorbar=dict(title=ctx.label.split(' ')[0]))
    st.plotly_chart(T.style(fig, h=540, title=f'The market at a glance - area = debt outstanding, colour = {ctx.label}', legend=False), use_container_width=True)
    a, b = st.columns(2)
    fig = px.scatter(summ, x='Years', y='Metric', size='Outstanding ($bn)', color='HY share (%)', text='Sector', size_max=55, hover_name='Sector',
                     color_continuous_scale=[[0, T.BLUE], [1, T.ORANGE]], range_color=[0, 100])
    fig.update_traces(textposition='top center', marker=dict(opacity=0.8, line=dict(width=0)))
    fig.update_xaxes(rangemode='nonnegative', title_text='Weighted average life (years)')
    fig.update_yaxes(title_text=ctx.label)
    a.plotly_chart(T.style(fig, h=470, title='Sector map: price of risk vs. tenor (bubble = size, colour = HY share)', legend=False), use_container_width=True)
    mix = z.groupby(['Sector', 'Rating Bucket'])['Outstanding Amount'].sum().reset_index()
    mix['share'] = mix['Outstanding Amount'] / mix.groupby('Sector')['Outstanding Amount'].transform('sum') * 100
    order = summ.sort_values('Metric')['Sector'].tolist()
    fig = px.bar(mix, x='share', y='Sector', color='Rating Bucket', orientation='h', color_discrete_map=T.RATING_COLORS, category_orders={'Sector': order, 'Rating Bucket': T.RATING_ORDER})
    fig.update_xaxes(ticksuffix='%', range=[0, 100], title_text='')
    b.plotly_chart(T.style(fig, h=470, title='Rating mix by sector (sorted by price of risk, tightest at bottom)'), use_container_width=True)
    top = summ.nlargest(16, 'Outstanding ($bn)')['Sector'].tolist()
    med = z[z['Sector'].isin(top)].groupby('Sector')[MET].median().sort_values()
    fig = go.Figure()
    for i, sec in enumerate(med.index):
        fig.add_trace(go.Box(x=z.loc[z['Sector'] == sec, MET], name=sec, orientation='h', boxpoints=False, marker_color=T.CAT[i % len(T.CAT)], line=dict(width=1.5)))
    fig.update_xaxes(rangemode='nonnegative', title_text=ctx.label)
    st.plotly_chart(T.style(fig, h=520, title=f'How much each sector\'s bonds disagree: {ctx.label} distribution (box = middle 50%)', legend=False), use_container_width=True)
    with st.expander('Sector table'):
        st.dataframe(summ, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------------------------ benchmarks
def _data_panel(ctx):
    with st.expander('Benchmark data (Treasury curve and ICE BofA OAS from FRED)'):
        b = ctx.bench
        c = st.columns(4)
        c[0].metric('Treasury curve', f'{ctx.curve[2]:%d %b %Y}' if ctx.curve[2] is not None else ('manual' if ctx.curve[0] is not None else 'missing'))
        c[1].metric('OAS series', f'{b.oas.shape[1]}' if b.has_oas else '0')
        c[2].metric('OAS through', f'{b.oas.index.max():%d %b %Y}' if b.has_oas else '-')
        c[3].metric('Saved snapshots', f'{ctx.hist["date"].nunique() if len(ctx.hist) else 0}')
        if st.button('Refresh benchmarks from FRED', key='sa_refresh'):
            with st.spinner('Downloading Treasury and ICE BofA series...'):
                ok, bad = bm.refresh(ctx.bench_dir)
            st.cache_data.clear()
            if bad:
                st.warning(f'{len(ok)} series updated, {len(bad)} failed (first error: {bad[0][1]}). Check your internet / proxy, or drop manual_curve.csv in data/benchmarks.')
            else:
                st.success(f'{len(ok)} series updated.')
            st.rerun()
        st.caption('Spreads are G-spreads: bond yield minus the interpolated Treasury yield at the bond\'s maturity (not an option-adjusted spread). The index comparison '
                   'uses the ICE BofA rating-bucket OAS series; FRED keeps only about 3 years, so the app merges every refresh into local files to keep older history. '
                   'Offline? Put columns tenor_years,yield_pct in data/benchmarks/manual_curve.csv.')


def _page_benchmarks(z, d_ok, ctx):
    _data_panel(ctx)
    if not ctx.has_spread:
        st.info('No Treasury curve yet, so visuals use yield (bp). Press "Refresh benchmarks from FRED" above to switch to G-spreads and unlock the ICE BofA index comparison.')
    c1, c2, c3 = st.columns([2, 2, 1])
    dim_label = c1.selectbox('Compare by', list(DIMS), key='sa_dim')
    options = ['ICE BofA OAS (rating-matched)', 'Screener universe (rating-matched)'] if ctx.has_index else ['Screener universe (rating-matched)']
    source_label = c2.radio('Benchmark', options, horizontal=True, key='sa_src')
    minb = c3.number_input('Min bonds', min_value=3, max_value=500, value=10, key='sa_minb')
    source = 'index' if source_label.startswith('ICE') else 'universe'
    dim = DIMS[dim_label]
    bb = _bucket_bench(d_ok, ctx, source)
    rows = []
    for k, g in z.groupby(dim):
        if len(g) < minb or k in ('', None):
            continue
        w = _weights(g)
        lvl = an.wavg(g, MET)
        bench = _blend(w, bb) if dim != 'Rating Bucket' else bb.get(k, np.nan)
        pctile = np.nan
        if source == 'index' and ctx.has_index and w:
            ser = bm.blend_oas(ctx.bench, w)
            pctile = bm.percentile_of(ser, ser.dropna().iloc[-1]) if ser.notna().any() else np.nan
        rows.append({dim: k, 'Bonds': len(g), 'Outstanding ($bn)': g['Outstanding Amount'].sum() / 1e9, 'Level': lvl, 'Benchmark': bench, 'Excess': lvl - bench if pd.notna(bench) else np.nan, 'Bench pctile': pctile})
    bt = pd.DataFrame(rows).dropna(subset=['Benchmark'])
    if bt.empty:
        st.warning('No group has enough rated bonds for a benchmark comparison. Lower "Min bonds" or choose another grouping.')
        return
    bt = bt.sort_values('Excess')
    cols = [T.RED if v > 0 else T.TEAL for v in bt['Excess']]
    fig = go.Figure(go.Bar(y=bt[dim], x=bt['Excess'], orientation='h', marker_color=cols, text=[f'{v:+.0f}' for v in bt['Excess']], textposition='outside'))
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text='bp above (+) / below (-) the benchmark for the same rating mix')
    a, b = st.columns(2)
    a.plotly_chart(T.style(fig, h=max(380, 26 * len(bt) + 120), title=f'Priced vs. benchmark by {dim_label.lower()} ({ctx.label})', legend=False), use_container_width=True)
    mx = float(max(bt['Level'].max(), bt['Benchmark'].max())) * 1.05
    fig = px.scatter(bt, x='Benchmark', y='Level', size='Outstanding ($bn)', color='Excess', text=dim, size_max=40, color_continuous_scale=T.DIVERGING, color_continuous_midpoint=0)
    fig.add_scatter(x=[0, mx], y=[0, mx], mode='lines', line=dict(color=T.GREY, dash='dash'), name='= benchmark', showlegend=False)
    fig.update_traces(textposition='top center', selector=dict(mode='markers+text'))
    fig.update_xaxes(rangemode='nonnegative', title_text='Benchmark for its rating mix (bp)')
    fig.update_yaxes(rangemode='nonnegative', title_text=f'Actual ({ctx.label})')
    b.plotly_chart(T.style(fig, h=max(380, 26 * len(bt) + 120), title='Above the line = paying more than the index would for the same ratings', legend=False), use_container_width=True)
    st.caption('Why a gap can exist: genuine extra risk the rating misses, longer maturity than the index, callable / illiquid / small issues, or cheapness. '
               + ('Index = ICE BofA rating-bucket OAS (option-adjusted); ours is G-spread, so callable-heavy groups look wider.' if source == 'index' else 'Universe benchmark = this screener\'s own weighted average for each rating bucket.'))
    T.section('Heat map', 'Where in the market investors are paid more or less than the rating bucket benchmark')
    mode = st.radio('Show', ['Excess vs. rating-bucket benchmark (bp)', f'{ctx.label} level'], horizontal=True, key='sa_heat')
    rows = []
    for (k, bucket), g in z.groupby([dim, 'Rating Bucket']):
        if len(g) >= 3 and bucket in bb and k in set(bt[dim]):
            lvl = an.wavg(g, MET)
            rows.append({dim: k, 'Bucket': bucket, 'v': lvl - bb[bucket] if mode.startswith('Excess') else lvl})
    if rows:
        hm = pd.DataFrame(rows).pivot(index=dim, columns='Bucket', values='v').reindex(columns=[b_ for b_ in T.RATING_ORDER if b_ in set(r['Bucket'] for r in rows)])
        hm = hm.reindex(bt[dim].tolist()[::-1])
        fig = px.imshow(hm, text_auto='.0f', aspect='auto', color_continuous_scale=T.DIVERGING if mode.startswith('Excess') else T.SEQUENTIAL,
                        color_continuous_midpoint=0 if mode.startswith('Excess') else None)
        st.plotly_chart(T.style(fig, h=max(380, 28 * len(hm) + 120), title=mode, legend=False), use_container_width=True)
    T.section('Over time', 'The benchmark path for a group\'s rating mix, with where the group trades today')
    member = st.selectbox(dim_label, bt.sort_values('Outstanding ($bn)', ascending=False)[dim].tolist(), key=f'sa_member_{dim}')
    g = z[z[dim] == member]
    w = _weights(g)
    ser = bm.blend_oas(ctx.bench, w) if ctx.bench.has_oas else pd.Series(dtype=float)
    fig = go.Figure()
    if len(ser.dropna()):
        fig.add_scatter(x=ser.index, y=ser.values, mode='lines', name='ICE BofA OAS for this rating mix', line=dict(color=T.BLUE, width=2.5))
    lvl = an.wavg(g, MET)
    fig.add_scatter(x=[ctx.asof], y=[lvl], mode='markers', name=f'{member} today ({ctx.label})', marker=dict(size=15, color=T.RED, symbol='diamond'))
    h = ctx.hist
    hl = h[(h['level'] == HIST_LEVEL.get(dim)) & (h['key'] == member)] if len(h) and dim in HIST_LEVEL else h.iloc[0:0]
    if hl['date'].nunique() >= 2:
        col = 'gspread' if ctx.has_spread else 'ytm'
        fig.add_scatter(x=hl['date'], y=hl[col] * (1 if ctx.has_spread else 100), mode='lines+markers', name=f'{member} (saved snapshots)', line=dict(color=T.RED, width=3))
    fig.update_yaxes(rangemode='nonnegative', title_text='bp')
    st.plotly_chart(T.style(fig, h=400, title=f'{member}: pricing vs. its benchmark path'), use_container_width=True)
    if len(ser.dropna()):
        cur = float(ser.dropna().iloc[-1])
        st.caption(f'Benchmark today {cur:.0f}bp ({bm.percentile_of(ser, cur):.0f}th percentile of its {len(ser.dropna()):,}-day history); {member} trades {lvl - cur:+.0f}bp from it. '
                   f'Saved snapshots: {h["date"].nunique() if len(h) else 0} - a line of your own history appears once there are two.')
    else:
        st.caption('Refresh benchmarks to draw the index path. Saved snapshots (your own history) appear here once there are two.')
    if ctx.bench.has_oas:
        T.section('Index OAS by rating', 'ICE BofA option-adjusted spreads')
        names = [c for c in ['IG', 'HY'] + bm.BUCKETS if c in ctx.bench.oas.columns]
        pick = st.multiselect('Series', names, default=[n for n in ['IG', 'HY', 'BBB', 'BB'] if n in names], key='sa_oas')
        if pick:
            fig = go.Figure()
            for n in pick:
                s = ctx.bench.oas[n].dropna()
                fig.add_scatter(x=s.index, y=s.values, mode='lines', name=n, line=dict(color=T.RATING_COLORS.get(n, T.NAVY if n == 'IG' else T.ORANGE), width=2))
            fig.update_xaxes(rangeslider_visible=True)
            fig.update_yaxes(rangemode='nonnegative', title_text='OAS (bp)')
            st.plotly_chart(T.style(fig, h=420, title='ICE BofA US corporate OAS history'), use_container_width=True)


# ------------------------------------------------------------------------------------------------ sector / industry story
def _sector_facts(s, z, d_ok, ctx, bb):
    out = s['Outstanding Amount'].sum()
    top5 = s.groupby('Ticker')['Outstanding Amount'].sum().nlargest(5).sum() / out * 100 if out else np.nan
    w = _weights(s)
    wk = s.groupby('Rating Bucket')['Outstanding Amount'].sum() / out * 100
    weak = float(wk[[b for b in ['BB', 'B', 'CCC'] if b in wk.index]].sum()) if len(wk) else 0.0
    lvl = an.wavg(s, MET)
    bench = _blend(w, bb)
    mid = s[s['Years to Maturity'].between(8, 12)][MET]
    short = s[s['Years to Maturity'].between(2, 4)][MET]
    slope = (mid.median() - short.median()) if len(mid) >= 5 and len(short) >= 5 else np.nan
    near = s.loc[s['Years to Maturity'] <= 3, 'Outstanding Amount'].sum() / out * 100 if out else np.nan
    disp = (s[MET].quantile(0.75) - s[MET].quantile(0.25)) if len(s) >= 8 else np.nan
    return dict(out=out / 1e9, issuers=s['Ticker'].nunique(), bonds=len(s), hy=s.loc[s['Universe'] == 'HY', 'Outstanding Amount'].sum() / out * 100 if out else np.nan, top5=top5,
                weak=weak, lvl=lvl, bench=bench, excess=lvl - bench if pd.notna(bench) else np.nan, slope=slope, near=near, life=an.wavg(s, 'Years to Maturity'), disp=disp, weights=w)


def _cards(Fs, ctx, ops):
    cards = []
    cards.append({'title': 'Size & structure', 'tone': 'info', 'headline': f'${Fs["out"]:,.0f}bn across {Fs["issuers"]} issuers',
                  'bullets': [f'{Fs["bonds"]} bonds; {Fs["hy"]:.0f}% of debt is high yield.', f'Top 5 issuers hold {Fs["top5"]:.0f}% of the debt - ' + ('concentrated.' if Fs['top5'] > 50 else 'diversified.')]})
    cards.append({'title': 'Credit quality', 'tone': 'warn' if Fs['weak'] > 40 else ('good' if Fs['weak'] < 10 else 'neutral'), 'headline': f'{Fs["weak"]:.0f}% rated BB or below',
                  'bullets': [', '.join(f'{b} {v * 100:.0f}%' for b, v in sorted(Fs['weights'].items(), key=lambda kv: T.RATING_ORDER.index(kv[0]))) or 'No rated bonds.']})
    if pd.notna(Fs['excess']):
        cards.append({'title': 'Pricing vs. benchmark', 'tone': 'warn' if Fs['excess'] > 75 else ('good' if Fs['excess'] < -25 else 'info'),
                      'headline': f'{Fs["lvl"]:.0f}bp vs. {Fs["bench"]:.0f}bp benchmark ({Fs["excess"]:+.0f}bp)',
                      'bullets': ['Benchmark = rating-bucket index levels weighted by this sector\'s own rating mix.',
                                  f'Bond-level spread dispersion (middle 50%): {Fs["disp"]:.0f}bp.' if pd.notna(Fs['disp']) else 'Too few bonds for a dispersion read.']})
    if pd.notna(Fs['slope']):
        cards.append({'title': 'Curve shape', 'tone': 'neutral', 'headline': f'8-12y trades {Fs["slope"]:+.0f}bp vs. 2-4y',
                      'bullets': ['Positive = upward-sloping credit curve (investors paid to extend); negative = inverted, often a sign of near-term stress.']})
    cards.append({'title': 'Refinancing wall', 'tone': 'warn' if Fs['near'] > 30 else 'neutral', 'headline': f'{Fs["near"]:.0f}% matures within 3 years',
                  'bullets': [f'Weighted average life {Fs["life"]:.1f} years.']})
    if ops:
        cards.append({'title': 'Operating picture (Capital IQ issuers)', 'tone': 'info', 'headline': f'{len(ops)} issuer(s) with financials loaded',
                      'bullets': [f'Median revenue growth {np.nanmedian([o["growth"] for o in ops]) * 100:.1f}%' if any(pd.notna(o['growth']) for o in ops) else 'Growth n/a',
                                  f'Median EBITDA margin {np.nanmedian([o["margin"] for o in ops]) * 100:.1f}%' if any(pd.notna(o['margin']) for o in ops) else 'Margin n/a',
                                  f'Median debt / EBITDA {np.nanmedian([o["lev"] for o in ops]):.1f}x' if any(pd.notna(o['lev']) for o in ops) else 'Leverage n/a']})
    return cards


def _ops(cos, tickers):
    out = []
    for t in tickers:
        c = cos.get(t)
        if c is None:
            continue
        ks = c.statements.get('key_stats')
        g, _ = S.last_actual(ks, 'Total Revenue Growth Over Prior Year')
        m, _ = S.last_actual(ks, 'EBITDA Margin %')
        lev, _ = c.cap_summary.latest('Total Debt/EBITDA', 'Credit Ratios') if c.cap_summary is not None else (np.nan, None)
        out.append({'ticker': t, 'name': c.name, 'growth': g, 'margin': m, 'lev': lev})
    return out


def _page_story(z, d_ok, ctx, capiq_dir, notes_dir, summ):
    sectors = summ['Sector'].tolist()
    fix_state('sa_sector', sectors)
    c1, c2 = st.columns(2)
    sector = c1.selectbox('Sector', sectors, key='sa_sector')
    s_all = z[z['Sector'] == sector]
    inds = ['All industries'] + sorted(i for i in s_all['Industry'].replace('', np.nan).dropna().unique())
    fix_state('sa_industry', inds)
    industry = c2.selectbox('Industry', inds, key='sa_industry')
    s = s_all if industry == 'All industries' else s_all[s_all['Industry'] == industry]
    if len(s) < 3:
        st.info('Too few bonds in this selection.')
        return
    bb = _bucket_bench(d_ok, ctx, 'index' if ctx.has_index else 'universe')
    Fs = _sector_facts(s, z, d_ok, ctx, bb)
    scope = sector if industry == 'All industries' else f'{sector} / {industry}'
    kind, key = ('sector', sector) if industry == 'All industries' else ('industry', scope)
    notes = S.load_notes(notes_dir, kind, key)
    cos = get_companies(capiq_dir)
    ops = _ops(cos, sorted(set(s['Ticker'])))
    head = f'{scope}: ${Fs["out"]:,.0f}bn across {Fs["issuers"]} issuers, {Fs["hy"]:.0f}% high yield' + (f'; trades {Fs["excess"]:+.0f}bp versus the index for its rating mix.' if pd.notna(Fs['excess']) else '.')
    chips = ''
    if notes.get('stance') in ('Positive', 'Negative', 'Neutral'):
        chips = T.chip(f'Analyst stance: {notes["stance"]}', {'Positive': 'good', 'Negative': 'bad', 'Neutral': 'neutral'}[notes['stance']])
    T.story_banner('Sector credit story' if industry == 'All industries' else 'Industry credit story', head, notes.get('thesis') or 'Add a thesis below to make this the headline of the story.', chips)
    cards = _cards(Fs, ctx, ops)
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, c_ in zip(cols, cards[i:i + 3]):
            with col:
                T.signal_card(c_['title'], c_['headline'], c_['bullets'], c_['tone'])
        st.write('')
    a, b = st.columns(2)
    if industry == 'All industries' and s['Industry'].replace('', np.nan).nunique() > 1:
        fig = px.treemap(s.assign(Industry=s['Industry'].replace('', 'Unclassified')), path=[px.Constant(sector), 'Industry', 'Ticker'], values='Outstanding Amount', color=MET,
                         color_continuous_scale=T.SEQUENTIAL)
        a.plotly_chart(T.style(fig, h=440, title='Industries and issuers (area = debt, colour = price of risk)', legend=False), use_container_width=True)
    else:
        by = s.groupby('Ticker')['Outstanding Amount'].sum().nlargest(12).reset_index()
        fig = go.Figure(go.Bar(y=by['Ticker'][::-1], x=by['Outstanding Amount'][::-1] / 1e9, orientation='h', marker_color=T.BLUE))
        a.plotly_chart(T.style(fig, h=440, title='Largest issuers ($bn)', legend=False), use_container_width=True)
    yr = s.assign(Year=pd.to_datetime(s['Maturity'], errors='coerce').dt.year).dropna(subset=['Year'])
    if not yr.empty:
        wall = yr.groupby(['Year', 'Rating Bucket'])['Outstanding Amount'].sum().reset_index()
        wall['bn'] = wall['Outstanding Amount'] / 1e9
        fig = px.bar(wall, x='Year', y='bn', color='Rating Bucket', color_discrete_map=T.RATING_COLORS, category_orders={'Rating Bucket': T.RATING_ORDER})
        b.plotly_chart(T.style(fig, h=440, title='Maturity wall by rating ($bn)'), use_container_width=True)
    c, d = st.columns(2)
    sc = s[s['Years to Maturity'].between(0.01, 60)]
    fig = px.scatter(sc, x='Years to Maturity', y=MET, color='Rating Bucket', color_discrete_map=T.RATING_COLORS, category_orders={'Rating Bucket': T.RATING_ORDER}, opacity=0.7,
                     hover_name='Label')
    for bk in [k for k in Fs['weights'] if bb.get(k) is not None and Fs['weights'][k] > 0.25]:
        fig.add_hline(y=bb[bk], line_dash='dot', line_color=T.RATING_COLORS.get(bk, T.GREY), annotation_text=f'{bk} benchmark {bb[bk]:.0f}bp', annotation_position='top right')
    fig.update_xaxes(rangemode='nonnegative', title_text='Years to maturity')
    fig.update_yaxes(title_text=ctx.label)
    c.plotly_chart(T.style(fig, h=440, title='Bond curve vs. rating benchmarks'), use_container_width=True)
    if ops:
        od = pd.DataFrame(ops).dropna(subset=['lev'])
        if len(od):
            fig = go.Figure(go.Bar(y=od['ticker'], x=od['lev'], orientation='h', marker_color=T.RED, text=[f'{v:.1f}x' for v in od['lev']], textposition='outside'))
            d.plotly_chart(T.style(fig, h=440, title='Leverage of issuers with Capital IQ data (debt / EBITDA)', legend=False), use_container_width=True)
    else:
        d.info('Load Capital IQ exports for issuers in this sector to see operating metrics (growth, margins, leverage) here.')
    T.section('Industry & management narrative', 'Your qualitative view - saved per sector / industry in data/stories')
    shown = False
    for k, label in S.SECTOR_FIELDS:
        if notes.get(k):
            shown = True
            with st.container(border=True):
                st.markdown(f'**{label}**')
                st.markdown(notes[k])
    with st.expander('Edit narrative' if shown else 'Write the narrative', expanded=not shown):
        with st.form(f'sector_notes_{kind}_{key}'):
            cur = notes.get('stance', 'Not set')
            stance = st.selectbox('Stance', S.STANCES, index=S.STANCES.index(cur) if cur in S.STANCES else 0)
            vals = {k: st.text_area(label, value=notes.get(k, ''), height=100) for k, label in S.SECTOR_FIELDS}
            if st.form_submit_button('Save'):
                S.save_notes(notes_dir, kind, key, dict(vals, stance=stance))
                st.success('Saved.')
                st.rerun()


# ------------------------------------------------------------------------------------------------ issuers
def _page_issuers(z, ctx, summ):
    sectors = summ['Sector'].tolist()
    fix_state('sa_iss_sector', sectors)
    sector = st.selectbox('Sector', sectors, key='sa_iss_sector')
    s = z[z['Sector'] == sector]
    it = an.issuer_table(s, MET)
    sw = an.wavg(s, MET)
    big = it.nlargest(60, 'Outstanding ($bn)')
    fig = px.scatter(big, x='Years', y='Metric', size='Outstanding ($bn)', color='Bucket', text='Ticker', hover_name='Name', size_max=50, color_discrete_map=T.RATING_COLORS,
                     category_orders={'Bucket': T.RATING_ORDER})
    fig.update_traces(textposition='top center', marker=dict(opacity=0.75, line=dict(width=0)))
    fig.add_hline(y=sw, line_dash='dash', line_color=T.GREY, annotation_text=f'{sector} average {sw:.0f}bp')
    fig.update_xaxes(rangemode='nonnegative', title_text='Weighted average life (years)')
    fig.update_yaxes(title_text=ctx.label)
    st.plotly_chart(T.style(fig, h=520, title=f'{sector} issuers: price of risk vs. tenor (bubble = debt outstanding)'), use_container_width=True)
    top = it.nlargest(25, 'Outstanding ($bn)').copy()
    top['vs'] = top['Metric'] - sw
    top = top.sort_values('vs')
    fig = go.Figure(go.Bar(y=top['Ticker'], x=top['vs'], orientation='h', marker_color=[T.RED if v > 0 else T.TEAL for v in top['vs']], text=[f'{v:+.0f}' for v in top['vs']], textposition='outside'))
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text=f'bp vs. {sector} average')
    st.plotly_chart(T.style(fig, h=640, title='The 25 largest issuers: rich (left) to cheap (right) versus the sector', legend=False), use_container_width=True)
    st.caption('Not adjusted for rating, maturity or seniority - a prompt to look, not a spread measure. Use Relative Value for peer-adjusted comparisons.')
    with st.expander('Issuer table'):
        st.dataframe(it.sort_values('Outstanding ($bn)', ascending=False), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------------------------ entry point
def render_sector_tab(f, d, ctx, capiq_dir, notes_dir, notch):
    if f is None or f.empty:
        st.info('No bonds match the current sidebar filters.')
        return
    cap = st.number_input('Hide yields above (%)', min_value=1.0, value=30.0, step=5.0, key='sa_cap', help='Stale or distressed yields distort sector averages. Raise to include them.')
    z = f.copy()
    for c in ['YTM', 'Years to Maturity', 'Outstanding Amount', 'Notch', MET]:
        z[c] = pd.to_numeric(z[c], errors='coerce')
    total = len(z)
    z = z[z['YTM'].between(0.001, cap) & (z['Outstanding Amount'] > 0) & z[MET].notna()].copy()
    z['Sector'] = z['Sector'].replace('', np.nan).fillna('Unclassified')
    z['Rating Bucket'] = z['Rating Bucket'].replace('', np.nan).fillna('NR')
    if z.empty:
        st.warning('No bonds left after removing missing yields and applying the yield cap.')
        return
    d_ok = d[d['YTM'].between(0.001, cap) & (d['Outstanding Amount'] > 0) & d[MET].notna()].copy()
    d_ok['Rating Bucket'] = d_ok['Rating Bucket'].replace('', np.nan).fillna('NR')
    summ = _summ(z, 'Sector')
    out = z['Outstanding Amount'].sum()
    T.kpi_row([('Sectors', f'{len(summ)}', '', 'neutral'), ('Bonds', f'{len(z):,}', f'of {total:,} in screen', 'neutral'), ('Debt outstanding', f'${out / 1e9:,.0f}bn', '', 'neutral'),
               (f'Weighted {ctx.label}', f'{an.wavg(z, MET):.0f}bp', f'as of {ctx.asof:%d %b %Y}', 'info'),
               ('High-yield share', f'{z.loc[z["Universe"] == "HY", "Outstanding Amount"].sum() / out * 100:.0f}%', 'of debt outstanding', 'warn')])
    page = nav(PAGES, 'sa_page')
    if page == 'Landscape':
        _page_landscape(z, ctx, summ)
    elif page == 'Benchmarks':
        _page_benchmarks(z, d_ok, ctx)
    elif page == 'Sector & industry story':
        _page_story(z, d_ok, ctx, capiq_dir, notes_dir, summ)
    else:
        _page_issuers(z, ctx, summ)
