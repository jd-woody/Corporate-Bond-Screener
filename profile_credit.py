"""profile_credit.py - Credit Profile pages: Credit health & comps, Ratios, Scenarios & recovery; plus the capital-stack and price-yield sections."""
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as an
import comps as C
import story as S
import theme as T
from context import MET
from ui_helpers import fix_state

QCOLOR = {1: T.TEAL, 2: '#7FD1C6', 3: '#F7D794', 4: T.RED}
QLABEL = {1: 'Top', 2: 'Above average', 3: 'Below average', 4: 'Bottom'}
QTONE = {'Top': 'good', 'Above Average': 'info', 'Below Average': 'warn', 'Bottom': 'bad'}
HEAT = [[0.0, T.TEAL], [0.33, '#A5DDD6'], [0.66, '#F7D794'], [1.0, T.RED]]


def _short(n, k=30):
    n = str(n)
    return n if len(n) <= k else n[:k - 1] + '...'


# ================================================================================================ credit health & comps
def _health_cards(chp, F):
    peers = chp['peers']
    sub = peers[peers['Is Subject']] if len(peers) else peers
    sp = sub['S&P Rating'].iloc[0] if len(sub) and pd.notna(sub['S&P Rating'].iloc[0]) else (F.get('ratings') or ['NR'])[0]
    items = [('Overall credit health', chp['overall'] or 'n/a', f'vs {len(peers)} public industry peers (quartile)', QTONE.get(chp['overall'], 'neutral'))]
    for name, score in chp['categories']:
        items.append((name, score, 'relative to peers', QTONE.get(score, 'neutral')))
    if pd.notna(chp['pd']):
        pct = (peers['PD (%)'].dropna() <= chp['pd']).mean() * 100 if len(peers) else np.nan
        items.append(('Market-signal default prob.', f'{chp["pd"]:.3f}%', f'{pct:.0f}th percentile of peers (lower is better)' if pd.notna(pct) else '', 'good' if pd.notna(pct) and pct <= 40 else ('warn' if pd.notna(pct) and pct >= 70 else 'info')))
    if chp['creditmodel']:
        cm, sn = C.notch_num(chp['creditmodel']), C.notch_num(sp)
        gap = (sn - cm) if pd.notna(cm) and pd.notna(sn) else np.nan
        sub_txt = f'S&P {sp}' + (f' ({gap:+.0f} notches vs model)' if pd.notna(gap) else '')
        items.append(('CreditModel score', chp['creditmodel'].upper(), sub_txt, 'info'))
    T.kpi_row(items)


def _peer_heatmap(chp):
    p = chp['peers'].copy()
    if p.empty:
        return
    p = p.sort_values(['Overall', 'Solvency', 'Operational', 'Liquidity', 'PD (%)']).reset_index(drop=True)
    labels = [('\u25b6 ' if s else '') + _short(n) for n, s in zip(p['Name'], p['Is Subject'])]
    cols = ['Overall', 'Operational', 'Solvency', 'Liquidity']
    z = p[cols].values.astype(float)
    text = [[('' if np.isnan(v) else str(int(v))) for v in row] for row in z]
    hover = [[f'{labels[i]}<br>{cols[j]}: {QLABEL.get(int(z[i][j]), "n/a") if not np.isnan(z[i][j]) else "n/a"}' for j in range(len(cols))] for i in range(len(p))]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=labels, text=text, texttemplate='%{text}', hovertext=hover, hoverinfo='text', colorscale=HEAT, zmin=1, zmax=4, showscale=False,
                               xgap=3, ygap=3))
    fig.update_yaxes(autorange='reversed', showgrid=False)
    fig.update_xaxes(side='top', showgrid=False)
    st.plotly_chart(T.style(fig, h=max(420, 27 * len(p) + 110), title='Peer credit-health scorecard (1 = top quartile, 4 = bottom; \u25b6 = this issuer)', legend=False), use_container_width=True)


def _pd_map(chp):
    p = chp['peers'].copy()
    p['x'] = p['CreditModel'].map(C.notch_num)
    p = p.dropna(subset=['x', 'PD (%)'])
    p = p[p['PD (%)'] > 0]
    if len(p) < 3:
        return
    fig = go.Figure()
    for q in [1, 2, 3, 4]:
        g = p[(p['Overall'] == q) & ~p['Is Subject']]
        if len(g):
            fig.add_scatter(x=g['x'], y=g['PD (%)'], mode='markers+text', text=[_short(n, 16) for n in g['Name']], textposition='top center', name=f'{QLABEL[q]} overall',
                            marker=dict(size=13, color=QCOLOR[q], opacity=0.85), hovertext=g['Name'], hoverinfo='text+y')
    me = p[p['Is Subject']]
    if len(me):
        fig.add_scatter(x=me['x'], y=me['PD (%)'], mode='markers+text', text=[_short(n, 16) for n in me['Name']], textposition='bottom center', name='This issuer',
                        marker=dict(size=22, color='rgba(0,0,0,0)', line=dict(color=T.INK, width=3)))
    ticks = list(range(1, 18))
    fig.update_xaxes(tickvals=ticks, ticktext=[C.NOTCH_LOWER[i - 1] for i in ticks], title_text='CreditModel score (better to the left)', range=[0.3, max(p['x'].max() + 1, 8)])
    fig.update_yaxes(type='log', title_text='Market-signal probability of default (%), log scale')
    st.plotly_chart(T.style(fig, h=470, title='Model rating vs. market-implied default risk'), use_container_width=True)


def _rating_gap(chp):
    p = chp['peers'].copy()
    p['cm'], p['sp'] = p['CreditModel'].map(C.notch_num), p['S&P Rating'].map(C.notch_num)
    p = p.dropna(subset=['cm', 'sp'])
    if len(p) < 3:
        return
    p['gap'] = p['sp'] - p['cm']
    p = p.sort_values('gap')
    fig = go.Figure(go.Bar(y=[_short(n, 28) for n in p['Name']], x=p['gap'], orientation='h', marker_color=[T.TEAL if v > 0 else (T.RED if v < 0 else T.GREY) for v in p['gap']],
                           text=[f'{v:+.0f}' for v in p['gap']], textposition='outside'))
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text='notches: S&P rating minus CreditModel (positive = model is more optimistic)')
    st.plotly_chart(T.style(fig, h=max(340, 26 * len(p) + 120), title='Where the model and the agency disagree', legend=False), use_container_width=True)


def _strengths(fin):
    last = fin['Period'].max()
    cur = fin[(fin['Period'] == last) & fin['Score'].notna()].copy()
    if cur.empty:
        return
    cur['strength'] = 5 - cur['Score']
    cur['label'] = cur['Category'].str.slice(0, 3) + ' | ' + cur['Metric'] + cur['Unit'].map(lambda u: f' ({u})' if u else '')
    cur = cur.sort_values(['Category', 'strength'], ascending=[False, True]) if False else cur.iloc[::-1]
    fig = go.Figure(go.Bar(y=cur['label'], x=cur['strength'], orientation='h', marker_color=[QCOLOR[int(s)] for s in cur['Score']],
                           text=[f'{c:,.2f} vs peers {g:,.2f}' for c, g in zip(cur['Company'], cur['Group Mean'])], textposition='inside', insidetextanchor='start',
                           hovertext=[f'{m}: quartile {int(s)} ({QLABEL[int(s)]}), weight {w}' for m, s, w in zip(cur['Metric'], cur['Score'], cur['Weight'])], hoverinfo='text'))
    fig.update_xaxes(tickvals=[1, 2, 3, 4], ticktext=['Bottom', 'Below avg', 'Above avg', 'Top'], range=[0, 4.2], title_text=f'position among peers, LTM to {last:%d %b %Y}')
    st.plotly_chart(T.style(fig, h=max(420, 25 * len(cur) + 120), title='Strengths and weaknesses versus industry peers (metric by metric)', legend=False), use_container_width=True)
    n_na = int(((fin['Period'] == last) & fin['Score'].isna()).sum())
    if n_na:
        st.caption(f'{n_na} metric(s) are not scored in the panel (not meaningful for this company, e.g. interest coverage when interest is not reported).')


def _metric_explorer(fin):
    names = list(dict.fromkeys(fin['Metric']))
    fix_state('cp_chp_metric', names)
    m = st.selectbox('Trend vs. peer-group mean', names, index=names.index('EBITDA Margin') if 'EBITDA Margin' in names else 0, key='cp_chp_metric')
    g = fin[fin['Metric'] == m].sort_values('Period')
    unit = g['Unit'].iloc[0]
    fig = go.Figure()
    if g['Company'].notna().any():
        fig.add_scatter(x=g['Period'], y=g['Company'], mode='lines+markers', name='This issuer', line=dict(color=T.BLUE, width=3.5), marker=dict(size=9))
    fig.add_scatter(x=g['Period'], y=g['Group Mean'], mode='lines+markers', name='Peer-group mean', line=dict(color=T.GREY, width=2.5, dash='dash'))
    fig.update_yaxes(ticksuffix={'%': '%', 'x': 'x', 'days': ' d'}.get(unit, ''))
    st.plotly_chart(T.style(fig, h=360, title=f'{m}: issuer vs. peers over five LTM periods'), use_container_width=True)
    if g['Company'].isna().all():
        st.caption('Capital IQ does not report this metric for the company (shown as "-" / NM in the panel); only the peer mean is plotted.')


def _market_check(chp, d_ok, ctx, pick, names):
    peers = chp['peers']
    mp = C.match_peers(peers, names)
    rows = []
    for _, r in peers.iterrows():
        t = mp.get(str(r['Ticker']))
        if t is None or pd.isna(r['PD (%)']) or r['PD (%)'] <= 0:
            continue
        g = d_ok[d_ok['Ticker'] == t]
        if len(g):
            rows.append({'Peer': r['Name'], 'Ticker': t, 'PD (%)': r['PD (%)'], 'Spread': an.wavg(g, MET), 'Debt ($bn)': g['Outstanding Amount'].sum() / 1e9, 'Overall': r['Overall'], 'Subject': bool(r['Is Subject'])})
    n_all = len(peers)
    if len(rows) < 3:
        st.caption(f'{len(rows)} of {n_all} panel peers have bonds in the screener (matched by ticker or company name), too few for a PD-vs-spread view.')
        return mp
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x='PD (%)', y='Spread', size='Debt ($bn)', color=df['Overall'].map(QLABEL), text='Ticker', size_max=45, hover_name='Peer',
                     color_discrete_map={QLABEL[k]: v for k, v in QCOLOR.items()})
    fig.update_traces(textposition='top center')
    me = df[df['Subject']]
    if len(me):
        fig.add_scatter(x=me['PD (%)'], y=me['Spread'], mode='markers', marker=dict(size=30, color='rgba(0,0,0,0)', line=dict(color=T.INK, width=3)), name='This issuer')
    fig.update_xaxes(type='log', title_text='Market-signal PD (%), log scale')
    fig.update_yaxes(rangemode='nonnegative', title_text=ctx.label)
    st.plotly_chart(T.style(fig, h=440, title=f'Do bond spreads reflect default risk? ({len(df)} of {n_all} peers have bonds in the screener)'), use_container_width=True)
    return mp


def _comparable_bonds(bonds, d_ok, ctx, pick, F, mp):
    """Template idea: the comparable bond of each peer, nearest in maturity to this issuer's reference bond."""
    if bonds is None or bonds.empty:
        return
    b = bonds[(bonds['YTM'] > 0) & (bonds['YTM'] <= 30) & (bonds['Years to Maturity'] > 0)]
    if b.empty:
        return
    mid = b[b['Years to Maturity'].between(3, 12)]
    ref = (mid if len(mid) else b).nlargest(1, 'Outstanding Amount').iloc[0]
    peer_t = sorted({t for t in (mp or {}).values() if t != pick})
    src = 'panel peers with bonds in the screener'
    if len(peer_t) < 2:
        sec = d_ok[(d_ok['Sector'] == F.get('sector')) & (d_ok['Rating Bucket'] == F.get('bucket')) & (d_ok['Ticker'] != pick)]
        peer_t = sec.groupby('Ticker')['Outstanding Amount'].sum().nlargest(12).index.tolist()
        src = f'largest {F.get("sector")} issuers in the same rating bucket'
    rows = []
    for t in peer_t:
        g = d_ok[(d_ok['Ticker'] == t) & (d_ok['Years to Maturity'] > 0)]
        if g.empty:
            continue
        big = g[g['Outstanding Amount'] >= 3e8]
        g = big if len(big) else g
        k = (g['Years to Maturity'] - ref['Years to Maturity']).abs().idxmin()
        r = g.loc[k]
        rows.append({'Ticker': t, 'Issuer': r['Name'], 'Bond': r['Label'], 'Years': r['Years to Maturity'], 'Price': r['Ask Price'], 'Yield (%)': r['YTM'], 'Spread': r[MET], 'Rating': r['Rating'], 'Size ($mm)': r['Outstanding Amount'] / 1e6})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig = px.scatter(df, x='Years', y='Spread', text='Ticker', size='Size ($mm)', color_discrete_sequence=[T.BLUE], size_max=34, hover_name='Bond')
    fig.update_traces(textposition='top center')
    fig.add_scatter(x=[ref['Years to Maturity']], y=[ref[MET]], mode='markers+text', text=[pick], textposition='bottom center', name='This issuer',
                    marker=dict(size=24, color=T.RED, symbol='diamond'))
    fig.add_hline(y=float(df['Spread'].median()), line_dash='dot', line_color=T.GREY, annotation_text=f'comp median {df["Spread"].median():.0f}', annotation_position='top right')
    fig.update_xaxes(rangemode='nonnegative', title_text='Years to maturity')
    fig.update_yaxes(title_text=ctx.label)
    st.plotly_chart(T.style(fig, h=440, title=f'Comparable debt: each peer\'s bond nearest {ref["Years to Maturity"]:.1f} years ({src})', legend=False), use_container_width=True)
    with st.expander('Comparable bonds table (price, yield, spread, rating)'):
        st.dataframe(pd.concat([pd.DataFrame([{'Ticker': pick, 'Issuer': ref['Name'], 'Bond': ref['Label'], 'Years': ref['Years to Maturity'], 'Price': ref['Ask Price'], 'Yield (%)': ref['YTM'],
                                               'Spread': ref[MET], 'Rating': ref['Rating'], 'Size ($mm)': ref['Outstanding Amount'] / 1e6}]), df], ignore_index=True),
                     hide_index=True, use_container_width=True)


COMP_METRICS = ['Revenue growth (%)', 'EBITDA margin (%)', 'FCF margin (%)', 'Debt / EBITDA (x)', 'Net debt / EBITDA (x)', 'EBITDA / interest (x)', 'TEV / EBITDA (x)', 'Debt / equity (%)']


def _capiq_comps(cos, pick):
    usable = {t: c for t, c in cos.items() if c.cap_summary is not None or c.statements.get('key_stats') is not None}
    if len(usable) < 2:
        st.caption('Credit comps from your own exports need at least two issuers with Capital IQ data in data/capiq. Add peers\' Financials workbooks (or load the Credit Health Panel above for a ready-made peer set).')
        return
    opts = sorted(usable)
    fix_state('cp_comps_sel', opts, multi=True)
    sel = st.multiselect('Issuers in the comp set', opts, default=opts[:10], key='cp_comps_sel', format_func=lambda t: f'{usable[t].name} ({t})')
    if pick not in sel and pick in usable:
        sel = [pick] + sel
    if len(sel) < 2:
        return
    df = pd.DataFrame([C.comps_row(usable[t]) for t in sel])
    show = [m for m in COMP_METRICS if df[m].notna().sum() >= 1]
    if not show:
        return
    ncol = 4
    nrow = math.ceil(len(show) / ncol)
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=show, vertical_spacing=0.18, horizontal_spacing=0.07)
    for i, m in enumerate(show):
        r, c = i // ncol + 1, i % ncol + 1
        d = df.dropna(subset=[m]).sort_values(m)
        fig.add_trace(go.Bar(y=d['Ticker'], x=d[m], orientation='h', marker_color=[T.RED if t == pick else T.BLUE for t in d['Ticker']], showlegend=False,
                             text=[f'{v:,.1f}' for v in d[m]], textposition='outside'), row=r, col=c)
        if len(d) >= 3:
            fig.add_vline(x=float(d[m].median()), line_dash='dash', line_color=T.GREY, row=r, col=c)
    st.plotly_chart(T.style(fig, h=300 * nrow + 60, title='Credit comps: this issuer (red) vs. the comp set (dashed = median)', legend=False), use_container_width=True)
    num = [c for c in df.columns if c not in ('Ticker', 'Issuer')]
    peers = df[df['Ticker'] != pick]
    stats = pd.DataFrame([{'Ticker': 'Median', 'Issuer': '', **peers[num].median().to_dict()}, {'Ticker': 'Mean', 'Issuer': '', **peers[num].mean().to_dict()}])
    with st.expander('Comps table: peers, median / mean, then this issuer (template layout)'):
        st.dataframe(pd.concat([peers, stats, df[df['Ticker'] == pick]], ignore_index=True), hide_index=True, use_container_width=True)


def page_health(F, co, bonds, d, d_ok, cos, ctx, pick, names):
    chp = co.statements.get('chp_summary') if co is not None else None
    fin = co.statements.get('chp_financials') if co is not None else None
    mp = {}
    if chp is None:
        st.info('Add the Capital IQ **Credit Health Panel** export (its Summary and Financials sheets) to see peer credit-health scores, default probability and metric-by-metric comparison. '
                'Credit comps from your other exports and the comparable-bond view below still work.')
    else:
        _health_cards(chp, F)
        c1, c2 = st.columns([1, 1])
        with c1:
            _peer_heatmap(chp)
        with c2:
            _pd_map(chp)
        if fin is not None:
            T.section('Metric by metric', 'Where the issuer is strongest and weakest versus its public industry peers')
            _strengths(fin)
            _metric_explorer(fin)
        T.section('Market cross-check', 'Does the bond market agree with the model?')
        _rating_gap(chp)
        mp = _market_check(chp, d_ok, ctx, pick, names) or {}
    T.section('Comparable debt and credit comps', 'Template-style comps: peer bonds near the same maturity, and peer financials with median / mean')
    _comparable_bonds(bonds, d_ok, ctx, pick, F, mp)
    _capiq_comps(cos, pick)
    st.caption('The Credit Health Panel scores each metric by quartile against the panel\'s public industry peer list (converted to USD at the panel\'s spot rate). '
               'PD is the panel\'s market-signal probability of default; CreditModel is S&P\'s model-based rating estimate - both are indicators, not ratings.')


# ================================================================================================ ratios
KEY_RATIOS = ['Total debt / EBITDA', 'Net debt / EBITDA', 'EBITDA / interest', 'FCF / debt', 'CFO / debt', 'Debt / capital', 'Current ratio', 'Cash conversion cycle (days)',
              'EBITDA margin', 'FCF margin', 'Pre-tax return on invested capital', 'Interest burden (% revenue)']
UNIT_SFX = {'x': 'x', '%': '%', 'days': ' d', '$mm': ''}


def _fmt(v, unit):
    return 'n/a' if pd.isna(v) else (f'{v:,.0f}' if unit == '$mm' else f'{v:,.1f}{UNIT_SFX.get(unit, "")}')


def page_ratios(co):
    rf = C.ratio_frame(co) if co is not None else None
    if rf is None or rf.empty:
        st.info('Add the Income Statement, Balance Sheet and Cash Flow exports (or the full Financials workbook) to compute the credit-ratio toolkit.')
        return
    meta = C.RATIO_META.set_index('Ratio')
    avail = [k for k in KEY_RATIOS if k in rf.index and rf.loc[k].notna().sum() >= 2]
    if avail:
        nrow = math.ceil(len(avail) / 4)
        titles = []
        for k in avail:
            s = rf.loc[k].dropna()
            titles.append(f'{k}: {_fmt(s.iloc[-1], meta.at[k, "Unit"])}')
        fig = make_subplots(rows=nrow, cols=4, subplot_titles=titles, vertical_spacing=0.2, horizontal_spacing=0.06)
        for i, k in enumerate(avail):
            s = rf.loc[k].dropna()
            ref = s.iloc[max(-3, -len(s))]
            better = meta.at[k, 'Better']
            move = s.iloc[-1] - ref
            good = (move < 0) if better == 'lower' else (move > 0)
            color = T.GREY if abs(move) < 1e-9 else (T.GREEN if good else T.RED)
            fig.add_trace(go.Scatter(x=list(s.index), y=list(s.values), mode='lines+markers', line=dict(color=color, width=3), marker=dict(size=7), showlegend=False,
                                     hovertemplate='%{x}: %{y:,.2f}<extra></extra>'), row=i // 4 + 1, col=i % 4 + 1)
        fig.update_xaxes(showticklabels=False)
        fig.update_annotations(font_size=12)
        st.plotly_chart(T.style(fig, h=250 * nrow + 60, title='Credit dashboard: 12 ratios at a glance (green = improving, red = deteriorating vs. two periods earlier)', legend=False), use_container_width=True)
    fam = st.radio('Ratio family', list(dict.fromkeys(C.RATIO_META['Family'])), horizontal=True, key='cp_ratio_family')
    names = [r for r in C.RATIO_META.loc[C.RATIO_META['Family'] == fam, 'Ratio'] if r in rf.index]
    if not names:
        st.caption('No ratios in this family could be computed from the loaded exports.')
        return
    fix_state('cp_ratio_pick', names, multi=True)
    pick = st.multiselect('Ratios to chart', names, default=names[:3], key='cp_ratio_pick')
    if pick:
        long = rf.loc[pick].reset_index().melt(id_vars='index', var_name='Period', value_name='Value').dropna()
        long = long.rename(columns={'index': 'Ratio'})
        fig = px.line(long, x='Period', y='Value', color='Ratio', markers=True, color_discrete_sequence=T.CAT, category_orders={'Period': list(rf.columns)})
        st.plotly_chart(T.style(fig, h=400, title=f'{fam} ratios over time (units as shown in the glossary)'), use_container_width=True)
    with st.expander('Ratio glossary: formula, unit, which direction is better, latest value'):
        g = C.RATIO_META.copy()
        g['Latest'] = [rf.loc[r].dropna().iloc[-1] if r in rf.index and rf.loc[r].notna().any() else np.nan for r in g['Ratio']]
        st.dataframe(g, hide_index=True, use_container_width=True)
    st.caption('Ratios are computed here from the loaded statements (period-end balances, same-period flows). Capital IQ\'s own pages use their own definitions, so a figure such as '
               'total debt / EBITDA can differ from the Capital Structure Summary or Ratios page. Percent ratios are shown in %; "n/a" = an input is missing or the denominator is zero.')


# ================================================================================================ scenarios & recovery
def _start(F, co):
    inc, cf = co.statements.get('income'), co.statements.get('cashflow')
    debt, ebitda, cash = F.get('debt'), F.get('ebitda'), F.get('cash')
    if pd.isna(debt) or pd.isna(ebitda) or ebitda <= 0:
        return None
    intexp = abs(S.last_actual(inc, 'Interest Expense')[0]) if inc is not None else np.nan
    rate = intexp / debt if pd.notna(intexp) and debt else 0.05
    tax = S.last_actual(inc, 'Income Tax Expense')[0]
    ebt = S.last_actual(inc, 'EBT Incl. Unusual Items')[0]
    tr = float(np.clip(tax / ebt, 0.0, 0.35)) if pd.notna(tax) and pd.notna(ebt) and ebt > 0 else 0.20
    ret = abs(F.get('div') or 0) + abs(F.get('buyback') or 0)
    ebit = S.last_actual(inc, 'EBIT')[0] if inc is not None else np.nan
    return dict(revenue=F.get('rev'), ebitda=ebitda, debt=debt, cash=cash or 0.0, interest_rate=float(np.clip(rate, 0.01, 0.15)), tax_rate=tr, shareholder_returns=ret,
                da=ebitda - ebit if pd.notna(ebit) else 0.0)


def _forecast_section(F, co):
    st_ = _start(F, co)
    street = C.street_path(co)
    if st_ is None or street.empty or street['ebitda'].isna().all():
        st.info('The forecast needs Capital IQ Key Stats with analyst estimates (columns marked E), plus the debt and EBITDA from the Capital Structure Summary.')
        return
    n = len(street)
    ltm_capex = -S.last_actual(co.statements.get('cashflow'), 'Capital Expenditure')[0] / st_['revenue'] * 100 if pd.notna(F.get('capex')) and st_['revenue'] else 20.0
    c = st.columns(5)
    cons = c[0].slider('Conservative: EBITDA vs. Street by final year (%)', -50, 0, -15, key='cp_sc_cons')
    opt = c[1].slider('Optimistic: EBITDA vs. Street by final year (%)', 0, 50, 10, key='cp_sc_opt')
    capex1 = c[2].slider('Capex % of revenue, year 1', 0, 150, int(round(min(max(ltm_capex, 0), 150))), key='cp_sc_capex')
    step = c[3].slider('Annual change in capex % (pts)', -40, 20, 0, key='cp_sc_step')
    radd = c[4].slider('Conservative: higher borrowing cost (bp)', 0, 400, 100, key='cp_sc_rate')
    capex_path = [max(capex1 + step * t, 0) for t in range(n)]
    ramp = lambda final: [final / 100.0 * (t + 1) / n for t in range(n)]
    scen = {'Conservative': dict(ebitda_shock=ramp(cons), capex_pct=capex_path, rate_add_bp=radd), 'Base (Street)': dict(ebitda_shock=[0.0], capex_pct=capex_path),
            'Optimistic': dict(ebitda_shock=ramp(opt), capex_pct=capex_path, rate_add_bp=-50)}
    res = {k: C.project(st_, street, v) for k, v in scen.items()}
    colors = {'Conservative': T.RED, 'Base (Street)': T.BLUE, 'Optimistic': T.GREEN}
    periods = list(street['period'])
    hist = S.series(co.statements.get('key_stats'), 'EBITDA')
    a, b = st.columns(2)
    fig = go.Figure()
    if hist is not None:
        h = hist[[p for p in hist.index if not str(p).endswith('E') and not str(p).startswith('LTM') and pd.notna(hist[p])]]
        fig.add_scatter(x=list(h.index), y=list(h.values), mode='lines+markers', name='Actual', line=dict(color=T.INK, width=3))
        anchor = h.index[-1] if len(h) else None
    else:
        anchor = None
    for k, df in res.items():
        xs = ([anchor] if anchor else []) + periods
        ys = ([float(h.iloc[-1])] if anchor else []) + list(df['ebitda'])
        fig.add_scatter(x=xs, y=ys, mode='lines+markers', name=k, line=dict(color=colors[k], width=3, dash='solid' if k.startswith('Base') else 'dot'))
    a.plotly_chart(T.style(fig, h=380, title='EBITDA: history and three scenarios ($mm)'), use_container_width=True)
    fig = go.Figure([go.Bar(x=periods, y=df['fcf_after'], name=k, marker_color=colors[k]) for k, df in res.items()])
    fig.update_layout(barmode='group')
    b.plotly_chart(T.style(fig, h=380, title='Free cash flow after shareholder returns ($mm) - negative = funded by new debt'), use_container_width=True)
    d1, d2, d3 = st.columns(3)
    fig = go.Figure([go.Bar(x=['Today'] + periods, y=[st_['debt']] + list(df['debt']), name=k, marker_color=colors[k]) for k, df in res.items()])
    fig.update_layout(barmode='group')
    d1.plotly_chart(T.style(fig, h=340, title='Total debt path ($mm)'), use_container_width=True)
    fig = go.Figure([go.Scatter(x=['Today'] + periods, y=[st_['debt'] / st_['ebitda']] + list(df['leverage']), mode='lines+markers', name=k, line=dict(color=colors[k], width=3)) for k, df in res.items()])
    fig.update_yaxes(ticksuffix='x')
    d2.plotly_chart(T.style(fig, h=340, title='Debt / EBITDA'), use_container_width=True)
    fig = go.Figure([go.Scatter(x=periods, y=df['coverage'], mode='lines+markers', name=k, line=dict(color=colors[k], width=3)) for k, df in res.items()])
    fig.update_yaxes(ticksuffix='x')
    d3.plotly_chart(T.style(fig, h=340, title='EBITDA / interest'), use_container_width=True)
    with st.expander('Assumptions and scenario tables'):
        st.write(f'Start: debt {S.mm(st_["debt"])}, LTM EBITDA {S.mm(st_["ebitda"])}, interest rate {st_["interest_rate"] * 100:.2f}% (LTM interest / debt), tax rate {st_["tax_rate"] * 100:.0f}%, '
                 f'shareholder returns {S.mm(st_["shareholder_returns"])} a year, D&A from Street EBITDA - EBIT. Revenue and the Base EBITDA are the Capital IQ consensus estimates.')
        for k, df in res.items():
            st.markdown(f'**{k}**')
            st.dataframe(df.round(1), hide_index=True, use_container_width=True)
    st.caption('Mechanics: FCF = EBITDA - capex - interest - cash taxes (on EBIT less interest); shortfalls are borrowed and surpluses repay debt at the same rate; working capital is held flat. '
               'Illustrative - a way to see how sensitive leverage is to the investment programme, not a forecast.')


def _recovery_section(F, co, bonds):
    if bonds is None or bonds.empty:
        st.info('No bonds for this issuer in the screener, so there is no capital structure to run a recovery waterfall on.')
        return
    ebitda = F.get('ebitda')
    stack = C.cap_stack(bonds, F.get('debt'), ebitda, F.get('tev'))
    if stack.empty:
        return
    cur_mult = None
    ks = co.statements.get('key_stats') if co is not None else None
    cm = ks.extra.get('current_multiples') if ks is not None else None
    if cm is not None and cm.row('TEV/EBITDA') is not None:
        v = cm.values.loc[cm.row('TEV/EBITDA'), cm.periods].astype(float).dropna()
        cur_mult = float(v.iloc[0]) if len(v) else None
    c = st.columns(5)
    incl_cash = c[4].checkbox('Add balance-sheet cash to value', value=False, key='cp_rec_cash', help='Cash is value creditors can claim, but a business also needs some to operate. Off = conservative.')
    base_mult = c[0].slider('Base-case EV multiple (x EBITDA)', 1.0, 20.0, float(round(min(max((cur_mult or 10.0) * 0.6, 3.0), 15.0), 1)), 0.5, key='cp_rec_mult')
    haircut = c[1].slider('EBITDA in the recovery case vs. LTM (%)', -80, 0, -20, key='cp_rec_hc')
    cost = c[2].slider('Costs and leakage (% of EV)', 0, 30, 10, key='cp_rec_cost')
    other_rank = c[3].radio('Other debt ranks', ['With senior unsecured', 'Ahead (secured)'], key='cp_rec_other', horizontal=False)
    if pd.isna(ebitda) or ebitda <= 0:
        base_ev = float(stack['Face ($mm)'].sum()) * 0.7
        st.caption('No EBITDA loaded: scenario EV is set from 70% of debt face for illustration.')
        evs = {k: base_ev * m for k, m in [('Stress', .5), ('Low', .75), ('Base', 1.0), ('High', 1.25), ('Strong', 1.5)]}
    else:
        eb = ebitda * (1 + haircut / 100.0)
        evs = {k: eb * base_mult * m for k, m in [('Stress', .5), ('Low', .75), ('Base', 1.0), ('High', 1.25), ('Strong', 1.5)]}
    probs = {'Stress': 10, 'Low': 20, 'Base': 40, 'High': 20, 'Strong': 10}
    ranks = {'Secured': 1, 'Senior unsecured': 3, 'Subordinated': 4}
    tr = []
    for _, r in stack.iterrows():
        nm = r['Tranche']
        rk = ranks.get(nm, 3 if other_rank.startswith('With') else 1)
        claim = r['Face ($mm)'] * (1 + (r['Coupon (%)'] / 100 * 0.5 if pd.notna(r['Coupon (%)']) else 0.0))
        tr.append((nm, rk, float(claim)))
    mat, expected, fulcrum = C.recovery_matrix(tr, evs, cost, probs, extra=float(F.get('cash') or 0.0) if incl_cash else 0.0)
    fig = go.Figure(go.Heatmap(z=(mat * 100).values, x=[f'{k}<br>EV {S.mm(v)}' for k, v in evs.items()], y=list(mat.index), text=[[f'{v * 100:.0f}%' for v in row] for row in mat.values],
                               texttemplate='%{text}', colorscale=[[0, T.RED], [0.5, '#F7D794'], [1, T.TEAL]], zmin=0, zmax=100, showscale=False, xgap=3, ygap=3))
    fig.update_yaxes(autorange='reversed')
    a, b = st.columns([3, 2])
    a.plotly_chart(T.style(fig, h=60 * len(mat) + 190, title='Recovery by tranche and scenario (% of claim)', legend=False), use_container_width=True)
    price = stack.set_index('Tranche')['Price']
    fig = go.Figure()
    fig.add_bar(x=list(expected.index), y=list(expected.values * 100), name='Model expected recovery', marker_color=T.BLUE)
    fig.add_scatter(x=list(price.index), y=list(price.values), mode='markers', name='Market price (per 100)', marker=dict(size=16, color=T.ORANGE, symbol='diamond'))
    fig.update_yaxes(title_text='% of par', range=[0, 110])
    b.plotly_chart(T.style(fig, h=60 * len(mat) + 190, title='Model recovery vs. what the market pays'), use_container_width=True)
    st.markdown('**Fulcrum security** (first tranche not paid in full): ' + '; '.join(f'{k}: {v}' for k, v in fulcrum.items()))
    with st.expander('Capital stack used in the waterfall'):
        st.dataframe(stack.round(2), hide_index=True, use_container_width=True)
    st.caption('A stylised waterfall: EV = LTM EBITDA x (1 + haircut) x multiple, less costs; paid in order of seniority, pro rata within a rank; claims = face plus half a year of coupon. '
               'Probabilities 10 / 20 / 40 / 20 / 10% across Stress ... Strong. * Other debt (loans, leases, CP) is the gap between Capital IQ total debt and the bonds in the screener; its ranking is unknown, so you choose. '
               'For a strong credit the answer is simply "everyone is covered" - the chart is most informative for leveraged or deteriorating issuers.')


def page_scenarios(F, co, bonds):
    if co is None:
        st.info('Scenarios need Capital IQ exports for this issuer (Key Stats with estimates, Capital Structure Summary, Cash Flow).')
        if bonds is not None:
            T.section('Recovery waterfall', 'Who gets paid, in what order, in a downside')
            _recovery_section(F, co, bonds)
        return
    T.section('Forecast scenarios', 'Conservative / Base (Street) / Optimistic: what do the investment programme and the cost of debt do to leverage?')
    _forecast_section(F, co)
    T.section('Recovery waterfall', 'Who gets paid, in what order, in a downside - and where the fulcrum security sits')
    _recovery_section(F, co, bonds)


# ================================================================================================ capital stack and price-yield sections
def capstack_section(F, co, bonds):
    """Capital-structure table with cumulative leverage and loan-to-value through each tranche."""
    if bonds is None or bonds.empty:
        return
    ebitda, tev = F.get('ebitda'), F.get('tev')
    stack = C.cap_stack(bonds, F.get('debt'), ebitda, tev)
    if stack.empty:
        return
    T.section('Capital stack', 'Debt in order of seniority: how far up the stack the company\'s earnings and enterprise value reach')
    colors = {'Secured': T.ORANGE, 'Senior unsecured': T.BLUE, 'Subordinated': T.PURPLE}
    fig = go.Figure()
    for _, r in stack.iterrows():
        txt = f'{r["Face ($mm)"]:,.0f}mm'
        if pd.notna(r['Cum. leverage (x EBITDA)']):
            txt += f'  |  {r["Cum. leverage (x EBITDA)"]:.1f}x EBITDA'
        if pd.notna(r['Loan-to-value (% TEV)']):
            txt += f'  |  LTV {r["Loan-to-value (% TEV)"]:.0f}%'
        fig.add_bar(y=[r['Tranche']], x=[r['Face ($mm)']], base=[r['Start ($mm)']], orientation='h', marker_color=colors.get(r['Tranche'], T.GREY), text=[txt],
                    textposition='inside', insidetextanchor='start', showlegend=False)
    total = float(stack['Face ($mm)'].sum())
    if pd.notna(tev) and tev > total:
        fig.add_bar(y=['Equity cushion (TEV less debt)'], x=[tev - total], base=[total], orientation='h', marker_color='rgba(19,168,158,.35)',
                    text=[f'{tev - total:,.0f}mm  |  TEV {S.mm(tev)}'], textposition='inside', insidetextanchor='start', showlegend=False)
    elif pd.notna(tev):
        fig.add_vline(x=tev, line_dash='dash', line_color=T.RED, annotation_text=f'TEV {S.mm(tev)} - debt exceeds enterprise value')
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(title_text='$mm, cumulative from the top of the stack', rangemode='nonnegative')
    st.plotly_chart(T.style(fig, h=60 * (len(stack) + 1) + 130, title='Capital stack with leverage and loan-to-value through each tranche', legend=False), use_container_width=True)
    show = stack[['Tranche', 'Face ($mm)', 'Price', 'Coupon (%)', 'YTM (%)', 'Interest ($mm)', 'Cum. debt ($mm)', 'Cum. leverage (x EBITDA)', 'Loan-to-value (% TEV)', 'Market value ($mm)']]
    with st.expander('Capitalization table (face, price, yield, interest, leverage and LTV through each tranche)'):
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)
        st.caption('Face and price are from the bond screener; * other debt = Capital IQ total debt less bonds in the screener (ranking unknown). Leverage and LTV use LTM EBITDA and the enterprise value in Capital IQ Key Stats.')


def price_yield_section(bonds, F):
    """Template idea (Convexity sheet): price-yield curve with the duration and duration+convexity approximations."""
    if bonds is None or bonds.empty:
        return
    b = bonds[(bonds['YTM'] > 0.1) & (bonds['Years to Maturity'] > 0.25) & bonds['Coupon'].notna()].nlargest(30, 'Outstanding Amount')
    if b.empty:
        return
    T.section('Rate sensitivity', 'What a move in yields does to the price - and what convexity is worth')
    opts = b.index.tolist()
    lbl = (b['Label'] + ' | ' + b['ISIN'].fillna('')).to_dict()
    fix_state('cp_py_bond', opts)
    i = st.selectbox('Bond (30 largest)', opts, format_func=lambda k: lbl[k], key='cp_py_bond')
    r = b.loc[i]
    df, dur, cvx, p0 = C.price_yield(float(r['Coupon']), float(r['YTM']), float(r['Years to Maturity']))
    fig = go.Figure()
    fig.add_scatter(x=df['shift_bp'], y=df['price'], mode='lines', name='Actual price', line=dict(color=T.INK, width=4))
    fig.add_scatter(x=df['shift_bp'], y=df['duration_only'], mode='lines', name='Duration only (straight line)', line=dict(color=T.GREY, dash='dash'))
    fig.add_scatter(x=df['shift_bp'], y=df['duration_convexity'], mode='lines', name='Duration + convexity', line=dict(color=T.TEAL, dash='dot', width=3))
    fig.add_scatter(x=[0], y=[p0], mode='markers', name='Model price today', marker=dict(size=14, color=T.RED))
    fig.update_xaxes(title_text='parallel yield shift (bp)')
    fig.update_yaxes(title_text='price per 100')
    st.plotly_chart(T.style(fig, h=420, title=f'Price-yield profile: {r["Label"]}'), use_container_width=True)
    up, dn = df[df['shift_bp'] == -100], df[df['shift_bp'] == 100]
    if len(up) and len(dn):
        gain = float(up['price'].iloc[0] - up['duration_only'].iloc[0]) + float(dn['price'].iloc[0] - dn['duration_only'].iloc[0])
        st.caption(f'Model duration {dur:.2f} and convexity {cvx:.0f}. The straight line (duration) understates the price when yields fall and overstates the loss when they rise; '
                   f'for a +/-100bp move convexity is worth about {gain / 2:.2f} points on average. Callable bonds give much of this away: this plain-bond model ignores call options.')
