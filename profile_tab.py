"""profile_tab.py - Credit Profile: the issuer hub. Story first, then capital structure, fundamentals, valuation & segments, market & peers, notes, data."""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as an
import profile_credit as PC
import story as S
import theme as T
from capiq_tab import fundamentals_row, get_companies, render_capiq_tab
from context import MET
from ui_helpers import fix_state, nav

PAGES = ['Story', 'Credit health & comps', 'Capital structure', 'Fundamentals', 'Ratios', 'Scenarios', 'Valuation & segments', 'Market & peers', 'Notes', 'Data & exports']
EST = '#A9C8FF'


def _norm(p):
    return re.sub(r'^(FY|LTM|\d+M) ', '', str(p))


def _colors(periods, base=T.BLUE):
    return [EST if str(p).endswith('E') else base for p in periods]


def _combo(title, periods, bars, line=None, h=330, line_pct=False, line_title=''):
    """Bars (list of (name, values, colour-or-None)) with an optional line on a secondary axis."""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    for name, vals, color in bars:
        fig.add_bar(x=list(periods), y=list(vals), name=name, marker_color=color or _colors(periods), secondary_y=False)
    if line is not None:
        name, vals = line
        fig.add_scatter(x=list(periods), y=list(vals), name=name, mode='lines+markers', line=dict(color=T.AMBER, width=3), marker=dict(size=7), secondary_y=True)
        fig.update_yaxes(ticksuffix='%' if line_pct else 'x', showgrid=False, title_text=line_title, secondary_y=True)
    fig.update_layout(barmode='group')
    return T.style(fig, h=h, title=title)


def _bench_for(bucket, d_ok, ctx):
    """Benchmark for a rating bucket: ICE BofA OAS (latest) if loaded, else the screener's own weighted average for that bucket."""
    if ctx.has_index and bucket in ctx.bench.oas.columns:
        s = ctx.bench.oas[bucket].dropna()
        if len(s):
            return float(s.iloc[-1]), 'ICE BofA OAS'
    ref = d_ok[d_ok['Rating Bucket'] == bucket]
    return (an.wavg(ref, MET), 'screener universe') if len(ref) else (np.nan, '')


def _kpis(F, ctx):
    items = []
    if F.get('has_capiq') and pd.notna(F.get('debt', np.nan)):
        nd = F.get('net_debt', np.nan)
        items.append(('Total debt', S.mm(F['debt']), F.get('debt_p') or '', 'neutral'))
        items.append(('Net debt' if pd.isna(nd) or nd >= 0 else 'Net cash', S.mm(abs(nd)) if pd.notna(nd) else 'n/a', 'debt less cash & ST investments', 'warn' if pd.notna(nd) and nd > 0 else 'good'))
        if pd.notna(F.get('lev', np.nan)):
            items.append(('Debt / EBITDA', f'{F["lev"]:.2f}x', F.get('lev_p') or '', 'neutral'))
        if pd.notna(F.get('fcf', np.nan)):
            items.append(('Free cash flow', S.mm(F['fcf']), 'CFO + capex, ' + str(F.get('cf_p')), 'bad' if F['fcf'] < 0 else 'good'))
        if pd.notna(F.get('due1', np.nan)) and pd.notna(F.get('cash', np.nan)) and F['due1']:
            c = F['cash'] / F['due1']
            items.append(('Cash vs 12m maturities', f'{c:.1f}x', f'{S.mm(F["cash"])} vs {S.mm(F["due1"])}', 'good' if c >= 1.5 else 'warn'))
    if 'n_bonds' in F:
        items.append(('Priced at', f'{F["spread"]:.0f}bp' if pd.notna(F['spread']) else 'n/a', f'{ctx.label}; yield {F["ytm"]:.2f}%', 'info'))
        items.append(('Bonds outstanding', S.mm(F['outstanding']), f'{F["n_bonds"]} bonds, avg life {F["maturity"]:.1f}y', 'neutral'))
    if items:
        T.kpi_row(items)


# ------------------------------------------------------------------------------------------------ story page
def _arc(co):
    ks, cf, bs, cs, rt = co.statements.get('key_stats'), co.statements.get('cashflow'), co.statements.get('balance'), co.cap_summary, co.ratios
    a, b = st.columns(2)
    rev = S.series(ks, 'Total Revenue')
    if rev is not None:
        mg = S.series(ks, 'EBITDA Margin %')
        with a:
            st.plotly_chart(_combo('Revenue ($mm) and EBITDA margin - light bars are estimates', rev.index, [('Revenue', rev.values, None)],
                                   line=('EBITDA margin', (mg.reindex(rev.index) * 100).values) if mg is not None else None, line_pct=True),
                            use_container_width=True)
    debt = S.series(bs, 'Total Debt') if bs is not None else None
    lev = S.series(rt, 'Total Debt/EBITDA')
    if debt is None and cs is not None:
        debt = S.series(cs, 'Total Debt')
    if debt is not None and debt.notna().any():
        debt = debt.dropna()
        line = None
        if lev is not None:
            ln = {_norm(p): v for p, v in lev.items() if pd.notna(v)}
            vals = [ln.get(_norm(p), np.nan) for p in debt.index]
            if any(pd.notna(v) for v in vals):
                line = ('Debt / EBITDA', vals)
        with b:
            st.plotly_chart(_combo('Total debt ($mm) and leverage', debt.index, [('Total debt', debt.values, T.NAVY)], line=line), use_container_width=True)
    c, d = st.columns(2)
    cfo, capex = S.series(cf, 'Cash from Ops.'), S.series(cf, 'Capital Expenditure')
    if cfo is not None and capex is not None:
        fcf = cfo + capex
        fig = go.Figure()
        fig.add_bar(x=list(cfo.index), y=list(cfo.values), name='Cash from operations', marker_color=T.TEAL)
        fig.add_bar(x=list(capex.index), y=list(capex.values), name='Capex', marker_color=T.ORANGE)
        fig.add_scatter(x=list(fcf.index), y=list(fcf.values), name='Free cash flow', mode='lines+markers', line=dict(color=T.INK, width=3))
        fig.update_layout(barmode='relative')
        with c:
            st.plotly_chart(T.style(fig, h=330, title='Cash generation vs. investment ($mm)'), use_container_width=True)
    if cs is not None:
        cash, _ = cs.latest('Total Cash & ST Investments')
        dues = [cs.latest(f'LT Debt (Incl. Cap. Leases) Due +{k}')[0] for k in (1, 2, 3)]
        if pd.notna(cash) and not all(pd.isna(x) for x in dues):
            cum = np.nancumsum(dues)
            fig = go.Figure()
            fig.add_bar(x=['Cash & ST investments'], y=[cash], marker_color=T.TEAL, name='Cash')
            fig.add_bar(x=['Due in 1y', 'Due by 2y', 'Due by 3y'], y=list(cum), marker_color=T.RED, name='Cumulative maturities')
            with d:
                st.plotly_chart(T.style(fig, h=330, title='Liquidity: cash vs. cumulative debt maturities ($mm)', legend=False), use_container_width=True)


def _page_story(F, co, bonds, notes, ctx, pick):
    _kpis(F, ctx)
    if co is not None:
        T.section('The arc', 'Growth, funding and balance sheet over time')
        _arc(co)
    else:
        st.info('Market view only - no Capital IQ exports loaded for this issuer. Add the multi-sheet Financials workbook (Data & exports page) to unlock the financial story.')
    sig = S.signals(F)
    if sig:
        T.section('What the numbers say', 'Automated read of the loaded data - heuristics to prompt a view, not a rating')
        for i in range(0, len(sig), 3):
            cols = st.columns(3)
            for col, s_ in zip(cols, sig[i:i + 3]):
                with col:
                    T.signal_card(s_['title'], s_['headline'], s_['bullets'], s_['tone'])
            st.write('')
    T.section('Management & operations')
    shown = False
    for key, label in S.ISSUER_FIELDS:
        if notes.get(key):
            shown = True
            with st.container(border=True):
                st.markdown(f'**{label}**')
                st.markdown(notes[key])
    if not shown:
        st.caption('No analyst notes yet - add the thesis, management and operating story on the Notes page and it appears here.')


# ------------------------------------------------------------------------------------------------ capital structure
def _page_capstruct(F, co, bonds, d_ok, ctx):
    if bonds is not None and len(bonds):
        b = bonds[bonds['Outstanding Amount'] > 0].copy()
        b['Maturity Type'] = b['Maturity Type'].replace('', 'n/a')
        left, right = st.columns([3, 2])
        with left:
            fig = px.treemap(b, path=[px.Constant('Bonds'), 'Collateral Type', 'Label'], values='Outstanding Amount', color=MET, color_continuous_scale=T.SEQUENTIAL)
            fig.update_layout(coloraxis_colorbar=dict(title=ctx.label.split(' ')[0]))
            st.plotly_chart(T.style(fig, h=470, title=f'Bond stack by seniority - area = amount outstanding, colour = {ctx.label}', legend=False), use_container_width=True)
        with right:
            by = b.groupby('Maturity Type')['Outstanding Amount'].sum().reset_index()
            fig = go.Figure(go.Pie(labels=by['Maturity Type'], values=by['Outstanding Amount'], hole=0.62, marker=dict(colors=T.CAT), textinfo='label+percent'))
            st.plotly_chart(T.style(fig, h=470, title='How the bonds can be repaid (call / bullet)', legend=False), use_container_width=True)
        yr = b.assign(Year=pd.to_datetime(b['Maturity'], errors='coerce').dt.year).dropna(subset=['Year'])
        if not yr.empty:
            w = yr.groupby('Year')['Outstanding Amount'].sum().reset_index()
            w['mm'] = w['Outstanding Amount'] / 1e6
            fig = go.Figure(go.Bar(x=w['Year'], y=w['mm'], marker_color=T.BLUE, name='Bonds maturing'))
            if co is not None and pd.notna(F.get('cash', np.nan)):
                fig.add_hline(y=F['cash'], line_dash='dash', line_color=T.TEAL, annotation_text=f'Cash & ST investments {S.mm(F["cash"])}', annotation_position='top left')
            st.plotly_chart(T.style(fig, h=380, title='Maturity wall ($mm) vs. liquidity', legend=False), use_container_width=True)
    PC.capstack_section(F, co, bonds)
    cs = co.cap_summary if co is not None else None
    if cs is not None:
        T.section('Capital IQ debt stack', 'All debt including loans and leases, from the Capital Structure Summary')
        a, c = st.columns(2)
        items = [i for i in ['Total Commercial Paper', 'Total Senior Bonds and Notes', 'Total Lease Liabilities'] if cs.row(i) is not None]
        if items:
            long = cs.values.loc[[cs.row(i) for i in items], ['Item'] + cs.periods].melt('Item', var_name='Period', value_name='$mm').dropna()
            fig = px.bar(long, x='Period', y='$mm', color='Item', color_discrete_sequence=T.CAT, category_orders={'Period': cs.periods})
            a.plotly_chart(T.style(fig, h=380, title='Debt composition ($mm)'), use_container_width=True)
        eq = [i for i in ['Total Debt', 'Total Common Equity'] if cs.row(i) is not None]
        if eq:
            long = cs.values.loc[[cs.row(i) for i in eq], ['Item'] + cs.periods].melt('Item', var_name='Period', value_name='$mm').dropna()
            fig = px.bar(long, x='Period', y='$mm', color='Item', barmode='stack', color_discrete_map={'Total Debt': T.RED, 'Total Common Equity': T.BLUE}, category_orders={'Period': cs.periods})
            c.plotly_chart(T.style(fig, h=380, title='Debt vs. book equity in total capital ($mm)'), use_container_width=True)
        sec, sp = cs.latest('Total Secured Debt')
        uns, up = cs.latest('Total Unsecured Debt')
        if pd.notna(sec) and pd.notna(uns):
            fig = go.Figure(go.Pie(labels=['Secured', 'Unsecured'], values=[sec, uns], hole=0.62, marker=dict(colors=[T.ORANGE, T.BLUE])))
            st.plotly_chart(T.style(fig, h=300, title=f'Secured vs. unsecured ({sp})', legend=True), use_container_width=True)
    with st.expander('Tables: bond list'):
        if bonds is not None and len(bonds):
            cols = [c for c in ['Label', 'Rating', 'Collateral Type', 'Maturity Type', 'Coupon', 'Ask Price', 'YTM', 'Years to Maturity', 'Outstanding Amount', 'ISIN'] if c in bonds]
            st.dataframe(bonds.sort_values(['Seniority Rank', 'Years to Maturity'])[cols], hide_index=True, use_container_width=True)
        st.caption('The Data & exports page holds the full Capital IQ capital-structure tables.')


# ------------------------------------------------------------------------------------------------ fundamentals
def _waterfall(cf, period):
    spec = [('Cash from operations', 'Cash from Ops.'), ('Capex', 'Capital Expenditure'), ('Acquisitions', 'Cash Acquisitions'), ('Dividends', 'Total Dividends Paid'),
            ('Buybacks', 'Repurchase of Common Stock'), ('Debt issued', 'Total Debt Issued'), ('Debt repaid', 'Total Debt Repaid'), ('Equity issued', 'Issuance of Common Stock')]
    vals = []
    for label, item in spec:
        s = S.series(cf, item)
        v = 0.0 if s is None or pd.isna(s.get(period, np.nan)) else float(s[period])
        if abs(v) > 0:
            vals.append((label, v))
    s = S.series(cf, 'Net Change in Cash')
    net = float(s[period]) if s is not None and pd.notna(s.get(period, np.nan)) else sum(v for _, v in vals)
    other = net - sum(v for _, v in vals)
    if vals and abs(other) > 0.01 * max(abs(v) for _, v in vals):
        vals.append(('Other (investments, FX, other)', other))
    return vals, net


def _sankey(vals, net):
    pos = [(l, v) for l, v in vals if v > 0]
    neg = [(l, -v) for l, v in vals if v < 0]
    if net < 0:
        pos.append(('Cash drawdown', -net))
    if net > 0:
        neg.append(('Cash build', net))
    labels = [l for l, _ in pos] + ['Cash'] + [l for l, _ in neg]
    hub = len(pos)
    src = list(range(len(pos))) + [hub] * len(neg)
    tgt = [hub] * len(pos) + list(range(hub + 1, hub + 1 + len(neg)))
    val = [v for _, v in pos] + [v for _, v in neg]
    colors = [T.TEAL] * len(pos) + [T.INK] + [T.ORANGE] * len(neg)
    fig = go.Figure(go.Sankey(node=dict(label=labels, color=colors, pad=18, thickness=16), link=dict(source=src, target=tgt, value=val, color='rgba(138,148,166,.35)')))
    return T.style(fig, h=420, title='Where the cash came from and where it went ($mm)', legend=False)


def _page_fundamentals(F, co):
    if co is None or not co.statements:
        st.info('Add Capital IQ exports (Key Stats, Income Statement, Balance Sheet, Cash Flow) for the fundamentals view.')
        return
    ks, cf, bs, rt = co.statements.get('key_stats'), co.statements.get('cashflow'), co.statements.get('balance'), co.ratios
    a, b = st.columns(2)
    rev = S.series(ks, 'Total Revenue')
    if rev is not None:
        gr = S.series(ks, 'Total Revenue Growth Over Prior Year')
        a.plotly_chart(_combo('Revenue ($mm) and growth', rev.index, [('Revenue', rev.values, None)],
                              line=('YoY growth', (gr.reindex(rev.index) * 100).values) if gr is not None else None, line_pct=True), use_container_width=True)
        fig = go.Figure()
        for item, color, nm in [('Gross Profit Margin %', T.GREEN, 'Gross'), ('EBITDA Margin %', T.BLUE, 'EBITDA'), ('EBIT Margin %', T.PURPLE, 'EBIT'), ('Net Income Margin %', T.ORANGE, 'Net income')]:
            s = S.series(ks, item)
            if s is not None:
                fig.add_scatter(x=list(s.index), y=list((s * 100).values), name=nm, mode='lines+markers', line=dict(color=color, width=2.5))
        fig.update_yaxes(ticksuffix='%')
        b.plotly_chart(T.style(fig, h=330, title='Margin ladder (estimates on the right)'), use_container_width=True)
    if cf is not None and S.series(cf, 'Cash from Ops.') is not None:
        T.section('Cash flow', 'Pick a period to see how cash was generated and used')
        cfo = S.series(cf, 'Cash from Ops.')
        pers = [p for p in cf.periods if pd.notna(cfo.get(p, np.nan))]
        per = st.selectbox('Period', pers, index=len(pers) - 1, key='cp_cfper') if pers else None
        if per:
            vals, net = _waterfall(cf, per)
            c1, c2 = st.columns(2)
            fig = go.Figure(go.Waterfall(x=[l for l, _ in vals] + ['Net change in cash'], y=[v for _, v in vals] + [0], measure=['relative'] * len(vals) + ['total'],
                                         text=[S.mm(v) for _, v in vals] + [S.mm(net)], textposition='outside',
                                         increasing=dict(marker=dict(color=T.TEAL)), decreasing=dict(marker=dict(color=T.ORANGE)), totals=dict(marker=dict(color=T.INK)),
                                         connector=dict(line=dict(color='#C7CEDA'))))
            c1.plotly_chart(T.style(fig, h=420, title=f'Cash bridge, {per} ($mm)', legend=False), use_container_width=True)
            c2.plotly_chart(_sankey(vals, net), use_container_width=True)
    if bs is not None:
        T.section('Balance sheet', 'What the company owns and how it is funded')
        per_b = bs.periods

        def row(item):
            s = S.series(bs, item)
            return s.fillna(0.0) if s is not None else pd.Series(0.0, index=per_b)
        assets = pd.DataFrame({'Cash & ST investments': row('Total Cash & ST Investments'), 'Receivables': row('Total Receivables'), 'Net PP&E': row('Net Property, Plant & Equipment'),
                               'Goodwill & intangibles': row('Goodwill') + row('Other Intangibles')})
        ta = row('Total Assets')
        assets['Other assets'] = (ta - assets.sum(axis=1)).clip(lower=0)
        fund = pd.DataFrame({'Debt': row('Total Debt'), 'Equity': row('Total Equity')})
        tle = row('Total Liabilities And Equity')
        fund['Other liabilities'] = (tle - fund.sum(axis=1)).clip(lower=0)
        c1, c2 = st.columns(2)
        fig = go.Figure([go.Bar(x=per_b, y=assets[c], name=c, marker_color=col) for c, col in zip(assets.columns, [T.TEAL, T.AMBER, T.BLUE, T.PURPLE, T.GREY])])
        fig.update_layout(barmode='stack')
        c1.plotly_chart(T.style(fig, h=380, title='Assets ($mm)'), use_container_width=True)
        fig = go.Figure([go.Bar(x=per_b, y=fund[c], name=c, marker_color=col) for c, col in zip(fund.columns, [T.RED, T.BLUE, T.GREY])])
        fig.update_layout(barmode='stack')
        c2.plotly_chart(T.style(fig, h=380, title='Funding: debt, equity, other ($mm)'), use_container_width=True)
    if rt is not None:
        cov = S.series(rt, 'EBITDA / Interest Exp.')
        lv = S.series(rt, 'Total Debt/EBITDA')
        shown = [(n, s) for n, s in [('Total debt / EBITDA', lv), ('EBITDA / interest', cov)] if s is not None and s.notna().sum() >= 2]
        if shown:
            fig = go.Figure()
            for (n, s), col in zip(shown, [T.RED, T.TEAL]):
                s = s.dropna()
                fig.add_scatter(x=list(s.index), y=list(s.values), name=n, mode='lines+markers', line=dict(color=col, width=3))
            fig.update_yaxes(ticksuffix='x')
            st.plotly_chart(T.style(fig, h=340, title='Leverage and coverage (Capital IQ ratios page)'), use_container_width=True)


# ------------------------------------------------------------------------------------------------ valuation & segments
def _page_val_seg(F, co):
    if co is None:
        st.info('Add Capital IQ exports (Multiples, Capitalization, Segments) for this view.')
        return
    cap, mu, sg = co.statements.get('capitalization'), co.statements.get('multiples'), co.statements.get('segments')
    if cap is None and mu is None and sg is None:
        st.info('Add the Multiples, Capitalization and Segments exports (or the full Financials workbook) for this view.')
        return
    if cap is not None:
        T.section('Equity cushion', 'Market value of equity versus debt at each balance-sheet date')
        mc, db = S.series(cap, 'Market Capitalization'), S.series(cap, '+ Total Debt')
        if mc is not None and db is not None:
            share = db / (mc + db) * 100
            st.plotly_chart(_combo('Market cap and total debt ($mm); line = debt as % of (market cap + debt)', cap.periods,
                                   [('Market cap', mc.values, T.BLUE), ('Total debt', db.values, T.RED)], line=('Debt / (equity + debt)', share.values), line_pct=True), use_container_width=True)
    if mu is not None:
        T.section('Valuation versus its own history', 'Quarterly high-low range, average and close of each trading multiple')
        names = list(dict.fromkeys(mu.values['Section']))
        pick = st.selectbox('Multiple', names, index=names.index('TEV/LTM EBITDA') if 'TEV/LTM EBITDA' in names else 0, key='cp_mult')
        g = {k: mu.values[(mu.values['Section'] == pick) & (mu.values['Item'] == k)] for k in ['High', 'Low', 'Average', 'Close']}
        if all(len(v) for v in g.values()):
            vals = {k: v[mu.periods].iloc[0].astype(float).values for k, v in g.items()}
            fig = go.Figure()
            fig.add_scatter(x=mu.periods, y=vals['High'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip')
            fig.add_scatter(x=mu.periods, y=vals['Low'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31,111,235,.15)', name='High-low range')
            fig.add_scatter(x=mu.periods, y=vals['Average'], mode='lines', line=dict(color=T.GREY, dash='dash'), name='Average')
            fig.add_scatter(x=mu.periods, y=vals['Close'], mode='lines+markers', line=dict(color=T.BLUE, width=3), name='Close')
            fig.update_yaxes(ticksuffix='x')
            st.plotly_chart(T.style(fig, h=380, title=f'{pick}'), use_container_width=True)
            cl = pd.Series(vals['Close']).dropna()
            if len(cl) >= 3:
                st.caption(f'Latest close {cl.iloc[-1]:.1f}x is at the {(cl <= cl.iloc[-1]).mean() * 100:.0f}th percentile of the {len(cl)} quarters exported.')
    if sg is not None:
        T.section('Segments', 'Where revenue and profit come from')
        secs = list(dict.fromkeys(sg.values['Section']))
        rev_s = next((s for s in secs if 'Business' in s and 'Revenue' in s), secs[0])
        prof_s = next((s for s in secs if 'Business' in s and 'Operating Profit' in s), None)
        geo_s = next((s for s in secs if 'Geographic' in s and 'Revenue' in s), None)
        v = sg.values[(sg.values['Section'] == rev_s) & ~sg.values['Item'].str.lower().str.startswith('total')]
        c1, c2 = st.columns(2)
        if not v.empty:
            long = v.melt(id_vars=['Item'], value_vars=sg.periods, var_name='Period', value_name='$mm').dropna()
            fig = px.bar(long, x='Period', y='$mm', color='Item', color_discrete_sequence=T.CAT, category_orders={'Period': sg.periods})
            c1.plotly_chart(T.style(fig, h=380, title='Revenue by segment ($mm)'), use_container_width=True)
            piv = v.set_index('Item')[sg.periods].astype(float)
            growth = (piv.pct_change(axis=1) * 100).iloc[:, 1:].reset_index().melt('Item', var_name='Period', value_name='YoY %').dropna()
            fig = px.line(growth, x='Period', y='YoY %', color='Item', markers=True, color_discrete_sequence=T.CAT, category_orders={'Period': sg.periods})
            fig.update_yaxes(ticksuffix='%')
            c2.plotly_chart(T.style(fig, h=380, title='Segment revenue growth'), use_container_width=True)
            if prof_s:
                p = sg.values[sg.values['Section'] == prof_s].set_index('Item')[sg.periods].astype(float)
                r = v.set_index('Item')[sg.periods].astype(float)
                common = [i for i in r.index if i in p.index]
                if common:
                    mg = (p.loc[common] / r.loc[common].where(r.loc[common] != 0) * 100).reset_index().melt('Item', var_name='Period', value_name='Margin %').dropna()
                    c3, c4 = st.columns(2)
                    fig = px.line(mg, x='Period', y='Margin %', color='Item', markers=True, color_discrete_sequence=T.CAT, category_orders={'Period': sg.periods})
                    fig.update_yaxes(ticksuffix='%')
                    c3.plotly_chart(T.style(fig, h=360, title='Segment operating margin before unallocated costs'), use_container_width=True)
                    if geo_s:
                        gv = sg.values[(sg.values['Section'] == geo_s) & ~sg.values['Item'].str.lower().str.startswith('total')]
                        last = sg.periods[-1]
                        fig = go.Figure(go.Pie(labels=gv['Item'], values=gv[last], hole=0.6, marker=dict(colors=T.CAT)))
                        c4.plotly_chart(T.style(fig, h=360, title=f'Revenue by geography, {last}', legend=True), use_container_width=True)


# ------------------------------------------------------------------------------------------------ market & peers
def _page_market(F, bonds, d_ok, ctx, cos, pick):
    if bonds is None or bonds.empty:
        st.info('No bonds for this issuer in the loaded screener files.')
        return
    b = bonds[(bonds['YTM'] > 0) & (bonds['YTM'] <= 30) & bonds['Years to Maturity'].notna() & (bonds['Years to Maturity'] > 0)].copy()
    bucket = F.get('bucket') or 'NR'
    sector = F.get('sector')
    bench, bsrc = _bench_for(bucket, d_ok, ctx)
    ref = d_ok[(d_ok['Rating Bucket'] == bucket) & (d_ok['Years to Maturity'] > 0)]
    fig = go.Figure()
    for sen, g in b.groupby('Collateral Type'):
        fig.add_scatter(x=g['Years to Maturity'], y=g[MET], mode='markers', name=sen, text=g['Label'], hovertemplate='%{text}<br>%{x:.1f}y  %{y:.0f}bp<extra></extra>',
                        marker=dict(size=np.clip(np.sqrt(g['Outstanding Amount'] / 1e6) * 1.2, 7, 30), opacity=0.8))
    cur = ref.groupby('Maturity Bucket', observed=True).agg(med=(MET, 'median'), yrs=('Years to Maturity', 'median'), n=(MET, 'size'))
    cur = cur[cur['n'] >= 5]
    if len(cur):
        fig.add_scatter(x=cur['yrs'], y=cur['med'], mode='lines+markers', name=f'{bucket} median (screener)', line=dict(color=T.GREY, dash='dot'))
    if pd.notna(bench) and bsrc == 'ICE BofA OAS':
        fig.add_hline(y=bench, line_dash='dash', line_color=T.TEAL, annotation_text=f'{bucket} index OAS {bench:.0f}bp', annotation_position='bottom right')
    fig.update_xaxes(rangemode='nonnegative', title_text='Years to maturity')
    fig.update_yaxes(title_text=ctx.label)
    st.plotly_chart(T.style(fig, h=460, title=f'{pick} bond curve vs. {bucket} peers (bubble = size)'), use_container_width=True)
    sec_ok = d_ok[d_ok['Sector'] == sector]
    it = an.issuer_table(sec_ok, MET)
    me = it[it['Ticker'] == pick]
    sp = F.get('spread')
    a, c = st.columns(2)
    labels = ['This issuer', f'{sector} avg', f'{bucket} benchmark ({bsrc})' if bsrc else f'{bucket} benchmark', 'Whole screener']
    vals = [sp, an.wavg(sec_ok, MET), bench, an.wavg(d_ok, MET)]
    fig = go.Figure(go.Bar(y=labels[::-1], x=vals[::-1], orientation='h', marker_color=[T.GREY, T.TEAL, T.BLUE, T.RED][::-1],
                           text=[f'{v:.0f}bp' if pd.notna(v) else 'n/a' for v in vals[::-1]], textposition='outside'))
    a.plotly_chart(T.style(fig, h=340, title=f'Priced vs. benchmarks ({ctx.label})', legend=False), use_container_width=True)
    if len(it) >= 5:
        fig = go.Figure(go.Box(x=it['Metric'], name=sector, boxpoints=False, marker_color=T.GREY, orientation='h'))
        if len(me):
            fig.add_scatter(x=[float(me['Metric'].iloc[0])], y=[sector], mode='markers', marker=dict(size=16, color=T.RED, symbol='diamond'), name=pick)
        fig.update_xaxes(rangemode='nonnegative', title_text=ctx.label)
        pct_rank = (it['Metric'] <= (me['Metric'].iloc[0] if len(me) else sp)).mean() * 100
        c.plotly_chart(T.style(fig, h=340, title=f'Where it sits among {len(it)} {sector} issuers ({pct_rank:.0f}th percentile)', legend=False), use_container_width=True)
    hist = ctx.hist
    if len(hist):
        h1 = hist[(hist['level'] == 'Ticker') & (hist['key'] == pick)]
        if h1['date'].nunique() >= 2:
            col = 'gspread' if ctx.has_spread else 'ytm'
            sc = 1 if ctx.has_spread else 100
            fig = go.Figure()
            fig.add_scatter(x=h1['date'], y=h1[col] * sc, name=pick, mode='lines+markers', line=dict(color=T.RED, width=3))
            h2 = hist[(hist['level'] == 'Sector') & (hist['key'] == sector)]
            fig.add_scatter(x=h2['date'], y=h2[col] * sc, name=f'{sector} avg', mode='lines', line=dict(color=T.GREY, dash='dash'))
            st.plotly_chart(T.style(fig, h=320, title=f'Spread history from saved screener snapshots ({ctx.label})'), use_container_width=True)
        else:
            st.caption(f'History builds automatically: each time you load a new screener CSV the app saves a snapshot ({hist["date"].nunique()} so far). Lines appear once there are two.')
    T.section('Peers', f'Other {sector} issuers in the screener')
    peers = it[(it['Ticker'] != pick) & (it['Bonds'] >= 2)]
    show = pd.concat([peers.nlargest(40, 'Outstanding ($bn)'), me])
    if len(show) >= 3:
        fig = px.scatter(show, x='Years', y='Metric', size='Outstanding ($bn)', color='Bucket', hover_name='Name', text='Ticker', size_max=45,
                         color_discrete_map=T.RATING_COLORS, category_orders={'Bucket': T.RATING_ORDER})
        fig.update_traces(textposition='top center', marker=dict(opacity=0.75, line=dict(width=0)))
        if len(me):
            fig.add_scatter(x=me['Years'], y=me['Metric'], mode='markers', marker=dict(size=26, color='rgba(0,0,0,0)', line=dict(color=T.INK, width=3)), name=pick, showlegend=True)
        fig.update_xaxes(rangemode='nonnegative', title_text='Weighted average life (years)')
        fig.update_yaxes(title_text=ctx.label)
        st.plotly_chart(T.style(fig, h=460, title='Peer map: spread vs. maturity (bubble = debt outstanding, ring = this issuer)'), use_container_width=True)
    prows = [fundamentals_row(c) for t, c in cos.items() if t and t != pick and c.cap_summary is not None and d_ok[d_ok['Ticker'] == t]['Sector'].eq(sector).any()]
    if pick in cos and cos[pick].cap_summary is not None:
        prows.append(fundamentals_row(cos[pick]))
    if len(prows) >= 2:
        pf = pd.DataFrame(prows).dropna(subset=['Debt / EBITDA'])
        mp = it.set_index('Ticker')['Metric']
        pf['Spread'] = pf['Ticker'].map(mp)
        pf = pf.dropna(subset=['Spread'])
        if len(pf) >= 2:
            fig = px.scatter(pf, x='Debt / EBITDA', y='Spread', text='Ticker', size='Total debt ($bn)', color_discrete_sequence=[T.BLUE], size_max=40)
            fig.update_traces(textposition='top center')
            fig.update_xaxes(rangemode='nonnegative', ticksuffix='x')
            fig.update_yaxes(title_text=ctx.label)
            st.plotly_chart(T.style(fig, h=400, title='Does the market pay for leverage? Debt/EBITDA vs. spread (issuers with Capital IQ data)', legend=False), use_container_width=True)
    PC.price_yield_section(bonds, F)


# ------------------------------------------------------------------------------------------------ notes
def _page_notes(pick, notes_dir, notes):
    st.caption('Your qualitative view - saved as a small JSON file in data/stories and shown on the Story page.')
    with st.form(f'notes_{pick}'):
        cur = notes.get('stance', 'Not set')
        stance = st.selectbox('Stance', S.STANCES, index=S.STANCES.index(cur) if cur in S.STANCES else 0)
        vals = {k: st.text_area(label, value=notes.get(k, ''), height=110) for k, label in S.ISSUER_FIELDS}
        if st.form_submit_button('Save notes'):
            S.save_notes(notes_dir, 'issuer', pick, dict(vals, stance=stance))
            st.success('Saved.')
            st.rerun()
    if notes.get('updated'):
        st.caption(f'Last saved {notes["updated"]}.')


# ------------------------------------------------------------------------------------------------ entry point
def render_profile_tab(d, f, ctx, capiq_dir, notes_dir):
    cos = {t: c for t, c in get_companies(capiq_dir).items() if t and c.files}
    d_ok = d[(d['YTM'] > 0) & (d['YTM'] <= 30)]
    names = d.groupby('Ticker')['Name'].first().to_dict()
    tickers = sorted(set(f['Ticker'].dropna()) | set(cos))
    if not tickers:
        st.info('No issuers available - loosen the sidebar filters.')
        return
    fix_state('cp_issuer', tickers)
    default = next((i for i, t in enumerate(tickers) if t in cos), 0)
    pick = st.selectbox('Issuer', tickers, index=default, format_func=lambda t: f'{names.get(t) or (cos[t].name if t in cos else t)}  ({t})', key='cp_issuer')
    co = cos.get(pick)
    bonds = d[d['Ticker'] == pick].copy()
    bonds = bonds if len(bonds) else None
    sector_med = bench = None
    if bonds is not None:
        sec_ok = d_ok[d_ok['Sector'] == bonds['Sector'].iloc[0]]
        sector_med = an.wavg(sec_ok, MET)
        bench, _ = _bench_for(bonds['Rating Bucket'].mode().iloc[0], d_ok, ctx)
    F = S.facts(co, bonds, MET, sector_med, bench)
    if 'name' not in F:
        F['name'] = names.get(pick, pick)
    notes = S.load_notes(notes_dir, 'issuer', pick)
    tone = {'Positive': 'good', 'Negative': 'bad', 'Neutral': 'neutral'}
    chips = ''
    if 'universe' in F:
        chips += ''.join(T.chip(r, 'info') for r in F.get('ratings', [])[:3]) + T.chip(F['universe'], 'neutral')
    if F.get('sector'):
        chips += T.chip(F['sector'], 'neutral')
    if notes.get('stance') in tone:
        chips += T.chip(f'Analyst stance: {notes["stance"]}', tone[notes['stance']])
    asof = f'Bonds as of {ctx.asof:%d %b %Y}' + (f' | Capital IQ through {F.get("rev_p") or F.get("debt_p")}' if co is not None else ' | no Capital IQ data')
    T.story_banner('Issuer credit story', S.headline(F), notes.get('thesis') or asof, chips)
    page = nav(PAGES, 'cp_page')
    if page == 'Story':
        _page_story(F, co, bonds, notes, ctx, pick)
    elif page == 'Credit health & comps':
        PC.page_health(F, co, bonds, d, d_ok, cos, ctx, pick, names)
    elif page == 'Capital structure':
        _page_capstruct(F, co, bonds, d_ok, ctx)
    elif page == 'Ratios':
        PC.page_ratios(co)
    elif page == 'Scenarios':
        PC.page_scenarios(F, co, bonds)
    elif page == 'Fundamentals':
        _page_fundamentals(F, co)
    elif page == 'Valuation & segments':
        _page_val_seg(F, co)
    elif page == 'Market & peers':
        _page_market(F, bonds, d_ok, ctx, cos, pick)
    elif page == 'Notes':
        _page_notes(pick, notes_dir, notes)
    else:
        if pick in cos:
            st.session_state['cq_issuer'] = pick
        render_capiq_tab(capiq_dir, bonds)
