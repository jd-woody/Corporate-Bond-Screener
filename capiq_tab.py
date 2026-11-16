"""capiq_tab.py - 'Credit Profile (Capital IQ)' tab for credit_analytics_bot.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import capiq_loader as cq

SUFFIXES = ('.xls', '.xlsx')


def _signature(directory):
    d = Path(directory)
    if not d.exists():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(d.iterdir())
                 if p.suffix.lower() in SUFFIXES and not p.name.startswith(('~', '.')))


@st.cache_data(show_spinner='Reading Capital IQ files...')
def _load(sig, directory):
    return cq.load_dir(directory)


def get_companies(directory):
    """{TICKER: CapIQCompany}; cached on the file list + modified times, so new exports are picked up automatically."""
    return _load(_signature(directory), str(directory))


def fmt_mm(v):
    if pd.isna(v):
        return '-'
    sign, a = ('-' if v < 0 else ''), abs(v)
    return f'{sign}${a / 1000:,.1f}bn' if a >= 1000 else f'{sign}${a:,.0f}mm'


def _wavg(df, col, w='Amt. Outstdg. ($mm)'):
    z = df[[col, w]].dropna()
    z = z[z[w] > 0]
    return float(np.average(z[col], weights=z[w])) if len(z) else np.nan


def _date_str(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ''
    return v.strftime('%Y-%m-%d') if hasattr(v, 'strftime') else str(v)


# ------------------------------------------------------------------------------------------------ sections
def _uploader(directory):
    with st.expander('Add or update Capital IQ exports'):
        st.caption(f'Files are saved to `{directory}`. You can also copy them there yourself - the tab refreshes on the next rerun. '
                   'Re-exporting a report simply replaces the older one (newest file per report wins).')
        up = st.file_uploader('Capital IQ .xls / .xlsx exports', type=['xls', 'xlsx'], accept_multiple_files=True, key='capiq_upload')
        if up and st.button('Save files', key='capiq_save'):
            Path(directory).mkdir(parents=True, exist_ok=True)
            for f in up:
                (Path(directory) / Path(f.name).name).write_bytes(f.getbuffer())
            st.cache_data.clear()
            st.rerun()


def _inventory(co):
    rows = []
    for key, label in cq.REPORT_NAMES.items():
        if key in co.files:
            name, mt = co.files[key]
            rows.append({'Report': label, 'Status': 'loaded', 'File': name, 'File saved': mt.strftime('%Y-%m-%d %H:%M')})
        else:
            rows.append({'Report': label, 'Status': 'missing', 'File': '', 'File saved': ''})
    with st.expander('Files loaded for this issuer' + (f' - {len(co.problems)} note(s)' if co.problems else '')):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for p in co.problems:
            st.caption(f'• {p}')


def _headline(co):
    cs, rt = co.cap_summary, co.ratios
    if cs is None:
        st.info('Add the **Financials > Capital Structure Summary** export to see headline credit metrics.')
        return
    td, per = cs.latest('Total Debt')
    cash, _ = cs.latest('Total Cash & ST Investments')
    nd, _ = cs.latest('Net Debt')
    lev, lev_p = cs.latest('Total Debt/EBITDA', 'Credit Ratios')
    dcap = cs.latest_pct('Total Debt', per) if per else np.nan
    sec, sec_p = cs.latest('Total Secured Debt')
    sec_pct = cs.latest_pct('Total Secured Debt', sec_p) if sec_p else np.nan
    a = st.columns(4)
    a[0].metric('Total debt', fmt_mm(td), help=f'Capital IQ "Total Debt", period {per}')
    a[1].metric('Cash & ST investments', fmt_mm(cash))
    a[2].metric('Net debt', fmt_mm(nd) if nd >= 0 else f'{fmt_mm(abs(nd))} net cash' if pd.notna(nd) else '-',
                help='Capital IQ net debt = total debt minus cash & short-term investments. Long-term marketable securities are not deducted.')
    a[3].metric('Debt / capital', f'{dcap * 100:.1f}%' if pd.notna(dcap) else '-')
    b = st.columns(4)
    b[0].metric('Total debt / EBITDA', f'{lev:.2f}x' if pd.notna(lev) else '-', help=f'Capital Structure Summary, {lev_p}')
    if rt is not None:
        cov, cov_p = rt.latest('EBITDA / Interest Exp.')
        z, z_p = rt.latest('Altman Z Score')
        cr, cr_p = rt.latest('Current Ratio')
    else:
        cov = z = cr = np.nan
        cov_p = z_p = cr_p = None
    b[1].metric(f'EBITDA / interest ({cov_p})' if cov_p else 'EBITDA / interest', f'{cov:.1f}x' if pd.notna(cov) else '-', help=f'Latest period Capital IQ populated: {cov_p}' if cov_p else 'Needs Ratios export')
    b[2].metric(f'Secured share ({sec_p})' if sec_p else 'Secured share of debt', f'{sec_pct * 100:.1f}%' if pd.notna(sec_pct) else '-', help=f'Total Secured Debt, {sec_p}' if sec_p else None)
    b[3].metric('Altman Z / current ratio', f'{z:.1f} / {cr:.2f}' if pd.notna(z) and pd.notna(cr) else '-', help=f'{z_p}')
    st.caption(f'Debt figures as of {per}' + (f'; filed {cs.filing[per]:%Y-%m-%d}' if per in cs.filing else '') + '. $ in millions of reporting currency.')


def _tab_cap_structure(co):
    cs = co.cap_summary
    if cs is None:
        st.info('Add the **Capital Structure Summary** export for this issuer.')
        return
    sections = list(dict.fromkeys(cs.values['Section']))
    sec = st.selectbox('Section', ['All'] + sections, key='cq_sec')
    st.dataframe(cq.display_cap_summary(cs, None if sec == 'All' else sec), hide_index=True, use_container_width=True, height=min(900, 38 + 35 * 22))
    left, right = st.columns(2)
    comp_items = [i for i in ['Total Commercial Paper', 'Total Senior Bonds and Notes', 'Total Lease Liabilities'] if cs.row(i) is not None]
    if comp_items:
        long = cs.values.loc[[cs.row(i) for i in comp_items], ['Item'] + cs.periods].melt('Item', var_name='Period', value_name='$mm').dropna()
        left.plotly_chart(px.bar(long, x='Period', y='$mm', color='Item', title='Debt composition ($mm)', category_orders={'Period': cs.periods}),
                          use_container_width=True)
    eq_items = [i for i in ['Total Debt', 'Total Common Equity'] if cs.row(i) is not None]
    if eq_items:
        long = cs.values.loc[[cs.row(i) for i in eq_items], ['Item'] + cs.periods].melt('Item', var_name='Period', value_name='$mm').dropna()
        right.plotly_chart(px.bar(long, x='Period', y='$mm', color='Item', title='Debt vs. equity in total capital ($mm)', category_orders={'Period': cs.periods}),
                           use_container_width=True)
    st.caption('"Principal Due" can exceed "Total Debt Outstanding" (percentages above 100%) because Capital IQ applies the adjustments shown on the '
               '"Total Adjustments" row. Lease lines can be blank in an interim quarter, so check the adjustments before comparing quarters line by line. '
               'Net-debt multiples show NM when the company has net cash.')


def _tab_debt_detail(co):
    det = co.cap_details
    if det is None:
        st.info('Add the **Capital Structure Details** export for this issuer.')
        return
    periods = list(dict.fromkeys(det['Period']))
    per = st.selectbox('Reporting period', periods, key='cq_det_per')
    x = det[det['Period'] == per].copy()
    total = x['Principal Due'].sum()
    m = st.columns(3)
    m[0].metric('Principal listed', fmt_mm(total))
    m[1].metric('Instruments', f'{len(x)}')
    foreign = x[x['Repayment Currency'] != x['Reporting Currency']]['Principal Due'].sum() if 'Repayment Currency' in x else np.nan
    m[2].metric('Repayable in non-reporting currency', f'{foreign / total * 100:.1f}%' if total else '-')
    left, right = st.columns(2)
    by_type = x.groupby('Type', dropna=False)['Principal Due'].sum().reset_index()
    left.plotly_chart(px.bar(by_type, x='Type', y='Principal Due', color='Type', title='Principal by instrument type ($mm)'), use_container_width=True)
    if 'Repayment Currency' in x:
        by_ccy = x.groupby('Repayment Currency', dropna=False)['Principal Due'].sum().reset_index()
        right.plotly_chart(px.bar(by_ccy, x='Repayment Currency', y='Principal Due', color='Repayment Currency', title='Principal by repayment currency ($mm, USD-equivalent)'),
                           use_container_width=True)
    cols = [c for c in ['Description', 'Type', 'Principal Due', 'Coupon/Base Rate', 'Floating Rate', 'Maturity', 'Seniority', 'Secured', 'Convertible', 'Repayment Currency'] if c in x]
    show = x[cols].copy()
    if 'Maturity' in show:
        show['Maturity'] = show['Maturity'].map(_date_str)
    for c in ['Coupon/Base Rate', 'Floating Rate']:
        if c in show:
            show[c] = show[c].map(lambda v: '' if pd.isna(v) else str(v))
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={'Principal Due': st.column_config.NumberColumn('Principal Due ($mm)', format='%.1f')})
    st.caption('Capital IQ can group most notes into a single "Fixed-rate Notes" line in this report; the bond-by-bond view is on the '
               'Maturities & bonds tab (Securities Summary export).')


def _screener_check(co, sec, bonds):
    if bonds is None or bonds.empty or 'Ticker' not in bonds:
        return
    tickers = sorted(bonds['Ticker'].dropna().unique())
    default = tickers.index(co.ticker) + 1 if co.ticker in tickers else 0
    pick = st.selectbox('Compare with screener ticker', ['(none)'] + tickers, index=default, key='cq_chk',
                        help='Capital IQ tickers usually match the screener ticker. Pick a different one if they differ.')
    if pick == '(none)':
        return
    sb = bonds[bonds['Ticker'] == pick]
    sc_out = sb['Outstanding Amount'].sum() / 1e6
    cq_out = sec['Amt. Outstdg. ($mm)'].sum()
    c = st.columns(4)
    c[0].metric('Bonds - Capital IQ', f'{len(sec)}')
    c[1].metric('Bonds - screener files', f'{len(sb)}')
    c[2].metric('Outstanding - Capital IQ', fmt_mm(cq_out))
    c[3].metric('Outstanding - screener', fmt_mm(sc_out), f'{sc_out - cq_out:+,.0f}mm vs Capital IQ', delta_color='off')
    st.caption('Differences are expected: the two sources are pulled at different times, can treat 144A/Reg S twins, foreign-currency issues and '
               'small tranches differently, and the screener only contains bonds present in the loaded screener files.')


def _tab_bonds(co, bonds):
    cs, sec = co.cap_summary, co.securities
    if cs is not None:
        v, per = cs.latest('LT Debt (Incl. Cap. Leases) Due +1')
        if per:
            names = [f'LT Debt (Incl. Cap. Leases) Due +{k}' for k in range(1, 6)] + ['LT Debt (Incl. Cap. Leases) Due, After 5 Yrs']
            labels = [f'Year +{k}' for k in range(1, 6)] + ['After 5 yrs']
            vals = [cs.latest(n)[0] if cs.latest(n)[1] == per else np.nan for n in names]
            lad = pd.DataFrame({'Bucket': labels, '$mm': vals}).dropna()
            st.plotly_chart(px.bar(lad, x='Bucket', y='$mm', title=f'Long-term debt maturity ladder, as of {per} ($mm)'), use_container_width=True)
    if sec is None:
        st.info('Add the **Fixed Income > Securities Summary** export for the bond-by-bond view.')
        return
    mm = st.checkbox('Include money-market instruments (commercial paper)', value=False, key='cq_mm')
    s = sec if mm else sec[~sec['Is Money Market']]
    s = s.copy()
    n_unpriced = int(s['Current Price'].isna().sum())
    c = st.columns(4)
    c[0].metric('Securities', f'{len(s)}')
    c[1].metric('Outstanding', fmt_mm(s['Amt. Outstdg. ($mm)'].sum()))
    w = _wavg(s, 'Current YTW')
    c[2].metric('Wtd avg YTW', f'{w:.2f}%' if pd.notna(w) else '-')
    c[3].metric('S&P rating', ' / '.join(sorted(s['S&P Rating'].dropna().unique())) if 'S&P Rating' in s else '-')
    left, right = st.columns(2)
    wall = s.dropna(subset=['Maturity Date', 'Amt. Outstdg. ($mm)']).assign(Year=lambda d: d['Maturity Date'].dt.year)
    if not wall.empty:
        left.plotly_chart(px.bar(wall.groupby('Year', as_index=False)['Amt. Outstdg. ($mm)'].sum(), x='Year', y='Amt. Outstdg. ($mm)', title='Maturity wall ($mm outstanding)'),
                          use_container_width=True)
    curve = s.dropna(subset=['Years to Maturity', 'Current YTW', 'Amt. Outstdg. ($mm)'])
    if not curve.empty:
        curve = curve.assign(Maturity=curve['Maturity Date'].map(_date_str))
        right.plotly_chart(px.scatter(curve, x='Years to Maturity', y='Current YTW', size='Amt. Outstdg. ($mm)', color='Seniority',
                                      hover_data=['Maturity', 'Coupon', 'Current Price'], title='Capital IQ yield curve (current YTW)'),
                           use_container_width=True)
    cols = [c for c in ['Maturity Date', 'Security Type', 'Seniority', 'Coupon', 'Coupon Type', 'Amt. Outstdg. ($mm)', 'Offering Amt. ($mm)', 'Current Price',
                        'Current YTW', 'S&P Rating', 'S&P CreditWatch/Outlook', 'Issued Currency', 'Next Call Date', 'Call Price'] if c in s]
    show = s[cols].sort_values('Maturity Date').copy()
    for c in ['Maturity Date', 'Next Call Date']:
        if c in show:
            show[c] = show[c].map(_date_str)
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button('Export Capital IQ securities', show.to_csv(index=False).encode(), f'capiq_securities_{co.ticker}.csv', 'text/csv', key='cq_dl')
    if n_unpriced:
        st.caption(f'{n_unpriced} security(ies) have no current price in the export (shown blank, excluded from yield stats).')
    st.caption('Prices and YTW are as of when the file was exported from Capital IQ, not live. YTW is yield-to-worst as defined by Capital IQ.')
    st.markdown('**Cross-check against the screener files**')
    _screener_check(co, s, bonds)


def _tab_ratios(co):
    rt = co.ratios
    if rt is None:
        st.info('Add the **Financials > Ratios** export for this issuer.')
        return
    sections = list(dict.fromkeys(rt.values['Section']))
    sec = st.selectbox('Section', sections, index=sections.index('Long Term Solvency') if 'Long Term Solvency' in sections else 0, key='cq_rsec')
    st.dataframe(cq.display_ratios(rt, sec), hide_index=True, use_container_width=True)
    v = rt.values[rt.values['Section'] == sec]
    items = v['Item'].tolist()
    default = [i for i in ['Total Debt/EBITDA', 'Total Debt/Capital'] if i in items] or items[:2]
    pick = st.multiselect('Chart items', items, default=default, key='cq_ritems')
    if pick:
        rows = []
        for i in v.index[v['Item'].isin(pick)]:
            kind = cq.ratio_kind(sec, v.at[i, 'Item'])
            for p in rt.periods:
                val = v.at[i, p]
                if pd.notna(val):
                    rows.append({'Item': v.at[i, 'Item'], 'Period': p, 'Value': val * 100 if kind == 'pct' else val})
        if rows:
            st.plotly_chart(px.line(pd.DataFrame(rows), x='Period', y='Value', color='Item', markers=True, category_orders={'Period': rt.periods},
                                    title='Selected ratios (percent items plotted in %, others in x / days)'), use_container_width=True)
    st.caption('Values are shown exactly as exported. The Ratios page and the Capital Structure Summary page are calculated by Capital IQ on slightly different '
               'bases, so a ratio such as Total Debt/EBITDA can differ between them for the same period. "-" means the cell is blank in the export '
               '(in this file, interest coverage is only populated for the first period); NM means not meaningful.')



# ------------------------------------------------------------------------------------------------ financial statements / valuation / segments
def _last_actual(m, item, section=None):
    """Latest non-estimate value of `item` (estimate columns end with 'E'); (nan, None) when absent."""
    i = m.row(item, section)
    if i is None:
        return np.nan, None
    for p in reversed([p for p in m.periods if not p.endswith('E')]):
        if pd.notna(m.values.at[i, p]):
            return float(m.values.at[i, p]), p
    return np.nan, None


def fundamentals_row(co):
    """One-line fundamentals for an issuer (used by the Sector Analysis tab); NaN where an export is missing."""
    row = {'Ticker': co.ticker, 'Issuer': co.name, 'Total debt ($bn)': np.nan, 'Net debt ($bn)': np.nan, 'Debt / EBITDA': np.nan,
           'Debt / capital': np.nan, 'LTM revenue ($bn)': np.nan, 'EBITDA margin': np.nan, 'Debt as of': ''}
    cs, ks = co.cap_summary, co.statements.get('key_stats')
    if cs is not None:
        td, per = cs.latest('Total Debt')
        nd, _ = cs.latest('Net Debt')
        lev, _ = cs.latest('Total Debt/EBITDA', 'Credit Ratios')
        row.update({'Total debt ($bn)': td / 1000, 'Net debt ($bn)': nd / 1000, 'Debt / EBITDA': lev,
                    'Debt / capital': cs.latest_pct('Total Debt', per) if per else np.nan, 'Debt as of': per or ''})
    if ks is not None:
        rev, _ = _last_actual(ks, 'Total Revenue')
        mg, _ = _last_actual(ks, 'EBITDA Margin %')
        row.update({'LTM revenue ($bn)': rev / 1000, 'EBITDA margin': mg})
    return row


def _operating_metrics(co):
    ks, cf = co.statements.get('key_stats'), co.statements.get('cashflow')
    if ks is None and cf is None:
        return
    c = st.columns(4)
    if ks is not None:
        rev, rp = _last_actual(ks, 'Total Revenue')
        gr, _ = _last_actual(ks, 'Total Revenue Growth Over Prior Year')
        eb, _ = _last_actual(ks, 'EBITDA')
        mg, _ = _last_actual(ks, 'EBITDA Margin %')
        ni, _ = _last_actual(ks, 'Net Income')
        c[0].metric('Revenue', fmt_mm(rev), f'{gr * 100:+.1f}% YoY' if pd.notna(gr) else None, help=f'Key Stats, {rp}')
        c[1].metric('EBITDA', fmt_mm(eb), f'{mg * 100:.1f}% margin' if pd.notna(mg) else None, delta_color='off')
        c[2].metric('Net income', fmt_mm(ni))
    if cf is not None:
        cfo, p = _last_actual(cf, 'Cash from Ops.')
        cx, _ = _last_actual(cf, 'Capital Expenditure')
        if pd.notna(cfo) and pd.notna(cx):
            fcf = cfo + cx
            c[3].metric('Free cash flow (CFO + capex)', fmt_mm(fcf), f'CFO {fmt_mm(cfo)}  |  capex {fmt_mm(cx)}', delta_color='off', help=f'Cash Flow export, {p}')


STMT_DEFAULTS = {'key_stats': ['Total Revenue', 'EBITDA'], 'income': ['Total Revenue', 'Net Income'],
                 'balance': ['Total Cash & ST Investments', 'Total Assets'], 'cashflow': ['Cash from Ops.', 'Capital Expenditure']}


def _tab_statements(co):
    avail = [k for k in cq.STATEMENT_KEYS if k in co.statements]
    if not avail:
        st.info('Add the **Key Stats / Income Statement / Balance Sheet / Cash Flow** exports (or the full multi-sheet **Financials** workbook).')
        return
    key = st.selectbox('Statement', avail, format_func=lambda k: cq.REPORT_NAMES[k], key='cq_stmt')
    m = co.statements[key]
    sections = list(dict.fromkeys(m.values['Section']))
    sec = st.selectbox('Section', ['All'] + sections, key=f'cq_stmt_sec_{key}')
    tbl = cq.display_statement(m, None if sec == 'All' else sec)
    if sec != 'All' or len(sections) <= 1:
        tbl = tbl.drop(columns='Section')
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(760, 38 + 35 * len(tbl)))
    st.caption('$ in millions of the reporting currency unless the row says otherwise (per-share rows, %). Columns ending in "E" are analyst estimates. '
               '"-" = blank in the export, NM = not meaningful.')
    items = list(dict.fromkeys(m.values['Item'] if sec == 'All' else m.values.loc[m.values['Section'] == sec, 'Item']))
    default = [i for i in STMT_DEFAULTS.get(key, []) if i in items] or items[:2]
    pick = st.multiselect('Chart rows', items, default=default, key=f'cq_stmt_items_{key}')
    style = st.radio('Chart style', ['Bars', 'Lines'], horizontal=True, key=f'cq_stmt_style_{key}')
    if pick:
        long = cq.chart_frame(m, pick, None if sec == 'All' else sec)
        if not long.empty:
            if style == 'Bars':
                fig = px.bar(long, x='Period', y='Value', color='Item', barmode='group', category_orders={'Period': m.periods})
            else:
                fig = px.line(long, x='Period', y='Value', color='Item', markers=True, category_orders={'Period': m.periods})
            fig.update_layout(title='Selected rows (percent rows in %, others as exported)', height=420)
            st.plotly_chart(fig, use_container_width=True)
    if key == 'cashflow':
        _fcf_block(m)
    if key == 'key_stats':
        lc, cm = m.extra.get('latest_cap'), m.extra.get('current_multiples')
        if lc is not None and not lc.empty:
            with st.expander('Latest capitalization (from Key Stats)'):
                t = lc.assign(Value=lc['Value'].map(lambda v: cq.fmt_value(v, 'auto')))
                st.dataframe(t, hide_index=True, use_container_width=True)
        if cm is not None:
            with st.expander('Valuation multiples on current capitalization'):
                st.dataframe(cq.display_statement(cm, kind_fn=cq.multiple_kind).drop(columns='Section'), hide_index=True, use_container_width=True)


def _fcf_block(cf):
    needed = ['Cash from Ops.', 'Capital Expenditure']
    if any(cf.row(n) is None for n in needed):
        return
    v = cf.values
    cfo, cx = v.loc[cf.row('Cash from Ops.'), cf.periods], v.loc[cf.row('Capital Expenditure'), cf.periods]
    div_i = cf.row('Total Dividends Paid')
    div = v.loc[div_i, cf.periods].fillna(0) if div_i is not None else 0
    fcf = cfo + cx
    out = pd.DataFrame({'Period': cf.periods, 'Cash from operations': cfo.values, 'Capital expenditure': cx.values,
                        'Free cash flow': fcf.values, 'FCF after dividends': (fcf + div).values}).dropna(subset=['Free cash flow'])
    long = out.melt('Period', var_name='Series', value_name='$mm')
    fig = px.bar(long, x='Period', y='$mm', color='Series', barmode='group', category_orders={'Period': cf.periods})
    fig.update_layout(title='Cash generation vs. investment ($mm; capex shown as outflow)', height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Free cash flow here = cash from operations + capital expenditure (capex is negative in the export). '
               'Debt issued / repaid and dividends are on the Cash Flow table above.')


def _tab_valuation(co):
    cap, mu, ks = co.statements.get('capitalization'), co.statements.get('multiples'), co.statements.get('key_stats')
    if cap is None and mu is None and ks is None:
        st.info('Add the **Multiples** and **Capitalization** exports (or the full multi-sheet **Financials** workbook).')
        return
    if cap is not None:
        st.markdown('**Historical capitalization**')
        st.dataframe(cq.display_statement(cap).drop(columns='Section'), hide_index=True, use_container_width=True)
        wanted = ['Market Capitalization', '= Total Enterprise Value (TEV)', '+ Total Debt']
        have = [i for i in wanted if cap.row(i) is not None]
        if have:
            long = cq.chart_frame(cap, have)
            fig = px.bar(long, x='Period', y='Value', color='Item', barmode='group', category_orders={'Period': cap.periods})
            fig.update_layout(title='Market cap, enterprise value and debt ($mm)', height=420)
            st.plotly_chart(fig, use_container_width=True)
        st.caption('Pricing is as of each balance-sheet filing date, so equity value moves with the share price at that date.')
    if mu is not None:
        st.markdown('**Trading multiples**')
        names = list(dict.fromkeys(mu.values['Section']))
        mult = st.selectbox('Multiple', names, key='cq_mult')
        stats = [x for x in mu.values.loc[mu.values['Section'] == mult, 'Item']]
        pick = st.multiselect('Statistic', stats, default=[x for x in ['Average', 'Close'] if x in stats], key='cq_mult_stats')
        sub = mu.values[(mu.values['Section'] == mult) & (mu.values['Item'].isin(pick))]
        if not sub.empty:
            long = sub.melt(id_vars=['Section', 'Item'], value_vars=mu.periods, var_name='Period', value_name='x').dropna()
            fig = px.line(long, x='Period', y='x', color='Item', markers=True, category_orders={'Period': mu.periods})
            fig.update_layout(title=f'{mult} (x)', height=380)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cq.display_statement(mu, mult, cq.multiple_kind).drop(columns='Section'), hide_index=True, use_container_width=True)
    if ks is not None and ks.extra.get('current_multiples') is not None:
        st.markdown('**Multiples on current capitalization (Key Stats)**')
        st.dataframe(cq.display_statement(ks.extra['current_multiples'], kind_fn=cq.multiple_kind).drop(columns='Section'), hide_index=True, use_container_width=True)


def _tab_segments(co):
    sg = co.statements.get('segments')
    if sg is None:
        st.info('Add the **Segments** export (or the full multi-sheet **Financials** workbook).')
        return
    sections = list(dict.fromkeys(sg.values['Section']))
    sec = st.selectbox('Segment view', sections, key='cq_seg')
    st.dataframe(cq.display_statement(sg, sec).drop(columns='Section'), hide_index=True, use_container_width=True)
    v = sg.values[(sg.values['Section'] == sec) & ~sg.values['Item'].str.lower().str.startswith('total')]
    if not v.empty:
        long = v.melt(id_vars=['Section', 'Item'], value_vars=sg.periods, var_name='Period', value_name='$mm').dropna()
        fig = px.bar(long, x='Period', y='$mm', color='Item', barmode='stack', category_orders={'Period': sg.periods})
        fig.update_layout(title=f'{sec} ($mm)', height=420)
        st.plotly_chart(fig, use_container_width=True)
    block = sec.split(' \u203a ')[0]
    rev = next((s for s in sections if s.startswith(block) and 'Revenue' in s), None)
    prof = next((s for s in sections if s.startswith(block) and 'Operating Profit' in s), None)
    if rev and prof:
        r = sg.values[sg.values['Section'] == rev].set_index('Item')[sg.periods]
        p = sg.values[sg.values['Section'] == prof].set_index('Item')[sg.periods]
        common = [i for i in r.index if i in p.index and not i.lower().startswith('total')]
        if common:
            mg = (p.loc[common] / r.loc[common].where(r.loc[common] != 0) * 100).reset_index().melt('Item', var_name='Period', value_name='Margin %').dropna()
            fig = px.line(mg, x='Period', y='Margin %', color='Item', markers=True, category_orders={'Period': sg.periods})
            fig.update_layout(title='Segment operating margin before unallocated costs (%)', height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Margin = segment operating profit / segment revenue, before unallocated corporate costs, so it overstates company-level margin.')


# ------------------------------------------------------------------------------------------------ entry point
def render_capiq_tab(capiq_dir, bonds=None):
    capiq_dir = Path(capiq_dir)
    capiq_dir.mkdir(parents=True, exist_ok=True)
    st.write('Issuer fundamentals from S&P Capital IQ exports: financial statements, debt stack, leverage, coverage, valuation, segments and the Capital IQ bond list.')
    _uploader(capiq_dir)
    try:
        companies = get_companies(capiq_dir)
    except Exception as e:  # defensive: loader already isolates bad files
        st.error(f'Could not read Capital IQ files: {e}')
        return
    stray = companies.pop('', None)
    if stray and stray.problems:
        with st.expander(f'{len(stray.problems)} file(s) could not be used'):
            for p in stray.problems:
                st.caption(f'• {p}')
    usable = {t: c for t, c in companies.items() if c.files}
    if not usable:
        st.info(f'No usable Capital IQ exports found in `{capiq_dir}`. Export these from Capital IQ Pro and drop them there (or use the uploader above): '
                'the multi-sheet **Financials** workbook (Key Stats, Income Statement, Balance Sheet, Cash Flow, Multiples, Capitalization, Capital Structure, Ratios, Segments) '
                'and/or **Fixed Income > Securities Summary**; single-page exports work too.')
        for c in companies.values():
            for p in c.problems:
                st.caption(f'• {c.ticker}: {p}')
        return
    tickers = sorted(usable)
    pick = st.selectbox('Issuer', tickers, format_func=lambda t: f'{usable[t].name} ({t})', key='cq_issuer')
    co = usable[pick]
    _inventory(co)
    _headline(co)
    _operating_metrics(co)
    t1, t2, t3, t4, t5, t6, t7 = st.tabs(['Capital structure', 'Debt detail', 'Maturities & bonds', 'Ratios',
                                          'Financial statements', 'Valuation', 'Segments'])
    with t1:
        _tab_cap_structure(co)
    with t2:
        _tab_debt_detail(co)
    with t3:
        _tab_bonds(co, bonds)
    with t4:
        _tab_ratios(co)
    with t5:
        _tab_statements(co)
    with t6:
        _tab_valuation(co)
    with t7:
        _tab_segments(co)
    st.caption('Research tool only. Capital IQ data is proprietary - keep exports within your licence terms.')
