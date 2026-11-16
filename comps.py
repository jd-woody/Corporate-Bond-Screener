"""comps.py - credit analytics behind the Credit Profile's comps, ratio, scenario, recovery and capital-stack views (pure pandas / numpy).

Ideas ported from the fixed-income Excel templates: credit-ratio glossary (Comps), Conservative / Base (Street) / Optimistic projections (Forecast),
tranche-by-tranche leverage and loan-to-value (Capital Structure), distressed recovery waterfall with fulcrum security (Opportunistic),
price-yield curve with duration and convexity approximations (Convexity).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import analytics as an
import story as S

NOTCH_LOWER = ['aaa', 'aa+', 'aa', 'aa-', 'a+', 'a', 'a-', 'bbb+', 'bbb', 'bbb-', 'bb+', 'bb', 'bb-', 'b+', 'b', 'b-', 'ccc+', 'ccc', 'ccc-', 'cc', 'c', 'd']
SUFFIX = {'inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'llc', 'plc', 'sa', 'ag', 'nv', 'holdings', 'holding', 'group', 'technologies',
          'technology', 'lp', 'the', 'de', 'cv', 'bv', 'se', 'spa', 'sab', 'international', 'intl', 'enterprise', 'enterprises'}


def notch_num(r):
    """Rating text (any case) -> notch 1 (AAA) ... 22 (D); NaN if unknown."""
    if not isinstance(r, str):
        return np.nan
    r = r.strip().lower()
    return float(NOTCH_LOWER.index(r) + 1) if r in NOTCH_LOWER else np.nan


# ------------------------------------------------------------------------------------------------ peer matching
def name_key(s):
    s = re.sub(r'\([^)]*\)', ' ', str(s).lower())
    toks = [t for t in re.sub(r'[^a-z0-9]+', ' ', s).split() if t not in SUFFIX]
    return ' '.join(toks)


def match_peers(peers, names):
    """{panel peer ticker: screener ticker} using the ticker first, then the normalised company name. `names` = {screener ticker: issuer name}."""
    keys = {}
    for t, n in names.items():
        keys.setdefault(name_key(n), t)
    out = {}
    for _, r in peers.iterrows():
        pt, pk = str(r['Ticker']), name_key(r['Name'])
        if pt in names:
            out[pt] = pt
        elif pk and pk in keys:
            out[pt] = keys[pk]
        elif pk and len(pk) >= 4:
            hit = next((t for k, t in keys.items() if k.startswith(pk + ' ')), None)
            if hit:
                out[pt] = hit
    return out


# ------------------------------------------------------------------------------------------------ ratio toolkit
def _norm(p):
    return re.sub(r'^(FY|LTM|\d+M) ', '', str(p))


def _get(m, item, section=None):
    s = S.series(m, item, section)
    if s is None:
        return None
    s = s.copy()
    s.index = [_norm(p) for p in s.index]
    return s[~s.index.duplicated(keep='last')]


RATIOS = [  # family, name, formula, unit, better
    ('Liquidity', 'Current ratio', 'Current assets / current liabilities', 'x', 'higher'),
    ('Liquidity', 'Quick ratio', '(Cash & ST inv. + receivables) / current liabilities', 'x', 'higher'),
    ('Liquidity', 'Cash ratio', 'Cash & ST inv. / current liabilities', 'x', 'higher'),
    ('Liquidity', 'Operating cash flow ratio', 'Cash from operations / current liabilities', 'x', 'higher'),
    ('Liquidity', 'Defensive interval (days)', '(Cash & ST inv. + receivables) / daily operating costs', 'days', 'higher'),
    ('Liquidity', 'Working capital ($mm)', 'Current assets - current liabilities', '$mm', 'higher'),
    ('Liquidity', 'Working capital / revenue', 'Working capital / revenue', '%', 'higher'),
    ('Leverage', 'Total debt / EBITDA', 'Total debt / EBITDA', 'x', 'lower'),
    ('Leverage', 'Net debt / EBITDA', '(Total debt - cash & ST inv.) / EBITDA', 'x', 'lower'),
    ('Leverage', 'Net debt / (EBITDA - capex)', 'Net debt / (EBITDA - capital expenditure)', 'x', 'lower'),
    ('Leverage', 'Debt / capital', 'Total debt / (total debt + total equity)', '%', 'lower'),
    ('Leverage', 'Debt / assets', 'Total debt / total assets', '%', 'lower'),
    ('Leverage', 'Debt / equity', 'Total debt / total equity', 'x', 'lower'),
    ('Leverage', 'Liabilities / assets', 'Total liabilities / total assets', '%', 'lower'),
    ('Leverage', 'Debt / cash from operations', 'Total debt / cash from operations', 'x', 'lower'),
    ('Leverage', 'Debt / free cash flow', 'Total debt / FCF (only when FCF > 0)', 'x', 'lower'),
    ('Coverage', 'EBIT / interest', 'EBIT / interest expense', 'x', 'higher'),
    ('Coverage', 'EBITDA / interest', 'EBITDA / interest expense', 'x', 'higher'),
    ('Coverage', '(EBITDA - capex) / interest', '(EBITDA - capex) / interest expense', 'x', 'higher'),
    ('Coverage', 'CFO / interest', 'Cash from operations / interest expense', 'x', 'higher'),
    ('Coverage', 'FCF / interest', 'Free cash flow / interest expense', 'x', 'higher'),
    ('Coverage', 'CFO / debt', 'Cash from operations / total debt', '%', 'higher'),
    ('Coverage', 'FCF / debt', 'Free cash flow / total debt', '%', 'higher'),
    ('Coverage', 'EBITDA / debt', 'EBITDA / total debt', '%', 'higher'),
    ('Coverage', 'Interest burden (% revenue)', 'Interest expense / revenue', '%', 'lower'),
    ('Profitability', 'Gross margin', 'Gross profit / revenue', '%', 'higher'),
    ('Profitability', 'EBITDA margin', 'EBITDA / revenue', '%', 'higher'),
    ('Profitability', 'EBIT margin', 'EBIT / revenue', '%', 'higher'),
    ('Profitability', 'Net margin', 'Net income / revenue', '%', 'higher'),
    ('Profitability', 'CFO margin', 'Cash from operations / revenue', '%', 'higher'),
    ('Profitability', 'FCF margin', '(CFO - capex) / revenue', '%', 'higher'),
    ('Profitability', 'Return on assets', 'Net income / total assets', '%', 'higher'),
    ('Profitability', 'Return on equity', 'Net income / total equity', '%', 'higher'),
    ('Profitability', 'Pre-tax return on invested capital', 'EBIT / (debt + equity - cash)', '%', 'higher'),
    ('Turnover', 'Days sales outstanding', 'Receivables / revenue x 365', 'days', 'lower'),
    ('Turnover', 'Days inventory outstanding', 'Inventory / cost of goods sold x 365', 'days', 'lower'),
    ('Turnover', 'Days payables outstanding', 'Payables / cost of goods sold x 365', 'days', 'higher'),
    ('Turnover', 'Cash conversion cycle (days)', 'DSO + DIO - DPO', 'days', 'lower'),
    ('Turnover', 'Asset turnover', 'Revenue / total assets', 'x', 'higher'),
]
RATIO_META = pd.DataFrame(RATIOS, columns=['Family', 'Ratio', 'Formula', 'Unit', 'Better'])


def ratio_frame(co):
    """(DataFrame ratio x period label, in the units of RATIO_META['Unit'] with % already x100). None if there is no income statement."""
    inc, bs, cf = co.statements.get('income'), co.statements.get('balance'), co.statements.get('cashflow')
    if inc is None:
        return None
    labels = {_norm(p): p for p in inc.periods}
    idx = list(labels)
    z = pd.Series(np.nan, index=idx)

    def g(m, item, default=None):
        s = _get(m, item) if m is not None else None
        if s is None:
            return z.copy() if default is None else pd.Series(default, index=idx)
        return s.reindex(idx)
    rev, cogs, gp = g(inc, 'Total Revenue'), g(inc, 'Cost Of Goods Sold'), g(inc, 'Gross Profit')
    ebit, ebitda, opinc, ni = g(inc, 'EBIT'), g(inc, 'EBITDA'), g(inc, 'Operating Income'), g(inc, 'Net Income')
    intexp = g(inc, 'Interest Expense').abs()
    cfo, capex = g(cf, 'Cash from Ops.'), g(cf, 'Capital Expenditure').abs()
    cash, ar, ap = g(bs, 'Total Cash & ST Investments'), g(bs, 'Total Receivables'), g(bs, 'Accounts Payable')
    inv = g(bs, 'Inventory').fillna(0.0) if bs is not None and _get(bs, 'Inventory') is not None else z.copy()
    tca, tcl, ta, tl = g(bs, 'Total Current Assets'), g(bs, 'Total Current Liabilities'), g(bs, 'Total Assets'), g(bs, 'Total Liabilities')
    debt, eq = g(bs, 'Total Debt'), g(bs, 'Total Equity')
    fcf = cfo - capex
    nd = debt - cash
    wc = tca - tcl
    opex = (rev - opinc).where(lambda s: s > 0)
    dso, dio, dpo = ar / rev * 365, inv / cogs * 365, ap / cogs * 365
    div = lambda a, b: (a / b.where(b != 0))
    R = {
        'Current ratio': div(tca, tcl), 'Quick ratio': div(cash + ar, tcl), 'Cash ratio': div(cash, tcl), 'Operating cash flow ratio': div(cfo, tcl),
        'Defensive interval (days)': (cash + ar) / (opex / 365), 'Working capital ($mm)': wc, 'Working capital / revenue': div(wc, rev) * 100,
        'Total debt / EBITDA': div(debt, ebitda), 'Net debt / EBITDA': div(nd, ebitda), 'Net debt / (EBITDA - capex)': div(nd, (ebitda - capex).where(lambda x: x > 0)),
        'Debt / capital': div(debt, debt + eq) * 100, 'Debt / assets': div(debt, ta) * 100, 'Debt / equity': div(debt, eq), 'Liabilities / assets': div(tl, ta) * 100,
        'Debt / cash from operations': div(debt, cfo), 'Debt / free cash flow': div(debt, fcf.where(fcf > 0)),
        'EBIT / interest': div(ebit, intexp), 'EBITDA / interest': div(ebitda, intexp), '(EBITDA - capex) / interest': div(ebitda - capex, intexp),
        'CFO / interest': div(cfo, intexp), 'FCF / interest': div(fcf, intexp), 'CFO / debt': div(cfo, debt) * 100, 'FCF / debt': div(fcf, debt) * 100,
        'EBITDA / debt': div(ebitda, debt) * 100, 'Interest burden (% revenue)': div(intexp, rev) * 100,
        'Gross margin': div(gp, rev) * 100, 'EBITDA margin': div(ebitda, rev) * 100, 'EBIT margin': div(ebit, rev) * 100, 'Net margin': div(ni, rev) * 100,
        'CFO margin': div(cfo, rev) * 100, 'FCF margin': div(fcf, rev) * 100, 'Return on assets': div(ni, ta) * 100, 'Return on equity': div(ni, eq) * 100,
        'Pre-tax return on invested capital': div(ebit, debt + eq - cash) * 100,
        'Days sales outstanding': dso, 'Days inventory outstanding': dio, 'Days payables outstanding': dpo, 'Cash conversion cycle (days)': dso + dio - dpo, 'Asset turnover': div(rev, ta),
    }
    df = pd.DataFrame(R).T
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    df.columns = [labels[c] for c in df.columns]
    return df


# ------------------------------------------------------------------------------------------------ comps table
def comps_row(co):
    ks, cf, cs, rt = co.statements.get('key_stats'), co.statements.get('cashflow'), co.cap_summary, co.ratios
    rev, _ = S.last_actual(ks, 'Total Revenue')
    row = {'Ticker': co.ticker, 'Issuer': co.name, 'Revenue ($bn)': rev / 1000 if pd.notna(rev) else np.nan,
           'Revenue growth (%)': S.last_actual(ks, 'Total Revenue Growth Over Prior Year')[0] * 100,
           'EBITDA margin (%)': S.last_actual(ks, 'EBITDA Margin %')[0] * 100}
    cm = ks.extra.get('current_multiples') if ks is not None else None
    tev = np.nan
    if cm is not None:
        i = cm.row('TEV/EBITDA')
        if i is not None:
            ltm = [p for p in cm.periods if str(p).startswith('LTM')]
            col = ltm[0] if ltm else cm.periods[0]
            tev = float(cm.values.at[i, col]) if pd.notna(cm.values.at[i, col]) else np.nan
    row['TEV / EBITDA (x)'] = tev
    debt = nd = lev = nlev = np.nan
    if cs is not None:
        debt, per = cs.latest('Total Debt')
        nd, _ = cs.latest('Net Debt')
        lev, _ = cs.latest('Total Debt/EBITDA', 'Credit Ratios')
        eqv, _ = cs.latest('Total Common Equity')
    else:
        eqv = np.nan
    ebitda, _ = S.last_actual(ks, 'EBITDA')
    row['Total debt ($bn)'] = debt / 1000 if pd.notna(debt) else np.nan
    row['Debt / EBITDA (x)'] = lev
    row['Net debt / EBITDA (x)'] = nd / ebitda if pd.notna(nd) and pd.notna(ebitda) and ebitda else np.nan
    row['Debt / equity (%)'] = debt / eqv * 100 if pd.notna(debt) and pd.notna(eqv) and eqv else np.nan
    cov, _ = S.last_actual(rt, 'EBITDA / Interest Exp.')
    row['EBITDA / interest (x)'] = cov
    cfo, _ = S.last_actual(cf, 'Cash from Ops.')
    cap, _ = S.last_actual(cf, 'Capital Expenditure')
    row['FCF margin (%)'] = (cfo + cap) / rev * 100 if pd.notna(cfo) and pd.notna(cap) and pd.notna(rev) and rev else np.nan
    return row


# ------------------------------------------------------------------------------------------------ forecast (Conservative / Base / Optimistic)
def street_path(co):
    """Estimate columns of Key Stats -> DataFrame(period, revenue, ebitda, ebit); empty when there are no estimates."""
    ks = co.statements.get('key_stats') if co is not None else None
    if ks is None:
        return pd.DataFrame(columns=['period', 'revenue', 'ebitda', 'ebit'])
    rev, ebitda, ebit = S.series(ks, 'Total Revenue'), S.series(ks, 'EBITDA'), S.series(ks, 'EBIT')
    est = [p for p in ks.periods if str(p).endswith('E') and rev is not None and pd.notna(rev.get(p, np.nan))]
    return pd.DataFrame({'period': est, 'revenue': [rev[p] for p in est], 'ebitda': [ebitda[p] if ebitda is not None else np.nan for p in est],
                         'ebit': [ebit[p] if ebit is not None else np.nan for p in est]})


def project(start, street, scen):
    """Roll the credit forward. start = dict(revenue, ebitda, debt, cash, interest_rate, tax_rate, shareholder_returns, da).
    street = DataFrame from street_path. scen = dict(ebitda_shock=[per year], capex_pct=[per year, % of revenue], rate_add_bp=0).
    Returns one row per year: revenue, ebitda, capex, interest, tax, fcf, fcf_after, debt, leverage, coverage. Borrowing funds any shortfall; surplus repays debt."""
    rows, debt = [], float(start['debt'])
    for t, r in enumerate(street.itertuples(index=False)):
        shock = scen['ebitda_shock'][min(t, len(scen['ebitda_shock']) - 1)]
        capex_pct = scen['capex_pct'][min(t, len(scen['capex_pct']) - 1)]
        ebitda = r.ebitda * (1 + shock)
        da = (r.ebitda - r.ebit) if pd.notna(r.ebit) else start.get('da', 0.0)
        rate = max(start['interest_rate'] + scen.get('rate_add_bp', 0) / 1e4, 0.0)
        interest = debt * rate
        capex = r.revenue * capex_pct / 100.0
        tax = max(ebitda - da - interest, 0.0) * start['tax_rate']
        fcf = ebitda - capex - interest - tax
        fcf_after = fcf - start['shareholder_returns']
        debt_end = max(debt - fcf_after, 0.0)
        rows.append({'period': r.period, 'revenue': r.revenue, 'ebitda': ebitda, 'capex': capex, 'interest': interest, 'tax': tax, 'fcf': fcf, 'fcf_after': fcf_after,
                     'debt': debt_end, 'leverage': debt_end / ebitda if ebitda > 0 else np.nan, 'coverage': ebitda / interest if interest > 0 else np.nan})
        debt = debt_end
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ recovery waterfall
def waterfall(tranches, distributable):
    """tranches = [(name, rank, claim)]; lower rank is paid first, pro-rata inside a rank. Returns {name: recovery fraction}."""
    left, out = float(distributable), {}
    for rank in sorted({t[1] for t in tranches}):
        grp = [t for t in tranches if t[1] == rank]
        claim = sum(t[2] for t in grp)
        pay = min(left, claim)
        for name, _, c in grp:
            out[name] = (pay / claim) if claim > 0 else 1.0
        left -= pay
    return out


def recovery_matrix(tranches, evs, cost_pct, probs, extra=0.0):
    """-> (recovery DataFrame tranche x scenario, probability-weighted Series, fulcrum {scenario: tranche})."""
    names = [t[0] for t in tranches]
    cols, fulcrum = {}, {}
    for label, ev in evs.items():
        rec = waterfall(tranches, ev * (1 - cost_pct / 100.0) + extra)   # extra = cash made available to creditors
        cols[label] = pd.Series(rec).reindex(names)
        partial = [n for n in names if rec[n] < 0.999]
        fulcrum[label] = partial[0] if partial else 'None (all claims covered)'
    df = pd.DataFrame(cols)
    w = pd.Series(probs).reindex(df.columns).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else w
    return df, (df * w).sum(axis=1), fulcrum


# ------------------------------------------------------------------------------------------------ capital stack
def seniority_class(rank):
    return 'Secured' if rank <= 2 else ('Senior unsecured' if rank == 3 else 'Subordinated')


def cap_stack(bonds, total_debt_mm=None, ebitda_mm=None, tev_mm=None):
    """Tranches in order of seniority with face, price, yield, coupon, interest and cumulative leverage / loan-to-value ('through the tranche')."""
    b = bonds[bonds['Outstanding Amount'] > 0].copy()
    b['cls'] = b['Seniority Rank'].map(seniority_class)
    rows = []
    for cls in ['Secured', 'Senior unsecured', 'Subordinated']:
        g = b[b['cls'] == cls]
        if g.empty:
            continue
        face = g['Outstanding Amount'].sum() / 1e6
        rows.append({'Tranche': cls, 'Face ($mm)': face, 'Price': an.wavg(g, 'Ask Price'), 'Coupon (%)': an.wavg(g, 'Coupon'), 'YTM (%)': an.wavg(g, 'YTM'),
                     'Interest ($mm)': float((g['Outstanding Amount'] / 1e6 * g['Coupon'].fillna(0) / 100).sum()), 'Rating': g['Rating'].mode().iloc[0] if g['Rating'].notna().any() else ''})
    bond_face = sum(r['Face ($mm)'] for r in rows)
    if total_debt_mm is not None and pd.notna(total_debt_mm) and total_debt_mm - bond_face > max(0.02 * total_debt_mm, 50):
        other = {'Tranche': 'Other debt (loans, leases, CP) *', 'Face ($mm)': total_debt_mm - bond_face, 'Price': np.nan, 'Coupon (%)': np.nan, 'YTM (%)': np.nan, 'Interest ($mm)': np.nan, 'Rating': ''}
        at = 1 if rows and rows[0]['Tranche'] == 'Secured' else 0
        rows.insert(at, other)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['Start ($mm)'] = df['Face ($mm)'].cumsum() - df['Face ($mm)']
    df['Cum. debt ($mm)'] = df['Face ($mm)'].cumsum()
    df['Market value ($mm)'] = df['Face ($mm)'] * df['Price'].fillna(100.0) / 100.0
    df['Cum. leverage (x EBITDA)'] = df['Cum. debt ($mm)'] / ebitda_mm if ebitda_mm and ebitda_mm > 0 else np.nan
    df['Loan-to-value (% TEV)'] = df['Cum. debt ($mm)'] / tev_mm * 100 if tev_mm and tev_mm > 0 else np.nan
    return df


# ------------------------------------------------------------------------------------------------ price-yield (convexity)
def price_yield(coupon, ytm, years, span_bp=300, step_bp=25):
    """Price at yield shifts, with the duration-only and duration+convexity approximations - shows what convexity is worth."""
    dur, cvx = an.risk_measures([coupon], [ytm], [years])
    dur, cvx = float(dur[0]), float(cvx[0])
    shifts = np.arange(-span_bp, span_bp + step_bp, step_bp)
    p0 = float(an._price(coupon, ytm, years))
    ok = ytm + shifts / 100.0 > 0.05
    shifts = shifts[ok]
    price = np.array([float(an._price(coupon, ytm + s / 100.0, years)) for s in shifts])
    dy = shifts / 1e4
    return pd.DataFrame({'shift_bp': shifts, 'price': price, 'duration_only': p0 * (1 - dur * dy), 'duration_convexity': p0 * (1 - dur * dy + 0.5 * cvx * dy ** 2)}), dur, cvx, p0
