"""story.py - turns Capital IQ data + bond data into a credit story (automated read) and stores analyst notes per issuer / sector."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ISSUER_FIELDS = [('thesis', 'Credit thesis (one paragraph)'), ('management', 'Management & strategy'), ('operations', 'Operations & business model'),
                 ('catalysts', 'Catalysts / what to watch'), ('risks', 'Key risks'), ('governance', 'Governance, ESG & financial policy')]
SECTOR_FIELDS = [('thesis', 'Sector credit thesis'), ('drivers', 'Demand & earnings drivers'), ('operations', 'Operations, costs & margins'),
                 ('management', 'What management teams are signalling'), ('regulation', 'Regulation, policy & macro'), ('risks', 'Key risks & tail scenarios'),
                 ('outlook', 'Outlook (12-24 months)')]
STANCES = ['Not set', 'Positive', 'Neutral', 'Negative']


# ------------------------------------------------------------------------------------------------ notes (plain JSON files)
def _path(directory, kind, key):
    return Path(directory) / f'{kind}__{re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))}.json'


def load_notes(directory, kind, key):
    p = _path(directory, kind, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_notes(directory, kind, key, data):
    Path(directory).mkdir(parents=True, exist_ok=True)
    data = dict(data, updated=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    _path(directory, kind, key).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


# ------------------------------------------------------------------------------------------------ series helpers
def series(m, item, section=None):
    """Item row of a Capital IQ Matrix as a float Series indexed by period label (None if absent)."""
    if m is None:
        return None
    i = m.row(item, section)
    return None if i is None else m.values.loc[i, m.periods].astype(float)


def last_actual(m, item, section=None):
    """(value, period) of the right-most non-estimate, non-empty period."""
    s = series(m, item, section)
    if s is None:
        return np.nan, None
    for p in reversed([p for p in s.index if not str(p).endswith('E')]):
        if pd.notna(s[p]):
            return float(s[p]), p
    return np.nan, None


def prior_actual(m, item, section=None, skip=1):
    """Value `skip` annual (FY) periods before the latest annual period - a like-for-like comparison base."""
    s = series(m, item, section)
    if s is None:
        return np.nan, None
    fy = [p for p in s.index if str(p).startswith('FY') and not str(p).endswith('E') and pd.notna(s[p])]
    return (float(s[fy[-1 - skip]]), fy[-1 - skip]) if len(fy) > skip else (np.nan, None)


def mm(v):
    if v is None or pd.isna(v):
        return 'n/a'
    sign, a = ('-' if v < 0 else ''), abs(v)
    return f'{sign}${a / 1000:,.1f}bn' if a >= 1000 else f'{sign}${a:,.0f}mm'


def pct(v, d=1):
    return 'n/a' if v is None or pd.isna(v) else f'{v * 100:.{d}f}%'


# ------------------------------------------------------------------------------------------------ facts
def facts(co, bonds, mcol, sector_med=None, bucket_bench=None):
    """Collect the numbers the story needs. `co` may be None (market-only issuer)."""
    F = {'has_capiq': co is not None}
    if bonds is not None and len(bonds):
        w = bonds['Outstanding Amount'].clip(lower=1)
        F.update(n_bonds=len(bonds), outstanding=float(bonds['Outstanding Amount'].sum() / 1e6),
                 ytm=float(np.average(bonds['YTM'].fillna(bonds['YTM'].median()), weights=w)),
                 spread=float(np.average(bonds[mcol].fillna(bonds[mcol].median()), weights=w)) if mcol in bonds else np.nan,
                 maturity=float(np.average(bonds['Years to Maturity'].fillna(bonds['Years to Maturity'].median()), weights=w)),
                 ratings=sorted(bonds['Rating'].dropna().unique(), key=lambda r: bonds.loc[bonds['Rating'] == r, 'Notch'].min()),
                 sector=bonds['Sector'].iloc[0], industry=bonds['Industry'].iloc[0] if 'Industry' in bonds else '', universe=bonds['Universe'].mode().iloc[0],
                 bucket=bonds['Rating Bucket'].mode().iloc[0] if 'Rating Bucket' in bonds else '')
        F['sector_med'], F['bucket_bench'] = sector_med, bucket_bench
        near = bonds[bonds['Years to Maturity'] <= 3]['Outstanding Amount'].sum() / 1e6
        F['bond_due_3y'] = float(near)
    if co is None:
        return F
    ks, cf, bs, cs, rt = co.statements.get('key_stats'), co.statements.get('cashflow'), co.statements.get('balance'), co.cap_summary, co.ratios
    mc, cap = co.statements.get('multiples'), co.statements.get('capitalization')
    F['name'] = co.name
    F['rev'], F['rev_p'] = last_actual(ks, 'Total Revenue')
    F['growth'], _ = last_actual(ks, 'Total Revenue Growth Over Prior Year')
    s = series(ks, 'Total Revenue Growth Over Prior Year')
    est = [p for p in (s.index if s is not None else []) if str(p).endswith('E') and pd.notna(s[p])]
    F['growth_est'] = float(s[est[0]]) if est else np.nan
    F['growth_est_p'] = est[0] if est else None
    F['ebitda'], _ = last_actual(ks, 'EBITDA')
    F['margin'], F['margin_p'] = last_actual(ks, 'EBITDA Margin %')
    F['margin_prior'], _ = prior_actual(ks, 'EBITDA Margin %', skip=2)
    F['cfo'], F['cf_p'] = last_actual(cf, 'Cash from Ops.')
    F['capex'], _ = last_actual(cf, 'Capital Expenditure')
    F['div'], _ = last_actual(cf, 'Total Dividends Paid')
    F['buyback'], _ = last_actual(cf, 'Repurchase of Common Stock')
    F['debt_issued'], _ = last_actual(cf, 'Total Debt Issued')
    F['debt_repaid'], _ = last_actual(cf, 'Total Debt Repaid')
    F['net_debt_issued'], _ = last_actual(cf, 'Net Debt Issued')
    F['fcf'] = F['cfo'] + F['capex'] if pd.notna(F['cfo']) and pd.notna(F['capex']) else np.nan
    F['fcf_after'] = F['fcf'] + (F['div'] if pd.notna(F['div']) else 0) + (F['buyback'] if pd.notna(F['buyback']) else 0) if pd.notna(F['fcf']) else np.nan
    F['capex_rev'] = -F['capex'] / F['rev'] if pd.notna(F['capex']) and pd.notna(F['rev']) and F['rev'] else np.nan
    if cs is not None:
        F['debt'], F['debt_p'] = cs.latest('Total Debt')
        F['cash'], _ = cs.latest('Total Cash & ST Investments')
        F['net_debt'], _ = cs.latest('Net Debt')
        F['lev'], F['lev_p'] = cs.latest('Total Debt/EBITDA', 'Credit Ratios')
        F['due1'], _ = cs.latest('LT Debt (Incl. Cap. Leases) Due +1')
        due3 = [cs.latest(f'LT Debt (Incl. Cap. Leases) Due +{k}')[0] for k in (1, 2, 3)]
        F['due3'] = float(np.nansum(due3)) if not all(pd.isna(due3)) else np.nan
    lev_s = series(rt, 'Total Debt/EBITDA')
    if lev_s is not None:
        fy = [p for p in lev_s.index if pd.notna(lev_s[p])]
        if len(fy) >= 2:
            F['lev_prev'], F['lev_prev_p'] = float(lev_s[fy[-2]]), fy[-2]
            F['lev_now'], F['lev_now_p'] = float(lev_s[fy[-1]]), fy[-1]
    F['cov'], _ = last_actual(rt, 'EBITDA / Interest Exp.')
    F['debt_trend'] = None
    if bs is not None:
        d = series(bs, 'Total Debt')
        if d is not None and d.notna().sum() >= 2:
            ds = d.dropna()
            F['debt_trend'] = (ds.index[0], float(ds.iloc[0]), ds.index[-1], float(ds.iloc[-1]))
        F['cash_bs'], _ = last_actual(bs, 'Total Cash & ST Investments')
    if mc is not None:
        i = mc.row('Close', 'TEV/LTM EBITDA')
        if i is not None:
            row = mc.values.loc[i, mc.periods].astype(float).dropna()
            if len(row) >= 3:
                F['tev_ebitda'], F['tev_ebitda_avg'] = float(row.iloc[-1]), float(row.mean())
    if cap is not None:
        F['mcap'], _ = last_actual(cap, 'Market Capitalization')
    lc = ks.extra.get('latest_cap') if ks is not None else None
    if lc is not None and not lc.empty:
        for item, key in [('= Total Enterprise Value (TEV)', 'tev'), ('Market Capitalization', 'mcap')]:
            hit = lc[lc['Item'] == item]
            if len(hit):
                F[key] = float(hit['Value'].iloc[0])
    if 'tev' not in F and cap is not None:
        F['tev'], _ = last_actual(cap, '= Total Enterprise Value (TEV)')
    chp, fin = co.statements.get('chp_summary'), co.statements.get('chp_financials')
    if chp is not None:
        F['chp'] = {'overall': chp['overall'], 'categories': dict(chp['categories']), 'pd': chp['pd'], 'cm': chp['creditmodel'], 'n_peers': len(chp['peers'])}
        if fin is not None and len(fin):
            last_p = fin['Period'].max()
            cur = fin[(fin['Period'] == last_p) & fin['Score'].notna()]
            F['chp']['weak'] = cur[cur['Score'] >= 4]['Metric'].tolist()
            F['chp']['strong'] = cur[cur['Score'] <= 1].sort_values('Weight', ascending=False)['Metric'].tolist()
    return F


# ------------------------------------------------------------------------------------------------ signals
def signals(F):
    """Automated observations as cards: [{title, tone, headline, bullets}]. Heuristics, not a rating."""
    out = []
    g, m, mp = F.get('growth'), F.get('margin'), F.get('margin_prior')
    if pd.notna(F.get('rev', np.nan)):
        tone = 'good' if pd.notna(g) and g >= 0.08 else ('bad' if pd.notna(g) and g < 0 else 'neutral')
        b = [f'Revenue {mm(F["rev"])} ({F.get("rev_p")}); growth {pct(g)} year on year.']
        if pd.notna(F.get('growth_est', np.nan)):
            b.append(f'Consensus implies {pct(F["growth_est"])} growth in {F["growth_est_p"]}.')
        out.append({'title': 'Growth', 'tone': tone, 'headline': 'Top line ' + ('accelerating' if tone == 'good' else 'shrinking' if tone == 'bad' else 'steady'), 'bullets': b})
    if pd.notna(m):
        d = (m - mp) * 100 if pd.notna(mp) else np.nan
        tone = 'good' if pd.notna(d) and d >= 1 else ('warn' if pd.notna(d) and d <= -1 else 'neutral')
        b = [f'EBITDA margin {pct(m)} ({F.get("margin_p")}).']
        if pd.notna(d):
            b.append(f'{d:+.1f} pts versus two fiscal years earlier ({pct(mp)}).')
        out.append({'title': 'Profitability', 'tone': tone, 'headline': 'Margins ' + ('expanding' if tone == 'good' else 'compressing' if tone == 'warn' else 'stable'), 'bullets': b})
    if pd.notna(F.get('lev', np.nan)):
        prev, now = F.get('lev_prev'), F.get('lev_now')
        dl = (now - prev) if prev is not None and now is not None else np.nan
        tone = 'warn' if pd.notna(dl) and dl > 0.5 else ('good' if pd.notna(dl) and dl < -0.25 else 'neutral')
        b = [f'Total debt {mm(F.get("debt"))} against EBITDA of {mm(F.get("ebitda"))}: {F["lev"]:.2f}x ({F.get("lev_p")}).']
        if pd.notna(dl):
            b.append(f'{dl:+.2f}x versus {F["lev_prev_p"]} ({prev:.2f}x -> {now:.2f}x, Capital IQ ratios page).')
        if F.get('debt_trend'):
            p0, v0, p1, v1 = F['debt_trend']
            b.append(f'Debt has moved from {mm(v0)} ({p0}) to {mm(v1)} ({p1}).')
        if pd.notna(F.get('cov', np.nan)):
            b.append(f'EBITDA / interest {F["cov"]:.1f}x (latest period Capital IQ populates).')
        head = 'Leverage ' + ('rising' if tone == 'warn' else 'falling' if tone == 'good' else 'little changed')
        if F.get('debt_trend'):
            p0, v0, p1, v1 = F['debt_trend']
            if v0 and v1 / v0 - 1 > 0.5:              # a falling ratio can hide a debt build - say so
                tone = 'warn'
                head = ('Ratio easing, but absolute debt up ' if pd.notna(dl) and dl < 0 else 'Leverage and debt both rising: debt up ') + f'{(v1 / v0 - 1) * 100:.0f}%'
        out.append({'title': 'Leverage', 'tone': tone, 'headline': head, 'bullets': b})
    if pd.notna(F.get('cfo', np.nan)):
        fcf, fa = F.get('fcf'), F.get('fcf_after')
        tone = 'bad' if pd.notna(fcf) and fcf < 0 else ('warn' if pd.notna(fa) and fa < 0 else 'good')
        b = [f'Cash from operations {mm(F["cfo"])}, capex {mm(abs(F["capex"]))} ({pct(F.get("capex_rev"))} of revenue) -> free cash flow {mm(fcf)} ({F.get("cf_p")}).']
        if pd.notna(F.get('div', np.nan)) or pd.notna(F.get('buyback', np.nan)):
            b.append(f'Dividends {mm(abs(F.get("div", 0) or 0))} and buybacks {mm(abs(F.get("buyback", 0) or 0))} take free cash flow to {mm(fa)} after shareholder returns.')
        if pd.notna(F.get('net_debt_issued', np.nan)):
            b.append(f'Net debt issued {mm(F["net_debt_issued"])}' + (' - the investment programme is being funded with borrowing.' if pd.notna(fcf) and fcf < 0 and F['net_debt_issued'] > 0 else '.'))
        head = 'Self-funding' if tone == 'good' else ('Funding gap covered by debt' if tone == 'bad' else 'Self-funding before shareholder returns')
        out.append({'title': 'Cash flow & funding', 'tone': tone, 'headline': head, 'bullets': b})
    if pd.notna(F.get('due1', np.nan)) and pd.notna(F.get('cash', np.nan)):
        cover = F['cash'] / F['due1'] if F['due1'] else np.inf
        share = F['due1'] / F['debt'] if F.get('debt') else np.nan
        tone = 'good' if cover >= 1.5 else ('warn' if cover >= 0.75 else 'bad')
        b = [f'Cash and ST investments {mm(F["cash"])} versus {mm(F["due1"])} of debt due in the next year ({cover:.1f}x cover; {pct(share)} of total debt).']
        if pd.notna(F.get('due3', np.nan)) and F.get('debt'):
            b.append(f'{mm(F["due3"])} ({pct(F["due3"] / F["debt"])}) falls due within three years.')
        out.append({'title': 'Liquidity & maturities', 'tone': tone, 'headline': 'Near-term maturities ' + ('well covered' if tone == 'good' else 'need refinancing' if tone != 'good' else ''), 'bullets': b})
    if 'n_bonds' in F:
        sp, sm, bb = F.get('spread'), F.get('sector_med'), F.get('bucket_bench')
        b = [f'{F["n_bonds"]} bond(s), {mm(F["outstanding"])} outstanding, rated {" / ".join(F["ratings"]) or "NR"}; weighted yield {F["ytm"]:.2f}%.']
        tone = 'info'
        head = 'Priced at ' + (f'{sp:.0f}bp' if pd.notna(sp) else 'n/a')
        if pd.notna(sp) and sm is not None and pd.notna(sm):
            d = sp - sm
            b.append(f'{d:+.0f}bp versus the {F["sector"]} sector average ({sm:.0f}bp).')
            tone = 'warn' if d > 75 else ('good' if d < -25 else 'info')
        if pd.notna(sp) and bb is not None and pd.notna(bb):
            b.append(f'{sp - bb:+.0f}bp versus the rating-bucket benchmark ({bb:.0f}bp).')
        out.append({'title': 'Market view', 'tone': tone, 'headline': head, 'bullets': b})
    ch = F.get('chp')
    if ch:
        tone = {'Top': 'good', 'Above Average': 'info', 'Below Average': 'warn', 'Bottom': 'bad'}.get(ch['overall'], 'neutral')
        b = [', '.join(f'{k} {v}' for k, v in ch['categories'].items()) + f' (vs {ch["n_peers"]} public industry peers).']
        if pd.notna(ch.get('pd', np.nan)):
            b.append(f'Market-signal default probability {ch["pd"]:.3f}%; CreditModel score {ch["cm"]}' + (f' vs. screener rating {F["ratings"][0]}.' if F.get('ratings') else '.'))
        if ch.get('weak'):
            b.append('Weakest versus peers: ' + ', '.join(ch['weak'][:4]) + '.')
        if ch.get('strong'):
            b.append('Strongest: ' + ', '.join(ch['strong'][:4]) + '.')
        out.append({'title': 'Peer credit health', 'tone': tone, 'headline': f'{ch["overall"]} quartile overall among peers', 'bullets': b})
    if pd.notna(F.get('tev_ebitda', np.nan)):
        r = F['tev_ebitda'] / F['tev_ebitda_avg'] - 1
        b = [f'TEV / LTM EBITDA {F["tev_ebitda"]:.1f}x versus {F["tev_ebitda_avg"]:.1f}x average over the exported quarters ({r * 100:+.0f}%).']
        if pd.notna(F.get('mcap', np.nan)) and F.get('debt'):
            b.append(f'Market cap {mm(F["mcap"])} is {F["mcap"] / F["debt"]:.1f}x total debt - the equity cushion beneath the bonds.')
        out.append({'title': 'Equity cushion', 'tone': 'good' if r < 0 else 'neutral', 'headline': 'Valuation ' + ('below' if r < 0 else 'above') + ' recent average', 'bullets': b})
    return out


def headline(F):
    """One-sentence summary built from the most decision-relevant facts available."""
    parts = []
    name = F.get('name') or 'This issuer'
    if pd.notna(F.get('growth', np.nan)) and pd.notna(F.get('margin', np.nan)):
        parts.append(f'revenue {F["growth"] * 100:+.1f}% with a {F["margin"] * 100:.0f}% EBITDA margin')
    if pd.notna(F.get('fcf', np.nan)):
        verb = 'burning' if F['fcf'] < 0 else 'generating'
        parts.append(f'{verb} {mm(abs(F["fcf"]))} of free cash flow' + (f' (capex {mm(abs(F["capex"]))})' if pd.notna(F.get('capex', np.nan)) and F['fcf'] < 0 else ''))
    if pd.notna(F.get('lev', np.nan)):
        parts.append(f'{F["lev"]:.1f}x total debt / EBITDA')
    if parts:
        return f'{name}: ' + '; '.join(parts) + '.'
    if 'n_bonds' in F:
        return f'{F.get("name") or "This issuer"}: {F["n_bonds"]} bond(s), {mm(F["outstanding"])} outstanding, priced at {F["spread"]:.0f}bp.'
    return 'No data loaded for this issuer.'
