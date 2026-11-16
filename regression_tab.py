"""regression_tab.py - fair-value regression: factors, dummies, WLS / robust SE / trimming, CV, diagnostics, cheap-rich, predictor."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as an
import theme as T
from ui_helpers import fix_state, nav

NUM = {
    'Years to Maturity': lambda z: z['Years to Maturity'],
    'ln(Years to Maturity)': lambda z: np.log(z['Years to Maturity'].clip(lower=0.05)),
    'Maturity Squared': lambda z: z['Years to Maturity'] ** 2,
    'Modified Duration': lambda z: z['Mod Duration'],
    'Convexity': lambda z: z['Convexity'],
    'Rating Notch': lambda z: z['Notch'],
    'Seniority Rank': lambda z: z['Seniority Rank'],
    'Log Size': lambda z: np.log(z['Outstanding Amount'].clip(lower=1)),
    'Price Premium': lambda z: z['Ask Price'] - 100,
    'Coupon': lambda z: z['Coupon'],
    'HY dummy': lambda z: (z['Universe'] == 'HY').astype(float),
    'Callable dummy': lambda z: z['Maturity Type'].fillna('').astype(str).str.contains('CALL').astype(float),
}
CATS = ['Sector', 'Rating Bucket', 'Collateral Type', 'Maturity Type', 'Country']
DEFAULT_NUM = ['Years to Maturity', 'Rating Notch', 'Seniority Rank', 'Log Size']
SCOPES = ['Pooled (IG + HY)', 'IG only', 'HY only', 'Separate IG and HY models']
MIN_LEVEL = 10


def design(z, nums, cats, levels=None):
    """Design matrix (with Intercept). Fit mode (levels=None) pools rare categories into 'Other' and drops the first level as reference;
    predict mode reuses the fitted levels."""
    cols = {'Intercept': np.ones(len(z))}
    for n in nums:
        cols[n] = NUM[n](z).astype(float).values
    X = pd.DataFrame(cols, index=z.index)
    out_levels = {}
    for c in cats:
        s = z[c].fillna('Unknown').astype(str)
        if levels is None:
            keep = s.value_counts()
            s = s.where(s.isin(keep[keep >= MIN_LEVEL].index), 'Other')
            lv = sorted(s.unique())
            out_levels[c] = lv
        else:
            lv = levels[c]
            s = s.where(s.isin(lv), 'Other' if 'Other' in lv else lv[0])
        for l in lv[1:]:
            X[f'{c} = {l}'] = (s == l).astype(float).values
    return X, out_levels


def fit_model(z, nums, cats, ln_y, wls, robust, trim_k):
    """Fit one model on frame z; returns a dict with results, or {'error': text}."""
    y_all = np.log(z['YTM']) if ln_y else z['YTM']
    X, levels = design(z, nums, cats)
    ok = X.notna().all(axis=1) & np.isfinite(y_all)
    n_missing = int((~ok).sum())
    X, y, zz = X[ok], y_all[ok], z[ok]
    keep = [c for c in X.columns if c == 'Intercept' or X[c].std() > 0]
    X = X[keep]
    if len(X) < max(30, X.shape[1] + 10):
        return {'error': f'Only {len(X)} complete rows for {X.shape[1]} terms - broaden the screen or choose fewer factors.'}
    w = np.sqrt(zz['Outstanding Amount'].values / 1e6) if wls else None
    m = an.fit_wls(X.values, y.values, w, robust)
    n_trim = 0
    if trim_k > 0:
        mask = np.abs(m['resid']) <= trim_k * m['resid'].std()
        if not mask.all() and mask.sum() >= max(30, X.shape[1] + 10):
            n_trim = int((~mask).sum())
            X, y, zz = X[mask], y[mask], zz[mask]
            w = None if w is None else w[mask]
            m = an.fit_wls(X.values, y.values, w, robust)
    fair = np.exp(m['fitted']) if ln_y else m['fitted']
    act = zz['YTM'].values
    res_bp = (act - fair) * 100.0
    out = zz.copy()
    out['Fair YTM'], out['Residual bp'] = fair, res_bp
    out['Z-score'] = res_bp / res_bp.std() if res_bp.std() > 0 else np.nan
    coefs = pd.DataFrame({'Term': X.columns, 'Coefficient': m['beta'], 'Std Error': m['se'], 't-stat': m['t'], 'p-value': m['p']})
    coefs['Sig.'] = np.select([coefs['p-value'] < 0.01, coefs['p-value'] < 0.05, coefs['p-value'] < 0.1], ['***', '**', '*'], '')
    Xs = X.drop(columns='Intercept')
    cond = float(np.linalg.cond(((Xs - Xs.mean()) / Xs.std()).values)) if Xs.shape[1] > 1 else 1.0
    xstd = {c: (1.0 if X[c].nunique() <= 2 else float(X[c].std())) for c in X.columns if c != 'Intercept'}
    cv = an.kfold_rmse(X.values, y.values, w, inv=np.exp if ln_y else None, y_true=act if ln_y else None)
    return {'out': out, 'coefs': coefs, 'n': len(X), 'k': X.shape[1], 'r2': m['r2'], 'adj_r2': m['adj_r2'], 'rmse_bp': float(np.sqrt(np.mean(res_bp ** 2))),
            'cv_bp': cv * 100.0 if np.isfinite(cv) else np.nan, 'cond': cond, 'n_missing': n_missing, 'n_trim': n_trim, 'levels': levels, 'cols': list(X.columns),
            'beta': m['beta'], 'xstd': xstd, 'nums': nums, 'cats': cats, 'ln_y': ln_y}


def _predict(model, row):
    X, _ = design(row, model['nums'], model['cats'], model['levels'])
    X = X.reindex(columns=model['cols'], fill_value=0.0)
    fit = float(X.values[0] @ model['beta'])
    return float(np.exp(fit) if model['ln_y'] else fit)


# ------------------------------------------------------------------------------------------------ sections

def _forest(m):
    """Effect of a 1-std-dev move in each numeric factor (or switching a dummy on) on yield, with 95% intervals - comparable across factors."""
    rows = []
    for _, r in m['coefs'].iterrows():
        if r['Term'] == 'Intercept':
            continue
        sd = m['xstd'].get(r['Term'], 1.0)
        eff, lo, hi = (r['Coefficient'] * sd, (r['Coefficient'] - 1.96 * r['Std Error']) * sd, (r['Coefficient'] + 1.96 * r['Std Error']) * sd)
        if m['ln_y']:
            eff, lo, hi = [(np.exp(v) - 1) * 100 for v in (eff, lo, hi)]
        else:
            eff, lo, hi = eff * 100, lo * 100, hi * 100
        rows.append({'Term': r['Term'], 'eff': eff, 'lo': lo, 'hi': hi, 'sig': r['p-value'] < 0.05})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values('eff')
    fig = go.Figure(go.Scatter(x=df['eff'], y=df['Term'], mode='markers', marker=dict(size=11, color=[T.BLUE if s_ else T.GREY for s_ in df['sig']]),
                               error_x=dict(type='data', symmetric=False, array=(df['hi'] - df['eff']).values, arrayminus=(df['eff'] - df['lo']).values, color=T.GREY)))
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text='% change in yield' if m['ln_y'] else 'bp of yield')
    st.plotly_chart(T.style(fig, h=max(320, 30 * len(df) + 120), title='What drives yield: effect of a 1-std-dev move (or switching a category on), 95% interval', legend=False), use_container_width=True)
    st.caption('Blue = statistically significant at 5%, grey = not distinguishable from zero. Standardised, so a factor in years and one in notches can be compared directly.')


def _curves(m):
    """Model-implied yield curves by rating bucket, other factors held at that bucket's typical values."""
    out = m['out']
    grid = np.linspace(1, 30, 30)
    fig = go.Figure()
    samp = out.sample(min(len(out), 1500), random_state=1) if len(out) else out
    fig.add_scatter(x=samp['Years to Maturity'], y=samp['YTM'], mode='markers', marker=dict(size=4, color='rgba(138,148,166,.25)'), name='Actual bonds', hoverinfo='skip')
    shown = 0
    for b in T.RATING_ORDER:
        sub_ = out[out['Rating Bucket'] == b] if 'Rating Bucket' in out else out.iloc[0:0]
        if len(sub_) < 10 or b == 'NR':
            continue
        row = pd.concat([sub_.iloc[[0]]] * len(grid), ignore_index=True)
        for c in ['Notch', 'Seniority Rank', 'Outstanding Amount', 'Coupon', 'Ask Price', 'YTM']:
            if c in sub_:
                row[c] = sub_[c].median()
        for c in set(['Universe', 'Maturity Type'] + list(m['cats'])):
            if c in sub_:
                row[c] = sub_[c].mode().iloc[0]
        row['Years to Maturity'] = grid
        row['Mod Duration'], row['Convexity'] = an.risk_measures(row['Coupon'].values, row['YTM'].values, grid)
        X, _ = design(row, m['nums'], m['cats'], m['levels'])
        fit = X.reindex(columns=m['cols'], fill_value=0.0).values @ m['beta']
        fair = np.exp(fit) if m['ln_y'] else fit
        fig.add_scatter(x=grid, y=fair, mode='lines', name=b, line=dict(color=T.RATING_COLORS[b], width=3.5))
        shown += 1
    if not shown:
        return
    fig.update_xaxes(rangemode='nonnegative', title_text='Years to maturity')
    fig.update_yaxes(rangemode='nonnegative', title_text='Model-implied yield (%)')
    st.plotly_chart(T.style(fig, h=470, title='What the model says a bond should yield, by rating and maturity'), use_container_width=True)
    st.caption('Each line holds the other factors at the typical bond of that rating bucket. Gaps between lines are the price of credit; the slope is the price of time.')


def _explain(m):
    out = m['out']
    opts = out.nlargest(300, 'Outstanding Amount').index.tolist()
    lbl = out['Label'].to_dict()
    fix_state('rg_exp', opts)
    i = st.selectbox('Bond (300 largest in the model)', opts, format_func=lambda k: lbl[k], key='rg_exp')
    X, _ = design(out.loc[[i]], m['nums'], m['cats'], m['levels'])
    contrib = X.reindex(columns=m['cols'], fill_value=0.0).values[0] * m['beta']
    parts, cats = [], {}
    for name, v in zip(m['cols'], contrib):
        head = name.split(' = ')[0] if ' = ' in name else None
        if head:
            cats[head] = cats.get(head, 0.0) + v
        else:
            parts.append((name, float(v)))
    parts += [(f'{h} effect', float(v)) for h, v in cats.items()]
    fit = float(contrib.sum())
    fair = float(np.exp(fit) if m['ln_y'] else fit)
    unit = 'log-yield points' if m['ln_y'] else 'yield points (%)'
    fig = go.Figure(go.Waterfall(x=[p for p, _ in parts] + ['Model fit'], y=[v for _, v in parts] + [0], measure=['relative'] * len(parts) + ['total'],
                                 text=[f'{v:+.2f}' for _, v in parts] + [f'{fit:.2f}'], textposition='outside', increasing=dict(marker=dict(color=T.RED)),
                                 decreasing=dict(marker=dict(color=T.TEAL)), totals=dict(marker=dict(color=T.INK)), connector=dict(line=dict(color='#C7CEDA'))))
    fig.update_yaxes(title_text=unit)
    st.plotly_chart(T.style(fig, h=440, title='How the model builds this bond\'s fair yield (intercept + each factor)', legend=False), use_container_width=True)
    act = float(out.at[i, 'YTM'])
    c = st.columns(3)
    c[0].metric('Model fair yield', f'{fair:.2f}%')
    c[1].metric('Actual yield', f'{act:.2f}%')
    c[2].metric('Residual', f'{(act - fair) * 100:+.0f} bp', help='Positive = the bond yields more than the model predicts (cheap to the model).')


def _lollipop(df, col, title):
    pick = pd.concat([df.nsmallest(15, col), df.nlargest(15, col)]).drop_duplicates('Label').sort_values(col)
    if pick.empty:
        return
    fig = go.Figure(go.Bar(y=pick['Label'].str.slice(0, 48), x=pick[col], orientation='h', marker_color=[T.RED if v > 0 else T.TEAL for v in pick[col]],
                           text=[f'{v:+.0f}' for v in pick[col]], textposition='outside'))
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text='bp vs. model')
    st.plotly_chart(T.style(fig, h=760, title=title, legend=False), use_container_width=True)

def _summary(models):
    if len(models) > 1:
        st.dataframe(pd.DataFrame([{'Model': k, 'Bonds': v['n'], 'Terms': v['k'], 'R\u00b2': v['r2'], 'Adj. R\u00b2': v['adj_r2'], 'RMSE (bp)': v['rmse_bp'], 'CV RMSE (bp)': v['cv_bp']}
                                   for k, v in models.items()]), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format='%.3f' if '\u00b2' in c else '%.0f') for c in ['R\u00b2', 'Adj. R\u00b2', 'RMSE (bp)', 'CV RMSE (bp)']})
    fix_state('rg_view', list(models))
    name = st.selectbox('Model detail', list(models), key='rg_view') if len(models) > 1 else next(iter(models))
    m = models[name]
    c = st.columns(5)
    c[0].metric('Bonds in model', f'{m["n"]:,}')
    c[1].metric('R\u00b2 / adj.', f'{m["r2"]:.3f} / {m["adj_r2"]:.3f}')
    c[2].metric('In-sample RMSE', f'{m["rmse_bp"]:.0f} bp')
    c[3].metric('5-fold CV RMSE', f'{m["cv_bp"]:.0f} bp' if np.isfinite(m['cv_bp']) else '-', help='Out-of-sample error: fit on 80% of bonds, predict the other 20%, repeated 5 times. If much larger than in-sample RMSE the model is overfit.')
    c[4].metric('Collinearity (cond. no.)', f'{m["cond"]:.0f}', help='Condition number of the standardised factors. Above ~30 suggests overlapping factors (e.g. maturity, maturity squared and duration together).')
    _forest(m)
    with st.expander('Coefficient table (raw)'):
        st.dataframe(m['coefs'], hide_index=True, use_container_width=True, column_config={
            'Coefficient': st.column_config.NumberColumn(format='%.4f'), 'Std Error': st.column_config.NumberColumn(format='%.4f'),
            't-stat': st.column_config.NumberColumn(format='%.2f'), 'p-value': st.column_config.NumberColumn(format='%.4f')})
    refs = '; '.join(f'{c_}: {lv[0]}' for c_, lv in m['levels'].items())
    notes = []
    if m['n_missing']:
        notes.append(f'{m["n_missing"]:,} bonds dropped for a missing factor value (e.g. NR bonds have no rating notch)')
    if m['n_trim']:
        notes.append(f'{m["n_trim"]:,} outliers trimmed and the model refit')
    if refs:
        notes.append(f'reference categories (the baseline each dummy is measured against) - {refs}')
    st.caption('Coefficients are in yield points (or log-yield points for the ln(YTM) model); * p<0.10, ** p<0.05, *** p<0.01. p-values use a normal approximation. '
               + ('; '.join(notes) + '.' if notes else ''))


def _diagnostics(out):
    left, right = st.columns(2)
    fig = px.density_heatmap(out, x='Fair YTM', y='YTM', nbinsx=50, nbinsy=50, color_continuous_scale=T.SEQUENTIAL)
    hi = float(max(out['YTM'].max(), out['Fair YTM'].max()))
    fig.add_scatter(x=[0, hi], y=[0, hi], mode='lines', name='perfect fit', line=dict(dash='dash', color=T.RED))
    fig.update_xaxes(rangemode='nonnegative', title_text='Fitted yield (%)')
    fig.update_yaxes(rangemode='nonnegative', title_text='Actual yield (%)')
    left.plotly_chart(T.style(fig, h=440, title='Actual vs. fitted yield (darker = more bonds)', legend=False), use_container_width=True)
    fig = px.histogram(out, x='Residual bp', nbins=70, color_discrete_sequence=[T.BLUE])
    fig.add_vline(x=0, line_color=T.INK)
    fig.update_xaxes(title_text='Residual (bp)')
    right.plotly_chart(T.style(fig, h=440, title='Residual distribution', legend=False), use_container_width=True)
    left, right = st.columns(2)
    key = 'Rating Bucket' if 'Rating Bucket' in out else 'Rating'
    fig = px.scatter(out, x='Years to Maturity', y='Residual bp', color=key, hover_name='Label', opacity=0.55, color_discrete_map=T.RATING_COLORS, category_orders={key: T.RATING_ORDER})
    fig.add_hline(y=0, line_color=T.INK)
    fig.update_xaxes(rangemode='nonnegative')
    left.plotly_chart(T.style(fig, h=440, title='Residual by maturity (a pattern here = a missing term)'), use_container_width=True)
    order = out.groupby('Sector')['Residual bp'].median().sort_values().index.tolist()
    fig = px.violin(out, x='Residual bp', y='Sector', orientation='h', box=True, points=False, category_orders={'Sector': order[::-1]}, color_discrete_sequence=[T.TEAL])
    fig.add_vline(x=0, line_color=T.INK)
    right.plotly_chart(T.style(fig, h=440, title='Residual by sector (a sector sitting off zero = a sector effect)', legend=False), use_container_width=True)
    st.caption('Residual charts are signed by nature: positive = the bond yields more than the model predicts.')


def _cheap_rich(out):
    zmin = st.number_input('Min |z-score|', min_value=0.0, max_value=5.0, value=1.0, step=0.25, key='rg_zmin')
    out = out.assign(**{'Outstanding ($mm)': out['Outstanding Amount'] / 1e6})
    base = out[out['Z-score'].abs() >= zmin]
    show = [c for c in ['Label', 'Rating', 'Sector', 'Years to Maturity', 'YTM', 'Fair YTM', 'Residual bp', 'Z-score', 'Outstanding ($mm)'] if c in out]
    cfg = {c: st.column_config.NumberColumn(format=f) for c, f in [('Years to Maturity', '%.2f'), ('YTM', '%.2f'), ('Fair YTM', '%.2f'), ('Residual bp', '%.0f'), ('Z-score', '%.2f'), ('Outstanding ($mm)', '%.0f')]}
    _lollipop(base, 'Residual bp', '15 richest and 15 cheapest bonds versus the model')
    with st.expander('Tables: cheap, rich and issuers'):
        t1, t2, t3 = st.tabs(['Cheap (yield above model)', 'Rich (yield below model)', 'Issuers'])
        with t1:
            st.dataframe(base.nlargest(25, 'Residual bp')[show], hide_index=True, use_container_width=True, column_config=cfg)
        with t2:
            st.dataframe(base.nsmallest(25, 'Residual bp')[show], hide_index=True, use_container_width=True, column_config=cfg)
        with t3:
            rows = []
            for tk, g in out.groupby('Ticker'):
                if len(g) >= 2:
                    w = g['Outstanding Amount'].clip(lower=1)
                    rows.append({'Ticker': tk, 'Issuer': g['Name'].iloc[0], 'Bonds': len(g), 'Wtd residual (bp)': float(np.average(g['Residual bp'], weights=w)), 'Outstanding ($bn)': g['Outstanding Amount'].sum() / 1e9})
            if rows:
                it = pd.DataFrame(rows).sort_values('Wtd residual (bp)', ascending=False)
                st.dataframe(pd.concat([it.head(15), it.tail(15)]).drop_duplicates('Ticker'), hide_index=True, use_container_width=True)
    st.download_button('Export all residuals', out[show].to_csv(index=False).encode(), 'regression_residuals.csv', 'text/csv', key='rg_dl')
    st.caption('A large positive residual is a starting point, not a conclusion: call features, stale quotes, liquidity and factors the model lacks all show up as residual.')


def _predict_ui(models, notch):
    st.write('Fair yield for a hypothetical bond from the fitted model. Duration / convexity (if used as factors) are computed assuming the bond prices at par.')
    fix_state('rg_pm', list(models))
    name = st.selectbox('Model', list(models), key='rg_pm') if len(models) > 1 else next(iter(models))
    m = models[name]
    c = st.columns(4)
    yrs = c[0].number_input('Years to maturity', min_value=0.25, max_value=60.0, value=7.0, step=0.5, key='rg_py')
    rating = c[1].selectbox('Rating', [r for r in notch], index=list(notch).index('BBB') if 'BBB' in notch else 0, key='rg_pr')
    sen = c[2].selectbox('Seniority rank', [1, 2, 3, 4, 5, 6], index=2, key='rg_ps')
    size = c[3].number_input('Outstanding ($mm)', min_value=1.0, value=500.0, step=50.0, key='rg_pz')
    c = st.columns(4)
    cpn = c[0].number_input('Coupon (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.25, key='rg_pc')
    price = c[1].number_input('Ask price', min_value=1.0, max_value=200.0, value=100.0, step=0.5, key='rg_pp')
    uni = c[2].radio('Universe', ['IG', 'HY'], index=0 if notch[rating] <= 10 else 1, horizontal=True, key='rg_pu')
    mtype = c[3].selectbox('Maturity type', ['CALLABLE', 'AT MATURITY', 'SINKABLE', 'PUTABLE'], key='rg_pt')
    ref = pd.concat([mm['out'] for mm in models.values()])
    row = pd.DataFrame([{'Years to Maturity': yrs, 'Notch': float(notch[rating]), 'Seniority Rank': float(sen), 'Outstanding Amount': size * 1e6, 'Coupon': cpn,
                         'Ask Price': price, 'Universe': uni, 'Maturity Type': mtype, 'Rating': rating}])
    dur, cvx = an.risk_measures([cpn if cpn > 0 else 5.0], [cpn if cpn > 0 else 5.0], [yrs])
    row['Mod Duration'], row['Convexity'] = dur, cvx
    row['Rating Bucket'] = rating.rstrip('+-')
    for cat in m['cats']:
        if cat not in ('Maturity Type', 'Rating Bucket'):
            lv = m['levels'][cat]
            row[cat] = st.selectbox(cat, lv, index=0, key=f'rg_pcat_{cat}')
    try:
        fair = _predict(m, row)
    except Exception as e:
        st.error(f'Could not price this bond with the selected factors: {e}')
        return
    k = st.columns(3)
    k[0].metric('Model fair yield', f'{fair:.2f}%')
    k[1].metric('Typical error (RMSE)', f'\u00b1{m["rmse_bp"]:.0f} bp')
    k[2].metric('Range (\u00b11 RMSE)', f'{max(fair - m["rmse_bp"] / 100, 0):.2f}% - {fair + m["rmse_bp"] / 100:.2f}%')
    if m['ln_y'] is False and fair <= 0:
        st.warning('The model produces a non-positive yield for these inputs - the bond is outside the range the model was fitted on.')
    st.caption(f'Fitted on {m["n"]:,} bonds. Extrapolating beyond the fitted maturity / size range is unreliable. A bond quoting above this yield looks cheap to the model, below looks rich.')


# ------------------------------------------------------------------------------------------------ entry point
def render_regression_tab(f, notch):
    st.write('Cross-sectional fair-value regression of yield on bond characteristics. Residuals flag bonds that yield more (cheap) or less (rich) than the model implies.')
    with st.expander('Model setup', expanded=True):
        c1, c2, c3 = st.columns(3)
        scope = c1.selectbox('Fit scope', SCOPES, key='rg_scope')
        dep = c2.radio('Dependent variable', ['YTM (%)', 'ln(YTM)'], horizontal=True, key='rg_dep', help='ln(YTM) models yields proportionally, which helps when high-yield dispersion dwarfs investment grade.')
        cap = c3.number_input('Hide yields above (%)', min_value=1.0, value=30.0, step=5.0, key='rg_cap')
        nums = st.multiselect('Numeric factors', list(NUM), default=DEFAULT_NUM, key='rg_nums')
        cats = st.multiselect('Categorical controls (dummy variables)', CATS, default=[], key='rg_cats', help='Adds one 0/1 column per category (first category is the baseline); categories with under 10 bonds are pooled as "Other".')
        o = st.columns(3)
        wls = o[0].checkbox('Weight by size (WLS)', value=False, key='rg_wls', help='Weights each bond by the square root of its amount outstanding so large, liquid issues count more.')
        robust = o[1].checkbox('Robust standard errors (HC1)', value=True, key='rg_rob', help='Standard errors that stay valid when residual size grows with yield or maturity (it almost always does).')
        trim = o[2].number_input('Trim outliers beyond k std (0 = off)', min_value=0.0, max_value=6.0, value=0.0, step=0.5, key='rg_trim')
    if not nums and not cats:
        st.warning('Choose at least one factor.')
        return
    z = f.copy()
    for c_ in ['YTM', 'Years to Maturity', 'Ask Price', 'Outstanding Amount', 'Notch', 'Coupon', 'Seniority Rank', 'Mod Duration', 'Convexity']:
        z[c_] = pd.to_numeric(z[c_], errors='coerce')
    z = z[z['YTM'].notna() & (z['YTM'] > 0) & (z['YTM'] <= cap) & (z['Years to Maturity'] > 0) & (z['Outstanding Amount'] > 0)].copy()
    z['Sector'] = z['Sector'].replace('', np.nan).fillna('Unclassified')
    if 'Rating Bucket' in z:
        z['Rating Bucket'] = z['Rating Bucket'].replace('', np.nan).fillna('NR')
    ln_y = dep.startswith('ln')
    frames = {'Pooled (IG + HY)': {'Pooled (IG + HY)': z}, 'IG only': {'IG only': z[z['Universe'] == 'IG']}, 'HY only': {'HY only': z[z['Universe'] == 'HY']},
              'Separate IG and HY models': {'IG model': z[z['Universe'] == 'IG'], 'HY model': z[z['Universe'] == 'HY']}}[scope]
    models, errors = {}, []
    for name, fr in frames.items():
        res = fit_model(fr, nums, cats, ln_y, wls, robust, trim) if len(fr) else {'error': 'no bonds in this scope'}
        (errors.append(f'{name}: {res["error"]}') if 'error' in res else models.__setitem__(name, res))
    for e in errors:
        st.warning(e)
    if not models:
        return
    out = pd.concat([m['out'] for m in models.values()])
    page = nav(['Model summary', 'Fitted curves', 'Diagnostics', 'Cheap / rich', 'Explain a bond', 'Predict fair yield'], 'rg_page')
    if page == 'Model summary':
        _summary(models)
    else:
        fix_state('rg_mv', list(models))
        mname = st.selectbox('Model', list(models), key='rg_mv') if len(models) > 1 and page in ('Fitted curves', 'Explain a bond') else next(iter(models))
        m = models[mname]
        if page == 'Fitted curves':
            _curves(m)
        elif page == 'Diagnostics':
            _diagnostics(out)
        elif page == 'Cheap / rich':
            _cheap_rich(out)
        elif page == 'Explain a bond':
            _explain(m)
        else:
            _predict_ui(models, notch)
