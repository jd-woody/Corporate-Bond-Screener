"""analytics.py - bond risk measures, peer-relative value, curve fits, OLS with diagnostics, position sizing.

Pure pandas / numpy (no Streamlit) so it can be tested on its own. Conventions:
  * YTM and coupon are in percent; Years to Maturity in years; prices per 100 par; face amounts in $mm.
  * Duration / convexity are MODEL values: a plain semi-annual fixed-rate bond priced off its quoted yield (no call
    features, no accrued interest). Callable bonds - most of this universe - therefore get overstated duration.
  * DV01 ($ per 1bp, per $1mm face) = price x modified duration.   Dollar convexity ($ per bp^2, per $1mm face) = price x convexity x 1e-4.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAT_BINS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 1000]
MAT_LABELS = ['0-1y', '1-2y', '2-3y', '3-5y', '5-7y', '7-10y', '10-15y', '15-20y', '20-30y', '30y+']


# ----------------------------------------------------------------------------- bond maths
def _price(c, y, t):
    """Clean price per 100 par for coupon c%, yield y% (semi-annual), t years; continuous in t."""
    i, n, cp = y / 200.0, 2.0 * t, c / 2.0
    v = (1.0 + i) ** (-n)
    return cp * (1.0 - v) / i + 100.0 * v


def risk_measures(coupon, ytm, years, bump=0.01):
    """-> (modified duration in years, convexity in years^2) as float arrays; NaN where inputs are unusable."""
    c, y, t = (np.asarray(a, dtype=float) for a in (coupon, ytm, years))
    ok = np.isfinite(c) & np.isfinite(y) & np.isfinite(t) & (y > bump) & (t > 0)
    dur = np.full(c.shape, np.nan)
    cvx = np.full(c.shape, np.nan)
    if ok.any():
        cc, yy, tt = c[ok], y[ok], t[ok]
        p0, pu, pd_ = _price(cc, yy, tt), _price(cc, yy + bump, tt), _price(cc, yy - bump, tt)
        dy = bump / 100.0
        dur[ok] = (pd_ - pu) / (2.0 * p0 * dy)
        cvx[ok] = (pu + pd_ - 2.0 * p0) / (p0 * dy * dy)
    return dur, cvx


def dv01_per_mm(price, mod_dur):
    return np.asarray(price, dtype=float) * np.asarray(mod_dur, dtype=float)


def cvx_per_mm(price, convexity):
    return np.asarray(price, dtype=float) * np.asarray(convexity, dtype=float) * 1e-4


def pnl_one_leg(face_mm, dv01_mm, cvx_mm, shift_bp):
    """Dollar P&L of a LONG face_mm position for a yield shift (bp): -DV01*dy + 0.5*CVX*dy^2."""
    s = np.asarray(shift_bp, dtype=float)
    return face_mm * (-dv01_mm * s + 0.5 * cvx_mm * s * s)


# ----------------------------------------------------------------------------- enrich the screener table
def enrich(df):
    """Adds Country (ISIN prefix), CUSIP (US/CA ISINs), risk columns, maturity bucket and a lower-case search column."""
    x = df.copy()
    isin = x['ISIN'].fillna('').astype(str) if 'ISIN' in x else pd.Series('', index=x.index)
    if 'Country' not in x:
        x['Country'] = isin.str[:2].replace('', 'Unknown')
    if 'CUSIP' not in x:
        x['CUSIP'] = np.where(isin.str[:2].isin(['US', 'CA']), isin.str[2:11], '')
    dur, cvx = risk_measures(x['Coupon'] if 'Coupon' in x else np.nan, x['YTM'], x['Years to Maturity'])
    x['Mod Duration'], x['Convexity'] = dur, cvx
    x['DV01 per $1mm'] = dv01_per_mm(x['Ask Price'], dur)
    x['Maturity Bucket'] = pd.cut(pd.to_numeric(x['Years to Maturity'], errors='coerce'), MAT_BINS, labels=MAT_LABELS, include_lowest=True)
    parts = [x[c].fillna('').astype(str) for c in ['Name', 'Ticker', 'ISIN', 'CUSIP', 'ID Number', 'Alt ISINs (RegS/144A twins)'] if c in x]
    x['_search'] = (parts[0].str.cat(parts[1:], sep=' | ') if parts else pd.Series('', index=x.index)).str.lower()
    return x


# ----------------------------------------------------------------------------- peer relative value
def peer_relative(df, group_cols, min_peers=8, ycol='YTM'):
    """Yield vs the median of its peer group (group_cols + maturity bucket). Group median includes the bond itself."""
    z = df[df['Maturity Bucket'].notna() & df[ycol].notna()].copy()
    keys = list(dict.fromkeys(list(group_cols) + ['Maturity Bucket']))
    g = z.groupby(keys, observed=True, dropna=False)[ycol]
    z['Peer count'] = g.transform('count')
    z['Peer median YTM'] = g.transform('median')
    sd = g.transform('std')
    ok = z['Peer count'] >= min_peers
    z['vs peers (bp)'] = np.where(ok, (z[ycol] - z['Peer median YTM']) * 100.0, np.nan)
    z['Peer z'] = np.where(ok & (sd > 0), (z[ycol] - z['Peer median YTM']) / sd, np.nan)
    return z


# ----------------------------------------------------------------------------- curves
def curve_coef(x, y):
    """Quadratic (>=6 points) or linear (3-5 points) least-squares curve; None when too few points."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    deg = 2 if len(x) >= 6 else 1
    return np.polyfit(x, y, deg), float(x.min()), float(x.max())


def curve_at(fit, tenors):
    """Fitted yield at each tenor; NaN outside the observed maturity range (no extrapolation)."""
    if fit is None:
        return np.full(len(tenors), np.nan)
    coef, lo, hi = fit
    t = np.asarray(tenors, dtype=float)
    out = np.polyval(coef, t)
    out[(t < lo) | (t > hi)] = np.nan
    return out


# ----------------------------------------------------------------------------- OLS / WLS with diagnostics
def _norm_sf2(t):
    return np.array([math.erfc(abs(v) / math.sqrt(2.0)) if np.isfinite(v) else np.nan for v in np.atleast_1d(t)])


def fit_wls(X, y, w=None, robust=False):
    """Weighted least squares. Returns dict: beta, se, t, p (normal approx), r2, adj_r2, rmse, fitted, resid, dof, cond."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    rw = resid * sw
    dof = max(n - k, 1)
    bread = np.linalg.pinv(Xw.T @ Xw)
    if robust:
        meat = (Xw * (rw ** 2)[:, None]).T @ Xw
        cov = bread @ meat @ bread * (n / dof)
    else:
        cov = bread * (rw @ rw / dof)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = beta / se
    ybar = np.average(y, weights=w)
    sst = float(np.sum(w * (y - ybar) ** 2))
    sse = float(rw @ rw)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return {'beta': beta, 'se': se, 't': t, 'p': _norm_sf2(t), 'r2': r2, 'adj_r2': 1.0 - (1.0 - r2) * (n - 1) / dof if sst > 0 else np.nan,
            'rmse': float(np.sqrt(np.mean(resid ** 2))), 'fitted': fitted, 'resid': resid, 'dof': dof, 'cond': float(np.linalg.cond(Xw))}


def kfold_rmse(X, y, w=None, k=5, seed=42, inv=None, y_true=None):
    """Out-of-sample RMSE from k-fold cross-validation. By default in the units of y; pass inv (back-transform, e.g. np.exp)
    and y_true (untransformed target) to measure the error in original units. NaN when there is too little data."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    if n < 10 * k:
        return np.nan
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    yt = y if y_true is None else np.asarray(y_true, dtype=float)
    idx = np.random.default_rng(seed).permutation(n)
    err = np.empty(n)
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        sw = np.sqrt(w[tr])
        b = np.linalg.lstsq(X[tr] * sw[:, None], y[tr] * sw, rcond=None)[0]
        pred = X[f] @ b
        err[f] = yt[f] - (inv(pred) if inv is not None else pred)
    return float(np.sqrt(np.mean(err ** 2)))


# ----------------------------------------------------------------------------- position sizing
def dv01_neutral_short(long_face_mm, dv01_long, dv01_short):
    """Short-leg face ($mm) that offsets the long leg's DV01."""
    return long_face_mm * dv01_long / dv01_short if dv01_short else np.nan


def butterfly_weights(anchor_face_mm, anchor, wing1, wing2):
    """Face amounts (signed, $mm) of wing1 / wing2 that make net DV01 AND net dollar convexity zero for a given anchor face.
    Each leg is a tuple (dv01_per_mm, cvx_per_mm). Returns (n1, n2, det)."""
    a = np.array([[wing1[0], wing2[0]], [wing1[1], wing2[1]]], dtype=float)
    b = -anchor_face_mm * np.array([anchor[0], anchor[1]], dtype=float)
    det = float(np.linalg.det(a))
    if abs(det) < 1e-9 * max(1.0, abs(a).max() ** 2):
        return np.nan, np.nan, det
    n1, n2 = np.linalg.solve(a, b)
    return float(n1), float(n2), det


# ----------------------------------------------------------------------------- aggregation helpers
def wavg(df, col, w='Outstanding Amount'):
    """Outstanding-weighted mean of `col` (NaN when nothing to weight)."""
    z = df[[col, w]].dropna()
    z = z[z[w] > 0]
    return float(np.average(z[col], weights=z[w])) if len(z) else np.nan


def issuer_table(df, mcol):
    """One row per ticker: name, bonds, outstanding ($bn), weighted metric / yield / maturity, modal rating bucket, weighted notch."""
    rows = []
    for k, g in df.groupby('Ticker'):
        rows.append({'Ticker': k, 'Name': g['Name'].iloc[0], 'Bonds': len(g), 'Outstanding ($bn)': g['Outstanding Amount'].sum() / 1e9,
                     'Metric': wavg(g, mcol), 'YTM': wavg(g, 'YTM'), 'Years': wavg(g, 'Years to Maturity'), 'Notch': wavg(g, 'Notch'),
                     'Bucket': g['Rating Bucket'].mode().iloc[0] if 'Rating Bucket' in g and g['Rating Bucket'].notna().any() else 'NR',
                     'Sector': g['Sector'].iloc[0], 'Industry': g['Industry'].iloc[0] if 'Industry' in g else ''})
    return pd.DataFrame(rows)
