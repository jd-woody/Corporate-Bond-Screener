"""benchmarks.py - Treasury curve + ICE BofA OAS benchmarks (FRED), G-spreads, and a local snapshot history.

* refresh(dir): downloads the public FRED CSVs (no API key) and MERGES them into local files, so history keeps growing
  even though FRED now serves only the latest ~3 years of the ICE BofA series.
* load(dir): -> Bench with .ust (date x tenor-years, %), .oas (date x name, bp). Optional manual files:
      data/benchmarks/manual_curve.csv   columns: tenor_years,yield_pct
* gspread(): YTM minus the interpolated Treasury yield at the bond's maturity, in bp. NOT an OAS (ignores optionality).
* record_snapshot(): appends sector / industry / rating / universe / issuer aggregates for each new CSV snapshot to
  data/history/snapshots.csv, which turns the screener into a time series from the day you start using it.
ICE BofA data is licensed to FRED for display; keep it local and within FRED / ICE terms.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UST = {1 / 12: 'DGS1MO', 0.25: 'DGS3MO', 0.5: 'DGS6MO', 1.0: 'DGS1', 2.0: 'DGS2', 3.0: 'DGS3', 5.0: 'DGS5', 7.0: 'DGS7', 10.0: 'DGS10', 20.0: 'DGS20', 30.0: 'DGS30'}
OAS = {'IG': 'BAMLC0A0CM', 'HY': 'BAMLH0A0HYM2', 'AAA': 'BAMLC0A1CAAA', 'AA': 'BAMLC0A2CAA', 'A': 'BAMLC0A3CA', 'BBB': 'BAMLC0A4CBBB',
       'BB': 'BAMLH0A1HYBB', 'B': 'BAMLH0A2HYB', 'CCC': 'BAMLH0A3HYC'}
URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={}'
BUCKETS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']


@dataclass
class Bench:
    ust: pd.DataFrame = field(default_factory=pd.DataFrame)   # index date, columns tenor (years), values percent
    oas: pd.DataFrame = field(default_factory=pd.DataFrame)   # index date, columns IG/HY/AAA..CCC, values bp
    manual_curve: pd.DataFrame | None = None
    notes: list = field(default_factory=list)

    @property
    def has_curve(self):
        return (not self.ust.empty) or (self.manual_curve is not None and len(self.manual_curve) >= 3)

    @property
    def has_oas(self):
        return not self.oas.empty


def _parse_fred(text):
    df = pd.read_csv(io.StringIO(text))
    df.columns = ['date', 'value']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna(subset=['date']).set_index('date')['value']


def _fetch(series_id, timeout=25):
    req = urllib.request.Request(URL.format(series_id), headers={'User-Agent': 'Mozilla/5.0 (credit-screener)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def refresh(directory, fetch=_fetch):
    """Download every series, merge with what is already on disk (new values win). Returns (ok_ids, failed [(id, reason)])."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ok, bad = [], []
    for sid in list(UST.values()) + list(OAS.values()):
        try:
            new = _parse_fred(fetch(sid))
            f = d / f'fred_{sid}.csv'
            if f.exists():
                old = _parse_fred(f.read_text())
                new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            new.rename('value').rename_axis('date').to_csv(f)
            ok.append(sid)
        except Exception as e:  # network, DNS, proxy ...
            bad.append((sid, str(e)[:120]))
    return ok, bad


def _read_local(f):
    s = pd.read_csv(f)
    s['date'] = pd.to_datetime(s['date'], errors='coerce')
    return s.dropna(subset=['date']).set_index('date')['value'].astype(float)


def load(directory):
    d = Path(directory)
    b = Bench()
    if not d.exists():
        return b
    ust = {}
    for t, sid in UST.items():
        f = d / f'fred_{sid}.csv'
        if f.exists():
            ust[t] = _read_local(f)
    if ust:
        b.ust = pd.DataFrame(ust).sort_index()
    oas = {}
    for name, sid in OAS.items():
        f = d / f'fred_{sid}.csv'
        if f.exists():
            oas[name] = _read_local(f) * 100.0          # FRED gives percent -> bp
    if oas:
        b.oas = pd.DataFrame(oas).sort_index()
    m = d / 'manual_curve.csv'
    if m.exists():
        try:
            mc = pd.read_csv(m)
            mc.columns = [c.strip().lower() for c in mc.columns]
            b.manual_curve = mc[['tenor_years', 'yield_pct']].apply(pd.to_numeric, errors='coerce').dropna().sort_values('tenor_years')
        except Exception as e:
            b.notes.append(f'manual_curve.csv unreadable: {e}')
    return b


def signature(directory):
    d = Path(directory)
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(d.glob('*.csv'))) if d.exists() else ()


def curve_on(bench, date=None):
    """(tenors, yields %, date used) from the Treasury row on / just before `date` (latest if None); manual curve as fallback."""
    if not bench.ust.empty:
        u = bench.ust.dropna(how='all')
        if date is not None:
            u = u[u.index <= pd.Timestamp(date)] if (u.index <= pd.Timestamp(date)).any() else u
        row = u.iloc[-1].dropna()
        if len(row) >= 3:
            return row.index.values.astype(float), row.values.astype(float), u.index[-1]
    if bench.manual_curve is not None and len(bench.manual_curve) >= 3:
        return bench.manual_curve['tenor_years'].values, bench.manual_curve['yield_pct'].values, None
    return None, None, None


def gspread(ytm, years, curve):
    """G-spread in bp: ytm% minus Treasury yield interpolated at `years` (clamped at the ends of the curve)."""
    t, y, _ = curve
    yrs = np.asarray(years, dtype=float)
    return (np.asarray(ytm, dtype=float) - np.interp(yrs, t, y)) * 100.0


def blend_oas(bench, weights):
    """Benchmark OAS history (bp) for a rating mix: weighted average of the bucket OAS series that exist, re-normalised each day."""
    if bench.oas.empty:
        return pd.Series(dtype=float)
    cols = [c for c in weights if c in bench.oas.columns and weights[c] > 0]
    if not cols:
        return pd.Series(dtype=float)
    w = pd.Series({c: weights[c] for c in cols})
    o = bench.oas[cols]
    return (o * w).sum(axis=1, min_count=1) / (o.notna() * w).sum(axis=1).replace(0, np.nan)


def percentile_of(series, value):
    s = series.dropna()
    return float((s <= value).mean() * 100.0) if len(s) and np.isfinite(value) else np.nan


# ------------------------------------------------------------------------------------------------ history
def _wavg(g, col):
    z = g[[col, 'Outstanding Amount']].dropna()
    z = z[z['Outstanding Amount'] > 0]
    return float(np.average(z[col], weights=z['Outstanding Amount'])) if len(z) else np.nan


def record_snapshot(df, directory, asof):
    """Append (or replace) this snapshot's aggregates; returns the full history DataFrame."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    f = d / 'snapshots.csv'
    asof = pd.Timestamp(asof).normalize()
    rows = []
    for level, col in [('Sector', 'Sector'), ('Industry', 'Industry'), ('Rating Bucket', 'Rating Bucket'), ('Universe', 'Universe'), ('Ticker', 'Ticker')]:
        if col not in df:
            continue
        for k, g in df.groupby(col):
            if k in ('', None) or (isinstance(k, float) and np.isnan(k)):
                continue
            rows.append({'date': asof, 'level': level, 'key': k, 'bonds': len(g), 'outstanding_bn': g['Outstanding Amount'].sum() / 1e9,
                         'ytm': _wavg(g, 'YTM'), 'gspread': _wavg(g, 'G-Spread (bp)') if 'G-Spread (bp)' in g else np.nan,
                         'maturity': _wavg(g, 'Years to Maturity')})
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f, parse_dates=['date'])
        old = old[old['date'] != asof]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
    return new


def load_history(directory):
    f = Path(directory) / 'snapshots.csv'
    if not f.exists():
        return pd.DataFrame(columns=['date', 'level', 'key', 'bonds', 'outstanding_bn', 'ytm', 'gspread', 'maturity'])
    return pd.read_csv(f, parse_dates=['date'])
