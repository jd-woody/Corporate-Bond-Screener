"""context.py - builds the shared app context: benchmark data, as-of date, spread basis, snapshot history."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import benchmarks as bm

MET = 'Spread (bp)'      # the one column every visual uses: G-spread if a Treasury curve is available, else yield in bp


@dataclass
class Ctx:
    bench: bm.Bench = field(default_factory=bm.Bench)
    curve: tuple = (None, None, None)
    asof: pd.Timestamp = None
    hist: pd.DataFrame = field(default_factory=pd.DataFrame)
    has_spread: bool = False
    bench_dir: Path = None
    hist_dir: Path = None

    @property
    def label(self):
        return 'G-spread (bp)' if self.has_spread else 'Yield (bp)'

    @property
    def has_index(self):
        return self.has_spread and self.bench.has_oas


@st.cache_data(show_spinner=False)
def _load_bench(sig, path):
    return bm.load(path)


def build(d, root):
    """-> (d with 'G-Spread (bp)' / 'Yield (bp)' / 'Spread (bp)' columns, Ctx). Never raises on missing benchmark files."""
    root = Path(root)
    ctx = Ctx(bench_dir=root / 'data' / 'benchmarks', hist_dir=root / 'data' / 'history')
    files = [root / 'data' / f for f in ('IG_Bond_Screener.csv', 'HY_Bond_Screener.csv') if (root / 'data' / f).exists()]
    ctx.asof = pd.Timestamp.fromtimestamp(max(p.stat().st_mtime for p in files)).normalize() if files else pd.Timestamp.today().normalize()
    ctx.bench = _load_bench(bm.signature(ctx.bench_dir), str(ctx.bench_dir))
    ctx.curve = bm.curve_on(ctx.bench, ctx.asof)
    ctx.has_spread = ctx.curve[0] is not None
    x = d.copy()
    x['Yield (bp)'] = x['YTM'] * 100.0
    if ctx.has_spread:
        x['G-Spread (bp)'] = bm.gspread(x['YTM'], x['Years to Maturity'], ctx.curve)
        x[MET] = x['G-Spread (bp)']
    else:
        x[MET] = x['Yield (bp)']
    if 'Rating Bucket' in x:
        x['Rating Bucket'] = x['Rating Bucket'].replace('', np.nan).fillna('NR')
    try:
        ctx.hist = _snapshot(x, ctx)
    except Exception:           # history is a nice-to-have; never block the app on it
        ctx.hist = bm.load_history(ctx.hist_dir)
    return x, ctx


def _snapshot(x, ctx):
    ok = x[(x['YTM'] > 0) & (x['YTM'] <= 30)]
    return _record_cached(ok, str(ctx.asof.date()), len(ok), ctx.has_spread, str(ctx.hist_dir))


def _record_cached(_df, asof, n, spread_flag, hist_dir):
    key = (asof, n, spread_flag, hist_dir)
    if st.session_state.get('_hist_key') == key and '_hist' in st.session_state:
        return st.session_state['_hist']
    h = bm.record_snapshot(_df, hist_dir, asof)
    st.session_state['_hist_key'], st.session_state['_hist'] = key, h
    return h
