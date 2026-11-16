"""update_benchmarks.py - download Treasury yields and ICE BofA OAS series from FRED into data/benchmarks (no API key needed).

    python update_benchmarks.py

Each run MERGES with what is already on disk, so history keeps growing even though FRED now serves only ~3 years of the ICE BofA series.
The app's Sector Analysis > Benchmarks page has the same thing as a button.
"""
from pathlib import Path

import benchmarks as bm

if __name__ == '__main__':
    target = Path(__file__).resolve().parent / 'data' / 'benchmarks'
    ok, bad = bm.refresh(target)
    print(f'Updated {len(ok)} of {len(ok) + len(bad)} series into {target}')
    for sid, why in bad[:3]:
        print(f'  FAILED {sid}: {why}')
    if len(bad) > 3:
        print(f'  ... and {len(bad) - 3} more')
    if bad:
        print('Check your internet connection / proxy. Offline? Put columns tenor_years,yield_pct in data/benchmarks/manual_curve.csv instead.')
    b = bm.load(target)
    if b.has_curve:
        _, _, dt = bm.curve_on(b)
        print(f'Treasury curve available' + (f' (latest {dt:%d %b %Y})' if dt is not None else ' (manual)'))
    if b.has_oas:
        print(f'ICE BofA OAS: {b.oas.shape[1]} series through {b.oas.index.max():%d %b %Y}')
