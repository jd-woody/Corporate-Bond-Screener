"""capiq_loader.py - parse S&P Capital IQ Pro Excel exports into tidy pandas tables.

Drop exports in data/capiq/.  Supported report types (detected from the title cell inside
each file, so file names do not matter):

    Financials > Capital Structure Summary   -> CapIQCompany.cap_summary
    Financials > Capital Structure Details   -> CapIQCompany.cap_details
    Financials > Ratios                      -> CapIQCompany.ratios
    Fixed Income > Securities Summary        -> CapIQCompany.securities

    Financials > Key Stats / Income Statement / Balance Sheet / Cash Flow
    Financials > Multiples / (Historical) Capitalization / Segments / Supplemental / Pension-OPEB
                                             -> CapIQCompany.statements[<key>]

Credit Health Panel > Summary / Financials -> CapIQCompany.statements['chp_summary'] (dict) / ['chp_financials'] (DataFrame)

Both single-page exports and the multi-sheet "Financials" workbook are supported: every sheet is routed by its own title.
Anything else (e.g. Credit Health Panel, which exports charts only) is listed as skipped.
Files are grouped per company by the "(Exchange:TICKER)" text in the title cell.

Reading legacy .xls needs `xlrd>=2.0.1`; .xlsx needs `openpyxl`.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TITLE_RE = re.compile(r'^(?P<company>.+?)\s*\((?P<exch>[^:()]+):(?P<tick>[^()]+)\)\s*>\s*(?P<path>.+)$')
DATE_RE = re.compile(r'[A-Za-z]{3}-\d{1,2}-\d{4}')
MONTHS_RE = re.compile(r'(\d+)\s*months?', re.I)
BLOCK_RE = re.compile(r'^(?P<label>.+?)\s*\((?P<end>[A-Za-z]{3}-\d{1,2}-\d{4})\)\s*Capital Structure As Reported Details', re.I)
PCT_RE = re.compile(r'^\s*(-?[\d.]+)\s*%\s*$')

REPORTS = {  # report key -> accepted endings of the title path (lower-case)
    'cap_summary': ('capital structure summary',),
    'cap_details': ('capital structure details',),
    'ratios': ('ratios',),
    'securities': ('securities summary',),
    'key_stats': ('key stats',),
    'income': ('income statement',),
    'balance': ('balance sheet',),
    'cashflow': ('cash flow',),
    'multiples': ('multiples',),
    'capitalization': ('historical capitalization', 'capitalization'),
    'segments': ('segments',),
    'supplemental': ('supplemental',),
    'pension': ('pension/opeb',),
    'chp_summary': ('summary',),          # Credit Health Panel > Summary
    'chp_financials': ('financials',),    # Credit Health Panel > Financials
}
REPORT_NAMES = {
    'key_stats': 'Key Stats', 'income': 'Income Statement', 'balance': 'Balance Sheet', 'cashflow': 'Cash Flow',
    'multiples': 'Multiples', 'capitalization': 'Capitalization', 'cap_summary': 'Capital Structure Summary',
    'cap_details': 'Capital Structure Details', 'ratios': 'Ratios', 'segments': 'Segments',
    'supplemental': 'Supplemental', 'pension': 'Pension/OPEB', 'securities': 'Securities Summary',
    'chp_summary': 'Credit Health Panel - Summary', 'chp_financials': 'Credit Health Panel - Financials',
}
STATEMENT_KEYS = ('key_stats', 'income', 'balance', 'cashflow', 'supplemental', 'pension')
HEADER_PREFIXES = ('for the fiscal', 'balance sheet as of', 'for quarter', 'for the quarter')

# Ratios that Capital IQ stores as fractions but displays as percentages.
PCT_RATIOS = {'Total Debt/Capital', 'LT Debt/Capital', 'Total Debt/Equity', 'LT Debt/Equity',
              'Total Liabilities/Total Assets'}
SECURITY_NUM = ['Coupon', 'Offering Amt. ($mm)', 'Amt. Outstdg. ($mm)', 'Current Price', 'Current YTW',
                'Offering Price', 'Offering Yield', 'Call Price']
SECURITY_DATE = ['Maturity Date', 'Offer Date', 'S&P Rating Date', 'Dated Date', 'First Coupon Date',
                 'Last Coupon Date', 'Next Call Date']


@dataclass
class Matrix:
    """Rows = (Section, Item); columns = period labels. `nm` flags cells Capital IQ printed as 'NM'."""
    values: pd.DataFrame           # columns: Section, Item, <period labels...> (floats)
    nm: pd.DataFrame               # same shape, bool
    pct: pd.DataFrame | None       # '% of Total' companion columns (cap summary only)
    periods: list                  # period labels in file order
    filing: dict = field(default_factory=dict)  # period label -> filing date
    extra: dict = field(default_factory=dict)   # report-specific side tables (Key Stats: latest_cap, current_multiples)

    def row(self, item, section=None):
        v = self.values
        m = v[v['Item'] == item]
        if section is not None:
            m = m[m['Section'] == section]
        return m.index[0] if len(m) else None

    def latest(self, item, section=None):
        """(value, period label) for the right-most non-empty period of `item`, else (nan, None)."""
        i = self.row(item, section)
        if i is None:
            return np.nan, None
        for p in reversed(self.periods):
            if pd.notna(self.values.at[i, p]):
                return float(self.values.at[i, p]), p
        return np.nan, None

    def is_nm(self, item, period, section=None):
        i = self.row(item, section)
        return bool(i is not None and period in self.nm.columns and self.nm.at[i, period])

    def latest_pct(self, item, period, section=None):
        i = self.row(item, section)
        if i is None or self.pct is None or period not in self.pct.columns:
            return np.nan
        return float(self.pct.at[i, period])


@dataclass
class CapIQCompany:
    ticker: str
    exchange: str = ''
    name: str = ''
    files: dict = field(default_factory=dict)       # report key -> (file name, modified timestamp)
    cap_summary: Matrix | None = None
    cap_details: pd.DataFrame | None = None
    ratios: Matrix | None = None
    securities: pd.DataFrame | None = None
    statements: dict = field(default_factory=dict)  # key_stats/income/balance/cashflow/multiples/capitalization/segments/... -> Matrix
    problems: list = field(default_factory=list)    # human-readable load problems / skipped files


# ----------------------------------------------------------------------------- low level
def read_raw(path):
    """First sheet as an all-cells DataFrame (header=None). Engine is chosen from the file signature."""
    path = Path(path)
    with open(path, 'rb') as fh:
        head = fh.read(8)
    if head.startswith(b'\xd0\xcf\x11\xe0'):
        engine = 'xlrd'
    elif head.startswith(b'PK'):
        engine = 'openpyxl'
    else:
        raise ValueError('not an Excel file (unrecognised file signature)')
    try:
        return pd.read_excel(path, sheet_name=0, header=None, engine=engine)
    except ImportError as e:
        raise ImportError(f'{e}  ->  run: python -m pip install -r requirements.txt  (needs xlrd>=2.0.1 for .xls)') from e


def read_sheets(path):
    """{sheet name: all-cells DataFrame} for every sheet; engine chosen from the file signature."""
    path = Path(path)
    with open(path, 'rb') as fh:
        head = fh.read(8)
    if head.startswith(b'\xd0\xcf\x11\xe0'):
        engine = 'xlrd'
    elif head.startswith(b'PK'):
        engine = 'openpyxl'
    else:
        raise ValueError('not an Excel file (unrecognised file signature)')
    try:
        return pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    except ImportError as e:
        raise ImportError(f'{e}  ->  run: python -m pip install -r requirements.txt  (needs xlrd>=2.0.1 for .xls)') from e


def _cell(x):
    """-> (float | nan, is_nm)."""
    if x is None:
        return np.nan, False
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return (float(x), False) if pd.notna(x) else (np.nan, False)
    s = str(x).strip().replace(',', '')
    if s.upper() == 'NM':
        return np.nan, True
    if s in ('', '-', 'NA', 'N/A', 'nan', 'NaT'):
        return np.nan, False
    neg = s.startswith('(') and s.endswith(')')
    s = s.strip('()').rstrip('%x ')
    try:
        v = float(s)
    except ValueError:
        return np.nan, False
    return (-v if neg else v), False


def _is_text(x):
    return isinstance(x, str) and x.strip() != ''


def _row_is_blank(row):
    return row.isna().all()


def _is_section(raw, i):
    """A section header = short text in column 0 and nothing else on the row."""
    r = raw.iloc[i]
    return _is_text(r.iloc[0]) and 0 < len(r.iloc[0].strip()) <= 45 and r.iloc[1:].isna().all()


def _cell_date(x):
    """Timestamp for a real date cell or for text such as '12 months\nMay-31-2025A'; else None."""
    if isinstance(x, (pd.Timestamp, _dt.datetime, _dt.date, np.datetime64)):
        d = pd.Timestamp(x)
        return None if pd.isna(d) else d
    if isinstance(x, str):
        m = DATE_RE.search(x)
        if m:
            d = pd.to_datetime(m.group(0), format='%b-%d-%Y', errors='coerce')
            return None if pd.isna(d) else d
    return None


_EST_RE = re.compile(DATE_RE.pattern + r'[Ee]\b')


def _period_label(cell):
    """'12 months\nSep-27-2025' -> ('FY Sep-2025', Timestamp); 'LTM...' -> 'LTM Jun-2026'; '3 months ...' -> '3M Jun-2026';
    estimate columns ('May-31-2027E') get an 'E' suffix; plain period-end dates (balance sheet, quarters) -> 'May-2025'."""
    end = _cell_date(cell)
    if end is None:
        return None, None
    s = cell if isinstance(cell, str) else ''
    months = MONTHS_RE.search(s)
    if 'LTM' in s.upper():
        pre = 'LTM '
    elif months:
        n = int(months.group(1))
        pre = 'FY ' if n == 12 else f'{n}M '
    else:
        pre = ''
    return f'{pre}{end:%b-%Y}{"E" if _EST_RE.search(s) else ""}', end


def _is_header_text(name):
    return _is_text(name) and name.strip().lower().startswith(HEADER_PREFIXES)


def _no_data(raw):
    for i in range(min(25, len(raw))):
        v = raw.iat[i, 0]
        if _is_text(v) and v.strip().lower() == 'no data available':
            return True
    return False


def _build_matrix(raw, header_row, value_cols, pct_cols=None, prefix_block=False):
    """Walk the rows of `raw`, tracking block titles (the short line right above a period header) and groups
    (short lines inside a block), and build a Matrix. Section = group or block (or 'block > group' when prefix_block)."""
    labels = [_period_label(raw.iat[header_row, c])[0] for c in value_cols]
    rows, nm_rows, pct_rows, filing = [], [], [], {}
    block = group = ''
    for i in range(len(raw)):
        r = raw.iloc[i]
        name = r.iloc[0]
        if not _is_text(name):
            continue
        name = name.strip()
        if _is_header_text(name):
            continue
        if len(name) > 45 and r.iloc[1:].isna().all():
            continue                                    # footer / disclaimer text
        if name.lower().startswith(('currency', 'units')):
            continue
        if _is_section(raw, i):
            nxt = raw.iat[i + 1, 0] if i + 1 < len(raw) else None
            if _is_header_text(nxt):
                block, group = name, ''
            else:
                group = name
            continue
        if i <= header_row:
            continue
        section = (f'{block} \u203a {group}' if prefix_block and block and group else (group or block))
        if name.lower() == 'filing date':
            for c, lab in zip(value_cols, labels):
                d = _cell_date(r.iloc[c]) if c < len(r) else None
                if lab and d is not None:
                    filing[lab] = d
            continue
        vals, nms, pcts = [], [], []
        for k, c in enumerate(value_cols):
            v, nm = _cell(r.iloc[c]) if c < len(r) else (np.nan, False)
            vals.append(v); nms.append(nm)
            if pct_cols is not None:
                p, _ = _cell(r.iloc[pct_cols[k]]) if pct_cols[k] < len(r) else (np.nan, False)
                pcts.append(p)
        if all(pd.isna(v) for v in vals) and not any(nms):
            continue
        rows.append([section, name] + vals); nm_rows.append([section, name] + nms)
        if pct_cols is not None:
            pct_rows.append([section, name] + pcts)
    cols = ['Section', 'Item'] + labels
    values = pd.DataFrame(rows, columns=cols)
    nm = pd.DataFrame(nm_rows, columns=cols)
    pct = pd.DataFrame(pct_rows, columns=cols) if pct_cols is not None else None
    return Matrix(values=values, nm=nm, pct=pct, periods=labels, filing=filing)


def _find_period_header(raw, scan_rows=60):
    """Header row = a row whose first cell is 'For the Fiscal Period Ending' / 'Balance Sheet as of:' / 'For Quarter Ending'
    (preferred) or any row with period-date text; returns (row, [value cols]) where cells hold a date (text or real date)."""
    def cols_of(i, first=1):
        return [c for c in range(first, raw.shape[1]) if _cell_date(raw.iat[i, c]) is not None]
    n = min(scan_rows, len(raw))
    for i in range(n):
        if _is_header_text(raw.iat[i, 0]) and cols_of(i):
            return i, cols_of(i)
    for i in range(n):
        cs = [c for c in range(1, raw.shape[1]) if isinstance(raw.iat[i, c], str) and DATE_RE.search(raw.iat[i, c])]
        if cs:
            return i, cs
    return None, []


# ----------------------------------------------------------------------------- report parsers
def parse_cap_summary(raw):
    hdr, cols = _find_period_header(raw)
    if hdr is None:
        raise ValueError('period header row not found')
    return _build_matrix(raw, hdr, cols, pct_cols=[c + 1 for c in cols])


def parse_ratios(raw):
    hdr, cols = _find_period_header(raw)
    if hdr is None:
        raise ValueError('period header row not found')
    return _build_matrix(raw, hdr, cols)


def parse_statement(raw):
    """Income Statement, Balance Sheet, Cash Flow, Capitalization, Supplemental, Pension/OPEB: item rows x period columns."""
    hdr, cols = _find_period_header(raw)
    if hdr is None:
        raise ValueError('period header row not found')
    m = _build_matrix(raw, hdr, cols)
    if m.values.empty:
        raise ValueError('no data rows in this export')
    return m


def parse_segments(raw):
    hdr, cols = _find_period_header(raw)
    if hdr is None:
        raise ValueError('period header row not found')
    m = _build_matrix(raw, hdr, cols, prefix_block=True)
    if m.values.empty:
        raise ValueError('no segment rows in this export')
    return m


_SUBROWS = {'Margin %', 'Growth Over Prior Year'}


def _disambiguate(m):
    """Key Stats repeats 'Margin %' / 'Growth Over Prior Year' under each metric; prefix them with that metric."""
    last = ''
    for i in m.values.index:
        item = m.values.at[i, 'Item']
        if item in _SUBROWS:
            new = f'{last} {item}'.strip()
            m.values.at[i, 'Item'] = new
            m.nm.at[i, 'Item'] = new
        else:
            last = item


def parse_key_stats(raw):
    cut = len(raw)
    for i in range(len(raw)):
        v = raw.iat[i, 0]
        if _is_text(v) and v.strip().lower().startswith(('latest capitalization', 'current capitalization')):
            cut = i
            break
    main = raw.iloc[:cut].reset_index(drop=True)
    hdr, cols = _find_period_header(main)
    if hdr is None:
        raise ValueError('period header row not found')
    m = _build_matrix(main, hdr, cols)
    _disambiguate(m)
    tail = raw.iloc[cut:].reset_index(drop=True)
    mult_start = None
    for i in range(len(tail)):
        v = tail.iat[i, 0]
        if _is_text(v) and v.strip().lower().startswith('valuation multiples'):
            mult_start = i
            break
    cap_rows, block = [], 'Enterprise value'
    for i in range(len(tail) if mult_start is None else mult_start):
        name = tail.iat[i, 0]
        if not _is_text(name) or len(name.strip()) > 45 or name.strip().lower().startswith('currency'):
            continue
        name = name.strip()
        if name.lower().startswith('book value'):
            block = 'Total capital'
        v, _ = _cell(tail.iat[i, 1]) if tail.shape[1] > 1 else (np.nan, False)
        if pd.notna(v):
            cap_rows.append([block, name, v])
    m.extra['latest_cap'] = pd.DataFrame(cap_rows, columns=['Block', 'Item', 'Value'])
    if mult_start is not None:
        t2 = tail.iloc[mult_start:].reset_index(drop=True)
        h2, c2 = _find_period_header(t2)
        if h2 is not None:
            cm = _build_matrix(t2, h2, c2)
            if not cm.values.empty:
                m.extra['current_multiples'] = cm
    return m


def parse_multiples(raw):
    hdr = None
    for i in range(min(60, len(raw))):
        if _is_text(raw.iat[i, 0]) and raw.iat[i, 0].strip().lower().startswith('for quarter'):
            hdr = i
            break
    if hdr is None:
        raise ValueError('"For Quarter Ending" header row not found')
    cols = [c for c in range(2, raw.shape[1]) if _cell_date(raw.iat[hdr, c]) is not None]
    labels = [_period_label(raw.iat[hdr, c])[0] for c in cols]
    rows, nms, cur = [], [], None
    for i in range(hdr + 1, len(raw)):
        a, stat = raw.iat[i, 0], raw.iat[i, 1]
        if _is_text(a) and a.strip() and len(a.strip()) <= 60:
            cur = a.strip()
        if cur is None or not _is_text(stat):
            continue
        vals, nm = zip(*[_cell(raw.iat[i, c]) for c in cols])
        if all(pd.isna(v) for v in vals) and not any(nm):
            continue
        rows.append([cur, stat.strip()] + list(vals)); nms.append([cur, stat.strip()] + list(nm))
    if not rows:
        raise ValueError('no multiples rows found')
    columns = ['Section', 'Item'] + labels
    return Matrix(values=pd.DataFrame(rows, columns=columns), nm=pd.DataFrame(nms, columns=columns), pct=None, periods=labels)


def parse_cap_details(raw):
    out, i, n = [], 0, len(raw)
    while i < n:
        c0 = raw.iat[i, 0]
        m = BLOCK_RE.match(c0.strip()) if _is_text(c0) else None
        if not m:
            i += 1
            continue
        label, end = m.group('label').strip(), pd.to_datetime(m.group('end'), format='%b-%d-%Y', errors='coerce')
        i += 1
        while i < n and not (_is_text(raw.iat[i, 0]) and raw.iat[i, 0].strip() == 'Description'):
            i += 1
        if i >= n:
            break
        names = [str(x).strip() if pd.notna(x) else '' for x in raw.iloc[i]]
        ccy = ''
        for k, nm in enumerate(names):
            mm = re.match(r'Principal Due \((\w+)\)', nm)
            if mm:
                ccy, names[k] = mm.group(1), 'Principal Due'
        i += 1
        rows = []
        while i < n and _is_text(raw.iat[i, 0]) and not BLOCK_RE.match(raw.iat[i, 0].strip()):
            rows.append(list(raw.iloc[i])); i += 1
        if rows:
            df = pd.DataFrame(rows, columns=[nm or f'col{k}' for k, nm in enumerate(names)])
            df.insert(0, 'Period', label); df.insert(1, 'Period End', end); df['Reporting Currency'] = ccy
            out.append(df)
    if not out:
        raise ValueError('no "Capital Structure As Reported Details" blocks found')
    df = pd.concat(out, ignore_index=True)
    df['Principal Due'] = df['Principal Due'].map(lambda v: _cell(v)[0])
    df['Coupon %'] = df['Coupon/Base Rate'].map(lambda v: float(PCT_RE.match(str(v)).group(1)) if PCT_RE.match(str(v)) else np.nan) \
        if 'Coupon/Base Rate' in df else np.nan
    if 'Maturity' in df:
        df['Maturity Date'] = pd.to_datetime(df['Maturity'].where(df['Maturity'].map(lambda v: isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, 'year'))), errors='coerce')
    return df


def parse_securities(raw, today=None):
    hdr = None
    for i in range(min(40, len(raw))):
        if _is_text(raw.iat[i, 1]) and raw.iat[i, 1].strip() == 'Maturity Date':
            hdr = i
            break
    if hdr is None:
        raise ValueError('"Maturity Date" header row not found')
    names = [str(x).strip() if pd.notna(x) else '' for x in raw.iloc[hdr]]
    keep = [k for k, nm in enumerate(names) if nm]
    rows, i = [], hdr + 1
    while i < len(raw) and pd.notna(raw.iat[i, 1]) and not (_is_text(raw.iat[i, 1]) and len(raw.iat[i, 1]) > 40):
        rows.append([raw.iat[i, k] for k in keep]); i += 1
    df = pd.DataFrame(rows, columns=[names[k] for k in keep])
    for c in SECURITY_NUM:
        if c in df:
            df[c] = df[c].map(lambda v: _cell(v)[0])
    for c in SECURITY_DATE:
        if c in df:
            df[c] = pd.to_datetime(df[c].replace('-', pd.NaT), errors='coerce')
    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])):
            df[c] = df[c].map(lambda v: np.nan if isinstance(v, str) and v.strip() in ('-', '') else v)
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    df['Years to Maturity'] = (df['Maturity Date'] - today).dt.days / 365.25
    df['Is Money Market'] = df['Security Type'].astype(str).str.contains('Money Market', case=False)
    return df




# ----------------------------------------------------------------------------- Credit Health Panel
QUARTILE = {1: 'Top', 2: 'Above Average', 3: 'Below Average', 4: 'Bottom'}
COMPANY_RE = re.compile(r'^(?P<name>.+?)\s*\((?P<exch>[^:()]+):(?P<tick>[^()]+)\)\s*$')


def _split_company(text):
    m = COMPANY_RE.match(str(text).strip())
    return (m.group('name').strip(), m.group('exch').strip(), m.group('tick').strip()) if m else (str(text).strip(), '', '')


def _pct_value(v):
    """'0.0238%' -> 0.0238 (percent units); numbers pass through; anything else NaN."""
    if isinstance(v, str):
        return _cell(v.replace('%', ''))[0]
    return _cell(v)[0]


def _find(raw, text, col, start=0, stop=None):
    for i in range(start, min(stop or len(raw), len(raw))):
        v = raw.iat[i, col] if col < raw.shape[1] else None
        if _is_text(v) and v.strip().lower().startswith(text.lower()):
            return i
    return None


def parse_chp_summary(raw):
    """Relative credit-health score (overall + operational / solvency / liquidity quartile), quantitative indicators, and the public peer table."""
    h = _find(raw, 'overall score', 1)
    if h is None:
        raise ValueError('"Overall Score" header not found')
    overall = raw.iat[h + 1, 1] if h + 1 < len(raw) else None
    cats = []
    for r in range(h + 1, min(h + 8, len(raw))):
        nm = raw.iat[r, 3] if raw.shape[1] > 5 else None
        if not _is_text(nm):
            break                                   # the score block ends at the first blank row
        cats.append((nm.strip(), str(raw.iat[r, 5]).strip()))
    inds = {}
    for r in range(h + 1, min(h + 8, len(raw))):
        nm = raw.iat[r, 10] if raw.shape[1] > 12 else None
        if not _is_text(nm):
            break
        inds[nm.strip()] = {'value': raw.iat[r, 11], 'date': _cell_date(raw.iat[r, 12])}
    pd_ind = next((v for k, v in inds.items() if 'probability of default' in k.lower()), None)
    cm_ind = next((v for k, v in inds.items() if 'creditmodel' in k.lower()), None)
    ph = None
    for i in range(h, len(raw)):
        if _is_text(raw.iat[i, 1]) and raw.iat[i, 1].strip() == 'Company' and _is_text(raw.iat[i, 2]) and raw.iat[i, 2].strip() == 'Overall':
            ph = i
            break
    peers = pd.DataFrame()
    if ph is not None:
        names = [str(x).strip() if pd.notna(x) else '' for x in raw.iloc[ph]]
        rows = []
        for r in range(ph + 1, len(raw)):
            c = raw.iat[r, 1]
            if not _is_text(c) or c.strip().startswith('*'):
                break
            rows.append(list(raw.iloc[r]))
        df = pd.DataFrame(rows, columns=[n or f'col{k}' for k, n in enumerate(names)])
        ren = {}
        for c in df.columns:
            lc = c.lower()
            ren[c] = ('Company' if lc == 'company' else 'Overall' if lc == 'overall' else 'Operational' if lc == 'operational' else 'Solvency' if lc == 'solvency'
                      else 'Liquidity' if lc == 'liquidity' else 'LTM Period Ending' if lc.startswith('ltm') else 'Financials Updated' if lc.startswith('financials updated')
                      else 'S&P Rating' if lc.startswith('s&p') else 'CreditModel' if lc.startswith('creditmodel') else 'PD (%)' if 'probability of default' in lc
                      else 'Country' if lc == 'country' else 'Industry' if lc.startswith('primary industry') else c)
        df = df.rename(columns=ren)
        for c in ['Overall', 'Operational', 'Solvency', 'Liquidity']:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'PD (%)' in df:
            df['PD (%)'] = df['PD (%)'].map(_pct_value)
        for c in ['LTM Period Ending', 'Financials Updated']:
            if c in df:
                df[c] = pd.to_datetime(df[c], errors='coerce')
        for c in ['S&P Rating', 'CreditModel']:
            if c in df:
                df[c] = df[c].map(lambda v: np.nan if (not _is_text(v) or v.strip() in ('-', '')) else v.strip())
        parts = df['Company'].map(_split_company)
        df['Name'] = parts.map(lambda t: t[0]); df['Exchange'] = parts.map(lambda t: t[1]); df['Ticker'] = parts.map(lambda t: t[2])
        peers = df
    subject = ''
    for i in range(min(15, len(raw))):
        v = raw.iat[i, 0]
        m = TITLE_RE.match(v.strip()) if _is_text(v) else None
        if m:
            subject = m.group('tick').strip().upper()
            break
    if len(peers):
        peers['Is Subject'] = peers['Ticker'].str.upper() == subject
    return {'overall': str(overall).strip() if overall is not None else '', 'categories': cats,
            'pd': _pct_value(pd_ind['value']) if pd_ind else np.nan, 'pd_date': pd_ind['date'] if pd_ind else None,
            'creditmodel': (str(cm_ind['value']).strip() if cm_ind else ''), 'cm_date': cm_ind['date'] if cm_ind else None,
            'peers': peers, 'subject': subject}


def parse_chp_financials(raw):
    """Long table: Category, Metric, Unit, Weight, Score (quartile 1=top..4=bottom), Period, Company value, Group mean."""
    h = _find(raw, 'ltm period ending', 0)
    if h is None:
        raise ValueError('"LTM Period Ending" header not found')
    cols = [c for c in range(1, raw.shape[1]) if _cell_date(raw.iat[h, c]) is not None]
    if not cols:
        raise ValueError('no period dates found')
    dates = [_cell_date(raw.iat[h, c]) for c in cols]
    out, cat = [], ''
    for r in range(h + 2, len(raw)):
        a, b = raw.iat[r, 0], raw.iat[r, 1]
        if _is_text(a) and a.strip().startswith('*'):
            break
        if _is_text(b) and b.strip():
            name = b.strip()
            m = re.match(r'^(.*?)\s*\(([^()]*)\)\s*$', name)
            metric, unit = (m.group(1).strip(), m.group(2).strip()) if m else (name, '')
            weight = _cell(a)[0]
            score = pd.to_numeric(raw.iat[r, 2], errors='coerce')
            for c, dt in zip(cols, dates):
                comp, nm = _cell(raw.iat[r, c])
                mean, _ = _cell(raw.iat[r, c + 1]) if c + 1 < raw.shape[1] else (np.nan, False)
                out.append({'Category': cat, 'Metric': metric, 'Unit': unit, 'Weight': weight, 'Score': float(score) if pd.notna(score) else np.nan,
                            'Period': dt, 'Company': comp, 'Group Mean': mean, 'NM': nm})
        elif _is_text(a) and a.strip():
            cat = a.strip()
    if not out:
        raise ValueError('no metric rows found')
    return pd.DataFrame(out)


PARSERS = {'cap_summary': parse_cap_summary, 'cap_details': parse_cap_details, 'ratios': parse_ratios,
           'securities': parse_securities, 'key_stats': parse_key_stats, 'multiples': parse_multiples,
           'segments': parse_segments, 'income': parse_statement, 'balance': parse_statement, 'cashflow': parse_statement,
           'capitalization': parse_statement, 'supplemental': parse_statement, 'pension': parse_statement,
           'chp_summary': parse_chp_summary, 'chp_financials': parse_chp_financials}
OLD_ATTRS = ('cap_summary', 'cap_details', 'ratios', 'securities')


# ----------------------------------------------------------------------------- directory loader
def identify(raw, filename=''):
    """-> (report key | None, company, exchange, ticker) from the title cell of one sheet."""
    for i in range(min(15, len(raw))):
        v = raw.iat[i, 0]
        m = TITLE_RE.match(v.strip()) if _is_text(v) else None
        if m:
            tail = m.group('path').split('>')[-1].strip().lower()
            key = next((k for k, tails in REPORTS.items() if tail in tails), None)
            return key, m.group('company').strip(), m.group('exch').strip(), m.group('tick').strip().upper()
    return None, '', '', ''


def load_dir(directory):
    """Parse every sheet of every .xls/.xlsx in `directory`. Returns {TICKER: CapIQCompany}. Never raises on a bad file."""
    directory = Path(directory)
    companies: dict[str, CapIQCompany] = {}
    orphan_problems = []
    if not directory.exists():
        return companies
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in ('.xls', '.xlsx') and not p.name.startswith(('~', '.')))
    newest = {}   # (ticker, report) -> (mtime, path, sheet, raw): the newest export of each report wins
    for p in files:
        try:
            sheets = read_sheets(p)
        except Exception as e:
            orphan_problems.append(f'{p.name}: could not read ({e})')
            continue
        found, unsupported, last_co = False, [], None
        for sname, raw in sheets.items():
            if raw.shape[0] < 6 or raw.shape[1] < 2:
                continue
            key, name, exch, tick = identify(raw, p.name)
            if not tick:
                continue
            found = True
            co = companies.setdefault(tick, CapIQCompany(ticker=tick, exchange=exch, name=name))
            last_co = co
            if key is None:
                unsupported.append(sname)
                continue
            if _no_data(raw):
                continue
            mt = p.stat().st_mtime
            if (tick, key) in newest and newest[(tick, key)][0] >= mt:
                co.problems.append(f'{p.name} [{sname}]: older duplicate of {REPORT_NAMES[key]} - ignored')
                continue
            newest[(tick, key)] = (mt, p, sname, raw, len(sheets) > 1)
        if not found:
            orphan_problems.append(f'{p.name}: no "(Exchange:TICKER)" title found - skipped')
        elif unsupported and last_co is not None:
            last_co.problems.append(f'{p.name}: {len(unsupported)} unsupported sheet(s) skipped ({", ".join(unsupported[:4])}'
                                    f'{"..." if len(unsupported) > 4 else ""}) - chart-only sheets (e.g. Credit Health Panel charts) are not data')
    for (tick, key), (mt, p, sname, raw, multi) in newest.items():
        co = companies[tick]
        try:
            parsed = PARSERS[key](raw)
            if key in OLD_ATTRS:
                setattr(co, key, parsed)
            else:
                co.statements[key] = parsed
            co.files[key] = (f'{p.name} [{sname}]' if multi else p.name, pd.Timestamp.fromtimestamp(mt))
        except Exception as e:
            co.problems.append(f'{p.name} [{sname}]: could not parse as {REPORT_NAMES[key]} ({e})')
    if orphan_problems:
        companies.setdefault('', CapIQCompany(ticker='', name='(unassigned files)')).problems.extend(orphan_problems)
    return companies


# ----------------------------------------------------------------------------- display helpers
def ratio_kind(section, item):
    if 'growth' in str(section).lower() or str(item).rstrip().endswith('%') or item in PCT_RATIOS:
        return 'pct'
    if 'days' in str(item).lower() or 'cash conversion' in str(item).lower():
        return 'days'
    return 'x'


def fmt_value(v, kind, is_nm=False):
    if is_nm:
        return 'NM'
    if pd.isna(v):
        return '-'
    if kind == 'auto':
        return f'{v:,.0f}' if abs(v) >= 1000 or float(v).is_integer() else f'{v:,.2f}'
    return {'pct': f'{v * 100:,.1f}%', 'x': f'{v:,.2f}x', 'days': f'{v:,.0f}', 'mm': f'{v:,.0f}'}.get(kind, f'{v:,.2f}')


def display_ratios(m: Matrix, section=None):
    """Ratios matrix -> string DataFrame ready for st.dataframe."""
    v, nm = m.values, m.nm
    keep = v.index if section is None else v.index[v['Section'] == section]
    out = []
    for i in keep:
        kind = ratio_kind(v.at[i, 'Section'], v.at[i, 'Item'])
        out.append([v.at[i, 'Item']] + [fmt_value(v.at[i, p], kind, bool(nm.at[i, p])) for p in m.periods])
    return pd.DataFrame(out, columns=['Item'] + m.periods)


def display_cap_summary(m: Matrix, section=None):
    """Cap summary -> string DataFrame: $mm values, with '% of Total' shown in brackets where Capital IQ gives it."""
    v, nm, pc = m.values, m.nm, m.pct
    keep = v.index if section is None else v.index[v['Section'] == section]
    out = []
    for i in keep:
        item = v.at[i, 'Item']
        is_ratio = v.at[i, 'Section'] == 'Credit Ratios'
        row = [item]
        for p in m.periods:
            val, isnm = v.at[i, p], bool(nm.at[i, p])
            if is_ratio:
                row.append(fmt_value(val, 'x', isnm))
            elif v.at[i, 'Section'] == 'Interest Rate Data':
                row.append(fmt_value(val, 'pct', isnm))
            elif pd.isna(val):
                row.append('-')
            else:
                share = pc.at[i, p] if pc is not None else np.nan
                row.append(f'{val:,.0f}' + (f'  ({share * 100:.1f}%)' if pd.notna(share) else ''))
        out.append(row)
    return pd.DataFrame(out, columns=['Item'] + m.periods)


def stmt_kind(section, item):
    """Display kind for financial-statement rows: percentages for margins/growth, otherwise smart number format."""
    t = str(item)
    low = t.lower()
    return 'pct' if t.rstrip().endswith('%') or 'growth' in low or 'cagr' in low else 'auto'


def multiple_kind(section, item):
    return 'x'


def display_statement(m: Matrix, section=None, kind_fn=stmt_kind):
    """Any item x period Matrix -> string DataFrame (Section column kept when several sections are shown)."""
    v, nm = m.values, m.nm
    keep = v.index if section is None else v.index[v['Section'] == section]
    rows = []
    for i in keep:
        kind = kind_fn(v.at[i, 'Section'], v.at[i, 'Item'])
        rows.append([v.at[i, 'Section'], v.at[i, 'Item']] + [fmt_value(v.at[i, p], kind, bool(nm.at[i, p])) for p in m.periods])
    return pd.DataFrame(rows, columns=['Section', 'Item'] + m.periods)


def chart_frame(m: Matrix, items, section=None, kind_fn=stmt_kind):
    """Long DataFrame [Item, Period, Value] for charting; percent rows are scaled to % units."""
    v = m.values
    out = []
    for i in v.index:
        if v.at[i, 'Item'] not in items or (section is not None and v.at[i, 'Section'] != section):
            continue
        scale = 100 if kind_fn(v.at[i, 'Section'], v.at[i, 'Item']) == 'pct' else 1
        for p in m.periods:
            if pd.notna(v.at[i, p]):
                out.append({'Item': v.at[i, 'Item'], 'Period': p, 'Value': v.at[i, p] * scale})
    return pd.DataFrame(out, columns=['Item', 'Period', 'Value'])
