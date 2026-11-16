# IG/HY Credit Analytics

A research dashboard for corporate credit: screen the universe, find relative value, model fair yields, and tell the story of each issuer and sector.

## Pages
- **Clean Screener** - filtered, sorted bond table (the sidebar screener applies everywhere).
- **Visual Analytics** - yield / price / maturity / rating pictures.
- **Sector Analysis** - *Landscape* (treemap, sector map, rating mix, spread distributions) | *Benchmarks* (spread vs. rating-matched index by sector, industry,
  rating, universe, country or collateral; heat map; benchmark path over time; ICE BofA OAS history) | *Sector & industry story* (auto-generated read + your narrative on
  management, operations, regulation, outlook) | *Issuers*.
- **Relative Value** - Bond comparison (tale of the tape) | Cheap / rich vs peers (cheapness map) | Curve comparison | Radar | Position sizing (DV01-neutral pair, DV01 + convexity-neutral butterfly).
- **Regressions** - drivers (effect sizes), fitted curves by rating, diagnostics, cheap / rich, "explain a bond" waterfall, fair-yield predictor.
- **Credit Profile** - the issuer hub: *Story* (headline, KPIs, the arc, automated signals incl. peer credit health, your management & operations notes) |
  **Credit health & comps** (Capital IQ Credit Health Panel scorecard, default probability vs. CreditModel, strengths / weaknesses, comparable bonds, credit comps with median / mean) |
  Capital structure (now with a **capital stack showing leverage and loan-to-value through each tranche**) | Fundamentals (cash bridge, sources & uses, balance sheet) |
  **Ratios** (39-ratio toolkit, 12-ratio dashboard, glossary) | **Scenarios** (Conservative / Base (Street) / Optimistic forecast; recovery waterfall with fulcrum security) |
  Valuation & segments | Market & peers (now with a **price-yield / convexity profile**) | Notes | Data & exports.

## Data folders
```
data/IG_Bond_Screener.csv, HY_Bond_Screener.csv   bond snapshots
data/capiq/        Capital IQ .xls / .xlsx exports (multi-sheet Financials workbook, or single pages)
data/benchmarks/   Treasury + ICE BofA OAS from FRED (refresh button, or: python update_benchmarks.py)
data/history/      snapshots.csv - saved automatically each time a new screener CSV is loaded; builds YOUR OWN time series
data/stories/      your notes (JSON)
```

## Run
```bash
python -m pip install -r requirements.txt
python update_benchmarks.py          # optional but recommended: unlocks G-spreads and the index comparison
python -m streamlit run credit_analytics_bot.py
```

## Method notes
Spread = G-spread when a Treasury curve is loaded, else yield in bp. Duration / convexity are model values (plain fixed-rate bond priced off its quoted yield; callable bonds are overstated).
The issuer "signals" are heuristics that prompt a view - not a rating. Research tool only; validate data quality, call features, liquidity and fundamentals before acting.


## Ideas ported from the fixed-income Excel templates
Comps sheet -> peer comps with Median / Mean and the subject below, credit-health comps, credit-ratio glossary (Ratios page) | Capital Structure sheet -> capital stack with cumulative leverage and LTV |
Forecast sheet -> Conservative / Base (Street) / Optimistic projections of EBITDA, FCF, debt and leverage | Opportunistic sheet -> recovery waterfall, expected recovery vs. price, fulcrum security |
Convexity sheet -> price-yield curve with duration and duration + convexity approximations.
