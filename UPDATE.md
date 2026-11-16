# Updating an existing install

Your CSVs and existing data/capiq files are not overwritten (the zip only adds sample Capital IQ files with the same names).
This update REPLACES: capiq_loader.py, story.py, profile_tab.py. NEW: comps.py, profile_credit.py. Everything else is included unchanged so the set stays consistent.
The zip also carries the Apple Credit Health Panel export as a sample in data/capiq.

## Windows - PowerShell (from the project folder)
```powershell
cd "C:\50Five Office\Screener\IG_HY Screener"
copy credit_analytics_bot.py credit_analytics_bot_py_before_v3.bak
tar -xf "$env:USERPROFILE\Downloads\credit_analytics_bot_v3_credit_comps_update.zip" -C .
python -m pip install -r requirements.txt
python update_benchmarks.py
python -m streamlit run credit_analytics_bot.py
```

## Mac / Linux
```bash
cd /path/to/project
cp credit_analytics_bot.py credit_analytics_bot_py_before_v3.bak
unzip -o ~/Downloads/credit_analytics_bot_v3_credit_comps_update.zip -d .
python3 -m pip install -r requirements.txt
python3 update_benchmarks.py
python3 -m streamlit run credit_analytics_bot.py
```

## First run
1. Press Ctrl+C on any running app first, then start it fresh.
   Credit Profile > Credit health & comps lights up when the Credit Health Panel export (.xls) is in data/capiq; the Ratios, Scenarios and capital-stack views need the Financials workbook.
2. If `update_benchmarks.py` reports failures (no internet / proxy), the app still works: visuals use yield (bp) and the screener's own rating-bucket benchmark.
3. Credit Profile > Notes and Sector Analysis > Sector & industry story: add your thesis, management and operations notes - they become the story banner.

## Roll back
Copy the .bak back over credit_analytics_bot.py (the new modules are only imported by the new version). To restore the old red theme delete .streamlit/config.toml changes.

## Troubleshooting
- Navigation shows a radio list instead of pills: Streamlit older than 1.40 - run `python -m pip install -U streamlit`.
- "No Treasury curve yet": run `python update_benchmarks.py` or press the refresh button on Sector Analysis > Benchmarks.
- Charts look stale after replacing files: stop and restart the app, then refresh the browser.
