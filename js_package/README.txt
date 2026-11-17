JS yield-curve chart package
============================
js_widgets.py          the vanilla-JavaScript chart (embedded in Streamlit with components.html)
apply_patch.py         copies the files in and edits credit_analytics_bot.py (makes a .bak backup first)
data/sample/           synthetic IG/HY bonds so the app runs without Capital IQ data

Manual edits, if apply_patch.py cannot find a spot:
 1. after "from filters import render_sidebar" add:   from js_widgets import render_curve_scatter
 2. in load_one(), after "p=ROOT/'data'/file" add:    if not p.exists(): p=ROOT/'data'/'sample'/file
 3. in the Clean Screener page, before st.dataframe(srt,...) add:
        with st.expander('Yield vs maturity (interactive JS chart)'):
            render_curve_scatter(x,CTX.curve)
