"""apply_patch.py - installs the JS chart into the screener. Run it from your project folder (the one with credit_analytics_bot.py).
Safe to run twice. Makes a backup first: credit_analytics_bot_py_before_js.bak (your .gitignore already ignores *.bak)."""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
proj = Path.cwd()
app = proj / 'credit_analytics_bot.py'
if not app.exists():
    sys.exit('Run this from the folder that contains credit_analytics_bot.py (cd there first).')

shutil.copy2(here / 'js_widgets.py', proj / 'js_widgets.py')
(proj / 'data' / 'sample').mkdir(parents=True, exist_ok=True)
for f in (here / 'data' / 'sample').glob('*.csv'):
    shutil.copy2(f, proj / 'data' / 'sample' / f.name)

src = app.read_text(encoding='utf-8')
orig = src
bak = proj / 'credit_analytics_bot_py_before_js.bak'
if not bak.exists():          # keep the FIRST backup, so a second run never overwrites your original
    shutil.copy2(app, bak)

edits = [
    ('from js_widgets import render_curve_scatter', 'from filters import render_sidebar\n',
     'from filters import render_sidebar\nfrom js_widgets import render_curve_scatter\n'),
    ('p=ROOT/\'data\'/\'sample\'/file', "    p=ROOT/'data'/file\n    if not p.exists(): return pd.DataFrame()",
     "    p=ROOT/'data'/file\n    if not p.exists(): p=ROOT/'data'/'sample'/file\n    if not p.exists(): return pd.DataFrame()"),
    ('render_curve_scatter(x', "    st.dataframe(srt,hide_index=True",
     "    with st.expander('Yield vs maturity (interactive JS chart)'):\n        render_curve_scatter(x,CTX.curve)\n    st.dataframe(srt,hide_index=True"),
]
for marker, find, repl in edits:
    if marker in src:
        print('already applied:', marker[:40])
    elif find in src:
        src = src.replace(find, repl, 1)
        print('patched:', marker[:40])
    else:
        print('COULD NOT FIND the spot for:', marker[:40], '- add it by hand (see README.txt)')
if src != orig:
    app.write_text(src, encoding='utf-8')
print('Done. Start the app with:  streamlit run credit_analytics_bot.py')
