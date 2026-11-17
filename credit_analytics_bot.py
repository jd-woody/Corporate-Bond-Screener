import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Corporate Credit Screener (IG/HY)", layout="wide")
import analytics
import context
import theme as T
from filters import render_sidebar
from js_widgets import render_curve_scatter
from profile_tab import render_profile_tab
from regression_tab import render_regression_tab
from rv_tab import render_rv_tab
from sector_tab import render_sector_tab
from ui_helpers import nav
from visual_tab import render_visual_tab

ROOT=Path(__file__).resolve().parent
CAPIQ_DIR=ROOT/'data'/'capiq'        # Capital IQ .xls/.xlsx exports
NOTES_DIR=ROOT/'data'/'stories'      # analyst notes (issuer / sector / industry)
T.apply_css()
NOTCH={r:i+1 for i,r in enumerate(["AAA","AA+","AA","AA-","A+","A","A-","BBB+","BBB","BBB-","BB+","BB","BB-","B+","B","B-","CCC+","CCC","CCC-","CC","C","D"])}
SEN={"SR SECURED":1,"SECURED":1,"1ST LIEN":1,"2ND LIEN":2,"COMPANY GUARNT":3,"SR UNSECURED":3,"UNSECURED":3,"SR SUBORDINATED":4,"SUBORDINATED":5,"JR SUBORDINATED":6}

def load_one(file, universe):
    p=ROOT/'data'/file
    if not p.exists(): p=ROOT/'data'/'sample'/file
    if not p.exists(): return pd.DataFrame()
    x=pd.read_csv(p,dtype=str,keep_default_na=False).rename(columns={'Issuer':'Name','YTM':'YTM','YTM (%)':'YTM','Coupon (%)':'Coupon','Outstanding Amount mm':'Outstanding Amount','Outstanding Amount (mm)':'Outstanding Amount','Outstanding Amount ($mm)':'Outstanding Amount'})
    for c in ['Coupon','Ask Price','YTM','Years to Maturity','Outstanding Amount','Issued Amount mm']:
        if c in x: x[c]=pd.to_numeric(x[c],errors='coerce')
    if 'Outstanding Amount' in x: x['Outstanding Amount']*=1e6
    x['Rating']=x['Rating'].str.upper().str.strip(); x['Notch']=x['Rating'].map(NOTCH)
    x['Seniority Rank']=x['Collateral Type'].map(SEN).fillna(3)
    x['Universe']=universe
    x['Label']=x['Name'].fillna('Unknown')+' | '+x.get('Coupon',pd.Series(index=x.index,dtype=float)).map(lambda v:f'{v:.3g}%' if pd.notna(v) else 'FRN')+' | '+x.get('Maturity',pd.Series(index=x.index,dtype=str)).fillna('Perp')
    return x
@st.cache_data
def load():
    x=pd.concat([load_one('IG_Bond_Screener.csv','IG'),load_one('HY_Bond_Screener.csv','HY')],ignore_index=True)
    return analytics.enrich(x) if not x.empty else x   # Country, CUSIP, duration/convexity, maturity bucket, search text
def wavg(x,col):
    z=x[[col,'Outstanding Amount']].dropna()
    return np.average(z[col],weights=z['Outstanding Amount'].clip(lower=1)) if len(z) else np.nan

d=load()
st.title('Corporate Credit Screener (IG/HY)')
st.caption('Dashboard for screening corporate credit instruments. Outputs are screening signals only - not real-time prices.')
if d.empty:
    st.error('Add IG_Bond_Screener.csv and HY_Bond_Screener.csv to the data folder.'); st.stop()
d,CTX=context.build(d,ROOT)          # adds Spread (bp) columns, loads benchmarks, saves a history snapshot
f,SORT_COL,SORT_ASC,ACTIVE=render_sidebar(d,NOTCH)
if f.empty:
    st.warning('No bonds match the current filters. Loosen them or press "Reset filters" in the sidebar.'); st.stop()

PAGES=['Clean Screener','Visual Analytics','Sector Analysis','Relative Value','Regressions','Credit Profile']
page=nav(PAGES,'main_page')
if page=='Clean Screener':
    x=f.copy()
    a,b,c,e=st.columns(4); a.metric('Bonds',len(x)); b.metric('Issuers',x.Name.nunique()); c.metric('Weighted YTM',f'{wavg(x,"YTM"):.2f}%'); e.metric('Outstanding',f'${x["Outstanding Amount"].sum()/1e9:.1f}bn')
    st.caption('Active filters: '+('; '.join(ACTIVE) if ACTIVE else 'none'))
    cols=[z for z in ['Universe','Name','Ticker','Sector','Industry','Rating','Collateral Type','Maturity Type','Coupon','Ask Price','YTM','G-Spread (bp)','Maturity','Years to Maturity','Mod Duration','Outstanding Amount','Country','ISIN','CUSIP','ID Number'] if z in x]
    srt=x[cols].sort_values(SORT_COL,ascending=SORT_ASC,na_position='last')
    with st.expander('Yield vs maturity'):
        render_curve_scatter(x,CTX.curve)
    st.dataframe(srt,hide_index=True,use_container_width=True,column_config={'Mod Duration':st.column_config.NumberColumn(format='%.2f'),'G-Spread (bp)':st.column_config.NumberColumn(format='%.0f')})
    st.download_button('Export filtered bonds',srt.to_csv(index=False).encode(),'credit_screener.csv','text/csv')
elif page=='Visual Analytics':
    render_visual_tab(f,NOTCH)
elif page=='Sector Analysis':
    render_sector_tab(f,d,CTX,CAPIQ_DIR,NOTES_DIR,NOTCH)
elif page=='Relative Value':
    render_rv_tab(f,NOTCH)
elif page=='Regressions':
    render_regression_tab(f,NOTCH)
else:
    render_profile_tab(d,f,CTX,CAPIQ_DIR,NOTES_DIR)
