"""theme.py - one visual language for the whole app: palette, Plotly styling, CSS, KPI / story cards."""
import html

import streamlit as st

INK, MUTED, GRID = '#1B2430', '#6B7686', '#EDF0F5'
NAVY = '#0B2545'
BLUE, TEAL, AMBER, RED, ORANGE, PURPLE, GREEN, GREY = '#1F6FEB', '#13A89E', '#F2A900', '#D64045', '#EF7B45', '#7B61FF', '#2BB673', '#8A94A6'
CAT = [BLUE, TEAL, AMBER, RED, PURPLE, ORANGE, GREEN, GREY, '#C04AB5', '#00A6D6']
RATING_COLORS = {'AAA': '#0B3C8C', 'AA': '#1F6FEB', 'A': '#4DA3FF', 'BBB': TEAL, 'BB': AMBER, 'B': ORANGE, 'CCC': RED, 'NR': GREY}
RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'NR']
UNI_COLORS = {'IG': BLUE, 'HY': ORANGE}
TONE = {'good': GREEN, 'warn': AMBER, 'bad': RED, 'neutral': GREY, 'info': BLUE}
# cheap (wide / above peers) = red, rich (tight / below peers) = teal, centred on 0
DIVERGING = [[0.0, TEAL], [0.5, '#F4F6FA'], [1.0, RED]]
SEQUENTIAL = [[0.0, '#EAF2FF'], [0.5, '#6FA8FF'], [1.0, '#0B3C8C']]

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1500px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
[data-baseweb="tag"] {{background: #E8F0FE !important; color: {INK} !important; border-radius: 6px !important;}}
[data-baseweb="tag"] svg {{fill: {MUTED} !important;}}
.kpi-row {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{background: #fff; border: 1px solid #E3E8F0; border-radius: 12px; padding: 12px 14px; box-shadow: 0 1px 2px rgba(16,24,40,.04);}}
.kpi .l {{font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED};}}
.kpi .v {{font-size: 24px; font-weight: 650; color: {INK}; line-height: 1.25;}}
.kpi .s {{font-size: 12px; color: {MUTED};}}
.kpi.good {{border-left: 4px solid {GREEN};}} .kpi.warn {{border-left: 4px solid {AMBER};}} .kpi.bad {{border-left: 4px solid {RED};}}
.kpi.info {{border-left: 4px solid {BLUE};}} .kpi.neutral {{border-left: 4px solid #D6DBE4;}}
.story {{background: linear-gradient(135deg, #0B2545 0%, #13315C 100%); color: #fff; border-radius: 16px; padding: 20px 24px; margin: 4px 0 16px 0;}}
.story .t {{font-size: 12px; text-transform: uppercase; letter-spacing: .1em; opacity: .7;}}
.story .h {{font-size: 22px; font-weight: 650; line-height: 1.35; margin: 4px 0 8px 0;}}
.story .p {{font-size: 14px; opacity: .88; line-height: 1.55;}}
.chip {{display: inline-block; padding: 2px 10px; margin: 2px 6px 2px 0; border-radius: 999px; font-size: 12px; font-weight: 600; color: #fff;}}
.sig {{border: 1px solid #E3E8F0; border-radius: 12px; padding: 12px 14px; background: #fff; height: 100%;}}
.sig .tt {{font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px;}}
.sig .hh {{font-size: 15px; font-weight: 600; color: {INK}; margin-bottom: 4px;}}
.sig ul {{margin: 4px 0 0 16px; padding: 0; font-size: 13px; color: #3A4556;}}
.sec {{font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: {MUTED}; margin: 18px 0 2px 0; font-weight: 700;}}
.sub {{font-size: 13px; color: {MUTED}; margin-bottom: 8px;}}
</style>
"""


def apply_css():
    st.markdown(CSS, unsafe_allow_html=True)


def style(fig, h=420, title=None, legend=True):
    """House style for any Plotly figure (call last)."""
    fig.update_layout(template='plotly_white', height=h, colorway=CAT, font=dict(family='Inter, Segoe UI, Helvetica, Arial, sans-serif', size=12, color=INK),
                      margin=dict(l=8, r=8, t=48 if title else 18, b=8), hoverlabel=dict(bgcolor='white', font_size=12),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=legend,
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, title_text=''))
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15, color=INK)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='#D6DBE4', tickcolor='#D6DBE4')
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor='rgba(0,0,0,0)')
    return fig


def esc(x):
    return html.escape(str(x))


def kpi_row(items):
    """items = [(label, value, sub, tone)] -> a responsive row of cards."""
    cards = ''.join(f'<div class="kpi {esc(t)}"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
                    for l, v, s, t in [(i + ('', 'neutral')[len(i) - 2:] if len(i) < 4 else i) for i in items])
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def chip(text, tone='neutral'):
    return f'<span class="chip" style="background:{TONE.get(tone, GREY)}">{esc(text)}</span>'


def section(title, sub=''):
    st.markdown(f'<div class="sec">{esc(title)}</div>' + (f'<div class="sub">{esc(sub)}</div>' if sub else ''), unsafe_allow_html=True)


def story_banner(kicker, headline, body='', chips=''):
    st.markdown(f'<div class="story"><div class="t">{esc(kicker)}</div><div class="h">{esc(headline)}</div><div class="p">{esc(body)}</div>'
                f'<div style="margin-top:10px">{chips}</div></div>', unsafe_allow_html=True)


def signal_card(title, headline, bullets, tone='neutral'):
    lis = ''.join(f'<li>{esc(b)}</li>' for b in bullets)
    st.markdown(f'<div class="sig" style="border-top:4px solid {TONE.get(tone, GREY)}"><div class="tt" style="color:{TONE.get(tone, GREY)}">{esc(title)}</div>'
                f'<div class="hh">{esc(headline)}</div><ul>{lis}</ul></div>', unsafe_allow_html=True)
