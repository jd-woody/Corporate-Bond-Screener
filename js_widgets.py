"""js_widgets.py - small vanilla-JavaScript widgets embedded in the Streamlit app with components.html.

render_curve_scatter(bonds, curve): interactive "YTM vs years to maturity" chart of the filtered bonds with the
Treasury curve drawn on top. The chart is drawn client-side in the browser (SVG, no libraries, no Node.js):
hover a bond for its details, and toggle IG / HY on and off without a Streamlit rerun.
"""
import json

import numpy as np
import pandas as pd
import streamlit.components.v1 as components

MAX_POINTS = 1500   # keeps the embedded HTML small; a random sample is drawn when more bonds are filtered

_HTML = """
<div id="wrap" style="position:relative;font-family:system-ui,Segoe UI,Arial,sans-serif;color:#888">
  <div id="controls" style="font-size:13px;margin:0 0 6px 4px">
    <label style="margin-right:14px;cursor:pointer"><input type="checkbox" id="tIG" checked>
      <span style="color:#1f77b4">&#9679;</span> IG</label>
    <label style="margin-right:14px;cursor:pointer"><input type="checkbox" id="tHY" checked>
      <span style="color:#ff7f0e">&#9679;</span> HY</label>
    <span style="color:#888">&#9473; Treasury curve</span>
    <span id="count" style="float:right;margin-right:8px"></span>
  </div>
  <svg id="chart" width="100%" viewBox="0 0 900 420" style="display:block"></svg>
  <div id="tip" style="position:absolute;display:none;pointer-events:none;padding:6px 8px;font-size:12px;line-height:1.35;
       background:rgba(30,30,30,.92);color:#fff;border-radius:4px;white-space:nowrap;z-index:5"></div>
</div>
<script>
const DATA = __DATA__;   // {pts: [[years, ytm, name, rating, universe], ...], curve: [[tenor, yield], ...]}
const NS = "http://www.w3.org/2000/svg";
const W = 900, H = 420, M = {l: 52, r: 16, t: 10, b: 40};
const COLORS = {IG: "#1f77b4", HY: "#ff7f0e"};
const svg = document.getElementById("chart"), tip = document.getElementById("tip");
const show = {IG: true, HY: true};

const xs = DATA.pts.map(p => p[0]).concat(DATA.curve.map(c => c[0]));
const ys = DATA.pts.map(p => p[1]).concat(DATA.curve.map(c => c[1]));
const xMax = Math.min(30, Math.ceil(Math.max(...xs, 1)));
const yMin = Math.max(0, Math.floor(Math.min(...ys, 0)));
const yMax = Math.ceil(Math.max(...ys, 1));
const sx = v => M.l + (v / xMax) * (W - M.l - M.r);
const sy = v => H - M.b - ((v - yMin) / (yMax - yMin)) * (H - M.t - M.b);

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || svg).appendChild(e);
  return e;
}

function axes() {
  const xStep = xMax > 15 ? 5 : 2, yStep = Math.max(1, Math.round((yMax - yMin) / 6));
  for (let v = 0; v <= xMax; v += xStep) {
    el("line", {x1: sx(v), x2: sx(v), y1: M.t, y2: H - M.b, stroke: "#888", "stroke-opacity": .15});
    el("text", {x: sx(v), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11, fill: "#888"}).textContent = v;
  }
  for (let v = yMin; v <= yMax; v += yStep) {
    el("line", {x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), stroke: "#888", "stroke-opacity": .15});
    el("text", {x: M.l - 8, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, fill: "#888"}).textContent = v + "%";
  }
  el("text", {x: (W + M.l) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "#888"}).textContent = "Years to maturity";
  el("text", {transform: "translate(13," + H / 2 + ") rotate(-90)", "text-anchor": "middle", "font-size": 12, fill: "#888"}).textContent = "Yield to maturity";
}

let dots = [];
function draw() {
  svg.innerHTML = "";
  axes();
  if (DATA.curve.length > 1) {
    el("polyline", {points: DATA.curve.map(c => sx(c[0]) + "," + sy(c[1])).join(" "), fill: "none", stroke: "#888", "stroke-width": 2});
  }
  dots = [];
  for (const p of DATA.pts) {
    if (!show[p[4]]) continue;
    const x = sx(Math.min(p[0], xMax)), y = sy(p[1]);
    el("circle", {cx: x, cy: y, r: 3.2, fill: COLORS[p[4]] || "#999", "fill-opacity": .55});
    dots.push({x, y, p});
  }
  document.getElementById("count").textContent = dots.length.toLocaleString() + " bonds shown";
}

svg.addEventListener("mousemove", ev => {
  const r = svg.getBoundingClientRect(), k = W / r.width;
  const mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
  let best = null, bd = 12 * 12;
  for (const d of dots) {
    const dd = (d.x - mx) ** 2 + (d.y - my) ** 2;
    if (dd < bd) { bd = dd; best = d; }
  }
  if (!best) { tip.style.display = "none"; return; }
  const p = best.p;
  tip.innerHTML = "<b>" + p[2].replace(/[<>&]/g, "") + "</b><br>" + p[4] + " &middot; " + (p[3] || "NR") +
                  "<br>" + p[1].toFixed(2) + "% YTM &middot; " + p[0].toFixed(1) + "y";
  tip.style.display = "block";
  const wr = document.getElementById("wrap").getBoundingClientRect();
  tip.style.left = Math.min(ev.clientX - wr.left + 12, wr.width - 190) + "px";
  tip.style.top = (ev.clientY - wr.top - 54) + "px";
});
svg.addEventListener("mouseleave", () => { tip.style.display = "none"; });
for (const u of ["IG", "HY"]) {
  document.getElementById("t" + u).addEventListener("change", e => { show[u] = e.target.checked; draw(); });
}
draw();
</script>
"""


def render_curve_scatter(bonds, curve=(None, None, None), height=470):
    """bonds: DataFrame with Name, Years to Maturity, YTM, Rating, Universe. curve: (tenors, yields %, date) from context."""
    cols = ['Name', 'Years to Maturity', 'YTM', 'Rating', 'Universe']
    z = bonds[[c for c in cols if c in bonds]].copy()
    z = z[(z['YTM'] > 0) & (z['YTM'] <= 30) & z['Years to Maturity'].notna() & (z['Years to Maturity'] > 0)]
    if len(z) > MAX_POINTS:
        z = z.sample(MAX_POINTS, random_state=0)
    pts = [[round(float(r['Years to Maturity']), 2), round(float(r['YTM']), 3), str(r['Name']),
            str(r.get('Rating', '') or ''), str(r.get('Universe', ''))] for _, r in z.iterrows()]
    t, y, _ = curve
    crv = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(t, y)] if t is not None else []
    payload = json.dumps({'pts': pts, 'curve': crv}).replace('</', '<\\/')   # never let data close the <script> tag
    components.html(_HTML.replace('__DATA__', payload), height=height)
