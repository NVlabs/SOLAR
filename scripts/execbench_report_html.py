#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render report_data.json (from report_execbench.py) as a single self-contained HTML page.

    python scripts/execbench_report_html.py out/execbench_final/report_data.json -o out/execbench_final/sol_report.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("report_json", type=Path)
ap.add_argument("-o", "--out", type=Path, required=True)
ap.add_argument("--setup", default="configs/execbench/leaderboard_b200.yaml")
args = ap.parse_args()
D = json.loads(args.report_json.read_text())

# Compact point list for the chart: [subset, problem, sol_us, measured_us, ref_sol_us, confidence]
points = [[p["subset"], p["problem"], round(p["sol_us"], 4), p["measured_us"] and round(p["measured_us"], 4),
           round(p["ref_sol_us"], 4), p["confidence"]] for p in D["points"]]
problems = D["problems"]
conf = D["confidence"]
subs = D["per_subset"]
viol = D["violations"]
n = D["shapes"]


def pct(k): return f"{100 * k / n:.0f} %"


def esc(s): return html.escape(str(s if s is not None else ""))


rows_above = [p for p in problems if p["median_ratio"] > 1.1]
rows_below = [p for p in problems if p["median_ratio"] < 0.9]


def problem_rows(ps):
    out = []
    for p in ps:
        out.append(
            f"<tr><td>{esc(p['subset'])}</td><td class=mono>{esc(p['problem'])}</td><td class=num>{p['shapes']}</td>"
            f"<td class=num>{p['median_sol_us']:,.2f}</td><td class=num>{p['median_ref_sol_us']:,.2f}</td>"
            f"<td class=num>{'' if p['median_measured_us'] is None else format(p['median_measured_us'], ',.1f')}</td>"
            f"<td class=num>{p['median_ratio']:.2f}</td>"
            f"<td><span class='chip {esc(p['confidence'])}'>{esc(p['confidence'])}</span></td>"
            f"<td class=cause>{esc(p['cause'] or p['confidence_reason'])}</td></tr>")
    return "\n".join(out)


subset_rows = "\n".join(
    f"<tr><td>{esc(s)}</td><td class=num>{d['shapes']}</td><td class=num>{d['median']:.3f}</td>"
    f"<td class=num>{d['within10']} <span class=dim>({100*d['within10']/d['shapes']:.0f} %)</span></td>"
    f"<td class=num>{d['above10']}</td><td class=num>{d['below10']}</td><td class=num>{d['above_measured']}</td></tr>"
    for s, d in subs.items())

all_problem_rows = "\n".join(
    f"<tr><td>{esc(p['subset'])}</td><td class=mono>{esc(p['problem'])}</td><td class=num>{p['shapes']}</td>"
    f"<td class=num>{p['median_sol_us']:,.2f}</td><td class=num>{p['median_ref_sol_us']:,.2f}</td>"
    f"<td class=num>{'' if p['median_measured_us'] is None else format(p['median_measured_us'], ',.1f')}</td>"
    f"<td class=num>{p['median_ratio']:.2f}</td>"
    f"<td class=num>{'' if p['max_sol_over_measured'] is None else format(p['max_sol_over_measured'], '.3f')}</td>"
    f"<td>{esc(p['precision'])}</td><td>{esc(p['bottleneck'])}</td>"
    f"<td><span class='chip {esc(p['confidence'])}'>{esc(p['confidence'])}</span></td></tr>"
    for p in sorted(problems, key=lambda p: (p['subset'], p['problem'])))

page = f"""<title>Solar SOL for SOL-ExecBench</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15202b; --ink-2:#4b5563; --ink-3:#7b8794; --line:#dde2e8;
  --accent:#0b6e99; --accent-ink:#ffffff; --sol:#0b6e99; --meas:#c2410c; --ref:#6b7280;
  --high:#1b7f4a; --high-bg:#e3f4ea; --medium:#a16207; --medium-bg:#fdf3d7; --low:#b42318; --low-bg:#fde7e4;
  --l1:#2a6f97; --l2:#8c4a9e; --quant:#c2410c; --fi:#0f8b8d;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}}
@media (prefers-color-scheme: dark){{ :root:not([data-theme="light"]){{ color-scheme:dark;
  --bg:#0f1419; --panel:#161c23; --ink:#e6ebf0; --ink-2:#b4bdc7; --ink-3:#7f8b98; --line:#2a333d;
  --accent:#4fb3e8; --accent-ink:#0b1016; --sol:#4fb3e8; --meas:#fb923c; --ref:#9aa3ad;
  --high:#4ade80; --high-bg:#122a1c; --medium:#facc15; --medium-bg:#2e2609; --low:#f87171; --low-bg:#331512;
  --l1:#6fb1dc; --l2:#c08fd3; --quant:#fb923c; --fi:#4fd1c5; }} }}
:root[data-theme="dark"]{{ color-scheme:dark;
  --bg:#0f1419; --panel:#161c23; --ink:#e6ebf0; --ink-2:#b4bdc7; --ink-3:#7f8b98; --line:#2a333d;
  --accent:#4fb3e8; --accent-ink:#0b1016; --sol:#4fb3e8; --meas:#fb923c; --ref:#9aa3ad;
  --high:#4ade80; --high-bg:#122a1c; --medium:#facc15; --medium-bg:#2e2609; --low:#f87171; --low-bg:#331512;
  --l1:#6fb1dc; --l2:#c08fd3; --quant:#fb923c; --fi:#4fd1c5; }}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.5;padding-inline:clamp(16px,4vw,48px);padding-block:24px 64px;margin:0}}
h1{{font-size:clamp(26px,3.2vw,36px);font-weight:600;letter-spacing:-0.01em;margin:0 0 4px;text-wrap:balance}}
h2{{font-size:19px;font-weight:600;margin:40px 0 12px;text-wrap:balance}}
h3{{font-size:15px;font-weight:600;margin:20px 0 8px}}
p{{max-width:72ch;color:var(--ink-2);margin:6px 0}}
.lede{{font-size:16px;color:var(--ink-2);max-width:78ch}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
.strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0 8px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px}}
.tile .v{{font-family:var(--mono);font-size:26px;font-weight:500;font-variant-numeric:tabular-nums}}
.tile .k{{font-size:13px;color:var(--ink-2);margin-top:2px}}
.tile.ok .v{{color:var(--high)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px}}
.chart-wrap{{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:16px;align-items:start}}
@media (max-width:860px){{.chart-wrap{{grid-template-columns:1fr}}}}
svg text{{font-family:var(--mono);font-size:11px;fill:var(--ink-2)}}
.legend{{font-size:13px;color:var(--ink-2)}}
.legend .row{{display:flex;align-items:center;gap:8px;margin:6px 0}}
.sw{{width:12px;height:12px;border-radius:50%;display:inline-block;flex:none}}
.controls{{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 4px}}
.controls button{{font:inherit;font-size:13px;padding:5px 10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:4px;cursor:pointer}}
.controls button[aria-pressed="true"]{{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}}
.controls button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
#tip{{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:8px 10px;font-size:12px;line-height:1.4;box-shadow:0 4px 14px rgba(0,0,0,.15);max-width:320px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-weight:600;color:var(--ink-2);font-size:12.5px;letter-spacing:.02em;white-space:nowrap}}
td.num,th.num{{text-align:right;font-family:var(--mono);font-size:13px}}
td.mono{{font-family:var(--mono);font-size:12.5px}}
td.cause{{color:var(--ink-2);max-width:46ch}}
.dim{{color:var(--ink-3)}}
.scroll{{overflow-x:auto}}
.chip{{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:500;font-family:var(--mono)}}
.chip.high{{background:var(--high-bg);color:var(--high)}} .chip.medium{{background:var(--medium-bg);color:var(--medium)}} .chip.low{{background:var(--low-bg);color:var(--low)}} .chip.n\\/a{{background:var(--line)}}
ul.assume{{max-width:80ch;padding-left:20px;color:var(--ink-2)}} ul.assume li{{margin:6px 0}} ul.assume b{{color:var(--ink);font-weight:600}}
details summary{{cursor:pointer;font-weight:600;margin:14px 0 8px}}
code{{font-family:var(--mono);font-size:12.5px;background:var(--bg);border:1px solid var(--line);border-radius:3px;padding:0 4px}}
@media (prefers-reduced-motion: reduce){{*{{transition:none!important}}}}
</style>

<div class="eyebrow">Solar · SOL-ExecBench · B200 · {esc(D.get('label') or '')}</div>
<h1>Solar SOL for SOL-ExecBench</h1>
<p class="lede">Speed-of-light latency for every problem and workload shape in SOL-ExecBench, produced by Solar's traced einsum roofline model under the <code>{esc(args.setup)}</code> setup. Each shape is compared with the leaderboard SOL table and with the two measured implementations in that table (the reference implementation and the optimized baseline). A SOL is a lower bound, so it must never exceed a measured kernel.</p>

<div class="strip">
  <div class="tile"><div class="v">{n:,}</div><div class="k">workload shapes with a SOL (of {D['shapes_total']:,})</div></div>
  <div class="tile"><div class="v">{D['median_ratio']:.3f}</div><div class="k">median SOL / leaderboard SOL</div></div>
  <div class="tile"><div class="v">{pct(D['buckets'][0]['shapes'])}</div><div class="k">shapes within ±10 % of the leaderboard SOL</div></div>
  <div class="tile ok"><div class="v">{viol['reference_impl']}</div><div class="k">shapes with SOL above the measured reference implementation</div></div>
  <div class="tile ok"><div class="v">{viol['optimized_baseline']}</div><div class="k">shapes with SOL above the measured optimized baseline</div></div>
  <div class="tile"><div class="v">{conf.get('high',0)} / {conf.get('medium',0)} / {conf.get('low',0)}</div><div class="k">shapes at high / medium / low confidence</div></div>
</div>

<h2>SOL against the best measured runtime</h2>
<p>Every shape is one point: Solar's fused SOL on the vertical axis, the fastest measured implementation of that shape on the horizontal axis (both in microseconds, log scale). Points on the diagonal would mean the measured kernel already reaches the bound; everything must lie on or below it. Hover a point for its problem and numbers; the buttons filter by subset or confidence.</p>
<div class="panel">
  <div class="controls" id="filters" role="group" aria-label="Filter points"></div>
  <div class="chart-wrap">
    <div id="chart"></div>
    <div class="legend" id="legend"></div>
  </div>
</div>

<h2>Key assumptions of the SOL model</h2>
<ul class="assume">
  <li><b>Precision.</b> The MAC-rate class comes from the problem's declared input dtypes (narrowest floating class wins; fp8 and nvfp4 from the quant metadata). Problems whose inputs are all float32, and anything with no dtype specified, are priced at the <b>16-bit tensor-core rate</b>: optimized kernels for those problems run bf16/fp16 math, and the bound has to hold for them. Memory bytes always use each tensor's own width (bool 1 B, fp8 1 B, bf16 2 B, fp32 4 B, complex64 8 B).</li>
  <li><b>Architecture.</b> B200 at the sustained clocks of <code>configs/arch/B200.yaml</code>: 0.906 PFLOPS TF32, 1.81 PFLOPS fp16/bf16, 3.62 PFLOPS fp8, 7.25 PFLOPS nvfp4 dense; 8 TB/s HBM. No launch floor is added to Solar's numbers (the leaderboard table adds about 0.4 µs; the comparison accepts either).</li>
  <li><b>Fusion.</b> The reported SOL is the fully fused bound: every external byte is read from HBM at most once (access regions deduplicated per tensor), intermediates stay on chip, declared outputs are written once. The unfused figure (every op's inputs and outputs) is kept in the CSV.</li>
  <li><b>Only touched memory is charged.</b> Paged and ragged attention read the gathered pages, MoE expert loops read each expert slice once, a cast or copy that only feeds a gather reads the gathered footprint, creation ops read nothing, broadcast outputs are views, and sparse scatters into zero buffers write only the touched rows.</li>
  <li><b>Only needed work is charged.</b> Triangular and additive -inf masks halve the attention and chunk-scan einsums they mask, products whose output is only partly gathered are computed for those rows, and einsums over operands that are pure replications (one group expanded to all heads) are computed once.</li>
  <li><b>Reference conventions kept.</b> Packed FP4 pairs count as two MACs; dense declared outputs are written in full unless the reference itself updates them sparsely; masks that arrive as input data are not discounted.</li>
</ul>

<h2>Confidence of each SOL</h2>
<p>Classified per shape from the patterns in its traced graph. A shape takes the lowest level of any pattern it contains.</p>
<div class="scroll"><table>
<tr><th>level</th><th class="num">shapes</th><th>pattern</th><th>why the bound is less certain</th></tr>
<tr><td><span class="chip high">high</span></td><td class="num">{conf.get('high',0)}</td><td>dense or streaming kernels</td><td>the traced shapes determine the work exactly</td></tr>
<tr><td><span class="chip medium">medium</span></td><td class="num">{conf.get('medium',0)}</td><td>structural sparsity</td><td>triangular / masked operands, broadcast or gathered-output shortcuts are discounted statically; a kernel has to realise that structure</td></tr>
<tr><td><span class="chip low">low</span></td><td class="num">{conf.get('low',0)}</td><td>data-dependent behaviour, or in-kernel quantization</td><td>index / mask inputs, gathers, scatters, top-k fix the touched set for one input instance; quantizer compute and FP4 packing are modelled, not measured</td></tr>
</table></div>

<h2>Per subset</h2>
<div class="scroll"><table>
<tr><th>subset</th><th class="num">shapes</th><th class="num">median SOL / ref</th><th class="num">within ±10 %</th><th class="num">above 10 %</th><th class="num">below 10 %</th><th class="num">above measured</th></tr>
{subset_rows}
</table></div>

<h2>Problems whose median SOL is above the leaderboard SOL by more than 10 % ({len(rows_above)} of {len(problems)})</h2>
<p>Each has a hand-checked cause. None of them exceeds a measured kernel.</p>
<div class="scroll"><table>
<tr><th>subset</th><th>problem</th><th class="num">shapes</th><th class="num">median SOL (µs)</th><th class="num">leaderboard SOL (µs)</th><th class="num">best measured (µs)</th><th class="num">SOL / ref</th><th>confidence</th><th>cause</th></tr>
{problem_rows(rows_above)}
</table></div>

<h2>Problems whose median SOL is below the leaderboard SOL by more than 10 % ({len(rows_below)})</h2>
<div class="scroll"><table>
<tr><th>subset</th><th>problem</th><th class="num">shapes</th><th class="num">median SOL (µs)</th><th class="num">leaderboard SOL (µs)</th><th class="num">best measured (µs)</th><th class="num">SOL / ref</th><th>confidence</th><th>cause</th></tr>
{problem_rows(rows_below)}
</table></div>

<details><summary>All {len(problems)} problems (medians over their shapes)</summary>
<div class="scroll"><table>
<tr><th>subset</th><th>problem</th><th class="num">shapes</th><th class="num">median SOL (µs)</th><th class="num">leaderboard SOL (µs)</th><th class="num">best measured (µs)</th><th class="num">SOL / ref</th><th class="num">max SOL / measured</th><th>precision</th><th>bottleneck</th><th>confidence</th></tr>
{all_problem_rows}
</table></div>
</details>

<h2>How to reproduce</h2>
<p>Install Solar (the install script applies the torchview patch), check out SOL-ExecBench next to it, then:</p>
<pre style="font-family:var(--mono);font-size:12.5px;overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px">python scripts/sweep_execbench.py --all-workloads --jobs 1 --mem-gb 55 --timeout 3600 \\
    --setup-config configs/execbench/leaderboard_b200.yaml --out-root out/execbench_full
python scripts/confidence_execbench.py --results-root out/execbench_full -o out/execbench_full/confidence.csv
python scripts/export_execbench_csv.py --ref &lt;sol_latencies.csv&gt; --results-root out/execbench_full \\
    --confidence out/execbench_full/confidence.csv -o out/execbench_full/sol_latencies
python scripts/report_execbench.py out/execbench_full/sol_latencies_compare.csv --label full
python scripts/execbench_report_html.py out/execbench_full/report_data.json -o out/execbench_full/sol_report.html</pre>
<p>Changing only the pricing policy or the arch is a perf-stage re-run: <code>scripts/reprice_execbench.py</code>. See the README section "Reproducing the SOL-ExecBench results".</p>
<div id="tip" hidden></div>

<script>
const POINTS = {json.dumps(points)};
const SUBS = ["L1","L2","Quant","FlashInfer-Bench"];
const COL = {{"L1":"var(--l1)","L2":"var(--l2)","Quant":"var(--quant)","FlashInfer-Bench":"var(--fi)"}};
const CONF = ["high","medium","low"];
const state = {{subset:new Set(SUBS), conf:new Set(CONF)}};
const filters = document.getElementById("filters");
function btn(label, on, cb){{ const b=document.createElement("button"); b.textContent=label; b.setAttribute("aria-pressed", on?"true":"false"); b.addEventListener("click",()=>{{ cb(); b.setAttribute("aria-pressed", b.getAttribute("aria-pressed")==="true"?"false":"true"); draw(); }}); return b; }}
SUBS.forEach(s=>filters.appendChild(btn(s,true,()=>state.subset.has(s)?state.subset.delete(s):state.subset.add(s))));
const sep=document.createElement("span"); sep.style.width="12px"; filters.appendChild(sep);
CONF.forEach(c=>filters.appendChild(btn("confidence: "+c,true,()=>state.conf.has(c)?state.conf.delete(c):state.conf.add(c))));
const legend=document.getElementById("legend");
legend.innerHTML = SUBS.map(s=>`<div class=row><span class=sw style="background:${{COL[s]}}"></span>${{s}}</div>`).join("") +
  `<div class=row style="margin-top:10px"><span class=sw style="background:var(--ink-3)"></span>filled: high confidence</div>
   <div class=row><span class=sw style="background:transparent;border:2px solid var(--ink-3)"></span>ring: medium</div>
   <div class=row><span class=sw style="background:transparent;border:2px dashed var(--ink-3)"></span>dashed ring: low</div>
   <div class=row style="margin-top:10px"><span style="width:18px;border-top:2px solid var(--meas);display:inline-block"></span>SOL = best measured runtime</div>
   <div class=row><span style="width:18px;border-top:1px dashed var(--ref);display:inline-block"></span>SOL = measured / 10</div>`;
const tip=document.getElementById("tip");
function fmt(x){{ return x>=1000? x.toLocaleString(undefined,{{maximumFractionDigits:0}}) : x>=10 ? x.toFixed(1) : x.toFixed(3); }}
function draw(){{
  const el=document.getElementById("chart");
  const W=Math.max(320, el.clientWidth||700), H=Math.round(Math.min(560, Math.max(360, W*0.72)));
  const m={{l:58,r:16,t:14,b:44}};
  const pts=POINTS.filter(p=>p[3]!=null && p[3]>0 && p[2]>0 && state.subset.has(p[0]) && state.conf.has(p[5]));
  const xs=pts.map(p=>p[3]), ys=pts.map(p=>p[2]);
  const lo=Math.pow(10,Math.floor(Math.log10(Math.min(...xs,...ys,1e-3)))), hi=Math.pow(10,Math.ceil(Math.log10(Math.max(...xs,...ys,1))));
  const sx=v=>m.l+(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(W-m.l-m.r);
  const sy=v=>H-m.b-(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(H-m.t-m.b);
  let s=`<svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="${{H}}" role="img" aria-label="Scatter of Solar SOL versus best measured runtime per workload shape">`;
  for(let e=Math.log10(lo); e<=Math.log10(hi); e++){{ const v=Math.pow(10,e);
    s+=`<line x1="${{sx(v)}}" y1="${{m.t}}" x2="${{sx(v)}}" y2="${{H-m.b}}" stroke="var(--line)" stroke-width="1"/>`;
    s+=`<line x1="${{m.l}}" y1="${{sy(v)}}" x2="${{W-m.r}}" y2="${{sy(v)}}" stroke="var(--line)" stroke-width="1"/>`;
    const lab = v>=1000? (v/1000)+" ms" : v>=1 ? v+" µs" : (v*1000)+" ns";
    s+=`<text x="${{sx(v)}}" y="${{H-m.b+16}}" text-anchor="middle">${{lab}}</text>`;
    s+=`<text x="${{m.l-6}}" y="${{sy(v)+4}}" text-anchor="end">${{lab}}</text>`; }}
  s+=`<line x1="${{sx(lo)}}" y1="${{sy(lo)}}" x2="${{sx(hi)}}" y2="${{sy(hi)}}" stroke="var(--meas)" stroke-width="2"/>`;
  s+=`<line x1="${{sx(lo*10)}}" y1="${{sy(lo)}}" x2="${{sx(hi)}}" y2="${{sy(hi/10)}}" stroke="var(--ref)" stroke-width="1" stroke-dasharray="4 4"/>`;
  s+=`<text x="${{(m.l+W-m.r)/2}}" y="${{H-8}}" text-anchor="middle">best measured runtime of the shape (min of reference implementation, optimized baseline)</text>`;
  s+=`<text transform="translate(14 ${{(m.t+H-m.b)/2}}) rotate(-90)" text-anchor="middle">Solar fused SOL</text>`;
  pts.forEach((p,i)=>{{ const c=COL[p[0]]; const conf=p[5];
    const common=`cx="${{sx(p[3]).toFixed(1)}}" cy="${{sy(p[2]).toFixed(1)}}" data-i="${{i}}"`;
    if(conf==="high") s+=`<circle ${{common}} r="3" fill="${{c}}" fill-opacity="0.75"/>`;
    else if(conf==="medium") s+=`<circle ${{common}} r="3.2" fill="none" stroke="${{c}}" stroke-width="1.6"/>`;
    else s+=`<circle ${{common}} r="3.2" fill="none" stroke="${{c}}" stroke-width="1.4" stroke-dasharray="2 1.5"/>`; }});
  s+=`</svg>`;
  el.innerHTML=s;
  const n=pts.length, above=pts.filter(p=>p[2]>p[3]).length;
  el.insertAdjacentHTML("beforeend", `<div class="dim" style="font-size:12.5px;margin-top:6px">${{n.toLocaleString()}} shapes shown · ${{above}} above the measured runtime</div>`);
  el.querySelectorAll("circle").forEach(cn=>{{
    cn.addEventListener("mousemove",ev=>{{ const p=pts[+cn.dataset.i];
      tip.innerHTML=`<b>${{p[0]}}/${{p[1]}}</b><br>SOL ${{fmt(p[2])}} µs · best measured ${{fmt(p[3])}} µs (${{(p[3]/p[2]).toFixed(1)}}× headroom)<br>leaderboard SOL ${{fmt(p[4])}} µs · confidence ${{p[5]}}`;
      tip.hidden=false; tip.style.left=Math.min(ev.clientX+14, window.innerWidth-340)+"px"; tip.style.top=(ev.clientY+14)+"px"; }});
    cn.addEventListener("mouseleave",()=>{{ tip.hidden=true; }});
  }});
}}
draw(); let rt; window.addEventListener("resize",()=>{{ clearTimeout(rt); rt=setTimeout(draw,150); }});
</script>
"""
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(page)
print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB, {len(points)} points)")
