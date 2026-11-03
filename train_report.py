#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a self-contained static HTML training report (no server, no CDN).

Reads a training run directory:

  <dir>/status.json    final/live state (optional)
  <dir>/metrics.jsonl  loss/PPL records
  <dir>/train.log      log tail (optional)
  <dir>/vocab.txt      vocabulary (optional; used for a token count)

Writes one HTML file with the data embedded.  Open it directly in a browser or
attach it to a release / experiment record.  This is the "archive" counterpart
to train_dashboard.py, which is for live monitoring.

Usage:
  python train_report.py --dir out --output report.html
  python train_report.py --dir out --title "GPT-2 6Lx192d" --output report.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import io
import json
import os
import sys


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{--bg:#070b12;--panel:#111827;--line:rgba(148,163,184,.16);
        --fg:#e8eefc;--dim:#93a4c3;--muted:#64748b;--acc:#6ea8ff;
        --green:#34d399;--cyan:#22d3ee;--violet:#8b5cf6;
        --mono:"Cascadia Code",ui-monospace,Consolas,monospace}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(900px 600px at 10% -10%,rgba(110,168,255,.14),transparent 60%),
       radial-gradient(800px 500px at 95% 0%,rgba(139,92,246,.10),transparent 55%),var(--bg);
       color:var(--fg);font:14px/1.55 "Segoe UI",system-ui,-apple-system,"PingFang SC",sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:34px 24px 50px}
  header{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;
         padding-bottom:18px;border-bottom:1px solid var(--line)}
  h1{font-size:24px;margin:0;letter-spacing:-.03em}
  .sub{color:var(--dim);font-size:13px;margin-top:4px}
  .stamp{color:var(--muted);font-size:12px;text-align:right}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}
  .card{background:linear-gradient(180deg,rgba(19,27,45,.9),rgba(12,18,32,.86));
        border:1px solid var(--line);border-radius:16px;padding:15px 17px}
  .k{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
  .v{font-size:24px;font-weight:700;letter-spacing:-.03em;margin-top:4px;
     font-variant-numeric:tabular-nums}
  .s{font-size:11.5px;color:var(--dim);margin-top:3px}
  .charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px}
  .chart{padding:16px 16px 8px}
  .chart.wide{grid-column:1 / -1}
  h2{font-size:13px;margin:0 0 4px;font-weight:600}
  canvas{display:block;width:100%;height:230px}
  .chart.wide canvas{height:180px}
  .bottom{display:grid;grid-template-columns:.9fr 1.4fr;gap:14px;margin-top:16px}
  dl{display:grid;grid-template-columns:1fr 1fr;gap:9px 18px;margin:0}
  dt{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
  dd{margin:1px 0 0;font-size:13px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
  pre{margin:0;max-height:260px;overflow:auto;padding:11px 13px;border-radius:12px;
      background:rgba(5,9,16,.5);border:1px solid rgba(148,163,184,.1);
      font:11.5px/1.55 var(--mono);color:#c7d2e6;white-space:pre-wrap}
  footer{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
         margin-top:18px;color:var(--muted);font-size:11.5px}
  @media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}
    .charts,.bottom{grid-template-columns:1fr}.chart.wide{grid-column:auto}}
  @media(max-width:560px){.cards{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
<header>
  <div><h1>__TITLE__</h1><div class="sub">__SUBTITLE__</div></div>
  <div class="stamp">generated __GENERATED__<br>run dir: __DIR__</div>
</header>

<section class="cards">
  <div class="card"><div class="k">Loss</div><div class="v" id="loss">–</div><div class="s" id="lossSub">–</div></div>
  <div class="card"><div class="k">Perplexity</div><div class="v" id="ppl">–</div><div class="s" id="pplSub">–</div></div>
  <div class="card"><div class="k">Best val PPL</div><div class="v" id="bestVal">–</div><div class="s" id="valSub">–</div></div>
  <div class="card"><div class="k">Progress</div><div class="v" id="progress">–</div><div class="s" id="progressSub">–</div></div>
  <div class="card"><div class="k">Throughput</div><div class="v" id="speed">–</div><div class="s" id="speedSub">–</div></div>
  <div class="card"><div class="k">Processed tokens</div><div class="v" id="processed">–</div><div class="s">token-passes</div></div>
  <div class="card"><div class="k">Parameters</div><div class="v" id="params">–</div><div class="s" id="arch">–</div></div>
  <div class="card"><div class="k">Vocabulary</div><div class="v" id="vocab">–</div><div class="s">tokens</div></div>
</section>

<section class="charts">
  <div class="card chart"><h2>Loss</h2><canvas id="lossChart"></canvas></div>
  <div class="card chart"><h2>Perplexity (log scale)</h2><canvas id="pplChart"></canvas></div>
</section>
<section class="charts">
  <div class="card chart wide"><h2>Throughput (derived from step / elapsed)</h2><canvas id="speedChart"></canvas></div>
</section>

<section class="bottom">
  <div class="card" style="padding:16px 18px"><h2>Run details</h2><dl>
    <div><dt>State</dt><dd id="state">–</dd></div>
    <div><dt>Step</dt><dd id="step">–</dd></div>
    <div><dt>Epoch</dt><dd id="epoch">–</dd></div>
    <div><dt>Elapsed</dt><dd id="elapsed">–</dd></div>
    <div><dt>Data dir</dt><dd id="dir">–</dd></div>
    <div><dt>Log lines</dt><dd id="logLines">–</dd></div>
  </dl></div>
  <div class="card" style="padding:16px 18px"><h2>Log tail</h2><pre id="log">–</pre></div>
</section>

<footer><span>Generated by train_report.py · self-contained, no external assets</span><span>__DIR__</span></footer>
</div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (n, d=2) => (n===null||n===undefined||!isFinite(n)) ? '–'
  : Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const fmtInt = n => (n===null||n===undefined||!isFinite(n)) ? '–' : Math.round(n).toLocaleString('en-US');
const fmtCompact = n => {
  if (n===null||n===undefined||!isFinite(n)) return '–';
  const a=Math.abs(n);
  if (a>=1e9) return (n/1e9).toFixed(2)+'B';
  if (a>=1e6) return (n/1e6).toFixed(2)+'M';
  if (a>=1e3) return (n/1e3).toFixed(1)+'k';
  return String(Math.round(n*10)/10);
};
const fmtTime = s => {
  if (s===null||s===undefined||!isFinite(s)) return '–';
  s=Math.max(0,Math.round(s));
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;
  return h ? `${h}h ${m}m` : (m ? `${m}m ${x}s` : `${x}s`);
};

function draw(canvas, points, key, color, logScale) {
  const dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect();
  const w=Math.max(10,rect.width), h=Math.max(10,rect.height);
  canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  const data=(points||[]).filter(p=>p[key]!=null && isFinite(p[key]));
  if (!data.length) {
    ctx.fillStyle='rgba(148,163,184,.55)'; ctx.font='12px system-ui';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText('no data',w/2,h/2); return;
  }
  const m={l:56,r:16,t:16,b:26}, pw=w-m.l-m.r, ph=h-m.t-m.b;
  const xs=data.map(p=>p.step), ys=data.map(p=>p[key]);
  const x0=Math.min(...xs), x1=Math.max(...xs);
  let y0=Math.min(...ys), y1=Math.max(...ys);
  const log=!!logScale && y0>0;
  if (log) { y0=Math.log10(y0); y1=Math.log10(y1); }
  if (!(y1>y0)) y1=y0+1;
  const pad=(y1-y0)*.08; y0-=pad; y1+=pad;
  const X=v=>m.l+(x1===x0?pw/2:pw*(v-x0)/(x1-x0));
  const Y=v=>m.t+ph-(y1===y0?ph/2:ph*((log?Math.log10(Math.max(v,1e-12)):v)-y0)/(y1-y0));
  ctx.font='10.5px "Cascadia Code",ui-monospace,monospace';
  ctx.textAlign='right'; ctx.textBaseline='middle'; ctx.fillStyle='#6b7a99';
  for (let i=0;i<=4;i++) {
    const raw=log?Math.pow(10,y0+(y1-y0)*i/4):y0+(y1-y0)*i/4;
    const py=Y(raw);
    ctx.strokeStyle='rgba(148,163,184,.10)';
    ctx.beginPath(); ctx.moveTo(m.l,py); ctx.lineTo(w-m.r,py); ctx.stroke();
    ctx.fillText(fmtCompact(raw),m.l-9,py);
  }
  ctx.textAlign='center'; ctx.textBaseline='top';
  [x0,(x0+x1)/2,x1].forEach(v=>ctx.fillText(fmtCompact(v),X(v),h-m.b+7));
  const grad=ctx.createLinearGradient(0,m.t,0,m.t+ph);
  grad.addColorStop(0,color+'33'); grad.addColorStop(1,color+'00');
  const smooth=[]; let v=null;
  ys.forEach(x=>{v=v==null?x:.28*x+.72*v;smooth.push(v);});
  ctx.beginPath();
  smooth.forEach((y,i)=>{const px=X(xs[i]),py=Y(y);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
  ctx.lineTo(X(x1),m.t+ph); ctx.lineTo(X(x0),m.t+ph); ctx.closePath();
  ctx.fillStyle=grad; ctx.fill();
  ctx.beginPath();
  ys.forEach((y,i)=>{const px=X(xs[i]),py=Y(y);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
  ctx.strokeStyle=color+'44'; ctx.lineWidth=1; ctx.stroke();
  ctx.beginPath();
  smooth.forEach((y,i)=>{const px=X(xs[i]),py=Y(y);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke();
  const minV=Math.min(...ys), minI=ys.indexOf(minV);
  ctx.beginPath(); ctx.arc(X(xs[minI]),Y(minV),3,0,Math.PI*2);
  ctx.fillStyle=color; ctx.fill();
  ctx.fillStyle='#9fb2d6'; ctx.textAlign='left'; ctx.textBaseline='bottom';
  ctx.fillText('min '+fmtCompact(minV),Math.min(w-m.r-40,X(xs[minI])+6),Math.max(m.t+10,Y(minV)-6));
}

(function render() {
  const s=DATA.status||{}, metrics=DATA.metrics||[];
  const last=[...metrics].reverse().find(p=>p.loss!=null)||{};
  const first=metrics.find(p=>p.loss!=null)||{};
  const vals=metrics.filter(p=>p.val===1 && p.ppl!=null);
  const bestVal=vals.length?Math.min(...vals.map(p=>p.ppl)):null;
  const spe=s.steps_per_epoch||0;
  const total=s.total_steps||0;
  const step=s.step||last.step||0;
  const epochs=spe?Math.max(1,Math.round(total/spe)):(s.epochs||0);
  const perStep=spe?(s.tokens||0)/spe:0;
  const processed=step*perStep;
  const pct=total?Math.min(100,100*step/total):0;
  const loss=s.last_loss!=null?s.last_loss:last.loss;
  const ppl=s.last_ppl!=null?s.last_ppl:last.ppl;
  const arch=s.arch||'–';

  $('loss').textContent=fmt(loss,4);
  $('lossSub').textContent=first.loss!=null?('start '+fmt(first.loss,2)):'–';
  $('ppl').textContent=fmt(ppl,1);
  $('pplSub').textContent=first.ppl!=null?('start '+fmt(first.ppl,1)):'–';
  $('bestVal').textContent=bestVal!=null?fmt(bestVal,1):'–';
  $('valSub').textContent=vals.length?vals.length+' validation point(s)':'no validation yet';
  $('progress').textContent=pct.toFixed(1)+'%';
  $('progressSub').textContent=fmtInt(step)+' / '+(total?fmtInt(total):'–');
  $('speed').textContent=s.tokens_per_s?fmtCompact(s.tokens_per_s)+' tok/s':'–';
  $('speedSub').textContent=perStep?('≈ '+fmtCompact(perStep)+' token/step'):'–';
  $('processed').textContent=fmtCompact(processed)+' / '+fmtCompact((s.tokens||0)*epochs);
  $('params').textContent=s.params?fmtCompact(s.params):'–';
  $('arch').textContent=arch;
  $('vocab').textContent=s.vocab?fmtInt(s.vocab):(DATA.vocab_lines?fmtInt(DATA.vocab_lines):'–');
  $('state').textContent=s.state||'unknown';
  $('step').textContent=fmtInt(step);
  $('epoch').textContent=spe?('epoch '+Math.min(epochs,Math.floor(step/spe)+1)+' / '+epochs):'–';
  $('elapsed').textContent=fmtTime(s.elapsed_s);
  $('dir').textContent=DATA.dir||'–';
  $('logLines').textContent=fmtInt(DATA.log_lines||0);
  $('log').textContent=DATA.log||'–';

  draw($('lossChart'),metrics,'loss','#6ea8ff',false);
  draw($('pplChart'),metrics,'ppl','#34d399',true);
  const speed=[];
  for (let i=1;i<metrics.length;i++) {
    const a=metrics[i-1],b=metrics[i],ds=b.step-a.step,dt=b.t-a.t;
    if (ds>0&&dt>0&&perStep>0) speed.push({step:b.step,tps:ds*perStep/dt});
  }
  draw($('speedChart'),speed,'tps','#22d3ee',false);
})();
</script>
</body></html>
"""


def load_json(path: str):
    if not os.path.exists(path):
        return {}
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_metrics(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_log(path: str, max_lines: int) -> tuple[str, int]:
    if not os.path.exists(path):
        return "", 0
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    tail = "".join(lines[-max_lines:])
    return tail, len(lines)


def build_report(run_dir: str, output: str, title: str, subtitle: str,
                 max_log_lines: int = 80) -> str:
    run_dir = os.path.abspath(run_dir)
    status = load_json(os.path.join(run_dir, "status.json"))
    metrics = load_metrics(os.path.join(run_dir, "metrics.jsonl"))
    log, log_lines = load_log(os.path.join(run_dir, "train.log"), max_log_lines)
    vocab_path = os.path.join(run_dir, "vocab.txt")
    vocab_lines = 0
    if os.path.exists(vocab_path):
        with io.open(vocab_path, encoding="utf-8", errors="replace") as fh:
            vocab_lines = sum(1 for line in fh if line.strip())

    payload = {
        "status": status,
        "metrics": metrics[-1200:],
        "log": log,
        "log_lines": log_lines,
        "vocab_lines": vocab_lines,
        "dir": run_dir,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = (TEMPLATE.replace("__TITLE__", html.escape(title))
                    .replace("__SUBTITLE__", html.escape(subtitle))
                    .replace("__GENERATED__", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
                    .replace("__DIR__", html.escape(run_dir))
                    .replace("__DATA__", data_json))
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with io.open(output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out", help="training run directory")
    parser.add_argument("--output", default="report.html", help="output HTML")
    parser.add_argument("--title", default="Zimo Training Report")
    parser.add_argument("--subtitle",
                        default="char-level GPT-2 style language model")
    parser.add_argument("--max-log-lines", type=int, default=80)
    args = parser.parse_args()
    if not os.path.isdir(args.dir):
        print(f"run directory not found: {args.dir}", file=sys.stderr)
        return 2
    path = build_report(args.dir, args.output, args.title, args.subtitle,
                        args.max_log_lines)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
