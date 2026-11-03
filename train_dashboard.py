# SPDX-License-Identifier: MIT
"""Character LM training dashboard (local HTTP, single page, no CDN).

Live view of a training run directory:

  <dir>/status.json    current state (step/epoch/loss/ppl/ETA/speed)
  <dir>/metrics.jsonl  one loss/PPL record per 10 steps
  <dir>/train.log      training log (optional)

Usage:
  python train_dashboard.py --dir out --port 8765 --title "Zimo Character LM"
  # open http://127.0.0.1:8765

Endpoints: /status, /metrics, /log, /meta.  Everything is standard library +
Canvas; no external assets.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#070b12; --panel:rgba(17,24,39,.78); --line:rgba(148,163,184,.14);
    --line2:rgba(148,163,184,.08); --fg:#e8eefc; --dim:#93a4c3;
    --muted:#64748b; --acc:#6ea8ff; --violet:#8b5cf6; --cyan:#22d3ee;
    --green:#34d399; --amber:#fbbf24; --rose:#fb7185;
    --radius:18px; --shadow:0 24px 60px -28px rgba(0,0,0,.85);
    --mono:"Cascadia Code",ui-monospace,Consolas,"Courier New",monospace;
  }
  *{box-sizing:border-box}
  html,body{min-height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.55 "Segoe UI",system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
       -webkit-font-smoothing:antialiased;}
  body::before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
    background:
      radial-gradient(900px 620px at 10% -12%, rgba(110,168,255,.18), transparent 62%),
      radial-gradient(760px 520px at 92% -4%, rgba(139,92,246,.14), transparent 58%),
      radial-gradient(760px 520px at 52% 112%, rgba(34,211,238,.09), transparent 62%),
      linear-gradient(180deg,#080d16 0%,#070b12 45%,#060910 100%);}
  .app{max-width:1460px;margin:0 auto;padding:0 24px 42px}
  .topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;
    justify-content:space-between;gap:18px;flex-wrap:wrap;
    margin:0 -24px;padding:16px 24px;background:rgba(7,11,18,.76);
    backdrop-filter:blur(18px);border-bottom:1px solid var(--line2);}
  .brand{display:flex;align-items:center;gap:14px;min-width:0}
  .logo{width:44px;height:44px;border-radius:14px;display:grid;place-items:center;
    background:linear-gradient(135deg,#6ea8ff 0%,#8b5cf6 58%,#22d3ee 100%);
    color:#06101f;font-size:21px;font-weight:800;letter-spacing:-.04em;
    box-shadow:0 10px 28px -10px rgba(110,168,255,.8);}
  .brand h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.01em}
  .brand p{margin:2px 0 0;font-size:12px;color:var(--dim)}
  .topmeta{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap}
  .pill{display:inline-flex;align-items:center;gap:8px;padding:7px 12px;
    border:1px solid var(--line);border-radius:999px;background:rgba(17,24,39,.58);
    font-size:12px;color:var(--dim);white-space:nowrap;}
  .pill b{color:var(--fg);font-weight:600}
  .pill .dot{width:7px;height:7px;border-radius:50%;background:var(--muted);flex:none}
  .pill.running .dot{background:var(--green);animation:pulse 1.8s infinite}
  .pill.done .dot{background:var(--acc)}
  .pill.waiting .dot{background:var(--amber)}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(52,211,153,.55)}
    70%{box-shadow:0 0 0 8px rgba(52,211,153,0)}
    100%{box-shadow:0 0 0 0 rgba(52,211,153,0)}}
  .muted{color:var(--muted);font-size:12px}
  main{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;padding-top:22px}
  .card{position:relative;overflow:hidden;
    background:linear-gradient(180deg,rgba(19,27,45,.9),rgba(12,18,32,.84));
    border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow);}
  main > .card{grid-column:span 12}
  .card::after{content:"";position:absolute;inset:0;border-radius:inherit;
    pointer-events:none;background:linear-gradient(120deg,rgba(255,255,255,.045),transparent 32%);}
  .hero{display:grid;grid-template-columns:160px minmax(0,1fr) 250px;
    gap:26px;align-items:center;padding:24px 26px;}
  .ring-wrap{position:relative;width:140px;height:140px;display:grid;place-items:center}
  .ring-wrap svg{width:140px;height:140px;transform:rotate(-90deg)}
  .ring-bg{fill:none;stroke:rgba(148,163,184,.12);stroke-width:10}
  .ring-fg{fill:none;stroke:url(#ringGrad);stroke-width:10;stroke-linecap:round;
    transition:stroke-dashoffset .7s cubic-bezier(.22,1,.36,1)}
  .ring-label{position:absolute;text-align:center;pointer-events:none}
  .ring-label b{display:block;font-size:27px;line-height:1.05;letter-spacing:-.03em;
    font-variant-numeric:tabular-nums}
  .ring-label span{display:block;margin-top:2px;font-size:10px;color:var(--muted);
    letter-spacing:.14em;text-transform:uppercase}
  .eyebrow{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
  .big{font-size:20px;font-weight:650;margin:5px 0 2px;letter-spacing:-.01em}
  .step{font:12.5px/1.5 var(--mono);color:var(--dim);font-variant-numeric:tabular-nums}
  .progress{position:relative;height:10px;margin-top:16px;border-radius:999px;
    background:rgba(148,163,184,.11);overflow:hidden}
  .progress i{display:block;height:100%;width:0;border-radius:999px;
    background:linear-gradient(90deg,#6ea8ff 0%,#8b5cf6 52%,#22d3ee 100%);
    box-shadow:0 0 18px rgba(110,168,255,.5);
    transition:width .7s cubic-bezier(.22,1,.36,1)}
  .milestones{display:flex;justify-content:space-between;margin-top:8px;
    font:11px/1 var(--mono);color:var(--muted)}
  .hero-side{display:grid;gap:11px;padding-left:20px;border-left:1px solid var(--line)}
  .kv{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
    font-size:12px;color:var(--dim)}
  .kv b{color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
  .stats{grid-column:span 12;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
    gap:14px}
  .stat{padding:16px 18px;min-height:98px;display:flex;flex-direction:column;
    justify-content:space-between;gap:8px}
  .stat .label{font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
  .stat .value{font-size:26px;line-height:1.1;font-weight:700;letter-spacing:-.03em;
    font-variant-numeric:tabular-nums}
  .stat .sub{display:flex;align-items:center;justify-content:space-between;gap:8px;
    font-size:11.5px;color:var(--dim)}
  .delta{font-variant-numeric:tabular-nums;font-weight:600}
  .delta.good{color:var(--green)} .delta.bad{color:var(--rose)}
  .charts{grid-column:span 12;display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .chart-card{padding:18px 18px 8px}
  .chart-head{display:flex;align-items:baseline;justify-content:space-between;
    gap:12px;margin-bottom:4px}
  .chart-head h2{font-size:14px;margin:0;font-weight:600}
  .hint{font-size:11px;color:var(--muted)}
  canvas{display:block;width:100%;height:250px}
  .chart-card.wide{grid-column:1 / -1}
  .chart-card.wide canvas{height:190px}
  .bottom{grid-column:span 12;display:grid;
    grid-template-columns:minmax(0,.92fr) minmax(0,1.45fr);gap:16px}
  .info{padding:18px 20px}
  .info h2,.logcard h2{font-size:14px;margin:0 0 12px;font-weight:600}
  .info dl{display:grid;grid-template-columns:1fr 1fr;gap:10px 20px;margin:0}
  .info dt{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
  .info dd{margin:2px 0 0;font-size:13px;font-variant-numeric:tabular-nums;
    overflow-wrap:anywhere}
  .logcard{padding:18px 20px;display:flex;flex-direction:column;min-height:320px}
  .log-head{display:flex;align-items:center;justify-content:space-between;
    gap:10px;margin-bottom:10px}
  .log-head label{display:flex;align-items:center;gap:6px;font-size:12px;
    color:var(--dim);cursor:pointer;user-select:none}
  .log-head input{accent-color:var(--acc)}
  pre#log{margin:0;flex:1;max-height:310px;overflow:auto;padding:12px 14px;
    background:rgba(5,9,16,.48);border:1px solid var(--line2);border-radius:12px;
    font:12px/1.55 var(--mono);color:#c7d2e6;white-space:pre-wrap;
    word-break:break-word}
  .logline{display:block}
  .logline.val{color:var(--green)}
  .logline.epoch{color:var(--acc)}
  .logline.done{color:var(--cyan)}
  .logline.warn{color:var(--amber)}
  footer{grid-column:span 12;display:flex;align-items:center;justify-content:space-between;
    gap:12px;flex-wrap:wrap;padding:6px 2px 8px;color:var(--muted);font-size:12px}
  #tooltip{position:fixed;z-index:60;pointer-events:none;padding:8px 10px;
    background:rgba(8,13,24,.96);border:1px solid rgba(148,163,184,.26);
    border-radius:10px;box-shadow:0 14px 36px rgba(0,0,0,.55);
    font:12px/1.55 var(--mono);color:#dbe6fb;opacity:0;transform:translateY(4px);
    transition:opacity .12s,transform .12s;max-width:260px}
  #tooltip.show{opacity:1;transform:translateY(0)}
  #tooltip b{color:#fff;font-weight:600}
  @media (max-width:1120px){
    .hero{grid-template-columns:140px minmax(0,1fr)}
    .hero-side{grid-column:span 2;grid-template-columns:repeat(4,minmax(0,1fr));
      padding-left:0;border-left:0;border-top:1px solid var(--line);padding-top:14px}
    .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
    .charts,.bottom{grid-template-columns:1fr}
  }
  @media (max-width:680px){
    .app{padding:0 14px 30px}
    .topbar{margin:0 -14px;padding:14px;align-items:flex-start;flex-direction:column}
    .hero{grid-template-columns:1fr;padding:20px}
    .ring-wrap{margin:0 auto}
    .stats{grid-template-columns:1fr}
    .info dl{grid-template-columns:1fr}
  }
  @media (prefers-reduced-motion:reduce){
    *{animation:none!important;transition:none!important}
  }
</style></head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <div class="logo">G2</div>
      <div>
        <h1>__TITLE__</h1>
        <p>__SUBTITLE__</p>
      </div>
    </div>
    <div class="topmeta">
      <span class="pill waiting" id="statePill"><span class="dot"></span><span id="stateText">连接中…</span></span>
      <span class="pill"><b>架构</b> <span id="arch">—</span></span>
      <span class="pill"><b>参数</b> <span id="params">—</span></span>
      <span class="pill"><b>设备</b> <span id="device">—</span></span>
      <span class="muted" id="updated">—</span>
    </div>
  </header>

  <main>
    <section class="card hero">
      <div class="ring-wrap">
        <svg viewBox="0 0 140 140" aria-hidden="true">
          <defs>
            <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#6ea8ff"/>
              <stop offset=".55" stop-color="#8b5cf6"/>
              <stop offset="1" stop-color="#22d3ee"/>
            </linearGradient>
          </defs>
          <circle class="ring-bg" cx="70" cy="70" r="56"/>
          <circle class="ring-fg" id="ringFg" cx="70" cy="70" r="56"
                  stroke-dasharray="351.858" stroke-dashoffset="351.858"/>
        </svg>
        <div class="ring-label"><b id="ringPct">0%</b><span>完成</span></div>
      </div>
      <div class="hero-main">
        <div class="eyebrow">Training progress</div>
        <div class="big" id="heroEpoch">第 1 / 2 个 epoch</div>
        <div class="step" id="heroStep">Step — / —</div>
        <div class="progress"><i id="bar"></i></div>
        <div class="milestones"><span>E1</span><span id="msMid">E2</span><span>完成</span></div>
      </div>
      <div class="hero-side">
        <div class="kv"><span>已运行</span><b id="elapsed">—</b></div>
        <div class="kv"><span>剩余</span><b id="eta">—</b></div>
        <div class="kv"><span>预计完成</span><b id="finish">—</b></div>
        <div class="kv"><span>速度</span><b id="heroSpeed">—</b></div>
      </div>
    </section>

    <section class="stats">
      <div class="card stat">
        <div class="label">Loss</div>
        <div class="value" id="loss">—</div>
        <div class="sub"><span id="lossSub">—</span><span class="delta" id="lossDelta">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">Perplexity</div>
        <div class="value" id="ppl">—</div>
        <div class="sub"><span id="pplSub">—</span><span class="delta" id="pplDelta">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">吞吐</div>
        <div class="value" id="speedCard">—</div>
        <div class="sub"><span id="speedSub">—</span><span id="speedPct">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">已处理 token</div>
        <div class="value" id="processed">—</div>
        <div class="sub"><span id="processedSub">token-passes</span><span id="processedPct">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">最佳验证 PPL</div>
        <div class="value" id="bestVal">—</div>
        <div class="sub"><span>val split</span><span id="valCount">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">Epoch</div>
        <div class="value" id="epochCard">—</div>
        <div class="sub"><span id="epochSub">—</span><span id="epochPct">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">参数量</div>
        <div class="value" id="paramsCard">—</div>
        <div class="sub"><span>parameters</span><span id="paramsArch">—</span></div>
      </div>
      <div class="card stat">
        <div class="label">字符表</div>
        <div class="value" id="vocabCard">—</div>
        <div class="sub"><span>vocab</span><span id="vocabRate">—</span></div>
      </div>
    </section>

    <section class="charts">
      <div class="card chart-card">
        <div class="chart-head"><h2>Loss</h2><span class="hint">原始点 + 平滑曲线 · 悬停查看</span></div>
        <canvas id="lossChart"></canvas>
      </div>
      <div class="card chart-card">
        <div class="chart-head"><h2>Perplexity</h2><span class="hint">对数轴 · 空心点为验证集</span></div>
        <canvas id="pplChart"></canvas>
      </div>
    </section>

    <section class="charts">
      <div class="card chart-card wide">
        <div class="chart-head"><h2>吞吐</h2><span class="hint" id="speedHint">由 metrics 的 step / t 推导</span></div>
        <canvas id="speedChart"></canvas>
      </div>
    </section>

    <section class="bottom">
      <div class="card info">
        <h2>模型与数据</h2>
        <dl>
          <div><dt>架构</dt><dd id="infoArch">—</dd></div>
          <div><dt>参数量</dt><dd id="infoParams">—</dd></div>
          <div><dt>字符表</dt><dd id="infoVocab">—</dd></div>
          <div><dt>训练 token</dt><dd id="infoTokens">—</dd></div>
          <div><dt>每 epoch 步数</dt><dd id="infoSteps">—</dd></div>
          <div><dt>总步数</dt><dd id="infoTotal">—</dd></div>
          <div><dt>设备</dt><dd id="infoDevice">—</dd></div>
          <div><dt>语料</dt><dd id="infoCorpus">—</dd></div>
          <div><dt>数据目录</dt><dd id="infoDir">—</dd></div>
          <div><dt>状态</dt><dd id="infoState">—</dd></div>
        </dl>
      </div>
      <div class="card logcard">
        <div class="log-head">
          <h2>训练日志</h2>
          <label><input type="checkbox" id="follow" checked> 跟随最新</label>
        </div>
        <pre id="log">（等待日志…）</pre>
      </div>
    </section>

    <footer>
      <span>数据目录：<span id="dataDir">—</span></span>
      <span>自动刷新 1s · <span id="clock">—</span></span>
    </footer>
  </main>
</div>
<div id="tooltip"></div>
<script>
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var state = { status: null, metrics: [], meta: null, device: '', hover: { loss: -1, ppl: -1, speed: -1 } };
  var hoverRaf = 0;

  function fmtNum(n, d) {
    if (n === null || n === undefined || !isFinite(n)) return '–';
    d = (d === undefined) ? 2 : d;
    return Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function fmtInt(n) {
    if (n === null || n === undefined || !isFinite(n)) return '–';
    return Math.round(n).toLocaleString('en-US');
  }
  function fmtCompact(n) {
    if (n === null || n === undefined || !isFinite(n)) return '–';
    var a = Math.abs(n);
    if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (a >= 1e3) return (n / 1e3).toFixed(1) + 'k';
    return String(Math.round(n * 10) / 10);
  }
  function fmtTime(s) {
    if (s === null || s === undefined || !isFinite(s)) return '–';
    s = Math.max(0, Math.round(s));
    var h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
    if (h) return h + 'h ' + m + 'm';
    if (m) return m + 'm ' + sec + 's';
    return sec + 's';
  }
  function clock(ts) {
    return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  function setText(id, text) {
    var el = $(id);
    if (el && el.textContent !== text) el.textContent = text;
  }
  function prettyArch(arch) {
    if (!arch) return '—';
    var m = /gpt2\\s+(\\d+)x(\\d+)/i.exec(arch);
    if (m) return 'GPT-2 · ' + m[1] + 'L × ' + m[2] + 'd';
    return arch;
  }
  function deltaClass(value, lowerIsBetter) {
    if (!isFinite(value)) return '';
    if (lowerIsBetter) return value < 0 ? 'good' : (value > 0 ? 'bad' : '');
    return value > 0 ? 'good' : (value < 0 ? 'bad' : '');
  }

  function derived(s, metrics) {
    var total = s.total_steps || 0;
    var step = s.step || 0;
    var spe = s.steps_per_epoch || 0;
    var epochs = spe ? Math.max(1, Math.round(total / spe)) : (s.epochs || 0);
    var curEpoch = spe ? Math.min(epochs || 1, Math.floor(step / spe) + 1) : (s.epoch || 0) + 1;
    var perStep = spe ? (s.tokens || 0) / spe : 0;
    var processed = step * perStep;
    var totalPasses = (s.tokens || 0) * epochs;
    var pct = total ? Math.min(100, 100 * step / total) : 0;
    var vals = metrics.filter(function (p) { return p.val === 1 && p.ppl != null; });
    var bestVal = vals.length ? Math.min.apply(null, vals.map(function (p) { return p.ppl; })) : null;
    var first = null, last = null;
    for (var i = 0; i < metrics.length; i++) {
      if (metrics[i].loss != null) { if (!first) first = metrics[i]; last = metrics[i]; }
    }
    return {
      total: total, step: step, spe: spe, epochs: epochs, curEpoch: curEpoch,
      perStep: perStep, processed: processed, totalPasses: totalPasses, pct: pct,
      bestVal: bestVal, valCount: vals.length, first: first, last: last
    };
  }

  function setState(s) {
    var st = (s && s.state) || 'waiting';
    var pill = $('statePill');
    pill.className = 'pill ' + (st === 'running' ? 'running' : st === 'done' ? 'done' : 'waiting');
    setText('stateText', st === 'running' ? '训练中' : st === 'done' ? '已完成' : st === 'waiting' ? '等待训练' : st);
    setText('infoState', st === 'running' ? '训练中' : st === 'done' ? '已完成' : st);
  }

  function render(s, metrics) {
    var d = derived(s, metrics || []);
    var total = d.total, step = d.step, pct = d.pct;
    var running = s.state === 'running';

    var device = s.device || state.device || '—';
    setState(s);
    setText('arch', prettyArch(s.arch));
    setText('params', s.params ? fmtCompact(s.params) : '—');
    setText('device', device);
    setText('updated', '更新于 ' + clock(Date.now()));

    setText('ringPct', pct.toFixed(pct >= 100 ? 0 : 1) + '%');
    var C = 351.858;
    var ring = $('ringFg');
    if (ring) ring.style.strokeDashoffset = String(C * (1 - Math.min(1, pct / 100)));
    var bar = $('bar');
    if (bar) bar.style.width = Math.min(100, pct).toFixed(2) + '%';

    setText('heroEpoch', '第 ' + d.curEpoch + ' / ' + (d.epochs || '?') + ' 个 epoch');
    setText('heroStep', 'Step ' + fmtInt(step) + ' / ' + (total ? fmtInt(total) : '—'));
    setText('msMid', d.epochs > 1
      ? 'E' + Math.min(d.epochs, Math.floor(d.epochs / 2) + 1) : '完成');
    setText('elapsed', fmtTime(s.elapsed_s));
    setText('eta', s.state === 'done' ? '已完成' : fmtTime(s.eta_s));
    setText('finish', s.state === 'done' ? '已完成' : (s.eta_s ? clock(Date.now() + s.eta_s * 1000) : '—'));
    setText('heroSpeed', s.tokens_per_s ? fmtCompact(s.tokens_per_s) + ' tok/s' : '—');

    var loss = (s.last_loss != null) ? s.last_loss : (d.last && d.last.loss);
    var ppl = (s.last_ppl != null) ? s.last_ppl : (d.last && d.last.ppl);
    setText('loss', fmtNum(loss, 4));
    setText('ppl', fmtNum(ppl, 1));
    setText('lossSub', 'step ' + fmtInt(step));
    setText('pplSub', d.first ? '起始 ' + fmtNum(d.first.ppl, 1) : '—');
    var lossDelta = (d.first && d.first.loss) ? 100 * (loss - d.first.loss) / d.first.loss : NaN;
    var pplDelta = (d.first && d.first.ppl) ? 100 * (ppl - d.first.ppl) / d.first.ppl : NaN;
    var ld = $('lossDelta'), pd = $('pplDelta');
    if (ld) {
      ld.textContent = isFinite(lossDelta) ? (lossDelta > 0 ? '+' : '') + lossDelta.toFixed(1) + '%' : '—';
      ld.className = 'delta ' + deltaClass(lossDelta, true);
    }
    if (pd) {
      pd.textContent = isFinite(pplDelta) ? (pplDelta > 0 ? '+' : '') + pplDelta.toFixed(1) + '%' : '—';
      pd.className = 'delta ' + deltaClass(pplDelta, true);
    }

    setText('speedCard', s.tokens_per_s ? fmtCompact(s.tokens_per_s) : '—');
    setText('speedSub', d.perStep ? 'tok/s · ≈ ' + fmtCompact(d.perStep) + ' token/step' : '—');
    setText('speedPct', 'epoch ' + d.curEpoch);
    setText('processed', fmtCompact(d.processed) + ' / ' + fmtCompact(d.totalPasses));
    setText('processedSub', 'token-passes');
    setText('processedPct', pct.toFixed(1) + '%');
    setText('bestVal', d.bestVal != null ? fmtNum(d.bestVal, 1) : '—');
    setText('valCount', d.valCount ? d.valCount + ' 次验证' : '尚未验证');
    setText('epochCard', d.curEpoch + ' / ' + (d.epochs || '?'));
    setText('epochSub', 'steps/epoch ' + (d.spe ? fmtInt(d.spe) : '—'));
    setText('epochPct', pct.toFixed(1) + '%');
    setText('paramsCard', s.params ? fmtCompact(s.params) : '—');
    setText('paramsArch', prettyArch(s.arch));
    setText('vocabCard', s.vocab ? fmtInt(s.vocab) : '—');
    setText('vocabRate', '字级 token');

    setText('infoArch', prettyArch(s.arch));
    setText('infoParams', s.params ? fmtInt(s.params) : '—');
    setText('infoVocab', s.vocab ? fmtInt(s.vocab) : '—');
    setText('infoTokens', s.tokens ? fmtInt(s.tokens) : '—');
    setText('infoSteps', d.spe ? fmtInt(d.spe) : '—');
    setText('infoTotal', total ? fmtInt(total) : '—');
    setText('infoDevice', device);
    setText('infoCorpus', s.corpus || '—');
    if (state.meta && state.meta.dir) setText('infoDir', state.meta.dir);
    setText('infoState', s.state === 'running' ? '训练中' : s.state === 'done' ? '已完成' : s.state || '—');

    var modelTitle = prettyArch(s.arch);
    if (!modelTitle || modelTitle === '—') modelTitle = 'GPT-2';
    document.title = '[' + pct.toFixed(0) + '%] ' + modelTitle + ' · loss ' + fmtNum(loss, 2);
    drawAll(metrics || [], s, d);
  }

  function emaValues(values, alpha) {
    var v = null;
    return values.map(function (x) {
      v = (v == null) ? x : alpha * x + (1 - alpha) * v;
      return v;
    });
  }

  function niceStep(span, count) {
    if (!(span > 0)) return 1;
    var raw = span / Math.max(1, count);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
    return step * mag;
  }
  function ticks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return [lo];
    var step = niceStep(hi - lo, count);
    var start = Math.ceil(lo / step) * step;
    var out = [];
    for (var v = start; v <= hi + 1e-12; v += step) out.push(v);
    return out.length ? out : [lo, hi];
  }

  function prep(canvas) {
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(10, rect.width), h = Math.max(10, rect.height);
    var pw = Math.round(w * dpr), ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw; canvas.height = ph;
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function drawEmpty(ctx, w, h, text) {
    ctx.fillStyle = 'rgba(148,163,184,.55)';
    ctx.font = '12px "Segoe UI",system-ui,sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2);
  }

  function drawChart(canvas, opts) {
    if (!canvas) return;
    var p = prep(canvas);
    var ctx = p.ctx, w = p.w, h = p.h;
    var m = { left: 56, right: 18, top: 18, bottom: 28 };
    var pw = w - m.left - m.right, ph = h - m.top - m.bottom;
    if (pw <= 0 || ph <= 0) return;

    var data = (opts.points || []).filter(function (d) {
      return d[opts.yKey] != null && isFinite(d[opts.yKey]);
    });
    if (!data.length) {
      drawEmpty(ctx, w, h, opts.empty || '等待数据');
      canvas._chart = null;
      return;
    }

    var xs = data.map(function (d) { return d[opts.xKey]; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var values = data.map(function (d) { return d[opts.yKey]; });
    var y0 = Math.min.apply(null, values), y1 = Math.max.apply(null, values);
    var log = !!opts.logScale && y0 > 0 && y1 > 0;
    if (log) { y0 = Math.log10(y0); y1 = Math.log10(y1); }
    if (!(y1 > y0)) y1 = y0 + 1;
    var pad = (y1 - y0) * 0.08;
    y0 -= pad; y1 += pad;

    var X = function (v) {
      return m.left + (x1 === x0 ? pw / 2 : pw * (v - x0) / (x1 - x0));
    };
    var Y = function (v) {
      var t = log ? Math.log10(Math.max(v, 1e-12)) : v;
      return m.top + ph - (y1 === y0 ? ph / 2 : ph * (t - y0) / (y1 - y0));
    };

    ctx.lineWidth = 1;
    ctx.font = '10.5px "Cascadia Code",ui-monospace,monospace';
    ctx.fillStyle = '#6b7a99';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var yTicks = ticks(y0, y1, 4);
    yTicks.forEach(function (v) {
      var raw = log ? Math.pow(10, v) : v;
      var py = Y(raw);
      ctx.strokeStyle = 'rgba(148,163,184,.10)';
      ctx.beginPath(); ctx.moveTo(m.left, py); ctx.lineTo(w - m.right, py); ctx.stroke();
      ctx.fillText(opts.formatY ? opts.formatY(raw) : fmtNum(raw, 1), m.left - 10, py);
    });

    var spe = opts.epochSteps || 0;
    if (spe > 0 && x1 > x0) {
      ctx.save();
      ctx.setLineDash([4, 5]);
      ctx.strokeStyle = 'rgba(148,163,184,.18)';
      ctx.fillStyle = '#6b7a99';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      var epochCount = Math.ceil(x1 / spe);
      for (var k = 1; k < epochCount; k++) {
        var sx = X(k * spe);
        if (sx > m.left + 2 && sx < w - m.right - 2) {
          ctx.beginPath(); ctx.moveTo(sx, m.top); ctx.lineTo(sx, m.top + ph); ctx.stroke();
          ctx.fillText('E' + (k + 1), sx + 4, m.top + 1);
        }
      }
      ctx.restore();
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#6b7a99';
    var xTicks = x1 === x0 ? [x0] : [x0, x0 + (x1 - x0) / 2, x1];
    xTicks.forEach(function (v) {
      ctx.fillText(fmtCompact(v), X(v), h - m.bottom + 8);
    });

    var smooth = emaValues(values, opts.smoothAlpha || 0.28);
    var baseY = m.top + ph;

    var grad = ctx.createLinearGradient(0, m.top, 0, baseY);
    grad.addColorStop(0, opts.fill || 'rgba(110,168,255,.16)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    smooth.forEach(function (v, i) {
      var px = X(xs[i]), py = Y(v);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.lineTo(X(xs[xs.length - 1]), baseY);
    ctx.lineTo(X(xs[0]), baseY);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    data.forEach(function (d, i) {
      var px = X(xs[i]), py = Y(values[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = opts.rawColor || 'rgba(148,163,184,.22)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.beginPath();
    smooth.forEach(function (v, i) {
      var px = X(xs[i]), py = Y(v);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = opts.color || '#6ea8ff';
    ctx.lineWidth = 2;
    ctx.shadowColor = opts.color || '#6ea8ff';
    ctx.shadowBlur = 9;
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (opts.valKey) {
      data.forEach(function (d, i) {
        if (d[opts.valKey] === 1) {
          var px = X(xs[i]), py = Y(values[i]);
          ctx.beginPath();
          ctx.arc(px, py, 3.2, 0, Math.PI * 2);
          ctx.fillStyle = '#e8eefc';
          ctx.fill();
          ctx.strokeStyle = opts.color || '#6ea8ff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });
    }

    var minV = Math.min.apply(null, values);
    var minI = values.indexOf(minV);
    var minX = X(xs[minI]), minY = Y(minV);
    ctx.beginPath();
    ctx.arc(minX, minY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = opts.color || '#6ea8ff';
    ctx.fill();
    ctx.fillStyle = '#9fb2d6';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.font = '10.5px "Cascadia Code",ui-monospace,monospace';
    var minLabel = 'min ' + (opts.formatY ? opts.formatY(minV) : fmtNum(minV, 1));
    var lx = Math.min(w - m.right - ctx.measureText(minLabel).width - 4, Math.max(m.left + 4, minX + 7));
    ctx.fillText(minLabel, lx, Math.max(m.top + 10, minY - 7));

    var hi = opts.hoverIndex;
    if (hi != null && hi >= 0 && hi < data.length) {
      var hx = X(xs[hi]);
      ctx.save();
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = 'rgba(232,238,252,.35)';
      ctx.beginPath(); ctx.moveTo(hx, m.top); ctx.lineTo(hx, m.top + ph); ctx.stroke();
      ctx.restore();
      var hy = Y(values[hi]);
      ctx.beginPath();
      ctx.arc(hx, hy, 4.2, 0, Math.PI * 2);
      ctx.fillStyle = opts.color || '#6ea8ff';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    canvas._chart = { data: data, xs: xs, opts: opts };
  }

  function deriveSpeed(metrics, s, d) {
    var out = [];
    if (!d || !d.perStep) return out;
    for (var i = 1; i < metrics.length; i++) {
      var a = metrics[i - 1], b = metrics[i];
      var ds = b.step - a.step, dt = b.t - a.t;
      if (ds > 0 && dt > 0) {
        out.push({
          step: b.step, tps: ds * d.perStep / dt,
          loss: b.loss, ppl: b.ppl, t: b.t
        });
      }
    }
    return out;
  }

  function drawAll(metrics, s, d) {
    if (!s) s = state.status || {};
    if (!d) d = derived(s, metrics || []);
    drawChart($('lossChart'), {
      points: metrics || [], xKey: 'step', yKey: 'loss',
      color: '#6ea8ff', fill: 'rgba(110,168,255,.16)',
      epochSteps: d.spe, hoverIndex: state.hover.loss,
      formatY: function (v) { return fmtNum(v, 2); }
    });
    drawChart($('pplChart'), {
      points: metrics || [], xKey: 'step', yKey: 'ppl',
      color: '#34d399', fill: 'rgba(52,211,153,.14)',
      logScale: true, valKey: 'val', epochSteps: d.spe,
      hoverIndex: state.hover.ppl,
      formatY: function (v) { return fmtCompact(v); }
    });
    drawChart($('speedChart'), {
      points: deriveSpeed(metrics || [], s, d), xKey: 'step', yKey: 'tps',
      color: '#22d3ee', fill: 'rgba(34,211,238,.14)',
      epochSteps: d.spe, hoverIndex: state.hover.speed,
      formatY: function (v) { return fmtCompact(v); }
    });
  }

  function showTooltip(evt, p, kind) {
    var el = $('tooltip');
    var lines = [
      '<b>Step ' + fmtInt(p.step) + '</b>',
      'Loss <b>' + fmtNum(p.loss, 4) + '</b>',
      'PPL <b>' + fmtNum(p.ppl, 1) + '</b>',
      'Elapsed <b>' + fmtTime(p.t) + '</b>'
    ];
    if (kind === 'speed') lines.push('吞吐 <b>' + fmtCompact(p.tps) + ' tok/s</b>');
    el.innerHTML = lines.join('<br>');
    el.classList.add('show');
    var x = evt.clientX + 14, y = evt.clientY - 12;
    var rect = el.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - 14;
    if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - 14;
    el.style.left = Math.max(8, x) + 'px';
    el.style.top = Math.max(8, y) + 'px';
  }
  function hideTooltip() {
    var el = $('tooltip');
    if (el) el.classList.remove('show');
  }
  function scheduleDraw() {
    if (hoverRaf) return;
    hoverRaf = requestAnimationFrame(function () {
      hoverRaf = 0;
      drawAll(state.metrics, state.status);
    });
  }
  function attachHover(canvas, kind) {
    if (!canvas) return;
    canvas.addEventListener('mousemove', function (evt) {
      var chart = canvas._chart;
      if (!chart || !chart.xs.length) return;
      var rect = canvas.getBoundingClientRect();
      var x = evt.clientX - rect.left;
      var idx = 0, best = Infinity;
      chart.xs.forEach(function (px, i) {
        var d = Math.abs(px - x);
        if (d < best) { best = d; idx = i; }
      });
      state.hover[kind] = idx;
      scheduleDraw();
      var p = chart.data[idx];
      if (p) showTooltip(evt, p, kind);
    });
    canvas.addEventListener('mouseleave', function () {
      state.hover[kind] = -1;
      hideTooltip();
      scheduleDraw();
    });
  }

  function renderLog(text) {
    var pre = $('log');
    if (!pre) return;
    if (pre._raw === text) return;
    pre._raw = text;
    var frag = document.createDocumentFragment();
    text.split('\\n').forEach(function (line) {
      var span = document.createElement('span');
      span.className = 'logline';
      if (/val_|val=|val_loss/.test(line)) span.classList.add('val');
      else if (/epoch=/.test(line)) span.classList.add('epoch');
      else if (/done/.test(line)) span.classList.add('done');
      else if (/WARNING|ERROR|Traceback/.test(line)) span.classList.add('warn');
      span.textContent = line + '\\n';
      frag.appendChild(span);
    });
    pre.replaceChildren(frag);
    if ($('follow').checked) pre.scrollTop = pre.scrollHeight;
  }

  function getJSON(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.json();
    });
  }
  function getText(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.text();
    });
  }

  function tick() {
    Promise.all([getJSON('/status'), getJSON('/metrics'), getText('/log')])
      .then(function (res) {
        var s = res[0], metrics = res[1], log = res[2];
        state.status = s;
        state.metrics = metrics || [];
        var dm = /device=([A-Za-z0-9_]+)/.exec(log || '');
        if (dm) state.device = dm[1].toUpperCase();
        render(s, state.metrics);
        renderLog(log || '');
      })
      .catch(function () {
        if (!state.status) setState({ state: 'waiting' });
        setText('stateText', state.status ? '连接中断，重试中…' : '等待训练进程…');
      });
    setText('clock', '本地时间 ' + clock(Date.now()));
  }

  function init() {
    attachHover($('lossChart'), 'loss');
    attachHover($('pplChart'), 'ppl');
    attachHover($('speedChart'), 'speed');
    var follow = $('follow');
    if (follow) follow.addEventListener('change', function () {
      if (follow.checked) {
        var pre = $('log');
        pre.scrollTop = pre.scrollHeight;
      }
    });
    var resizeTimer = 0;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { drawAll(state.metrics, state.status); }, 120);
    });
    getJSON('/meta').then(function (m) {
      state.meta = m;
      setText('dataDir', m.dir || '—');
      setText('infoDir', m.dir || '—');
    }).catch(function () {});
    tick();
    setInterval(tick, 1000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    root = "."
    title = "Zimo Character LM"
    subtitle = "character-level language model · training dashboard"

    def log_message(self, *args):  # 静音访问日志
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            page = (HTML.replace("__TITLE__", html.escape(self.title))
                        .replace("__SUBTITLE__", html.escape(self.subtitle)))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/meta":
            payload = json.dumps({"dir": os.path.abspath(self.root)})
            self._send(200, payload.encode("utf-8"), "application/json")
            return
        if self.path == "/status":
            path = os.path.join(self.root, "status.json")
            if not os.path.exists(path):
                self._send(200, b'{"state":"waiting"}', "application/json")
                return
            with io.open(path, encoding="utf-8") as f:
                self._send(200, f.read().encode("utf-8"), "application/json")
            return
        if self.path == "/metrics":
            path = os.path.join(self.root, "metrics.jsonl")
            points = []
            if os.path.exists(path):
                with io.open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            points.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            self._send(200, json.dumps(points[-800:]).encode("utf-8"),
                       "application/json")
            return
        if self.path == "/log":
            path = os.path.join(self.root, "train.log")
            text = ""
            if os.path.exists(path):
                with io.open(path, encoding="utf-8", errors="ignore") as f:
                    text = "".join(f.readlines()[-80:])
            self._send(200, text.encode("utf-8"), "text/plain; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--title", default="Zimo Character LM")
    ap.add_argument("--subtitle",
                    default="character-level language model · training dashboard")
    args = ap.parse_args()
    Handler.root = args.dir
    Handler.title = args.title
    Handler.subtitle = args.subtitle
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}  数据目录={args.dir}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
