'use strict';

const $ = (s) => document.querySelector(s);
const state = { result: null, sim: 0 };
const COLORS = { hf: '#e0533d', pred: '#17a06b', lf: '#93a1b1' };

const API = {
  files: () => fetch('/api/files').then((r) => r.json()),
  results: (f) => fetch('/api/results?file=' + encodeURIComponent(f)).then((r) => r.json()),
  upload: (payload) =>
    fetch('/api/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then((r) => r.json()),
};

function fmt(v, d = 3) {
  return v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d);
}
function toast(msg, ok = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = ok ? 'show good' : 'show';
  setTimeout(() => (t.className = ''), 4000);
}
function loading(on) {
  document.body.classList.toggle('loading', on);
}

let mainChart, errChart;
function ensureCharts() {
  if (!mainChart) mainChart = echarts.init($('#chart'));
  if (!errChart) errChart = echarts.init($('#errChart'));
}
window.addEventListener('resize', () => {
  ensureCharts();
  if (mainChart) mainChart.resize();
  if (errChart) errChart.resize();
});

// row = [LF_X, PZ, LF_Z, HF_Z]; sort by x, return paired series
function sortedSeries(row) {
  const x = row[0];
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b] || a - b);
  const pick = (arr) => order.map((i) => [+x[i], +arr[i]]);
  return { hf: pick(row[3]), pred: pick(row[1]), lf: pick(row[2]), x: order.map((i) => +x[i]) };
}

function init() {
  ensureCharts();
  $('#fileSel').addEventListener('change', (e) => loadResult(e.target.value));
  $('#uploadBtn').addEventListener('click', () => $('#filePick').click());
  $('#filePick').addEventListener('change', onUpload);
  loadFiles();
}

function loadFiles() {
  API.files()
    .then((r) => {
      const sel = $('#fileSel');
      sel.innerHTML = '';
      if (!r.files.length) {
        sel.innerHTML = '<option value="">No result files found</option>';
        return;
      }
      for (const f of r.files) {
        const o = document.createElement('option');
        o.value = f.file;
        o.textContent = f.label + (f.hasChi ? '' : '   (no chi)');
        sel.appendChild(o);
      }
      loadResult(sel.value);
    })
    .catch((e) => toast('Could not load files: ' + e));
}

function loadResult(file) {
  if (!file) return;
  loading(true);
  API.results(file)
    .then(setResult)
    .catch((e) => {
      loading(false);
      toast('Error: ' + e);
    });
}

function onUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    loading(true);
    API.upload({ name: file.name, xyz: reader.result })
      .then(setResult)
      .catch((e) => {
        loading(false);
        toast('Upload error: ' + e);
      });
  };
  reader.readAsText(file);
  ev.target.value = '';
}

function setResult(r) {
  loading(false);
  state.result = r;
  state.sim = 0;
  const sel = $('#fileSel');
  if (![...sel.options].some((o) => o.value === r.file)) {
    const o = document.createElement('option');
    o.value = o.textContent = '· ' + (r.file || 'upload');
    sel.prepend(o);
  }
  sel.value = r.file;
  renderStats();
  renderTable();
  renderChart();
}

function card(label, val, sub, cls = '') {
  return `<div class="card ${cls}"><div class="k">${label}</div><div class="v">${val}</div><div class="s">${sub || ''}</div></div>`;
}

function renderStats() {
  const s = state.result.sims[state.sim];
  const good = (v) => (v == null ? '' : v < 0.02 ? 'good' : 'bad');
  $('#stats').innerHTML = [
    card('Prediction error χ²', fmt(s.chi_pred, 1), 'PRED vs HF'),
    card('LF baseline χ²', fmt(s.chi_lf, 1), 'LF vs HF'),
    card('RMSE PRED–HF', fmt(s.rmse_pred_hf), 'per point'),
    card('Shock x · PRED', fmt(s.shock_pred), '', 'accent'),
    card('Shock x · LF', fmt(s.shock_lf), '', 'accent'),
    card('Shock x · HF', fmt(s.shock_hf), '', 'accent'),
    card('Δshock PRED–HF', fmt(s.err_pred_hf, 4), 'location error', good(s.err_pred_hf)),
    card('Δshock PRED–LF', fmt(s.err_pred_lf, 4), 'location error', good(s.err_pred_lf)),
  ].join('');
}

function renderTable() {
  const c = (v) => (v == null ? '' : v < 0.02 ? 'good' : 'bad');
  const head =
    '<thead><tr><th>#</th><th>val</th><th>χ² pred</th><th>χ² LF</th><th>x·P</th><th>x·L</th><th>x·H</th><th>ΔP–H</th><th>ΔP–L</th></tr></thead>';
  const body = state.result.sims
    .map(
      (s) =>
        `<tr data-i="${s.index}" class="${s.index === state.sim ? 'active' : ''}">` +
        `<td>${s.index}</td><td>${s.is_validation ? '✓' : ''}</td>` +
        `<td>${fmt(s.chi_pred, 1)}</td><td>${fmt(s.chi_lf, 1)}</td>` +
        `<td>${fmt(s.shock_pred)}</td><td>${fmt(s.shock_lf)}</td><td>${fmt(s.shock_hf)}</td>` +
        `<td class="${c(s.err_pred_hf)}">${fmt(s.err_pred_hf, 4)}</td>` +
        `<td class="${c(s.err_pred_lf)}">${fmt(s.err_pred_lf, 4)}</td></tr>`
    )
    .join('');
  const t = $('#simTable');
  t.innerHTML = head + '<tbody>' + body + '</tbody>';
  t.querySelectorAll('tbody tr').forEach((tr) => (tr.onclick = () => selectSim(+tr.dataset.i)));
}

function selectSim(i) {
  state.sim = i;
  renderStats();
  renderChart();
  $('#simTable').querySelectorAll('tbody tr').forEach((tr) => tr.classList.toggle('active', +tr.dataset.i === i));
}

function renderChart() {
  const s = state.result.sims[state.sim];
  const d = sortedSeries(s.data);

  const marks = [
    { x: s.shock_hf, color: COLORS.hf, name: 'HF' },
    { x: s.shock_lf, color: COLORS.lf, name: 'LF' },
    { x: s.shock_pred, color: COLORS.pred, name: 'PRED' },
  ]
    .filter((m) => m.x != null)
    .map((m) => ({
      xAxis: m.x,
      name: m.name,
      lineStyle: { color: m.color, type: 'dashed', width: 1 },
      label: { show: true, position: 'end', formatter: m.name, color: m.color, fontSize: 10 },
    }));

  const base = {
    grid: { left: 55, right: 16, top: 42, bottom: 44 },
    tooltip: { trigger: 'axis', valueFormatter: (v) => fmt(v, 3) },
    legend: { data: ['HF', 'PRED', 'LF'], top: 6 },
    xAxis: { type: 'value', name: 'x (chord)', min: 0, max: 1, nameLocation: 'middle', nameGap: 28 },
    yAxis: { type: 'value', name: '−Cp' },
  };

  mainChart.setOption(
    {
      ...base,
      series: [
        {
          name: 'HF', type: 'line', showSymbol: false, lineStyle: { width: 2 },
          itemStyle: { color: COLORS.hf }, data: d.hf,
          markLine: { symbol: 'none', data: marks },
        },
        { name: 'PRED', type: 'line', showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: COLORS.pred }, data: d.pred },
        { name: 'LF', type: 'line', showSymbol: false, lineStyle: { width: 1.5, type: 'dashed' }, itemStyle: { color: COLORS.lf }, data: d.lf },
      ],
    },
    true
  );

  // deviation vs x
  const devPred = d.x.map((xx, i) => [xx, Math.abs(d.hf[i][1] - d.pred[i][1])]);
  const devLf = d.x.map((xx, i) => [xx, Math.abs(d.hf[i][1] - d.lf[i][1])]);
  errChart.setOption(
    {
      grid: { left: 55, right: 16, top: 40, bottom: 40 },
      tooltip: { trigger: 'axis', valueFormatter: (v) => fmt(v, 4) },
      legend: { data: ['|PRED−HF|', '|LF−HF|'], top: 6 },
      title: {
        text: 'Cp deviation vs x   (shock @ PRED ' + fmt(s.shock_pred, 3) + ')',
        left: 0, top: 0, textStyle: { fontSize: 12, fontWeight: 'normal', color: '#5b6472' },
      },
      xAxis: { type: 'value', min: 0, max: 1 },
      yAxis: { type: 'value' },
      series: [
        {
          name: '|PRED−HF|', type: 'line', showSymbol: false,
          areaStyle: { opacity: 0.15 }, lineStyle: { width: 1.5 }, itemStyle: { color: COLORS.pred }, data: devPred,
        },
        {
          name: '|LF−HF|', type: 'line', showSymbol: false,
          lineStyle: { width: 1.5 }, itemStyle: { color: COLORS.lf }, data: devLf,
        },
      ],
    },
    true
  );
}

document.addEventListener('DOMContentLoaded', init);
