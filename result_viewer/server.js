'use strict';

const express = require('express');
const fs = require('fs');
const path = require('path');
const { analyzeSims } = require('./lib/analyze');

const app = express();
const PORT = process.env.PORT || 8050;
const ROOT = path.resolve(__dirname, '..');
const RESULTS_DIR = process.env.RESULTS_DIR || path.join(ROOT, 'results');

app.use(express.json({ limit: '40mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// Serve the ECharts bundle locally (no CDN dependency).
app.get('/vendor/echarts.min.js', (req, res) => {
  res.sendFile(path.join(__dirname, 'node_modules', 'echarts', 'dist', 'echarts.min.js'));
});

function parseChi(text) {
  const rows = [];
  for (const line of String(text).split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    const [a, b, c] = t.split(',');
    rows.push({
      chi_pred: parseFloat(a),
      chi_lf: parseFloat(b),
      is_validation: c.trim() === 'True',
    });
  }
  return rows;
}

// Parse "<base>_xyz_<tag>_<pid>.csv" -> { base, tag, pid }. tag may be empty.
function parseName(f) {
  const i = f.indexOf('_xyz_');
  const base = i >= 0 ? f.slice(0, i) : f;
  const after = (i >= 0 ? f.slice(i + 5) : f).replace(/\.csv$/, '');
  const pidM = after.match(/(\d+)$/);
  const pid = pidM ? pidM[1] : '?';
  const tag = after.slice(0, after.length - pid.length).replace(/_+$/, '');
  return { base, tag, pid };
}

// List available result sets (xyz files) and whether a matching chi file exists.
app.get('/api/files', (req, res) => {
  if (!fs.existsSync(RESULTS_DIR)) return res.json({ files: [] });
  const files = fs
    .readdirSync(RESULTS_DIR)
    .filter((f) => /_xyz_.+\.csv$/.test(f))
    .sort();

  const out = files.map((f) => {
    const { base, tag, pid } = parseName(f);
    const chiFile = f.replace('_xyz_', '_chi_');
    const hasChi = fs.existsSync(path.join(RESULTS_DIR, chiFile));
    const st = fs.statSync(path.join(RESULTS_DIR, f));
    return {
      file: f,
      chiFile,
      hasChi,
      tag,
      label: base + ' · ' + (tag ? tag + ' ' : '') + pid,
      size: st.size,
      mtime: st.mtimeMs,
    };
  });
  out.sort((a, b) => b.mtime - a.mtime);
  res.json({ files: out });
});

// Load one result set (xyz + optional chi) and return computed per-sim data.
app.get('/api/results', (req, res) => {
  const name = path.basename(req.query.file || '');
  if (!/_xyz_.+\.csv$/.test(name)) return res.status(400).json({ error: 'bad file' });
  const xp = path.join(RESULTS_DIR, name);
  if (!fs.existsSync(xp)) return res.status(404).json({ error: 'not found' });

  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(xp, 'utf8'));
  } catch (e) {
    return res.status(500).json({ error: 'bad json: ' + e.message });
  }

  const chiName = name.replace('_xyz_', '_chi_');
  const chiPath = path.join(RESULTS_DIR, chiName);
  let chiRows = [];
  if (fs.existsSync(chiPath)) chiRows = parseChi(fs.readFileSync(chiPath, 'utf8'));

  const { tag } = parseName(name);
  const result = analyzeSims(parsed.data || [], chiRows);
  if (!chiRows.length) {
    // No chi file: infer train/valid membership from the filename tag.
    const v = tag === 'valid';
    result.sims.forEach((s) => (s.is_validation = v));
  }
  result.file = name;
  result.chiFile = chiName;
  result.hasChi = chiRows.length > 0;
  result.tag = tag;
  result.meta = parsed.meta || null;
  res.json(result);
});

// Upload a result file from the client (raw xyz csv/json text, optional chi text).
app.post('/api/upload', (req, res) => {
  try {
    const { name, xyz, chi } = req.body || {};
    if (!xyz) return res.status(400).json({ error: 'missing xyz payload' });
    const parsed = JSON.parse(xyz);
    let chiRows = [];
    if (typeof chi === 'string' && chi.trim()) chiRows = parseChi(chi);
    const result = analyzeSims(parsed.data || [], chiRows);
    result.file = name || 'upload';
    result.hasChi = chiRows.length > 0;
    result.meta = parsed.meta || null;
    res.json(result);
  } catch (e) {
    res.status(400).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`result_viewer on http://localhost:${PORT}`);
  console.log(`  results dir: ${RESULTS_DIR}`);
});
