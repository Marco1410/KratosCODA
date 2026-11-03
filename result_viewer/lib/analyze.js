'use strict';

// Locate a shock as the x of the largest absolute adjacent Cp jump on the
// chordwise-ordered upper surface. The shock is a genuine Cp discontinuity,
// so it shows up as the biggest single-step change; the leading-edge suction
// peak is a curved rise spread over several tiny-dx steps, so per-step it is
// smaller. Returns the bracketing midpoint.
function detectShock(x, cp) {
  const n = x.length;
  if (n < 2) return { x: null, slope: null };
  const order = x.map((_, i) => i).sort((a, b) => x[a] - x[b] || a - b);
  const xs = order.map((i) => x[i]);
  const cs = order.map((i) => cp[i]);

  let best = { jump: -1, lo: 0, hi: 1 };
  for (let k = 1; k < n; k++) {
    const a = cs[k - 1];
    const b = cs[k];
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const j = Math.abs(b - a);
    if (j > best.jump) best = { jump: j, lo: k - 1, hi: k };
  }
  if (best.jump < 0) return { x: null, slope: null };
  const dx = xs[best.hi] - xs[best.lo];
  return { x: (xs[best.lo] + xs[best.hi]) / 2, slope: dx > 1e-12 ? best.jump / dx : null };
}

// sum of squared differences, chi(a, b) = sum((a-b)^2)
function chiSq(a, b) {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

// data:  [[LF_X, PZ, LF_Z, HF_Z], ...]  (one entry per simulation)
// chiRows: [{ chi_pred, chi_lf, is_validation }, ...]  (may be empty)
function analyzeSims(data, chiRows) {
  const sims = data.map((row, i) => {
    const x = row[0];
    const pred = row[1];
    const lf = row[2];
    const hf = row[3];
    const n = x.length;

    const sp = detectShock(x, pred);
    const sl = detectShock(x, lf);
    const sh = detectShock(x, hf);

    const chi_pred = chiRows[i] ? chiRows[i].chi_pred : chiSq(hf, pred);
    const chi_lf = chiRows[i] ? chiRows[i].chi_lf : chiSq(hf, lf);

    return {
      index: i,
      chi_pred,
      chi_lf,
      rmse_pred_hf: Math.sqrt(chi_pred / n),
      rmse_lf_hf: Math.sqrt(chi_lf / n),
      is_validation: chiRows[i] ? !!chiRows[i].is_validation : false,
      shock_pred: sp.x,
      shock_lf: sl.x,
      shock_hf: sh.x,
      err_pred_hf: sp.x != null && sh.x != null ? Math.abs(sp.x - sh.x) : null,
      err_pred_lf: sp.x != null && sl.x != null ? Math.abs(sp.x - sl.x) : null,
      data: [x, pred, lf, hf],
    };
  });

  return { sims, n: sims.length };
}

module.exports = { analyzeSims, detectShock, chiSq };
