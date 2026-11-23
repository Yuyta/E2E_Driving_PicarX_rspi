/* PiCar-X Control Center — 外部のライブラリを使わない画面（サーバーは src/gui/web/server.py） */
"use strict";

const TOKEN = document.querySelector('meta[name="picarx-token"]').content;
const STAGES = {
  candidate: "候補",
  approved: "有効",
  rejected: "却下",
  archived: "廃止",
};
const STAGE_HELP = {
  candidate: "まだ判断していない",
  approved: "走行に使ってよい",
  rejected: "使わない（性能やトレーサビリティに問題）",
  archived: "以前は使っていたが今は使わない",
};
const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const state = { catalog: null, workflows: null, phases: [], jobs: [], timers: [], robot: null };

/* ------------------------------------------------------------------ DOM helpers */

function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  applyProps(el, props);
  appendKids(el, kids);
  return el;
}
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, props, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  applyProps(el, props, true);
  appendKids(el, kids);
  return el;
}
function applyProps(el, props, svg) {
  if (!props) return;
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") {
      // CSS の変数（--ph など）は setProperty でしか入らない
      for (const [name, value] of Object.entries(v)) {
        if (name.startsWith("--")) el.style.setProperty(name, value);
        else el.style[name] = value;
      }
    }
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else if (!svg && k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
}
function appendKids(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
}

const ICONS = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  cube: '<path d="M12 2l9 5v10l-9 5-9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
  compare: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  db: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  flow: '<circle cx="5" cy="6" r="2.5"/><circle cx="5" cy="18" r="2.5"/><circle cx="19" cy="12" r="2.5"/><path d="M7.5 6c6 0 4 6 9 6M7.5 18c6 0 4-6 9-6"/>',
  road: '<path d="M6 21L10 3M18 21L14 3M12 7v2M12 13v2M12 19v1"/>',
  play: '<path d="M7 4l13 8-13 8z"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  book: '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M4 21.5A2.5 2.5 0 0 1 6.5 19H20v3H6.5"/>',
};
document.querySelectorAll("i[data-icon]").forEach((i) => {
  i.innerHTML = `<svg viewBox="0 0 24 24">${ICONS[i.dataset.icon] || ""}</svg>`;
});

/* ------------------------------------------------------------------ format */

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const fmtInt = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("ja-JP"));
const pct = (v) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
const short = (hash, n = 8) => (hash ? String(hash).slice(0, n) : "—");
function fmtDate(iso, withYear) {
  if (!iso) return "—";
  const d = new Date(String(iso).replace(/([+-]\d{2})(\d{2})$/, "$1:$2"));
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n) => String(n).padStart(2, "0");
  const date = `${withYear ? d.getFullYear() + "/" : ""}${d.getMonth() + 1}/${d.getDate()}`;
  return `${date} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function duration(a, b) {
  if (!a) return "—";
  const t0 = new Date(String(a).replace(/([+-]\d{2})(\d{2})$/, "$1:$2"));
  const t1 = b ? new Date(String(b).replace(/([+-]\d{2})(\d{2})$/, "$1:$2")) : new Date();
  const sec = Math.max(0, Math.round((t1 - t0) / 1000));
  if (sec < 60) return `${sec}秒`;
  if (sec < 3600) return `${Math.floor(sec / 60)}分${sec % 60}秒`;
  return `${Math.floor(sec / 3600)}時間${Math.floor((sec % 3600) / 60)}分`;
}
const stageBadge = (stage) => h("span", { class: `badge stage-${stage}`, title: STAGE_HELP[stage] }, STAGES[stage] || stage);
const HEALTH_TEXT = { ok: "問題なし", warn: "注意", error: "要対応", unknown: "不明" };
const healthBadge = (level, text) => h("span", { class: `health ${level}` }, text || HEALTH_TEXT[level]);
const STATUS_TEXT = { running: "実行中", succeeded: "成功", failed: "失敗", stopped: "停止", unknown: "不明" };
const statusDot = (st) => h("span", { class: `status-dot ${st}` }, STATUS_TEXT[st] || st);
const BODY_SOURCE = { recorded: "収集の時に記録した版", date: "セッションの日時から決めた版" };
function bodyBadge(id, source, changed) {
  if (!id) return h("span", { class: "muted" }, "—");
  const title = [BODY_SOURCE[source], changed ? "記録した後に版の中身が書き換えられた" : null].filter(Boolean).join("・");
  return h("a", { class: `badge body${changed ? " changed" : ""}`, href: "#/body", title: title || "車体の版", onclick: (e) => e.stopPropagation() }, id, source === "date" ? h("span", { class: "est" }, "推定") : null);
}

/* ------------------------------------------------------------------ MLOps phases */

// フェーズの定義（名前、色、アイコン、画面）はサーバー（src/gui/workflows.py の PHASES）から読む。
// ランチャーと同じ色と絵を使う。番号は並び順にだけ使い、画面には出さない
const phaseById = (id) => state.phases.find((p) => p.id === id);
const loopPhases = () => state.phases.filter((p) => p.number !== null);
const routePhases = (route) => loopPhases().filter((p) => p.page === route);
const iconSvg = (markup) => `<svg viewBox="0 0 24 24" aria-hidden="true">${markup}</svg>`;
function phaseIcon(p, extra) {
  const el = h("span", { class: `ph-icon ${extra || ""}`, style: { "--ph": p.color }, title: `${p.title}: ${p.summary}` });
  el.innerHTML = iconSvg(p.icon);
  return el;
}

/** フェーズごとに「流れているもの」の量。0 の段階へは粒を流さない */
function phaseCounts(c) {
  if (!c) return null;
  return {
    prepare: (c.bodies || []).length,
    collect: c.summary.sessions,
    curate: c.sessions.filter((x) => x.annotation_hash || x.manifest_hash).length,
    train: c.models.length,
    evaluate: c.cv.length + c.models.filter((m) => m.stage !== "candidate").length,
    // デプロイの記録がなくても、Pi の走行記録にモデルの版があれば、そのモデルは Pi に入っていた
    deploy: (c.deployments || []).length + (c.summary.deployed.version ? 1 : 0) + (c.summary.last_pi_run?.model_version ? 1 : 0),
    operate: c.summary.runs,
  };
}

let flowSeq = 0;
/** MLOps の流れの図。段階を線でつなぎ、記録のある区間に粒を流す。実走行からデータ収集へ戻る輪も描く
 *  nodes: 各段階の下に出す中身（なければ名前だけの小さい図）。current: 強調する段階 */
function flowDiagram({ counts, current = [], details, compact }) {
  const ps = loopPhases();
  const gid = `flow-grad-${++flowSeq}`;
  const svg = s("svg", { class: "flow-track", "aria-hidden": "true" });
  const nodes = ps.map((p) => {
    const d = details?.[p.id];
    return h("a", { class: `flow-node ${current.includes(p) ? "on" : ""} ${counts && !counts[p.id] ? "idle" : ""}`, href: `#/${p.page}`, style: { "--ph": p.color }, title: p.summary, "aria-current": current.includes(p) ? "step" : null },
      phaseIcon(p, "orb"),
      h("span", { class: "flow-name" }, p.title),
      d ? [h("span", { class: "flow-value" }, d[0], h("small", {}, d[1])), h("span", { class: "flow-foot" }, d[2])] : null);
  });
  const row = h("div", { class: "flow-row", style: { gridTemplateColumns: `repeat(${ps.length}, minmax(0, 1fr))` } }, nodes);
  const box = h("div", { class: `flow ${compact ? "compact" : ""}`, role: "img", "aria-label": `開発の流れ: ${ps.map((p) => p.title).join(" → ")} → ${ps[1]?.title || ""}へ戻る` }, svg, row);
  // 置いた場所の実際の大きさで描く（縦横の伸び縮みで粒がつぶれないように）
  const draw = () => {
    const B = box.getBoundingClientRect();
    if (!B.width) return;
    const orbs = nodes.map((n) => n.querySelector(".orb").getBoundingClientRect());
    const xs = orbs.map((o) => o.left + o.width / 2 - B.left);
    const y = orbs[0].top + orbs[0].height / 2 - B.top;
    const W = B.width, H = B.height, r = compact ? 8 : 14;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    const grad = s("linearGradient", { id: gid, gradientUnits: "userSpaceOnUse", x1: xs[0], x2: xs[xs.length - 1], y1: 0, y2: 0 },
      ps.map((p, i) => s("stop", { offset: i / Math.max(1, ps.length - 1), "stop-color": p.color })));
    const parts = [s("defs", {}, grad)];
    const flowing = (i) => !counts || (counts[ps[i].id] > 0 && counts[ps[i + 1].id] > 0);
    for (let i = 0; i < xs.length - 1; i++) {
      const d = `M${xs[i]},${y} H${xs[i + 1]}`;
      parts.push(s("path", { class: "flow-base", d }));
      if (flowing(i)) parts.push(s("path", { class: "flow-dots", d, stroke: `url(#${gid})` }));
    }
    // 実走行 → データ収集 の輪（下を回って、データ収集の手前で本線に合流する）
    if (xs.length > 2) {
      const R = W - 4, L = (xs[0] + xs[1]) / 2, yb = H - 3;
      const d = `M${xs[xs.length - 1]},${y} H${R - r} Q${R},${y} ${R},${y + r} V${yb - r} Q${R},${yb} ${R - r},${yb} H${L + r} Q${L},${yb} ${L},${yb - r} V${y + r} Q${L},${y} ${L + r},${y}`;
      parts.push(s("path", { class: "flow-base loop", d }));
      if (!counts || counts.operate > 0) parts.push(s("path", { class: "flow-dots loop", d, stroke: `url(#${gid})` }));
    }
    svg.replaceChildren(...parts);
  };
  let last = "";
  new ResizeObserver(() => { const k = `${Math.round(box.clientWidth)}x${Math.round(box.clientHeight)}`; if (k !== last) { last = k; draw(); } }).observe(box);
  return h("div", { class: "flow-scroll" }, box);
}

/** サイドバーの「開発の流れ」。フェーズを縦に並べ、線に沿って粒が流れる */
function buildNavFlow() {
  const holder = document.getElementById("nav-flow");
  if (!holder || !state.phases.length) return;
  const ps = loopPhases();
  holder.style.setProperty("--flow-grad", `linear-gradient(to bottom, ${ps.map((p) => p.color).join(", ")})`);
  holder.replaceChildren(...ps.map((p) => h("a", { href: `#/${p.page}`, "data-route": p.page, style: { "--ph": p.color }, title: p.summary },
    phaseIcon(p), h("span", { class: "nav-text" }, h("span", {}, p.title),
      // 画面の名前がフェーズの名前と同じ時（デプロイ）は繰り返さない
      ROUTES[p.page]?.[0] && ROUTES[p.page][0] !== p.title ? h("small", {}, ROUTES[p.page][0]) : null))));
}

/** 画面の上に置く「今どのフェーズか」の流れの図と、そのフェーズで使うツール */
function phaseHeader(route) {
  const mine = routePhases(route);
  if (!mine.length) return null;
  const workflows = state.workflows || [];
  const blocks = mine.map((p) => h("div", { class: "phase-block", style: { "--ph": p.color } },
    h("div", { class: "phase-title" }, phaseIcon(p, "solid"), h("div", {}, h("b", {}, p.title), h("span", {}, p.summary))),
    h("div", { class: "phase-tools" }, workflows.filter((w) => w.phase === p.id).map((w) =>
      h("button", { type: "button", class: `small ${w.moves_car ? "danger-ghost" : "ghost"}`, title: w.description, onclick: () => launch(w.id) },
        w.title, w.moves_car ? h("span", { class: "tiny-badge" }, "車体が動く") : null)))));
  return h("section", { class: "phase-head" },
    flowDiagram({ counts: phaseCounts(state.catalog), current: mine, compact: true }),
    h("div", { class: "phase-blocks" }, blocks));
}
function setPagePhase(route) {
  const el = document.getElementById("page-phase");
  el.replaceChildren(...routePhases(route).map((p) => phaseIcon(p, "solid")));
}

/* ------------------------------------------------------------------ api */

async function api(path, { method = "GET", body } = {}) {
  const headers = { "X-PicarX-Token": TOKEN };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let res;
  try {
    res = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    // ブラウザの「Failed to fetch」はサーバーに届かなかった時に出る
    throw new Error("Control Center のサーバーに接続できません。サーバーが止まっています（ランチャーを閉じた、ターミナルで Ctrl+C を押したなど）。`uv run python -m src.gui.web` で起動し直してから、このページを開き直してください");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}
async function loadCatalog(fresh) {
  state.catalog = await api(`/catalog${fresh ? "?fresh=1" : ""}`);
  return state.catalog;
}
async function loadWorkflows() {
  if (!state.workflows) state.workflows = await api("/workflows");
  return state.workflows;
}
const frameUrl = (sid, i) => `/api/sessions/${encodeURIComponent(sid)}/frames/${i}.jpg?token=${encodeURIComponent(TOKEN)}`;

function toast(message, kind) {
  const el = h("div", { class: `toast ${kind || ""}` }, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 7000 : 4000);
}

/* ------------------------------------------------------------------ tooltip */

const tip = document.getElementById("tooltip");
function showTip(evt, title, rows) {
  tip.replaceChildren(
    h("div", { class: "tt-title" }, title),
    ...(rows || []).map(([color, label, value]) =>
      h("div", { class: "tt-row" }, h("span", {}, color ? h("i", { style: { background: color } }) : null, label), h("b", {}, value)),
    ),
  );
  tip.hidden = false;
  const pad = 14;
  const r = tip.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => (tip.hidden = true);

/* ------------------------------------------------------------------ charts */

/** 折れ線。series = [{name, color, points: [[x, y], ...]}]。凡例と線の端の名前を付け、なぞると値を出す。
 *  文字の大きさが変わらないよう、置いた場所の実際の幅で描き直す */
function lineChart(opts) {
  if (!opts.series.some((sr) => sr.points.length)) return h("div", { class: "empty" }, "データがありません");
  const legend = opts.series.length > 1
    ? h("div", { class: "legend" }, opts.series.map((sr) => h("span", {}, h("i", { style: { background: sr.color } }), sr.name)))
    : null;
  const plot = h("div");
  let lastWidth = 0;
  const ro = new ResizeObserver(() => {
    const w = Math.round(plot.clientWidth);
    if (w && Math.abs(w - lastWidth) > 4) { lastWidth = w; plot.replaceChildren(lineSvg(opts, w)); }
  });
  ro.observe(plot);
  return h("div", { class: "chart" }, legend, h("div", { class: "muted small" }, opts.yLabel), plot);
}
function lineSvg({ series, xLabel, yLabel, height = 250, marker, yFormat = (v) => fmt(v, 3) }, width) {
  const W = Math.max(320, width), H = height, m = { l: 52, r: 76, t: 12, b: 40 };
  const all = series.flatMap((sr) => sr.points);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(0, ...ys), y1 = Math.max(...ys);
  if (y1 === y0) y1 = y0 + 1;
  const step = niceStep((y1 - y0) / 4);
  y1 = Math.ceil(y1 / step) * step;
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${yLabel} の推移` });
  for (let v = y0; v <= y1 + 1e-9; v += step) {
    svg.append(s("line", { class: v === y0 ? "base-line" : "grid-line", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    svg.append(s("text", { class: "tick", x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, yFormat(v)));
  }
  const xticks = niceTicks(x0, x1, Math.max(3, Math.floor(W / 110)));
  for (const v of xticks) svg.append(s("text", { class: "tick", x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, v));
  svg.append(s("text", { class: "axis-title", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, xLabel));
  if (marker) {
    svg.append(s("line", { x1: X(marker.x), x2: X(marker.x), y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-dasharray": "3 3" }));
    svg.append(s("text", { class: "tick", x: X(marker.x) + 4, y: m.t + 10 }, marker.label));
  }
  const labels = [];
  series.forEach((sr) => {
    const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": sr.dash || null }));
    const last = sr.points[sr.points.length - 1];
    if (series.length <= 4 && last) labels.push({ y: Y(last[1]), x: X(last[0]), name: sr.name });
  });
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 14) labels[i].y = labels[i - 1].y + 14;
  labels.forEach((l) => svg.append(s("text", { class: "direct", x: l.x + 8, y: l.y + 4 }, l.name)));
  // なぞった位置の縦線と点
  const cross = s("line", { class: "cross", y1: m.t, y2: H - m.b, visibility: "hidden" });
  const dots = series.map((sr) => s("circle", { r: 4.5, fill: sr.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }));
  svg.append(cross, ...dots);
  const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
  const xsSorted = [...new Set(xs)].sort((a, b) => a - b);
  hit.addEventListener("mousemove", (evt) => {
    const box = svg.getBoundingClientRect();
    const vx = ((evt.clientX - box.left) / box.width) * W;
    const xv = x0 + ((vx - m.l) / (W - m.l - m.r)) * (x1 - x0);
    const near = xsSorted.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a), xsSorted[0]);
    cross.setAttribute("x1", X(near)); cross.setAttribute("x2", X(near)); cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((sr, i) => {
      const p = sr.points.find((q) => q[0] === near);
      if (p) {
        dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1])); dots[i].setAttribute("visibility", "visible");
        rows.push([sr.color, sr.name, yFormat(p[1])]);
      } else dots[i].setAttribute("visibility", "hidden");
    });
    showTip(evt, `${xLabel} ${near}`, rows);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); });
  svg.append(hit);
  return svg;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const n = raw / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}
function niceTicks(a, b, count) {
  if (a === b) return [a];
  const step = Math.max(1, Math.round(niceStep((b - a) / count)));
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b; v += step) out.push(v);
  if (out[0] - a >= step / 2) out.unshift(a);
  return out;
}

/** 横棒（値の大きさ）。rows = [{name, value, text, color}] */
function hbars(rows, { max, title } = {}) {
  const top = max || Math.max(...rows.map((r) => r.value || 0), 1e-9);
  const nameW = Math.min(280, Math.max(64, ...rows.map((r) => String(r.name).length * 13 + 8)));
  return h("div", { class: "hbars", style: { "--name-w": `${nameW}px` } }, rows.map((r) => {
    const fill = h("div", { class: "fill", style: { width: `${Math.max(0, (r.value || 0) / top) * 100}%`, background: r.color || "var(--series-1)" } });
    const row = h("div", { class: "hbar" }, h("span", { class: "name" }, r.name), h("div", { class: "track" }, fill), h("span", { class: "val" }, r.text ?? fmt(r.value, 3)));
    row.addEventListener("mousemove", (e) => showTip(e, title || r.name, [[r.color || "var(--series-1)", r.name, r.text ?? fmt(r.value, 3)], ...(r.extra || [])]));
    row.addEventListener("mouseleave", hideTip);
    return row;
  }));
}

/** 左・直進・右の割合（色は 左=青、直進=灰、右=赤 の対になる配色） */
function distBar(dist) {
  const l = dist.left_ratio || 0, st = dist.straight_ratio || 0, r = dist.right_ratio || 0;
  const el = h("div", { class: "dist", role: "img", "aria-label": `左 ${pct(l)} 直進 ${pct(st)} 右 ${pct(r)}` },
    h("span", { class: "l", style: { width: `${l * 100}%` } }),
    h("span", { class: "s", style: { width: `${st * 100}%` } }),
    h("span", { class: "r", style: { width: `${r * 100}%` } }));
  el.addEventListener("mousemove", (e) => showTip(e, "舵角の分布", [["var(--div-neg)", "左", pct(l)], ["var(--div-mid)", "直進", pct(st)], ["var(--div-pos)", "右", pct(r)]]));
  el.addEventListener("mouseleave", hideTip);
  return el;
}
const distLegend = () => h("div", { class: "legend" },
  h("span", {}, h("i", { class: "box", style: { background: "var(--div-neg)" } }), "左"),
  h("span", {}, h("i", { class: "box", style: { background: "var(--div-mid)" } }), "直進"),
  h("span", {}, h("i", { class: "box", style: { background: "var(--div-pos)" } }), "右"));

/* ------------------------------------------------------------------ table */

/** 並べ替えのできる表。cols = [{key, label, num, render(row), sort(row)}] */
function table(cols, rows, { onRow, empty = "データがありません", sortKey, desc = false, rowClass } = {}) {
  if (!rows.length) return h("div", { class: "empty" }, empty);
  let key = sortKey, down = desc;
  const tbody = h("tbody");
  const head = h("tr");
  const draw = () => {
    const col = cols.find((c) => c.key === key);
    const sorted = col ? [...rows].sort((a, b) => {
      const va = (col.sort || ((r) => r[col.key]))(a), vb = (col.sort || ((r) => r[col.key]))(b);
      const cmp = va === vb ? 0 : va === null || va === undefined ? 1 : vb === null || vb === undefined ? -1 : va < vb ? -1 : 1;
      return down ? -cmp : cmp;
    }) : rows;
    tbody.replaceChildren(...sorted.map((r) => h("tr", { class: [onRow ? "click" : "", rowClass ? rowClass(r) : ""].join(" "), onclick: onRow ? () => onRow(r) : null },
      cols.map((c) => h("td", { class: c.num ? "num" : "" }, c.render ? c.render(r) : r[c.key] ?? "—")))));
    head.replaceChildren(...cols.map((c) => h("th", {
      class: `${c.num ? "num" : ""} ${c.nosort ? "" : "sortable"}`,
      title: c.title,
      onclick: c.nosort ? null : () => { down = key === c.key ? !down : false; key = c.key; draw(); },
    }, c.label, key === c.key ? h("span", { class: "arrow" }, down ? "▼" : "▲") : null)));
  };
  draw();
  return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, head), tbody));
}

/* ------------------------------------------------------------------ modal */

function modal({ title, body, actions }) {
  const root = document.getElementById("modal-root");
  const close = () => { root.replaceChildren(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  const back = h("div", { class: "modal-back", onclick: (e) => { if (e.target === back) close(); } },
    h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title },
      h("header", {}, h("h2", {}, title)),
      h("div", { class: "body" }, body),
      h("footer", {}, actions(close))));
  root.replaceChildren(back);
  const first = back.querySelector("input, textarea, select, button");
  if (first) first.focus();
  return close;
}

/* ------------------------------------------------------------------ launching tools */

function nextVersion() {
  const nums = (state.catalog?.models || []).map((m) => parseInt(String(m.version).replace(/\D/g, ""), 10)).filter((n) => !Number.isNaN(n));
  return `v${String((nums.length ? Math.max(...nums) : 0) + 1).padStart(3, "0")}`;
}

/* 引数の文字列 ⇔ フォーム。フォルダやファイルの引数（src/gui/workflows.py の PATH_ARGS）は欄と「選ぶ…」にする */
function tokenize(text) {
  const out = [];
  const re = /"((?:[^"\\]|\\.)*)"|'([^']*)'|(\S+)/g;
  let m;
  while ((m = re.exec(text))) out.push(m[1] !== undefined ? m[1].replace(/\\(.)/g, "$1") : m[2] ?? m[3]);
  return out;
}
const quoteArg = (s) => (s === "" || /[\s"'\\]/.test(s) ? `"${s.replace(/(["\\])/g, "\\$1")}"` : s);
function parseArgs(text, specs) {
  const toks = tokenize(text);
  const fields = [], rest = [];
  for (let i = 0; i < toks.length; i++) {
    const [flag, inline] = toks[i].split(/=(.*)/s);
    if (specs[flag]) {
      const value = inline ?? (toks[i + 1] !== undefined && !toks[i + 1].startsWith("--") ? toks[++i] : "");
      fields.push({ flag, value });
    } else rest.push(toks[i]);
  }
  return { fields, rest: rest.map(quoteArg).join(" ") };
}
// 位置で決まる引数（push など）が先に来るよう、ほかの引数を前に置く
const composeArgs = (rest, fields) => [rest.trim(), ...fields.filter((f) => f.value.trim()).map((f) => `${f.flag} ${quoteArg(f.value.trim())}`)].filter(Boolean).join(" ");
async function loadPathArgs() {
  if (!state.pathArgs) state.pathArgs = await api("/path_args").catch(() => ({}));
  return state.pathArgs;
}

/** プロジェクト（python/）の中のフォルダやファイルを選ぶ。選んだ相対パスを返す（やめたら null） */
function pickPath(spec, current) {
  return new Promise((resolve) => {
    const isNew = spec.kind === "new";
    const parts = String(current || "").split("/").filter(Boolean);
    let dir = spec.kind === "dir" ? parts.join("/") : parts.slice(0, -1).join("/");
    const name = isNew ? h("input", { type: "text", class: "mono", value: parts[parts.length - 1] || "", spellcheck: false }) : null;
    const crumbs = h("div", { class: "picker-crumbs" });
    const list = h("div", { class: "picker-list", role: "listbox" });
    const here = h("span", { class: "mono small muted" });
    const join = (...xs) => xs.filter(Boolean).join("/");
    const fileOk = (f) => !spec.exts?.length || spec.exts.some((x) => f.name.toLowerCase().endsWith(x));
    let back;
    const onKey = (e) => { if (e.key === "Escape") { e.stopImmediatePropagation(); finish(null); } };
    function finish(value) { back.remove(); document.removeEventListener("keydown", onKey, true); resolve(value); }
    const row = (icon, label, onclick, extra) => h("div", { class: "picker-row", role: "option", tabindex: 0, onclick, onkeydown: (e) => { if (e.key === "Enter") onclick(); } },
      h("span", { class: "picker-icon" }, icon), h("span", { class: "picker-name" }, label), extra || null);
    const showHere = () => { here.textContent = isNew ? join(dir, name.value) || "—" : spec.kind === "dir" ? dir || "python（一番上）" : ""; };
    async function open(path) {
      list.replaceChildren(h("div", { class: "loading" }, h("span", { class: "spinner" })));
      let res;
      try { res = await api(`/browse?path=${encodeURIComponent(path)}`); } catch (err) { list.replaceChildren(h("div", { class: "alert error" }, err.message)); return; }
      dir = res.path;
      const segs = dir ? dir.split("/") : [];
      crumbs.replaceChildren(h("a", { href: "#", onclick: (e) => { e.preventDefault(); open(""); } }, "python"),
        ...segs.flatMap((sg, i) => [h("span", { class: "muted" }, " / "), h("a", { href: "#", onclick: (e) => { e.preventDefault(); open(segs.slice(0, i + 1).join("/")); } }, sg)]));
      showHere();
      const files = spec.kind === "dir" ? [] : res.files.filter(fileOk);
      list.replaceChildren(
        dir ? row("↰", "1つ上へ", () => open(segs.slice(0, -1).join("/"))) : "",
        ...res.dirs.map((d) => row("📁", d, () => open(join(dir, d)),
          spec.kind === "dir" ? h("button", { type: "button", class: "small ghost", onclick: (e) => { e.stopPropagation(); finish(join(dir, d)); } }, "選ぶ") : null)),
        ...files.map((f) => row("📄", f.name, () => { if (isNew) { name.value = f.name; showHere(); } else finish(join(dir, f.name)); },
          h("span", { class: "muted small" }, f.size >= 1048576 ? `${fmt(f.size / 1048576, 1)} MB` : `${fmt(f.size / 1024, 1)} KB`))),
        !res.dirs.length && !files.length ? h("div", { class: "empty small" }, spec.kind === "file" ? `${spec.exts.join(" / ")} のファイルがありません` : "フォルダがありません") : "",
        res.truncated ? h("div", { class: "muted small" }, "多いので先頭だけを出しています") : "");
    }
    if (name) name.addEventListener("input", showHere);
    const hint = spec.kind === "dir" ? "フォルダを開いて「このフォルダを選ぶ」か、行の「選ぶ」を押します"
      : spec.kind === "file" ? `${spec.exts.join(" / ")} のファイルを押して選びます`
      : "置き場所のフォルダを開き、名前を決めて「ここに作る」を押します（まだないフォルダやファイルは、ツールが作ります）";
    back = h("div", { class: "modal-back picker-back", onclick: (e) => { if (e.target === back) finish(null); } },
      h("div", { class: "modal picker", role: "dialog", "aria-modal": "true", "aria-label": `${spec.label}を選ぶ` },
        h("header", {}, h("h2", {}, `${spec.label}を選ぶ`)),
        h("div", { class: "body" }, h("div", { class: "muted small" }, hint), crumbs, list,
          name ? h("label", { class: "field" }, h("span", {}, "名前"), name) : null),
        h("footer", {}, here, h("span", { style: { flex: 1 } }),
          h("button", { type: "button", class: "ghost", onclick: () => finish(null) }, "キャンセル"),
          spec.kind === "dir" ? h("button", { type: "button", onclick: () => finish(dir) }, "このフォルダを選ぶ") : null,
          isNew ? h("button", { type: "button", onclick: () => { if (!name.value.trim()) return toast("名前を入れてください", "error"); finish(join(dir, name.value.trim())); } }, "ここに作る") : null)));
    document.body.append(back);
    document.addEventListener("keydown", onKey, true);
    open(dir);
  });
}

async function launch(workflowId, presetArgs) {
  const workflows = await loadWorkflows();
  const specs = await loadPathArgs();
  const wf = workflows.find((w) => w.id === workflowId);
  if (!wf) return toast("ツールが見つかりません", "error");
  let initial = presetArgs ?? wf.arguments;
  if (wf.id === "train" && presetArgs === undefined) initial = initial.replace(/models\/v\d+/, `models/${nextVersion()}`);
  const models = state.catalog?.models || [];
  const approved = models.filter((m) => m.usable).map((m) => m.version);
  // フォームの中身。fields はフォルダやファイルの引数、rest はそれ以外の引数
  let fields = [];
  const rest = h("input", { type: "text", class: "mono", spellcheck: false, style: { width: "100%" }, placeholder: "なし" });
  const formBox = h("div", { class: "stack", style: { gap: "10px" } });
  const preview = h("code", { class: "cmd-preview" });
  const warnBox = h("div", { class: "warn-box small", hidden: true }, "このモデルは「有効」になっていません。モデルの画面で採否を判断してからデプロイすることを勧めます。");
  const current = () => composeArgs(rest.value, fields);
  const refresh = () => {
    preview.textContent = `python -m ${wf.module} ${current()}`;
    const m = /models\/([A-Za-z0-9_.-]+)/.exec(current());
    warnBox.hidden = !(wf.id === "deploy_push" && m && !approved.includes(m[1]));
  };
  rest.addEventListener("input", refresh);
  const fill = (text) => {
    const parsed = parseArgs(text, specs);
    fields = parsed.fields;
    rest.value = parsed.rest;
    formBox.replaceChildren(...fields.map((f) => {
      const spec = specs[f.flag];
      const input = h("input", { type: "text", class: "mono", value: f.value, spellcheck: false, oninput: (e) => { f.value = e.target.value; refresh(); } });
      f.input = input;
      return h("div", { class: "field" },
        h("span", {}, spec.label, h("code", { class: "flag" }, f.flag)),
        h("div", { class: "path-row" }, input, h("button", { type: "button", class: "ghost", onclick: async () => {
          const picked = await pickPath(spec, input.value);
          if (picked !== null) { input.value = f.value = picked; refresh(); }
        } }, "選ぶ…")));
    }), h("label", { class: "field" }, h("span", {}, fields.length ? "ほかの引数" : "実行引数"), rest));
    refresh();
  };
  fill(initial);
  const preset = wf.examples.length
    ? h("select", { onchange: (e) => { if (e.target.value) fill(e.target.value); } },
      h("option", { value: "" }, "例から選ぶ…"), wf.examples.map(([label, value]) => h("option", { value }, label)))
    : null;
  // モデルのフォルダを含むツールは、版の一覧からも選べる（欄の models/<版> を置き換える）
  const swapModel = (text, v) => text.replace(/models\/[A-Za-z0-9_.-]+/, `models/${v}`);
  const modelPick = /models\/v\d+/.test(initial) && wf.id !== "train" && models.length
    ? h("select", { onchange: (e) => {
      for (const f of fields) if (/models\/[A-Za-z0-9_.-]+/.test(f.value)) { f.value = swapModel(f.value, e.target.value); f.input.value = f.value; }
      rest.value = swapModel(rest.value, e.target.value);
      refresh();
    } }, models.map((m) => h("option", { value: m.version, selected: initial.includes(`models/${m.version}`) }, `${m.version}（${STAGES[m.stage]}）`)))
    : null;
  const carCheck = wf.moves_car ? h("input", { type: "checkbox" }) : null;
  const confirmCheck = wf.confirm ? h("input", { type: "checkbox" }) : null;
  modal({
    title: `${wf.title} を起動`,
    body: [
      h("div", { class: "row", style: { gap: "8px", flexWrap: "nowrap", alignItems: "flex-start" } },
        phaseById(wf.phase) ? phaseIcon(phaseById(wf.phase)) : null,
        h("span", { class: "ink-2" }, phaseById(wf.phase) ? h("b", {}, `${phaseById(wf.phase).title}: `) : null, wf.description,
          " ", h("a", { href: manualHref(wf.id), onclick: () => document.getElementById("modal-root").replaceChildren() }, "使い方 →"))),
      modelPick ? h("label", { class: "field" }, h("span", {}, "モデル"), modelPick) : null,
      preset ? h("label", { class: "field" }, h("span", {}, "操作の例"), preset) : null,
      formBox,
      h("div", { class: "field" }, h("span", {}, "実行するコマンド"), preview),
      warnBox,
      wf.kind === "window" ? h("div", { class: "muted small" }, "別の画面（ウィンドウ）が開きます。") : null,
      confirmCheck ? h("label", { class: "check warn-box" }, confirmCheck, h("span", {}, wf.confirm)) : null,
      carCheck ? h("label", { class: "check danger-box" }, carCheck, h("span", {}, h("b", {}, "車体が動きます。"), " 周囲に人・物・段差がなく、すぐ緊急停止できることを確認しました。")) : null,
    ],
    actions: (close) => [
      h("button", { class: "ghost", type: "button", onclick: close }, "キャンセル"),
      h("button", {
        type: "button",
        class: wf.moves_car ? "danger" : "",
        onclick: async (e) => {
          if (carCheck && !carCheck.checked) return toast("安全の確認にチェックを入れてください", "error");
          if (confirmCheck && !confirmCheck.checked) return toast("確認にチェックを入れてください", "error");
          e.target.disabled = true;
          try {
            const job = await api("/jobs", { method: "POST", body: { workflow: wf.id, args: current(), confirm_moves_car: !!carCheck?.checked, confirmed: !!confirmCheck?.checked } });
            close();
            toast(`${wf.title} を起動しました`);
            location.hash = `#/jobs/${job.id}`;
          } catch (err) {
            e.target.disabled = false;
            toast(err.message, "error");
          }
        },
      }, wf.moves_car ? "安全を確認して起動" : "起動"),
    ],
  });
}

/* ------------------------------------------------------------------ views: overview */

function card(title, sub, ...body) {
  return h("section", { class: "card" }, title ? h("div", { class: "card-head" }, h("h2", {}, title), sub ? h("span", { class: "sub" }, sub) : null) : null, body);
}
function stat(label, value, unit, foot) {
  return h("div", { class: "card stat" }, h("span", { class: "label" }, label), h("span", { class: "value" }, value, unit ? h("small", {}, unit) : null), foot ? h("span", { class: "foot" }, foot) : null);
}

function collectAlerts(c) {
  const alerts = [];
  for (const m of c.models) {
    for (const ch of m.checks) if (ch.level !== "ok") alerts.push({ level: ch.level, text: `${m.version}: ${ch.message}`, href: `#/models/${m.version}` });
    if (m.stage === "candidate") alerts.push({ level: "info", text: `${m.version} は採否が未判断です（候補）`, href: `#/models/${m.version}` });
    if (m.field.body_mismatch) alerts.push({ level: "warn", text: `${m.version}: 学習データと違う車体の版で走った走行が ${m.field.body_mismatch} 件`, href: `#/models/${m.version}?tab=走行・デプロイ` });
    if (m.field.fingerprint_mismatch) alerts.push({ level: "error", text: `${m.version}: 走行の記録と学習データのハッシュが違う走行が ${m.field.fingerprint_mismatch} 件`, href: `#/models/${m.version}` });
  }
  const inc = c.summary.incomplete_sessions || [];
  if (inc.length) alerts.push({ level: "warn", text: `ラベルのないセッション（記録が途中で止まった）が ${inc.length} 件: ${inc.map((x) => x.session_id).join(", ")}`, href: "#/data" });
  if (c.summary.unused_sessions.length) alerts.push({ level: "info", text: `どのモデルの学習にも使っていないセッションが ${c.summary.unused_sessions.length} 件`, href: "#/data" });
  const dep = c.summary.deployed, run = c.summary.last_pi_run;
  if (c.summary.body_error) alerts.push({ level: "error", text: `車体の版（configs/body.toml）を読めません: ${c.summary.body_error}`, href: "#/body" });
  else if (!(c.bodies || []).length) alerts.push({ level: "info", text: "車体の版がまだ登録されていません。どの車体で集めたデータかを残すため、車体の画面から登録してください", href: "#/body" });
  if (dep.version && run && run.model_version !== dep.version) alerts.push({ level: "warn", text: `デプロイの記録（${dep.version}）と Pi の最後の走行の版（${run.model_version}）が違います。Pi のモデル一覧で確かめてください`, href: "#/deploy" });
  if (dep.version && !c.models.find((m) => m.version === dep.version)?.usable) alerts.push({ level: "warn", text: `Pi で使用中の ${dep.version} は「有効」ではありません`, href: `#/models/${dep.version}` });
  return alerts;
}

/** 開発の流れ（MLOps のフェーズ）ごとの今の状態。押すとそのフェーズの画面へ */
function pipeline(c) {
  const sm = c.summary;
  const cvCount = c.cv.filter((x) => x.kind === "cross_validation").length;
  const values = {
    prepare: [sm.body_current || "未登録", "今の車体", `${(c.bodies || []).length} 版を登録`],
    collect: [fmtInt(sm.sessions), "セッション", `${fmtInt(sm.frames)} フレーム・${fmt(sm.hours * 60, 1)} 分`],
    curate: [c.sessions.filter((x) => x.annotation_hash).length, "件のラベル", `マニフェスト ${c.sessions.filter((x) => x.manifest_hash).length} / ${c.sessions.length}`],
    train: [sm.models, "モデル", sm.latest ? `最新 ${sm.latest}` : "まだない"],
    evaluate: [cvCount, "件の交差検証", `有効なモデル ${sm.usable_models}`],
    deploy: sm.deployed.version
      ? [sm.deployed.version, "Pi で使用中", sm.deployed.at ? `${fmtDate(sm.deployed.at)} にデプロイ` : ""]
      : [sm.last_pi_run?.model_version || "—", "Pi で使用中", sm.last_pi_run ? "走行記録から" : "デプロイの記録なし"],
    operate: [sm.runs, "回の走行", `自律 ${fmt(c.runs.reduce((a, r) => a + r.autonomous_s, 0) / 60, 1)} 分`],
  };
  if (!state.phases.length) return null;
  return h("section", { class: "card flow-card" },
    h("div", { class: "card-head" }, h("h2", {}, "開発の流れ"), h("span", { class: "sub" }, "光の流れている区間に記録があります。実走行の記録は下の輪からデータ収集へ戻ります")),
    flowDiagram({ counts: phaseCounts(c), details: values }));
}

async function viewOverview(root) {
  const c = await loadCatalog();
  const sm = c.summary;
  const latest = c.models.find((m) => m.version === sm.latest);
  const facts = h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", {}, "モデルの状態"), h("span", { class: "sub" }, "どれが最新か・有効か・Pi で使われているか")),
    h("div", { class: "facts" },
      fact("Pi で使用中（デプロイの記録）", sm.deployed.version, sm.deployed.at ? `${fmtDate(sm.deployed.at)} にデプロイ` : "Control Center からのデプロイの記録なし"),
      fact("Pi の最後の走行で使った版", sm.last_pi_run?.model_version, sm.last_pi_run ? `${fmtDate(sm.last_pi_run.started_at)} の走行` : "走行の記録なし"),
      fact("最新のモデル", sm.latest, latest ? `${fmtDate(latest.created_at)} 作成・${STAGES[latest.stage]}` : ""),
      fact("最新の有効なモデル", sm.latest_approved, sm.latest_approved ? "採用済みでトレーサビリティに誤りなし" : "「有効」にしたモデルがまだない")));
  const alerts = collectAlerts(c);
  const jobs = await api("/jobs");
  root.append(
    h("p", { class: "lead" }, "走行データからモデルを作り、評価し、Pi へデプロイして走らせるまでを一つの画面で管理します。各モデルがどのデータとコードから作られたかを記録からたどれます。"),
    h("div", { class: "stack" },
      pipeline(c),
      facts,
      h("div", { class: "grid cols-2" },
        card("確認すること", `${alerts.length} 件`, alerts.length
          ? h("div", { class: "stack", style: { gap: "8px" } }, alerts.slice(0, 12).map((a) => h("a", { class: `alert ${a.level === "info" ? "info" : a.level}`, href: a.href, style: { color: "var(--ink)", textDecoration: "none" } }, healthBadge(a.level === "info" ? "unknown" : a.level, a.level === "info" ? "情報" : null), h("span", {}, a.text))))
          : h("div", { class: "empty" }, "確認することはありません")),
        card("最近のジョブ", "", table([
          { key: "status", label: "状態", render: (j) => statusDot(j.status) },
          { key: "title", label: "ツール" },
          { key: "started_at", label: "開始", render: (j) => fmtDate(j.started_at) },
          { key: "dur", label: "時間", render: (j) => duration(j.started_at, j.ended_at), nosort: true },
        ], jobs.slice(0, 6), { onRow: (j) => (location.hash = `#/jobs/${j.id}`), empty: "まだジョブはありません。「ツール」から起動できます" }))),
      card("最近の走行", "Pi の自律走行の記録（data/runs）", runsTable(c.runs.slice(-6).reverse()))));
}
function fact(label, value, foot) {
  return h("div", { class: "fact" }, h("div", { class: "label" }, label),
    h("div", { class: "value" }, value ? h("a", { href: `#/models/${value}` }, value) : h("span", { class: "muted" }, "—")),
    h("div", { class: "foot" }, foot));
}

/* ------------------------------------------------------------------ views: models */

async function viewModels(root, _params, query) {
  const c = await loadCatalog();
  let filter = query.get("stage") || "all";
  const holder = h("div");
  const draw = () => {
    const rows = c.models.filter((m) => filter === "all" || m.stage === filter);
    holder.replaceChildren(table([
      { key: "version", label: "版", render: (m) => h("span", { class: "row", style: { gap: "6px" } }, h("b", { class: "mono" }, m.version), m.version === c.summary.latest ? h("span", { class: "badge tag" }, "最新") : null, m.deployed ? h("span", { class: "badge outline" }, "Pi で使用中") : null), sort: (m) => m.version },
      { key: "stage", label: "採否", render: (m) => stageBadge(m.stage) },
      { key: "health", label: "トレーサビリティ", render: (m) => healthBadge(m.health), sort: (m) => ({ error: 0, warn: 1, ok: 2 })[m.health] },
      { key: "created_at", label: "作成", render: (m) => fmtDate(m.created_at, true) },
      { key: "sessions", label: "学習データ", render: (m) => `${m.train_sessions.length + m.val_sessions.length} セッション / ${fmtInt(m.total_samples)}`, sort: (m) => m.total_samples },
      { key: "body", label: "車体", title: "学習データを集めた車体の版", render: (m) => (m.body_versions?.length ? h("span", { class: "row", style: { gap: "4px" } }, m.body_versions.map((b) => bodyBadge(b, m.body_source === "date" ? "date" : "recorded"))) : "—"), sort: (m) => (m.body_versions || []).join(",") },
      { key: "mae", label: "検証 MAE", num: true, title: "学習時の検証セッションでの舵角の誤差（度）", render: (m) => m.validation.steering_mae_deg !== undefined ? `±${fmt(m.validation.steering_mae_deg, 2)}°` : "—", sort: (m) => m.validation.steering_mae_deg },
      { key: "cv", label: "交差検証", num: true, render: (m) => m.cv.length || "—", sort: (m) => m.cv.length },
      { key: "field", label: "人の介入/10分", num: true, title: "Pi の自律走行での人の介入の回数（10分あたり）", render: (m) => fmt(m.field.human_per_10min, 1), sort: (m) => m.field.human_per_10min },
      { key: "commit", label: "コード", render: (m) => h("span", { class: "hash" }, short(m.code.commit, 7) + (m.code.dirty ? "*" : "")), sort: (m) => m.code.commit },
    ], rows, { onRow: (m) => (location.hash = `#/models/${m.version}`), sortKey: "version", desc: true, empty: "このステージのモデルはありません" }));
  };
  const seg = h("div", { class: "seg" }, [["all", "すべて"], ...Object.entries(STAGES)].map(([k, label]) => {
    const count = k === "all" ? c.models.length : c.models.filter((m) => m.stage === k).length;
    return h("button", { type: "button", class: filter === k ? "on" : "", onclick: (e) => { filter = k; seg.querySelectorAll("button").forEach((b) => b.classList.remove("on")); e.currentTarget.classList.add("on"); draw(); } }, `${label} ${count}`);
  }));
  draw();
  root.append(
    h("p", { class: "lead" }, "学習したモデルの一覧です。採否（候補・有効・却下・廃止）は models/registry.json に履歴付きで残ります。「有効」は、採用と判断していてトレーサビリティに誤りがないモデルです。"),
    h("section", { class: "card" },
      h("div", { class: "card-head" }, seg, h("span", { class: "spacer" }),
        h("button", { type: "button", class: "ghost small", onclick: () => (location.hash = "#/compare") }, "比較する")),
      holder));
}

async function viewModel(root, [version]) {
  const c = await loadCatalog();
  const m = c.models.find((x) => x.version === version);
  if (!m) { root.append(h("div", { class: "empty" }, `モデル ${version} が見つかりません`)); return; }
  setTitle(`モデル ${m.version}`);
  const head = h("section", { class: "card" },
    h("div", { class: "row", style: { gap: "12px", alignItems: "flex-start" } },
      h("div", { style: { flex: 1, minWidth: "240px" } },
        h("div", { class: "row", style: { gap: "8px" } }, h("h2", { class: "mono", style: { fontSize: "22px" } }, m.version), stageBadge(m.stage), healthBadge(m.health),
          m.version === c.summary.latest ? h("span", { class: "badge tag" }, "最新") : null,
          m.deployed ? h("span", { class: "badge outline" }, "Pi で使用中") : null),
        h("div", { class: "muted small", style: { marginTop: "4px" } }, `${fmtDate(m.created_at, true)} 作成${m.created_at_source === "file_time" ? "（ファイルの時刻から推定）" : ""}・${m.note ? `メモ: ${m.note}` : "採否のメモなし"}`)),
      h("div", { class: "row" },
        h("button", { type: "button", class: "ghost", onclick: () => stageDialog(m) }, "採否を変える"),
        h("button", { type: "button", class: "ghost", onclick: () => verifyDialog(m) }, "データを検証"),
        h("button", { type: "button", class: "ghost", onclick: () => launch("bundle", `--model_dir models/${m.version} --output dist/models`) }, "バンドル作成"),
        h("button", { type: "button", onclick: () => launch("deploy_push", `push --model_dir models/${m.version}`) }, "Pi へデプロイ…"))));
  const tabs = [
    ["概要", () => modelSummary(m, c)],
    ["トレーサビリティ", () => modelLineage(m, c)],
    ["学習", () => modelTraining(m)],
    ["評価", () => modelEval(m, c)],
    ["走行・デプロイ", () => modelField(m, c)],
    ["環境", () => modelEnv(m)],
  ];
  root.append(h("div", { class: "stack" }, head, tabbed(tabs, new URLSearchParams(location.hash.split("?")[1] || "").get("tab"))));
}

function tabbed(tabs, initial) {
  const body = h("div");
  const bar = h("div", { class: "tabs", role: "tablist" });
  const show = (i) => {
    bar.querySelectorAll("button").forEach((b, j) => { b.classList.toggle("on", i === j); b.setAttribute("aria-selected", String(i === j)); });
    body.replaceChildren(tabs[i][1]());
  };
  tabs.forEach(([name], i) => bar.append(h("button", { type: "button", role: "tab", onclick: () => show(i) }, name)));
  const start = Math.max(0, tabs.findIndex(([n]) => n === initial));
  show(start);
  return h("div", {}, bar, body);
}

function checksList(checks) {
  return h("ul", { class: "checks" }, checks.map((ch) => h("li", {}, healthBadge(ch.level, " "), h("span", {}, ch.message))));
}

function modelSummary(m, c) {
  const v = m.validation;
  const hist = m.history || [];
  return h("div", { class: "grid cols-2" },
    card("トレーサビリティと成果物の確認", "", checksList(m.checks)),
    card("要点", "", h("dl", { class: "kv" },
      h("dt", {}, "検証の誤差"), h("dd", {}, v.steering_mae !== undefined ? `MAE ${fmt(v.steering_mae, 4)}（±${fmt(v.steering_mae_deg, 2)}°）` : "—"),
      h("dt", {}, "最良の検証損失"), h("dd", {}, fmt(m.best_val_loss, 4)),
      h("dt", {}, "学習データ"), h("dd", {}, `学習 ${m.train_sessions.length} / 検証 ${m.val_sessions.length} セッション、${fmtInt(m.train_samples)} / ${fmtInt(m.val_samples)} サンプル`),
      h("dt", {}, "エポック"), h("dd", {}, `${hist.length || m.params.epochs || "—"}`),
      h("dt", {}, "推論時間（PC）"), h("dd", {}, m.inference_ms_pc ? `${fmt(m.inference_ms_pc, 2)} ms` : "—"),
      h("dt", {}, "前方が塞がっているかの出力"), h("dd", {}, m.blocked_trained ? "学習済み" : "学習していない（使わない）"),
      h("dt", {}, "交差検証"), h("dd", {}, m.cv.length ? m.cv.join(", ") : "同じデータの交差検証なし"),
      h("dt", {}, "Pi での走行"), h("dd", {}, m.field.runs ? `${m.field.runs} 回・自律 ${fmt(m.field.autonomous_min, 1)} 分・人の介入 ${fmt(m.field.human_per_10min, 1)} 回/10分` : "まだ走っていない"))),
    card("採否の履歴", "models/registry.json", stageTimeline(m)));
}
function stageTimeline(m) {
  const hist = m.stage_history || [];
  return hist.length
    ? h("ul", { class: "timeline" }, [...hist].reverse().map((e) => h("li", {}, h("div", {}, `${STAGES[e.from] || e.from} → `, stageBadge(e.to)), e.note ? h("div", { class: "ink-2" }, e.note) : null, h("div", { class: "when" }, fmtDate(e.at, true)))))
    : h("div", { class: "empty" }, "まだ採否を判断していません");
}

function modelLineage(m, c) {
  const byId = Object.fromEntries(c.sessions.map((x) => [x.session_id, x]));
  const recorded = Object.fromEntries(m.data_sessions.map((x) => [x.session_id, x]));
  const rows = [...m.train_sessions.map((id) => ({ id, role: "学習" })), ...m.val_sessions.map((id) => ({ id, role: "検証" }))].map((r) => {
    const cur = byId[r.id];
    const rec = recorded[r.id] || {};
    let level = "unknown", text = "記録なし";
    if (!cur) { level = "warn"; text = "この PC にない"; }
    else if (rec.manifest_hash && cur.manifest_hash === rec.manifest_hash && rec.manifest_ok !== false) { level = "ok"; text = "学習時のまま"; }
    else if (rec.manifest_hash) { level = "error"; text = "学習後に変わった"; }
    return { ...r, cur, rec, level, text };
  });
  const job = m.created_by;
  const filters = Object.entries(m.data_filters || {}).filter(([, v]) => typeof v === "number");
  const FILTER_LABEL = {
    skipped_motion_mismatch: "指令と動きが食い違い、外した", skipped_stopped: "止まっていたので外した", kept_for_blocked_only: "塞がりの学習だけに使った",
    relabeled_turn_intent: "曲がりの意図を付け直した", stuck_blocked: "引っかかりを塞がりとした", stuck_unknown: "引っかかりの近くで不明にした",
  };
  return h("div", { class: "stack" },
    h("div", { class: "grid cols-2" },
      card("どのように作られたか", "", h("dl", { class: "kv" },
        h("dt", {}, "コマンド"), h("dd", {}, job ? h("code", {}, job.command) : h("span", { class: "muted" }, "Control Center の外で学習（コマンドの記録なし）。パラメーターは下の通り")),
        job ? [h("dt", {}, "ジョブ"), h("dd", {}, h("a", { href: `#/jobs/${job.id}` }, job.id), " ", statusDot(job.status))] : null,
        h("dt", {}, "コード"), h("dd", {}, m.code.commit ? h("span", { class: "hash" }, m.code.commit) : "—", m.code.dirty ? h("span", { class: "badge confirm-badge", style: { marginLeft: "6px" } }, "未コミットの変更あり") : null),
        h("dt", {}, "学習データのハッシュ"), h("dd", {}, h("span", { class: "hash", title: "学習に使ったサンプル（画像の中身・舵角・セッション）のハッシュ" }, m.data_fingerprint || "—")),
        h("dt", {}, "車体の版"), h("dd", {}, m.body_versions?.length ? m.body_versions.map((b) => bodyBadge(b, m.body_source === "date" ? "date" : "recorded")) : "—",
          h("span", { class: "muted small", style: { marginLeft: "6px" } }, m.body_source === "metadata" ? "学習の時に記録" : "セッションの日時から決めた")),
        h("dt", {}, "ONNX の SHA256"), h("dd", {}, h("span", { class: "hash" }, m.onnx_sha256 || "—")),
        h("dt", {}, "成果物"), h("dd", {}, Object.entries(m.artifacts).map(([k, v]) => h("span", { class: `badge ${v ? "outline" : "stage-rejected"}`, style: { marginRight: "4px" } }, `${v ? "✓" : "×"} ${k}`))))),
      card("パラメーター", "metadata.json の params", h("dl", { class: "kv" }, Object.entries(m.params).flatMap(([k, v]) => [h("dt", { class: "mono" }, k), h("dd", { class: "mono" }, JSON.stringify(v))])))),
    card("どのデータから作られたか", `${rows.length} セッション・マニフェストのハッシュで照合`, table([
      { key: "id", label: "セッション", render: (r) => h("a", { class: "mono", href: `#/data/${r.id}` }, r.id) },
      { key: "role", label: "用途" },
      { key: "frames", label: "フレーム", num: true, render: (r) => fmtInt(r.cur?.frames), sort: (r) => r.cur?.frames },
      { key: "collector", label: "収集", render: (r) => r.cur?.collector || "—" },
      { key: "body", label: "車体", render: (r) => bodyBadge(m.body_record?.sessions?.[r.id] || r.cur?.body_version, m.body_record?.sessions?.[r.id] ? "recorded" : r.cur?.body_source, r.cur?.body_changed), sort: (r) => r.cur?.body_version },
      { key: "hash", label: "マニフェスト（学習時）", render: (r) => h("span", { class: "hash" }, short(r.rec.manifest_hash, 12)) },
      { key: "level", label: "今のデータ", render: (r) => healthBadge(r.level, r.text) },
      { key: "ann", label: "ラベル", render: (r) => (m.annotations[r.id] ? h("span", { class: "badge tag" }, short(m.annotations[r.id])) : "—"), nosort: true },
    ], rows)),
    filters.length ? card("学習データの絞り込み", "学習の時に外した・付け直したフレームの数", hbars(filters.map(([k, v]) => ({ name: FILTER_LABEL[k] || k, value: v, text: fmtInt(v) })), { title: "フレーム数" })) : null,
    h("div", {}, h("a", { href: `#/lineage?focus=${encodeURIComponent(`model:${m.version}`)}` }, "トレーサビリティの図でこのモデルを見る →")));
}

function modelTraining(m) {
  const hist = m.history || [];
  if (!hist.length) return h("div", { class: "empty" }, "学習の履歴がありません");
  const best = hist.reduce((a, b) => (b.val_loss < a.val_loss ? b : a), hist[0]);
  return h("div", { class: "stack" },
    card("損失の推移", `最良の検証損失 ${fmt(best.val_loss, 4)}（エポック ${best.epoch}、この時点の重みを保存）`, lineChart({
      series: [
        { name: "学習", color: SERIES[0], points: hist.map((e) => [e.epoch, e.train_loss]) },
        { name: "検証", color: SERIES[1], points: hist.map((e) => [e.epoch, e.val_loss]) },
      ],
      xLabel: "エポック", yLabel: "損失", marker: { x: best.epoch, label: "最良" }, yFormat: (v) => fmt(v, 3),
    })),
    card("エポックごとの値", "", table([
      { key: "epoch", label: "エポック", num: true },
      { key: "train_loss", label: "学習の損失", num: true, render: (e) => fmt(e.train_loss, 4) },
      { key: "val_loss", label: "検証の損失", num: true, render: (e) => fmt(e.val_loss, 4) },
    ], hist, { rowClass: (e) => (e.epoch === best.epoch ? "selected" : "") })));
}

function modelEval(m, c) {
  const v = m.validation;
  const dirs = [["left", "左"], ["straight", "直進"], ["right", "右"]];
  const intents = [["any", "おまかせ"], ["left", "左へ"], ["right", "右へ"]];
  const cvs = c.cv.filter((x) => m.cv.includes(x.name));
  return h("div", { class: "stack" },
    h("div", { class: "grid cols-2" },
      card("向き別の誤差（検証）", "舵角の MAE（-1〜1）。小さいほどよい", v.steering_mae_by_direction
        ? hbars(dirs.map(([k, label]) => ({ name: label, value: v.steering_mae_by_direction[k], extra: [[null, "サンプル", fmtInt(v.samples_by_direction?.[k])]] })), { title: "向き別の MAE" })
        : h("div", { class: "empty" }, "記録なし")),
      card("行きたい向き（意図）別の誤差", "", v.steering_mae_by_intent
        ? hbars(intents.map(([k, label]) => ({ name: label, value: v.steering_mae_by_intent[k], extra: [[null, "サンプル", fmtInt(v.samples_by_intent?.[k])]] })), { title: "意図別の MAE" })
        : h("div", { class: "empty" }, "記録なし"))),
    card("交差検証（同じ学習データ）", "セッションを1つずつ抜いて評価。偏りのない比較に使う", cvs.length
      ? h("div", { class: "stack" }, cvs.map((x) => h("div", {},
        h("div", { class: "row", style: { marginBottom: "6px" } }, h("b", { class: "mono" }, x.name), h("span", { class: "muted small" }, `${x.method}・${fmtDate(x.modified_at, true)}`)),
        hbars([
          { name: "モデル", value: x.overall.steering_mae, color: SERIES[0] },
          { name: "意図の平均", value: x.baselines.intent_mean?.steering_mae, color: "var(--div-mid)" },
          { name: "常に直進", value: x.baselines.zero?.steering_mae, color: "var(--div-mid)" },
        ].filter((r) => r.value !== undefined), { title: "交差検証の MAE（基準と比べる）" }))))
      : h("div", { class: "empty" }, "このモデルと同じ学習データの交差検証はありません")),
    m.evaluation ? card("オフライン評価（eval_metrics.json）", "", h("dl", { class: "kv" }, Object.entries(m.evaluation).flatMap(([k, val]) => [h("dt", { class: "mono" }, k), h("dd", {}, typeof val === "number" ? fmt(val, 4) : JSON.stringify(val))]))) : null);
}

function modelField(m, c) {
  const runs = c.runs.filter((r) => r.model_version === m.version);
  return h("div", { class: "stack" },
    h("div", { class: "grid cols-4" },
      stat("走行", m.field.runs, "回"),
      stat("自律の時間", fmt(m.field.autonomous_min, 1), "分"),
      stat("人の介入", m.field.human, "回", `${fmt(m.field.human_per_10min, 1)} 回/10分`),
      stat("安全の介入", m.field.safety, "回", "障害物・段差・引っかかり")),
    card("デプロイの履歴", "data/deployments.jsonl", m.deployments.length
      ? h("ul", { class: "timeline" }, [...m.deployments].reverse().map((d) => h("li", {}, h("div", {}, `${{ push: "デプロイして使用中に", install: "デプロイのみ", rollback: "戻した" }[d.action] || d.action} `, healthBadge(d.ok ? "ok" : "error", d.ok ? "成功" : "失敗")), h("div", { class: "when" }, fmtDate(d.at, true)))))
      : h("div", { class: "empty" }, "Control Center からのデプロイの記録はありません")),
    card("このモデルの走行", "", runsTable(runs.reverse())));
}

function modelEnv(m) {
  return h("div", { class: "grid cols-2" },
    card("ソフトウェア", "", h("dl", { class: "kv" }, Object.entries(m.environment).flatMap(([k, v]) => [h("dt", { class: "mono" }, k), h("dd", { class: "mono" }, v ?? "—")]))),
    card("前処理（Pi と同じにする値）", "", h("dl", { class: "kv" }, Object.entries(m.preprocessing).flatMap(([k, v]) => [h("dt", { class: "mono" }, k), h("dd", { class: "mono" }, JSON.stringify(v))]))));
}

function stageDialog(m) {
  let chosen = m.stage === "approved" ? "archived" : "approved";
  const options = h("div", { class: "stack", style: { gap: "6px" } }, Object.entries(STAGES).map(([k, label]) => {
    const input = h("input", { type: "radio", name: "stage", value: k, checked: k === chosen, onchange: () => (chosen = k) });
    return h("label", { class: "check" }, input, h("span", {}, stageBadge(k), " ", h("span", { class: "ink-2 small" }, STAGE_HELP[k])));
  }));
  const note = h("textarea", { rows: 3, placeholder: "判断の理由（例: 交差検証で v005 より MAE が 0.02 小さい、走行で介入が減った）" });
  const errors = m.checks.filter((x) => x.level === "error");
  modal({
    title: `${m.version} の採否`,
    body: [
      h("div", {}, "今: ", stageBadge(m.stage)),
      options,
      h("label", { class: "field" }, h("span", {}, "理由（履歴に残ります）"), note),
      errors.length ? h("div", { class: "warn-box small" }, "トレーサビリティの確認に誤りがあります。「有効」にしても、直すまでは有効なモデルとして扱いません: ", errors.map((e) => e.message).join(" / ")) : null,
    ],
    actions: (close) => [
      h("button", { class: "ghost", type: "button", onclick: close }, "キャンセル"),
      h("button", { type: "button", onclick: async () => {
        if ((chosen === "approved" || chosen === "rejected") && !note.value.trim()) return toast("有効・却下にする時は理由を書いてください", "error");
        try {
          await api(`/models/${encodeURIComponent(m.version)}/stage`, { method: "POST", body: { stage: chosen, note: note.value.trim() } });
          close();
          toast(`${m.version} を「${STAGES[chosen]}」にしました`);
          await loadCatalog(true);
          render();
        } catch (err) { toast(err.message, "error"); }
      } }, "保存"),
    ],
  });
}

function verifyDialog(m) {
  const body = h("div", { class: "loading" }, h("span", { class: "spinner" }), "学習データの全ファイルのハッシュを確かめています…");
  modal({ title: `${m.version} の学習データを検証`, body, actions: (close) => [h("button", { type: "button", onclick: close }, "閉じる")] });
  api(`/models/${encodeURIComponent(m.version)}/verify`, { method: "POST", body: {} }).then((res) => {
    const bad = res.sessions.filter((x) => x.ok !== true);
    body.replaceWith(h("div", { class: "stack", style: { gap: "10px" } },
      bad.length ? h("div", { class: "alert error" }, healthBadge("error"), `${bad.length} セッションがマニフェストと合いません`) : h("div", { class: "alert info" }, healthBadge("ok"), "すべてのセッションがマニフェストと一致しました（学習時のデータのまま）"),
      table([
        { key: "session_id", label: "セッション", render: (x) => h("span", { class: "mono" }, x.session_id) },
        { key: "ok", label: "結果", render: (x) => healthBadge(x.ok === true ? "ok" : x.ok === false ? "error" : "warn", x.ok === true ? "一致" : x.ok === false ? "不一致" : "なし") },
        { key: "problems", label: "内容", render: (x) => x.problems.slice(0, 3).join(", ") || "—", nosort: true },
      ], res.sessions)));
  }).catch((err) => body.replaceWith(h("div", { class: "alert error" }, err.message)));
}

/* ------------------------------------------------------------------ views: compare */

async function viewCompare(root, _p, query) {
  const c = await loadCatalog();
  const picked = new Set((query.get("m") || c.models.slice(-3).map((m) => m.version).join(",")).split(",").filter(Boolean));
  const holder = h("div", { class: "stack" });
  const chooser = h("div", { class: "row" }, c.models.map((m) => h("label", { class: "check pill", style: { cursor: "pointer" } },
    h("input", { type: "checkbox", checked: picked.has(m.version), onchange: (e) => { e.target.checked ? picked.add(m.version) : picked.delete(m.version); draw(); } }),
    h("span", { class: "mono" }, m.version), stageBadge(m.stage))));
  const draw = () => {
    const ms = c.models.filter((m) => picked.has(m.version));
    if (!ms.length) { holder.replaceChildren(h("div", { class: "empty" }, "比べるモデルを選んでください")); return; }
    const paramKeys = [...new Set(ms.flatMap((m) => Object.keys(m.params)))].sort();
    const differs = (k) => new Set(ms.map((m) => JSON.stringify(m.params[k]))).size > 1;
    const rowsDef = [
      ["採否", (m) => stageBadge(m.stage)],
      ["トレーサビリティ", (m) => healthBadge(m.health)],
      ["作成", (m) => fmtDate(m.created_at, true)],
      ["検証 MAE（度）", (m) => `±${fmt(m.validation.steering_mae_deg, 2)}°`, (m) => m.validation.steering_mae_deg],
      ["最良の検証損失", (m) => fmt(m.best_val_loss, 4), (m) => m.best_val_loss],
      ["学習サンプル", (m) => fmtInt(m.train_samples)],
      ["検証セッション", (m) => m.val_sessions.join(", ") || "—"],
      ["学習データのハッシュ", (m) => h("span", { class: "hash" }, short(m.data_fingerprint))],
      ["車体の版", (m) => (m.body_versions?.length ? m.body_versions.join(", ") : "—"), null, new Set(ms.map((m) => (m.body_versions || []).join(","))).size > 1],
      ["コード", (m) => h("span", { class: "hash" }, short(m.code.commit, 7) + (m.code.dirty ? "*" : ""))],
      ["人の介入/10分", (m) => fmt(m.field.human_per_10min, 1), (m) => m.field.human_per_10min],
      ...paramKeys.map((k) => [`params.${k}`, (m) => h("span", { class: "mono", style: differs(k) ? { fontWeight: 700, color: "var(--accent-ink)" } : {} }, JSON.stringify(m.params[k] ?? null)), null, differs(k)]),
    ];
    const best = (fn) => { const vals = ms.map(fn).filter((x) => x !== null && x !== undefined); return vals.length > 1 ? Math.min(...vals) : null; };
    const grid = h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "項目"), ms.map((m) => h("th", {}, h("a", { class: "mono", href: `#/models/${m.version}` }, m.version))))),
      h("tbody", {}, rowsDef.map(([label, render, value, diff]) => {
        const b = value ? best(value) : null;
        return h("tr", {}, h("td", { class: diff ? "" : "ink-2", style: diff ? { fontWeight: 600 } : {} }, label, diff ? h("span", { class: "badge tag", style: { marginLeft: "6px" } }, "違い") : null),
          ms.map((m) => h("td", { style: b !== null && value(m) === b ? { fontWeight: 700 } : {} }, render(m), b !== null && value(m) === b ? " ◀" : null)));
      }))));
    const curves = ms.length <= 3 && ms.some((m) => (m.history || []).length)
      ? card("検証の損失の推移", "", lineChart({ series: ms.map((m, i) => ({ name: m.version, color: SERIES[i], points: (m.history || []).map((e) => [e.epoch, e.val_loss]) })), xLabel: "エポック", yLabel: "検証の損失" }))
      : ms.length > 3 ? card("検証の損失の推移", "", h("div", { class: "empty" }, "色で見分けられるよう、損失の図は3つまでのモデルで表示します")) : null;
    holder.replaceChildren(card("モデルの比較", "太字 ◀ は最も小さい（よい）値。違うパラメーターに印", grid), curves);
  };
  draw();
  const cvRows = c.cv.filter((x) => x.kind === "cross_validation");
  const cmpRows = c.cv.filter((x) => x.kind === "comparison");
  // compare_cv の判定（ADOPT / KEEP / NOT_COMPARABLE と同じ値）
  const VERDICT = { adopt: ["ok", "採用してよい"], keep_current: ["error", "今の設定のまま（採用しない）"], not_comparable: ["warn", "比べられない"] };
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "モデルの値とパラメーターを並べて比べます。偏りのない比較には、同じデータの交差検証の結果を使ってください。"),
    card("比べるモデル", "", chooser),
    holder,
    card("交差検証の結果", "data/cv（同じ学習データのハッシュのモデルにつながる）", table([
      { key: "name", label: "名前", render: (x) => h("b", { class: "mono" }, x.name) },
      { key: "modified_at", label: "日時", render: (x) => fmtDate(x.modified_at, true) },
      { key: "mae", label: "MAE（度）", num: true, render: (x) => `±${fmt(x.overall.steering_mae_deg, 2)}°`, sort: (x) => x.overall.steering_mae_deg },
      { key: "intent_mean", label: "基準: 意図の平均", num: true, render: (x) => `±${fmt(x.baselines.intent_mean?.steering_mae_deg, 2)}°`, sort: (x) => x.baselines.intent_mean?.steering_mae_deg },
      { key: "folds", label: "抜いたセッション", num: true, render: (x) => x.folds.length },
      { key: "params", label: "主な設定", nosort: true, render: (x) => ["turn_intent", "flip_augment", "blocked_loss_weight", "blocked_stuck"].filter((k) => k in x.params).map((k) => `${k}=${JSON.stringify(x.params[k])}`).join(" ") },
      { key: "models", label: "モデル", nosort: true, render: (x) => c.models.filter((m) => m.data_fingerprint && m.data_fingerprint === x.data_fingerprint).map((m) => h("a", { class: "badge tag", href: `#/models/${m.version}`, style: { marginRight: "4px" } }, m.version)) },
    ], cvRows, { sortKey: "modified_at", desc: true, empty: "交差検証の結果はありません（ツール → 交差検証）" })),
    card("採用の判定（交差検証の比較）", "src.e2e.evaluation.compare_cv の結果", table([
      { key: "name", label: "名前", render: (x) => h("b", { class: "mono" }, x.name) },
      { key: "verdict", label: "判定", render: (x) => { const [lv, t] = VERDICT[x.verdict] || ["unknown", x.verdict || "—"]; return healthBadge(lv, t); } },
      { key: "files", label: "比べたもの", nosort: true, render: (x) => Object.entries(x.files || {}).map(([k, v]) => `${k}: ${String(v).split(/[\\/]/).pop()}`).join(" / ") },
      { key: "reasons", label: "理由", nosort: true, render: (x) => (x.reasons || []).join(" / ") || "—" },
    ], cmpRows, { empty: "比較の結果はありません" }))));
}

/* ------------------------------------------------------------------ views: data */

async function viewData(root) {
  const c = await loadCatalog();
  const inc = c.summary.incomplete_sessions || [];
  const total = c.sessions.reduce((a, x) => a + x.frames, 0);
  const avg = (k) => c.sessions.reduce((a, x) => a + (x.distribution[k] || 0) * x.frames, 0) / (total || 1);
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "走行データのセッションです。マニフェスト（全ファイルの SHA256）で、学習の後にデータが変わっていないかを確かめられます。"),
    h("div", { class: "grid cols-4" },
      stat("セッション", c.sessions.length, "件", inc.length ? `ほかにラベルのないもの ${inc.length} 件` : ""),
      stat("フレーム", fmtInt(total), "", `${fmt(c.summary.hours * 60, 1)} 分`),
      stat("マニフェストあり", c.sessions.filter((x) => x.manifest_hash).length, `/ ${c.sessions.length}`, "ないものは上の「マニフェスト作成・検証」で作る"),
      h("div", { class: "card stat" }, h("span", { class: "label" }, "全体の舵角の分布"), h("div", { style: { margin: "10px 0 4px" } }, distBar({ left_ratio: avg("left_ratio"), straight_ratio: avg("straight_ratio"), right_ratio: avg("right_ratio") })), distLegend())),
    inc.length ? h("div", { class: "alert warn" }, healthBadge("warn"), h("span", {}, `ラベル（labels.jsonl）のないセッションは学習に使われません: ${inc.map((x) => `${x.session_id}（画像 ${x.images}）`).join("、")}`)) : null,
    card("セッション", "行を押すと画像と詳細", table([
      { key: "session_id", label: "セッション", render: (x) => h("b", { class: "mono" }, x.session_id) },
      { key: "collector", label: "収集", render: (x) => x.collector || "—" },
      { key: "body_version", label: "車体", title: "このデータを集めた車体の版", render: (x) => bodyBadge(x.body_version, x.body_source, x.body_changed) },
      { key: "frames", label: "フレーム", num: true, render: (x) => fmtInt(x.frames) },
      { key: "duration_sec", label: "時間", num: true, render: (x) => `${fmt(x.duration_sec, 0)} 秒` },
      { key: "dist", label: "舵角の分布（左・直進・右）", nosort: true, render: (x) => distBar(x.distribution) },
      { key: "manifest_hash", label: "マニフェスト", render: (x) => (x.manifest_hash ? h("span", { class: "hash" }, short(x.manifest_hash)) : healthBadge("warn", "なし")) },
      { key: "annotation_hash", label: "ラベル", render: (x) => (x.annotation_hash ? h("span", { class: "badge tag" }, "あり") : "—") },
      { key: "used_by", label: "使ったモデル", render: (x) => (x.used_by.length ? x.used_by.map((v) => h("span", { class: "badge outline", style: { marginRight: "4px" } }, v)) : h("span", { class: "muted" }, "未使用")), sort: (x) => x.used_by.length },
    ], c.sessions, { onRow: (x) => (location.hash = `#/data/${x.session_id}`), sortKey: "session_id", desc: true }),
    h("div", { style: { marginTop: "10px" } }, distLegend()))));
}

async function viewSession(root, [sid]) {
  const c = await loadCatalog();
  const x = c.sessions.find((y) => y.session_id === sid);
  if (!x) { root.append(h("div", { class: "empty" }, `セッション ${sid} が見つかりません`)); return; }
  setTitle(`セッション ${sid}`);
  const n = Math.min(24, x.frames);
  const idx = [...new Set(Array.from({ length: n }, (_, i) => Math.round((i * (x.frames - 1)) / Math.max(1, n - 1))))];
  root.append(h("div", { class: "stack" },
    h("div", { class: "grid cols-2" },
      card("概要", "", h("dl", { class: "kv" },
        h("dt", {}, "フォルダ"), h("dd", { class: "mono" }, x.path),
        h("dt", {}, "収集の方法"), h("dd", {}, x.collector || "—"),
        h("dt", {}, "車体の版"), h("dd", {}, bodyBadge(x.body_version, x.body_source, x.body_changed), x.body_source ? h("span", { class: "muted small", style: { marginLeft: "6px" } }, BODY_SOURCE[x.body_source]) : null),
        h("dt", {}, "フレーム"), h("dd", {}, `${fmtInt(x.frames)}（${fmt(x.duration_sec, 1)} 秒・${fmt(x.fps, 1)} fps）`),
        h("dt", {}, "舵角の分布"), h("dd", {}, distBar(x.distribution), h("div", { class: "small muted", style: { marginTop: "4px" } }, `左 ${pct(x.distribution.left_ratio)}・直進 ${pct(x.distribution.straight_ratio)}・右 ${pct(x.distribution.right_ratio)}`)),
        h("dt", {}, "マニフェスト"), h("dd", {}, x.manifest_hash ? h("span", { class: "hash" }, x.manifest_hash) : healthBadge("warn", "なし"), x.manifest_created_at ? h("div", { class: "small muted" }, `${fmtDate(x.manifest_created_at, true)} 作成`) : null),
        h("dt", {}, "手作業のラベル"), h("dd", {}, x.annotation_hash ? h("span", { class: "hash" }, x.annotation_hash) : "なし"))),
      card("このデータで学習したモデル", "", x.used_by.length
        ? h("div", { class: "stack", style: { gap: "8px" } }, x.used_by.map((v) => { const m = c.models.find((mm) => mm.version === v); return h("a", { href: `#/models/${v}`, class: "row", style: { color: "var(--ink)" } }, h("b", { class: "mono" }, v), stageBadge(m.stage), h("span", { class: "muted small" }, m.val_sessions.includes(sid) ? "検証に使用" : "学習に使用")); }))
        : h("div", { class: "empty" }, "まだどのモデルの学習にも使っていません"))),
    card("画像（等間隔に抜き出し）", `${idx.length} 枚 / ${fmtInt(x.frames)}`, h("div", { class: "frames" }, idx.map((i) => h("figure", {},
      h("img", { src: frameUrl(sid, i), loading: "lazy", alt: `frame ${i}`, onerror: (e) => e.target.closest("figure").remove() }),
      h("figcaption", {}, `#${i}`)))))));
}

/* ------------------------------------------------------------------ views: lineage */

const KIND_LABEL = { session: "走行データ", annotation: "手作業のラベル", model: "モデル", cv: "交差検証", field: "Pi での走行" };
const COLUMN_TITLES = ["走行データ", "手作業のラベル", "モデル", "評価・走行"];
// 節の種類ごとの色（開発の流れのフェーズと同じ色）
const KIND_PHASE = { session: "collect", annotation: "curate", model: "train", cv: "evaluate", field: "operate" };
const kindColor = (kind) => phaseById(KIND_PHASE[kind])?.color || "var(--axis)";
async function viewLineage(root, _p, query) {
  const c = await loadCatalog();
  const { nodes, edges } = c.graph;
  if (!nodes.length) { root.append(h("div", { class: "empty" }, "まだたどれる記録がありません")); return; }
  const cols = [0, 1, 2, 3].filter((i) => nodes.some((n) => n.column === i));
  const NW = 200, NH = 42, GAP = 10, CG = 110, TOP = 40;
  const pos = {};
  const counts = Object.fromEntries(cols.map((col) => [col, nodes.filter((n) => n.column === col).length]));
  const most = Math.max(...Object.values(counts));
  cols.forEach((col, ci) => {
    const list = nodes.filter((n) => n.column === col);
    const offset = ((most - list.length) * (NH + GAP)) / 2;
    list.forEach((n, i) => {
      // ラベルは元のセッションの横に置く（線が他の節の上を通らないように）
      const src = col === 1 ? pos[`session:${n.id.split(/:(.+)/)[1]}`] : null;
      pos[n.id] = { x: 20 + ci * (NW + CG), y: src ? src.y : TOP + offset + i * (NH + GAP) };
    });
  });
  const W = 20 + cols.length * (NW + CG) - CG + 20;
  const H = TOP + most * (NH + GAP) + 10;
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "データとモデルのトレーサビリティの図" });
  const COLUMN_KIND = ["session", "annotation", "model", "cv"];
  cols.forEach((col, ci) => svg.append(
    s("circle", { cx: 24 + ci * (NW + CG), cy: 18, r: 4, fill: kindColor(COLUMN_KIND[col]) }),
    s("text", { class: "lin-col-title", x: 34 + ci * (NW + CG), y: 22 }, COLUMN_TITLES[col])));
  // つながりだけに線を引き、元から先へ光の粒を流す（色は元の種類の色から先の種類の色へ）
  const kindOf = Object.fromEntries(nodes.map((n) => [n.id, n.kind]));
  const defs = s("defs");
  svg.append(defs);
  const edgeEls = edges.map((e, i) => {
    const a = pos[e.from], b = pos[e.to];
    if (!a || !b) return null;
    const x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2;
    // 列を飛ばすつながり（走行データ → モデル）は、間の列を自分の行のまま横切り、最後の隙間で曲がる。
    // 間の列の節（ほかのセッションのラベル）の後ろを通って、つながっているように見えないようにする
    const xa = b.x - CG > x1 + 1 ? b.x - CG : x1;
    // 間の列の、同じ行に節がある時（そのセッションのラベル）は、節のすぐ下の隙間を通る
    const blocked = xa > x1 && Object.entries(pos).some(([id, q]) => q.x > a.x && q.x < b.x && Math.abs(q.y - a.y) < NH && id !== e.from && id !== e.to);
    const yr = blocked ? a.y + NH + GAP / 2 : y1;
    const mx = (xa + x2) / 2;
    const d = blocked
      ? `M${x1},${y1} C${x1 + 24},${y1} ${x1 + 24},${yr} ${x1 + 48},${yr} H${xa} C${mx},${yr} ${mx},${y2} ${x2},${y2}`
      : `M${x1},${y1} H${xa} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    const gid = `lin-grad-${i}`;
    defs.append(s("linearGradient", { id: gid, gradientUnits: "userSpaceOnUse", x1, x2, y1: 0, y2: 0 },
      s("stop", { offset: 0, "stop-color": kindColor(kindOf[e.from]) }),
      s("stop", { offset: 1, "stop-color": kindColor(kindOf[e.to]) })));
    const el = s("g", { class: `lin-link ${e.kind}` },
      s("path", { class: `lin-edge ${e.kind}`, d }),
      s("path", { class: "lin-flow", d, stroke: `url(#${gid})` }));
    el.dataset.from = e.from; el.dataset.to = e.to;
    svg.append(el);
    return el;
  }).filter(Boolean);
  const linked = new Set(edgeEls.flatMap((el) => [el.dataset.from, el.dataset.to]));
  const STAGE_COLOR = { approved: "var(--good)", rejected: "var(--bad)", candidate: "var(--muted)", archived: "var(--axis)" };
  const nodeEls = {};
  nodes.forEach((n) => {
    const p = pos[n.id];
    const lone = !linked.has(n.id);
    const sub = lone && n.kind === "session" ? `${fmtInt(n.frames)} フレーム・まだ使われていない`
      : n.kind === "session" ? (n.missing ? "この PC にない" : `${fmtInt(n.frames)} フレーム${n.body ? `・車体 ${n.body}` : ""}`)
      : n.kind === "model" ? `${STAGES[n.stage]}・${HEALTH_TEXT[n.health]}`
      : n.kind === "field" ? `${n.runs} 回・${n.autonomous_min} 分・人の介入 ${n.human}` : KIND_LABEL[n.kind];
    const g = s("g", { class: `lin-node ${n.kind} ${n.missing ? "missing" : ""} ${lone ? "lone" : ""}`, transform: `translate(${p.x},${p.y})`, tabindex: 0, style: `--kind:${kindColor(n.kind)}` },
      s("rect", { width: NW, height: NH, rx: 7 }),
      n.kind !== "model" ? s("rect", { class: "kind-bar", width: 3, height: NH - 14, x: 4, y: 7, rx: 1.5 }) : null,
      n.kind === "model" ? s("rect", { class: "stripe", width: 4, height: NH - 12, x: 6, y: 6, rx: 2, style: `fill:${STAGE_COLOR[n.stage]};stroke:none` }) : null,
      s("text", { class: "t", x: n.kind === "model" ? 16 : 10, y: 17 }, n.kind === "annotation" ? n.label.replace("ラベル ", "") : n.kind === "field" ? "Pi での走行" : n.label),
      s("text", { class: "k", x: n.kind === "model" ? 16 : 10, y: 32 }, sub));
    g.addEventListener("click", () => focus(n.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter") focus(n.id); });
    nodeEls[n.id] = g;
    svg.append(g);
  });
  const panel = h("div");
  const wrap = h("div", { class: "lineage-wrap" }, svg);
  function connected(id) {
    const up = new Set([id]), down = new Set([id]);
    let changed = true;
    while (changed) { changed = false; for (const e of edges) { if (down.has(e.from) && !down.has(e.to)) { down.add(e.to); changed = true; } if (up.has(e.to) && !up.has(e.from)) { up.add(e.from); changed = true; } } }
    return new Set([...up, ...down]);
  }
  function focus(id) {
    const set = connected(id);
    svg.classList.add("dim");
    Object.entries(nodeEls).forEach(([k, el]) => el.classList.toggle("hi", set.has(k)));
    edgeEls.forEach((el) => el.classList.toggle("hi", set.has(el.dataset.from) && set.has(el.dataset.to)));
    const n = nodes.find((x) => x.id === id);
    const [kind, key] = id.split(/:(.+)/);
    const link = kind === "model" ? `#/models/${key}` : kind === "session" || kind === "annotation" ? `#/data/${key}` : kind === "cv" ? "#/compare" : "#/runs";
    const up = [...set].filter((k) => k !== id && nodes.find((x) => x.id === k)?.column < n.column);
    const down = [...set].filter((k) => k !== id && nodes.find((x) => x.id === k)?.column > n.column);
    panel.replaceChildren(card(`${KIND_LABEL[n.kind]}: ${n.label}`, "", h("dl", { class: "kv" },
      h("dt", {}, "元になったもの"), h("dd", {}, up.length ? up.map((k) => h("span", { class: "badge outline", style: { margin: "0 4px 4px 0" } }, k.replace(":", " "))) : "—"),
      h("dt", {}, "ここから作られたもの"), h("dd", {}, down.length ? down.map((k) => h("span", { class: "badge outline", style: { margin: "0 4px 4px 0" } }, k.replace(":", " "))) : "—")),
      h("div", { class: "row", style: { marginTop: "10px" } }, h("a", { href: link }, "詳細を開く →"), h("button", { type: "button", class: "ghost small", onclick: clear }, "選択を外す"))));
  }
  function clear() {
    svg.classList.remove("dim");
    Object.values(nodeEls).forEach((el) => el.classList.remove("hi"));
    edgeEls.forEach((el) => el.classList.remove("hi"));
    panel.replaceChildren(h("div", { class: "muted small" }, "線はつながっているところだけに引き、光の粒が元から作られたものへ流れます。節を押すと、その節につながる流れだけを強調します。どこにもつながっていない節は薄く表示します。"));
  }
  clear();
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "どのモデルがどのデータから作られ、どう評価され、どこで走ったかをたどれます（走行データ → 手作業のラベル → モデル → 交差検証・Pi での走行）。記録（metadata.json のセッション、データのハッシュ、run.json の版）だけから作っています。"),
    h("div", { class: "legend" },
      h("span", {}, s("svg", { width: 34, height: 8 }, s("line", { class: "legend-flow", x1: 2, x2: 32, y1: 4, y2: 4 })), "流れの向き（元 → 作られたもの）"),
      // つながりの種類は、図と同じ流れる印の形（丸い粒・長い棒・棒と粒）で見せる。下地の線はどれも同じ実線
      ...[["train", "学習に使用"], ["val", "検証に使用"], ["same_data", "同じデータで評価"]].map(([kind, text]) =>
        h("span", {}, s("svg", { width: 60, height: 8, class: `lin-link ${kind}` },
          s("line", { class: "lin-edge", x1: 0, x2: 60, y1: 4, y2: 4 }),
          s("line", { class: "lin-flow legend-kind", x1: 4, x2: 60, y1: 4, y2: 4 })), text)),
      h("span", {}, h("i", { class: "box", style: { background: "var(--good)" } }), "有効"),
      h("span", {}, h("i", { class: "box", style: { background: "var(--muted)" } }), "候補"),
      h("span", {}, h("i", { class: "box", style: { background: "var(--bad)" } }), "却下")),
    panel, wrap));
  const f = query.get("focus");
  if (f && nodeEls[f]) { focus(f); nodeEls[f].scrollIntoView?.({ block: "center", inline: "center" }); }
}

/* ------------------------------------------------------------------ views: runs */

function runsTable(runs) {
  return table([
    { key: "run_id", label: "走行", render: (r) => h("span", { class: "mono" }, r.run_id) },
    { key: "model_version", label: "モデル", render: (r) => (r.model_version ? h("a", { class: "mono", href: `#/models/${r.model_version}` }, r.model_version) : "—") },
    { key: "started_at", label: "開始", render: (r) => fmtDate(r.started_at) },
    { key: "body_version", label: "車体", render: (r) => bodyBadge(r.body_version, "date") },
    { key: "autonomous_s", label: "自律", num: true, render: (r) => `${fmt(r.autonomous_s / 60, 1)} 分` },
    { key: "human", label: "人の介入", num: true },
    { key: "safety", label: "安全の介入", num: true },
    { key: "rate", label: "人/10分", num: true, render: (r) => (r.autonomous_s > 0 ? fmt((r.human * 600) / r.autonomous_s, 1) : "—"), sort: (r) => (r.autonomous_s > 0 ? r.human / r.autonomous_s : null) },
    { key: "end_reason", label: "終わった理由", render: (r) => r.end_reason || (r.ended ? "—" : "途中") },
  ], runs, { empty: "走行の記録はありません（実走行の「走行記録の取り込み」で Pi から持ってくる）" });
}

async function viewRuns(root) {
  const c = await loadCatalog();
  const per = c.models.filter((m) => m.field.runs);
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "Pi の自律走行の記録（data/runs/*/run.json）です。モデルごとに、人と安全機能が介入した回数を比べられます。"),
    card("モデル別", "人の介入が少ないほどよい", per.length
      ? hbars(per.map((m) => ({ name: m.version, value: m.field.human_per_10min || 0, text: `${fmt(m.field.human_per_10min, 1)} 回/10分`, extra: [[null, "走行", `${m.field.runs} 回`], [null, "自律", `${fmt(m.field.autonomous_min, 1)} 分`]] })), { title: "人の介入（10分あたり）" })
      : h("div", { class: "empty" }, "走行の記録はありません")),
    card("走行", `${c.runs.length} 件`, runsTable([...c.runs].reverse()))));
}

/* ------------------------------------------------------------------ views: tools */

async function viewTools(root, _p, query) {
  const workflows = await loadWorkflows();
  if (!state.catalog) await loadCatalog();
  const phases = state.phases.length ? state.phases : [{ id: null, number: null, title: "ツール", summary: "", color: "var(--muted)", page: "tools", icon: "" }];
  const inPhase = (p) => workflows.filter((w) => (p.id ? w.phase === p.id : true));
  const used = phases.filter((p) => inPhase(p).length);
  root.append(
    h("p", { class: "lead" }, "既存のツールを、使う段階（フェーズ）の順に並べています。どれも別のプロセスで起動し、出力は「ジョブ」に残ります。車体が動くツールは安全の確認が必要です。"),
    // フェーズへ飛ぶ目次。並びの順が作業の順
    h("nav", { class: "phase-jump", "aria-label": "フェーズへ移動" }, used.map((p) => h("a", {
      href: `#/tools?phase=${p.id}`, style: { "--ph": p.color },
      onclick: (e) => { e.preventDefault(); document.getElementById(`phase-${p.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" }); },
    }, phaseIcon(p), p.title, h("small", {}, inPhase(p).length)))));
  for (const p of used) {
    root.append(h("section", { class: "tool-group", id: `phase-${p.id}`, style: { "--ph": p.color } },
      h("header", {}, phaseIcon(p, "solid"), h("h2", {}, p.title), h("span", { class: "sub" }, p.summary),
        p.number !== null ? h("a", { class: "small open-page", href: `#/${p.page}` }, `${ROUTES[p.page]?.[0] || ""}の画面へ →`) : null),
      h("div", { class: "tools" }, inPhase(p).map((w) => h("div", { class: "card tool" },
        h("div", { class: "title" }, w.title,
          w.moves_car ? h("span", { class: "badge moves" }, "車体が動く") : null,
          w.confirm ? h("span", { class: "badge confirm-badge", title: w.confirm }, "変更の確認") : null,
          w.kind === "window" ? h("span", { class: "badge outline" }, "画面") : null),
        h("div", { class: "desc" }, w.description),
        h("div", { class: "module" }, w.module),
        h("div", { class: "actions" }, h("a", { href: manualHref(w.id), class: "small", style: { marginRight: "auto", alignSelf: "center" } }, "使い方"), h("button", { type: "button", class: w.moves_car ? "danger small" : "small", onclick: () => launch(w.id) }, w.moves_car ? "確認して起動…" : "起動…")))))));
  }
  const target = query?.get("phase");
  if (target) requestAnimationFrame(() => document.getElementById(`phase-${target}`)?.scrollIntoView({ block: "start" }));
}

/* ------------------------------------------------------------------ views: body (0 準備) */

async function viewBody(root) {
  const c = await loadCatalog();
  const bodies = [...(c.bodies || [])].reverse();
  const cur = bodies.find((b) => b.current);
  const err = c.summary.body_error;
  const warnings = c.models.flatMap((m) => m.checks.filter((ch) => ch.key === "body" && ch.level !== "ok").map((ch) => ({ m, ch })));
  const calib = (b) => Object.entries(b.calibration || {});
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "車体の版は、走行データを集めた時の装備と校正の記録です。装備や校正を変えたら新しい版を登録します。セッション・モデル・走行は、どの版の車体のものかでつながり、違う車体のデータを混ぜた時や、学習した時と車体が変わった時に印が付きます（configs/body.toml）。"),
    err ? h("div", { class: "alert error" }, healthBadge("error"), h("span", {}, `configs/body.toml を読めません: ${err}`)) : null,
    h("div", { class: "grid cols-2" },
      card("今の車体", cur ? `${fmtDate(cur.since, true)} から使用` : "", cur
        ? h("div", { class: "stack", style: { gap: "12px" } },
          h("div", { class: "row" }, h("span", { class: "body-id" }, cur.id), h("b", {}, cur.title)),
          h("dl", { class: "kv" },
            h("dt", {}, "装備"), h("dd", {}, cur.equipment.length ? h("div", { class: "row", style: { gap: "4px" } }, cur.equipment.map((x) => h("span", { class: "badge outline" }, x))) : "—"),
            h("dt", {}, "校正"), h("dd", {}, calib(cur).length ? h("div", { class: "row", style: { gap: "4px" } }, calib(cur).map(([k, v]) => h("span", { class: "hash" }, `${k} = ${v}`))) : "—"),
            cur.notes ? [h("dt", {}, "メモ"), h("dd", {}, cur.notes)] : null,
            h("dt", {}, "この版のデータ"), h("dd", {}, `${cur.sessions.length} セッション・${fmtInt(cur.frames)} フレーム・走行 ${cur.runs} 回`)))
        : h("div", { class: "empty" }, "車体の版がまだありません。上の「車体の版を登録」で登録してください")),
      card("車体の違いの確認", "モデルのトレーサビリティの確認のうち、車体に関するもの", warnings.length
        ? h("div", { class: "stack", style: { gap: "8px" } }, warnings.map(({ m, ch }) => h("a", { class: "alert warn", href: `#/models/${m.version}`, style: { color: "var(--ink)", textDecoration: "none" } }, healthBadge("warn"), h("span", {}, h("b", { class: "mono" }, m.version), ` ${ch.message}`))))
        : h("div", { class: "empty" }, bodies.length ? "車体の違いによる問題はありません" : "車体の版を登録すると確認できます"))),
    card("車体の版の履歴", `${bodies.length} 版・新しい順`, table([
      { key: "id", label: "版", render: (b) => h("span", { class: "row", style: { gap: "6px" } }, h("b", { class: "mono" }, b.id), b.current ? h("span", { class: "badge tag" }, "使用中") : null) },
      { key: "since", label: "使い始め", render: (b) => fmtDate(b.since, true) },
      { key: "title", label: "内容", render: (b) => h("div", {}, h("div", {}, b.title), b.notes ? h("div", { class: "muted small" }, b.notes) : null), nosort: true },
      { key: "calibration", label: "校正", nosort: true, render: (b) => h("span", { class: "mono small" }, calib(b).map(([k, v]) => `${k}=${v}`).join("  ") || "—") },
      { key: "frames", label: "データ", num: true, render: (b) => `${b.sessions.length} セッション / ${fmtInt(b.frames)}`, sort: (b) => b.frames },
      { key: "models", label: "学習したモデル", nosort: true, render: (b) => (b.models.length ? b.models.map((v) => h("a", { class: "badge outline", href: `#/models/${v}`, style: { marginRight: "4px" } }, v)) : h("span", { class: "muted" }, "—")) },
      { key: "runs", label: "走行", num: true },
    ], bodies, { empty: "車体の版がまだありません" }))));
}

/* ------------------------------------------------------------------ views: deploy (5 デプロイ) */

async function viewDeploy(root) {
  const c = await loadCatalog();
  const sm = c.summary;
  const deps = [...(c.deployments || [])].reverse();
  const ACTION = { push: "デプロイして使用中に", install: "デプロイのみ", rollback: "戻した" };
  const alerts = collectAlerts(c).filter((a) => a.href === "#/deploy" || /Pi で使用中/.test(a.text));
  const models = [...c.models].sort((a, b) => Number(b.usable) - Number(a.usable) || String(b.version).localeCompare(String(a.version)));
  root.append(h("div", { class: "stack" },
    h("p", { class: "lead" }, "使うと決めた（有効な）モデルを Pi へ送り、使用中にします。Pi は送られたモデルのハッシュを確かめてから切り替え、問題があれば1つ前の版へ戻せます。"),
    h("section", { class: "card" }, h("div", { class: "facts" },
      fact("Pi で使用中（デプロイの記録）", sm.deployed.version, sm.deployed.at ? `${fmtDate(sm.deployed.at)} にデプロイ` : "Control Center からのデプロイの記録なし"),
      fact("Pi の最後の走行で使った版", sm.last_pi_run?.model_version, sm.last_pi_run ? `${fmtDate(sm.last_pi_run.started_at)} の走行` : "走行の記録なし"),
      fact("最新の有効なモデル", sm.latest_approved, sm.latest_approved ? "採用済みでトレーサビリティに誤りなし" : "「有効」にしたモデルがまだない"),
      h("div", { class: "fact" }, h("div", { class: "label" }, "今の車体"), h("div", { class: "value" }, sm.body_current ? h("a", { href: "#/body" }, sm.body_current) : h("span", { class: "muted" }, "—")), h("div", { class: "foot" }, "学習データの車体と同じかを下の表で確かめる")))),
    // 記録の外（手作業など）でデプロイしたモデルで走っている時は、記録がないことをはっきり出す
    !deps.length && sm.last_pi_run?.model_version
      ? h("div", { class: "alert warn" }, healthBadge("warn", "記録なし"),
        h("span", {}, `デプロイの記録がありません。Pi は ${sm.last_pi_run.model_version} で走っています（${fmtDate(sm.last_pi_run.started_at)} の走行記録から）。Control Center の外でデプロイしたと考えられます。次から上の「Pi へデプロイ」を使うと、いつ・どの版を入れたかが記録に残ります。`))
      : null,
    alerts.length ? h("div", { class: "stack", style: { gap: "8px" } }, alerts.map((a) => h("div", { class: `alert ${a.level}` }, healthBadge(a.level), h("span", {}, a.text)))) : null,
    card("デプロイするモデル", "有効なモデルを上に並べています", table([
      { key: "version", label: "版", render: (m) => h("span", { class: "row", style: { gap: "6px" } }, h("b", { class: "mono" }, m.version), m.deployed ? h("span", { class: "badge outline" }, "Pi で使用中") : null), sort: (m) => m.version },
      { key: "stage", label: "採否", render: (m) => stageBadge(m.stage) },
      { key: "health", label: "トレーサビリティ", render: (m) => healthBadge(m.health), sort: (m) => ({ error: 0, warn: 1, ok: 2 })[m.health] },
      { key: "body", label: "学習データの車体", render: (m) => (m.body_versions?.length ? h("span", { class: "row", style: { gap: "4px" } }, m.body_versions.map((b) => bodyBadge(b)), sm.body_current && !m.body_versions.includes(sm.body_current) ? h("span", { class: "health warn" }, "今と違う") : null) : "—"), nosort: true },
      { key: "mae", label: "検証 MAE", num: true, render: (m) => (m.validation.steering_mae_deg !== undefined ? `±${fmt(m.validation.steering_mae_deg, 2)}°` : "—"), sort: (m) => m.validation.steering_mae_deg },
      { key: "go", label: "", nosort: true, render: (m) => h("button", { type: "button", class: m.usable ? "small" : "small ghost", onclick: (e) => { e.stopPropagation(); launch("deploy_push", `push --model_dir models/${m.version}`); } }, "Pi へデプロイ…") },
    ], models, { onRow: (m) => (location.hash = `#/models/${m.version}`), empty: "モデルがありません" })),
    card("デプロイの履歴", "data/deployments.jsonl・新しい順", deps.length
      ? h("ul", { class: "timeline" }, deps.map((d) => h("li", {}, h("div", {}, h("b", { class: "mono" }, d.version || "—"), ` ${ACTION[d.action] || d.action} `, healthBadge(d.ok ? "ok" : "error", d.ok ? "成功" : "失敗")), h("div", { class: "when" }, fmtDate(d.at, true)))))
      : h("div", { class: "empty" }, "Control Center からのデプロイの記録はありません"))));
}

/* ------------------------------------------------------------------ views: jobs */

async function viewJobs(root) {
  const jobs = await api("/jobs");
  state.jobs = jobs;
  root.append(h("p", { class: "lead" }, "起動したツールの記録です（data/jobs）。何をどの引数で動かし、どう終わったかが残ります。"),
    card("ジョブ", `${jobs.length} 件`, table([
      { key: "status", label: "状態", render: (j) => statusDot(j.status) },
      { key: "title", label: "ツール", render: (j) => h("b", {}, j.title) },
      { key: "command", label: "コマンド", render: (j) => h("code", { class: "small" }, j.command), nosort: true },
      { key: "started_at", label: "開始", render: (j) => fmtDate(j.started_at, true) },
      { key: "dur", label: "時間", render: (j) => duration(j.started_at, j.ended_at), nosort: true },
      { key: "produces", label: "成果物", render: (j) => j.produces || "—" },
    ], jobs, { onRow: (j) => (location.hash = `#/jobs/${j.id}`), sortKey: "started_at", desc: true, empty: "まだジョブはありません" })));
  if (jobs.some((j) => j.status === "running")) state.timers.push(setTimeout(render, 3000));
}

async function viewJob(root, [id]) {
  const jobs = await api("/jobs");
  let job = jobs.find((j) => j.id === id);
  if (!job) { root.append(h("div", { class: "empty" }, "ジョブが見つかりません")); return; }
  setTitle(job.title);
  const log = h("div", { class: "log", role: "log" });
  const statusEl = h("span", {}, statusDot(job.status));
  const stopBtn = h("button", { type: "button", class: job.moves_car ? "danger" : "ghost", hidden: job.status !== "running", onclick: async () => {
    stopBtn.disabled = true;
    try { const r = await api(`/jobs/${id}/stop`, { method: "POST", body: {} }); toast(r.stop_note ? `止めました（${r.stop_note}）` : "止めました"); } catch (err) { toast(err.message, "error"); }
  } }, job.moves_car ? "■ 車体を止めて終了" : "止める");
  const producedLink = h("span");
  // 出力がまだない間は、大きな黒い枠ではなく、なぜ空なのかを1行で書く
  const emptyNote = h("div", { class: "log-empty" });
  log.classList.add("is-empty");
  log.append(emptyNote);
  const showEmpty = (status) => {
    if (!log.classList.contains("is-empty")) return;
    emptyNote.textContent = status === "running"
      ? (job.kind === "window" ? "この画面のツールは、ここに出力を書き出しません。操作は開いたウィンドウで行います。" : "出力を待っています…")
      : "出力はありませんでした。";
  };
  const hideEmpty = () => { if (log.classList.contains("is-empty")) { log.classList.remove("is-empty"); emptyNote.remove(); } };
  showEmpty(job.status);
  let since = 0, stick = true;
  log.addEventListener("scroll", () => { stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 20; });
  const poll = async () => {
    try {
      const res = await api(`/jobs/${id}/log?since=${since}`);
      since = res.next;
      if (res.lines.length) hideEmpty(); else showEmpty(res.status);
      for (const line of res.lines) log.append(h("div", { class: /error|traceback|failed|失敗/i.test(line) ? "err" : /✓|success|finished|成功/i.test(line) ? "ok" : "" }, line || " "));
      if (stick) log.scrollTop = log.scrollHeight;
      statusEl.replaceChildren(statusDot(res.status), res.exit_code !== null && res.exit_code !== undefined ? h("span", { class: "muted small" }, `  終了コード ${res.exit_code}`) : null);
      stopBtn.hidden = res.status !== "running";
      if (res.status === "running") state.timers.push(setTimeout(poll, document.hidden ? 5000 : 1000));
      else if (job.produces && /^models\//.test(job.produces)) {
        await loadCatalog(true);
        const v = job.produces.split("/").pop();
        if (state.catalog.models.find((m) => m.version === v)) producedLink.replaceChildren(h("a", { href: `#/models/${v}` }, `作ったモデル ${v} を開く →`));
      }
    } catch (err) { hideEmpty(); log.append(h("div", { class: "err" }, err.message)); }
  };
  root.append(h("div", { class: "stack" },
    h("section", { class: "card" },
      h("div", { class: "row", style: { alignItems: "flex-start" } },
        h("dl", { class: "kv", style: { flex: 1 } },
          h("dt", {}, "状態"), h("dd", {}, statusEl),
          h("dt", {}, "コマンド"), h("dd", {}, h("code", {}, job.command)),
          h("dt", {}, "開始"), h("dd", {}, fmtDate(job.started_at, true), job.ended_at ? `（${duration(job.started_at, job.ended_at)}）` : ""),
          job.produces ? [h("dt", {}, "成果物"), h("dd", {}, h("span", { class: "mono" }, job.produces), " ", producedLink)] : null,
          job.stop_note ? [h("dt", {}, "停止"), h("dd", {}, job.stop_note)] : null),
        h("div", { class: "row" }, stopBtn, h("button", { type: "button", class: "ghost", onclick: () => launch(job.workflow, job.args.map((a) => (/\s/.test(a) ? `"${a}"` : a)).join(" ")) }, "同じ引数でもう一度…")))),
    job.kind === "window" ? h("div", { class: "alert info" }, healthBadge("unknown", "画面"),
      job.status === "running" ? "別のウィンドウで開いています。閉じるとこのジョブも終わります。" : "別のウィンドウで開く画面のツールです。ウィンドウを閉じた時にこのジョブも終わりました。") : null,
    log));
  poll();
}

/* ------------------------------------------------------------------ views: manual */

// 画面（ウィンドウ）のツールは、マニュアルの画面の説明へつなぐ
const MANUAL_GUI = { app: "gui-app", collector: "gui-collector", data_viewer: "gui-viewer" };
const manualHref = (id) => `#/manual?s=${MANUAL_GUI[id] || `tool-${id}`}`;

async function viewManual(root, _p, query) {
  if (!state.manual) {
    const res = await fetch("manual.html", { cache: "no-store" }).catch(() => null);
    if (!res || !res.ok) throw new Error("マニュアルを読み込めません（サーバーが止まっていないか確かめてください）");
    state.manual = await res.text();
  }
  // 同じサーバーの静的なファイル（src/gui/web/static/manual.html）なので、そのまま入れる
  const body = h("div", { class: "manual-body card" });
  body.innerHTML = state.manual;
  // フェーズの印に、画面と同じアイコンを入れる
  body.querySelectorAll(".ph-icon[data-phase]").forEach((el) => { const p = phaseById(el.dataset.phase); if (p) el.innerHTML = iconSvg(p.icon); });
  const toc = h("nav", { class: "manual-toc", "aria-label": "目次" });
  const search = h("input", { type: "search", placeholder: "言葉で探す（例: 引数、デプロイ、fps）", class: "manual-search" });
  toc.append(search);
  const list = h("ol");
  body.querySelectorAll("section[id]").forEach((sec) => {
    const title = sec.querySelector("h2");
    const item = h("li", {}, h("a", { href: `#/manual?s=${sec.id}` }, title ? title.textContent : sec.id));
    const subs = [...sec.querySelectorAll("h3[id], .tool[id]")];
    if (subs.length) {
      item.append(h("ol", {}, subs.map((el) => {
        // フェーズの見出し（「学習 — 説明」）は、目次では名前だけにする
        const label = el.matches(".tool") ? el.querySelector("h4").firstChild.textContent.trim() : el.textContent.split(" — ")[0].trim();
        return h("li", { class: el.matches(".tool") ? "tool-item" : "" }, h("a", { href: `#/manual?s=${el.id}` }, label));
      })));
    }
    list.append(item);
  });
  toc.append(list);
  search.addEventListener("input", () => {
    const q = search.value.trim().toLowerCase();
    body.querySelectorAll(".tool, section > table tbody tr").forEach((el) => {
      el.classList.toggle("dim-out", !!q && !el.textContent.toLowerCase().includes(q));
    });
  });
  root.append(h("div", { class: "manual-layout" }, toc, body));
  const target = query.get("s");
  requestAnimationFrame(() => {
    const el = target && document.getElementById(target);
    if (!el) { window.scrollTo(0, 0); return; }
    el.scrollIntoView({ block: "start" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1600);
  });
}

/* ------------------------------------------------------------------ robot & emergency stop */

document.addEventListener("visibilitychange", () => {
  // 表に戻った時は、待たずに最新にする
  if (!document.hidden) { pollRobotNow(); }
});
let robotPollAt = 0;
function pollRobotNow() { if (Date.now() - robotPollAt > 3000) pollRobot(true); }
async function pollRobot(once) {
  robotPollAt = Date.now();
  const pill = document.getElementById("robot-pill");
  try {
    const r = await api("/robot/status");
    state.robot = r;
    if (r.online) {
      const st = r.status;
      const bat = st.sensors?.battery?.voltage;
      pill.className = `pill ${st.emergency_stop ? "bad" : "good"}`;
      pill.replaceChildren(h("span", { class: "dot" }), st.emergency_stop ? "Pi 緊急停止中" : `Pi 接続中${bat ? `・${fmt(bat, 2)}V` : ""}`);
      pill.title = `モード ${st.mode}`;
    } else {
      pill.className = "pill neutral";
      pill.replaceChildren(h("span", { class: "dot" }), "Pi 未接続");
      pill.title = `Pi の API サーバーに接続できません（${r.error}）`;
    }
  } catch { pill.className = "pill bad"; pill.replaceChildren(h("span", { class: "dot" }), "サーバー停止中"); pill.title = "Control Center のサーバーが止まっています。起動し直してください"; }
  // タブが裏にある間は間隔を延ばす（見ていない時に Pi へ問い合わせ続けない）
  if (once === true) return;
  setTimeout(pollRobot, document.hidden ? 30000 : state.robot?.online ? 4000 : 10000);
}
async function pollJobs() {
  if (document.hidden) { setTimeout(pollJobs, 15000); return; }
  try {
    const jobs = await api("/jobs");
    const n = jobs.filter((j) => j.status === "running").length;
    const el = document.getElementById("nav-running");
    el.hidden = !n; el.textContent = n;
  } catch { /* 次の回で */ }
  setTimeout(pollJobs, 3000);
}
document.getElementById("estop").addEventListener("click", async () => {
  try {
    const r = await api("/robot/emergency_stop", { method: "POST", body: {} });
    toast(`${r.robot}${r.stopped_jobs.length ? `・走行ジョブ ${r.stopped_jobs.length} 件を停止` : ""}`, r.ok ? "" : "error");
    if (r.ok) {
      modal({
        title: "緊急停止中",
        body: [h("div", {}, "Pi は緊急停止の状態です。解除するまで走行は拒否されます。"), h("div", { class: "muted small" }, "周囲の安全を確かめてから解除してください。")],
        actions: (close) => [
          h("button", { type: "button", class: "ghost", onclick: close }, "このまま"),
          h("button", { type: "button", onclick: async () => { try { const x = await api("/robot/emergency_stop/reset", { method: "POST", body: {} }); toast(x.ok ? "緊急停止を解除しました" : `解除できません（${x.error}）`, x.ok ? "" : "error"); close(); } catch (err) { toast(err.message, "error"); } } }, "解除する"),
        ],
      });
    }
  } catch (err) { toast(err.message, "error"); }
});

/* ------------------------------------------------------------------ router */

const ROUTES = {
  overview: ["概要", viewOverview],
  body: ["車体", viewBody],
  data: ["データ", viewData, viewSession],
  models: ["モデル", viewModels, viewModel],
  compare: ["比較・評価", viewCompare],
  deploy: ["デプロイ", viewDeploy],
  runs: ["走行記録", viewRuns],
  lineage: ["トレーサビリティ", viewLineage],
  tools: ["ツール", viewTools],
  jobs: ["ジョブ", viewJobs, viewJob],
  manual: ["操作マニュアル", viewManual],
};
function setTitle(t) { document.getElementById("page-title").textContent = t; document.title = `${t} — PiCar-X Control Center`; }

let renderSeq = 0;
async function render() {
  state.timers.forEach(clearTimeout);
  state.timers = [];
  hideTip();
  const seq = ++renderSeq;
  const [path, qs] = (location.hash.replace(/^#\/?/, "") || "overview").split("?");
  const [name, ...rest] = path.split("/").map(decodeURIComponent);
  const route = ROUTES[name] || ROUTES.overview;
  const routeName = ROUTES[name] ? name : "overview";
  document.querySelectorAll(".sidebar a[data-route]").forEach((a) => a.classList.toggle("active", a.dataset.route === routeName));
  setTitle(route[0]);
  setPagePhase(routeName);
  document.body.classList.remove("menu-open");
  const view = document.getElementById("view");
  const root = h("div");
  view.replaceChildren(h("div", { class: "loading" }, h("span", { class: "spinner" }), "読み込み中…"));
  try {
    const detail = rest.length && route[2];
    const fn = detail ? route[2] : route[1];
    await fn(root, rest, new URLSearchParams(qs || ""));
    // 一覧の画面には、開発の流れの中の位置と、そのフェーズのツールを上に出す
    if (!detail) {
      await loadWorkflows().catch(() => null);
      const head = phaseHeader(routeName);
      if (head) root.prepend(head);
    }
    if (seq === renderSeq) view.replaceChildren(root);
  } catch (err) {
    if (seq === renderSeq) view.replaceChildren(h("div", { class: "alert error" }, healthBadge("error"), h("span", {}, `読み込めません: ${err.message}`)));
  }
}
window.addEventListener("hashchange", render);
document.getElementById("refresh").addEventListener("click", async () => { await loadCatalog(true); render(); toast("最新の状態を読み直しました"); });
document.getElementById("menu-toggle").addEventListener("click", () => document.body.classList.toggle("menu-open"));
// 表示の明暗の切り替え。ボタンには押すと何になるかを出す（「テーマ」だけでは分からないため）
const themeButton = document.getElementById("theme-toggle");
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && darkQuery.matches);
function syncThemeButton() {
  const next = isDark() ? "明るい表示" : "暗い表示";
  themeButton.textContent = `${isDark() ? "☀" : "☾"} ${next}にする`;
  themeButton.title = `画面を${next}に切り替えます（今は${isDark() ? "暗い" : "明るい"}表示）`;
}
themeButton.addEventListener("click", () => {
  document.documentElement.dataset.theme = isDark() ? "light" : "dark";
  try { localStorage.setItem("picarx-theme", document.documentElement.dataset.theme); } catch { /* 保存できなくても表示は変える */ }
  syncThemeButton();
});
darkQuery.addEventListener("change", syncThemeButton);
try { const t = localStorage.getItem("picarx-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* なし */ }
syncThemeButton();

(async () => {
  try {
    state.phases = await api("/phases");
    buildNavFlow();
  } catch { /* フェーズを読めなくても画面は出す（サーバーが止まっている時は各画面が理由を出す） */ }
  render();
  pollRobot();
  pollJobs();
})();
