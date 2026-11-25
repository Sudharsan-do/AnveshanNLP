const PRESETS = [
  { label: "Hinglish · IoT Lights", text: "kripaya rasoeeghar ki battiyaan band kar den" },
  { label: "Hindi · Weather", text: "शनिवार का पूर्वानुमान क्या है" },
  { label: "Noisy ASR · Train", text: "dilli se dehraadun k lie tren tikat buk kr plz" },
  { label: "Mixed · Smart Socket", text: "wemo plug socket बंद करें" },
  { label: "Hindi · Stock QA", text: "रिलायंस का वर्तमान स्टॉक मूल्य क्या है" },
  { label: "Hinglish · Calendar", text: "kal subah 11 baje meeting yaad dila dena" },
];

const OOS_PRESETS = [
  { label: "Class 55 · Podcast", text: "mera sabse zyada suna jane wala podcast kaunsa hai" },
  { label: "Class 55 · Alarm Query", text: "subah 8 baje ka alarm query check karo" },
  { label: "True OOS · Nonsense", text: "quantum banana spaceship purple 404 protocol" },
  { label: "True OOS · Off-Domain", text: "mujhe apna aadhaar card renew karwana hai" },
];

const MS_SUITES = [
  {
    name: "Train Ticket (2)",
    dev: "दिल्ली से देहरादून के लिए ट्रेन टिकट बुक करो",
    hing: "dilli se dehradun ke liye train ticket book karo",
    noisy: "dilli se dehraadun k lie tren tikat buk kr plz",
    mixed: "दिल्ली se dehradun के लिए train ticket book करो",
  },
  {
    name: "Light Off (39)",
    dev: "कृपया रसोईघर की बत्तियां बंद कर दें",
    hing: "kripaya rasoighar ki battiyaan band kar den",
    noisy: "plz rasoeeghar ki battiya bnd kr do",
    mixed: "kitchen ki बत्तियां band kar den",
  },
  {
    name: "Weather (13)",
    dev: "क्या कल दिल्ली में बारिश होने वाली है",
    hing: "kya kal dilli mein barish hone wali hai",
    noisy: "kia kl dilli m baarish hogi kya",
    mixed: "कल delhi में weather कैसा रहेगा",
  },
  {
    name: "Currency (10)",
    dev: "रुपये और डॉलर का एक्सचेंज रेट क्या है",
    hing: "rupaye aur dollar ka exchange rate kya hai",
    noisy: "rupye or dolar ka xchange ret ky h",
    mixed: "INR और USD का exchange rate बताओ",
  },
];

const state = {
  mode: "ensemble",
  systemStats: null,
  latencyHistory: [],
};

const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function initTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

function initModeToggle() {
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      runPredict();
    });
  });
}

function initSlidersAndPresets() {
  $("oos-conf-slider").addEventListener("input", (e) => {
    $("oos-conf-val").textContent = Number(e.target.value).toFixed(2);
  });
  $("oos-conf-slider").addEventListener("change", () => runPredict());

  $("oos-guard-slider").addEventListener("input", (e) => {
    $("oos-guard-val").textContent = Number(e.target.value).toFixed(2);
  });
  $("oos-guard-slider").addEventListener("change", () => runPredict());

  $("tail-multipliers-toggle").addEventListener("change", () => runPredict());

  const presetWrap = $("preset-chips");
  PRESETS.forEach((p) => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.textContent = p.label;
    btn.addEventListener("click", () => {
      $("query-input").value = p.text;
      runPredict();
    });
    presetWrap.appendChild(btn);
  });

  const oosWrap = $("oos-preset-chips");
  OOS_PRESETS.forEach((p) => {
    const btn = document.createElement("button");
    btn.className = "chip chip-oos";
    btn.textContent = p.label;
    btn.addEventListener("click", () => {
      $("query-input").value = p.text;
      runPredict();
    });
    oosWrap.appendChild(btn);
  });

  const suiteWrap = $("ms-suite-chips");
  MS_SUITES.forEach((s) => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.textContent = s.name;
    btn.addEventListener("click", () => {
      $("ms-dev").value = s.dev;
      $("ms-hing").value = s.hing;
      $("ms-noisy").value = s.noisy;
      $("ms-mixed").value = s.mixed;
      runMultiScript();
    });
    suiteWrap.appendChild(btn);
  });

  $("predict-btn").addEventListener("click", runPredict);
  $("query-input").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runPredict();
  });
  $("run-multiscript-btn").addEventListener("click", runMultiScript);
  $("reset-telemetry-btn").addEventListener("click", () => {
    state.latencyHistory = [];
    renderSessionLatency();
  });
  $("taxonomy-search").addEventListener("input", (e) => renderTaxonomy(e.target.value));
}

async function fetchSystemStats() {
  const resp = await fetch("/api/system-stats");
  const data = await resp.json();
  state.systemStats = data;

  const mi = data.model_info;
  const totalM = (mi.total_active_params / 1e6).toFixed(2);
  $("compliance-pill-text").textContent = `${totalM}M / 300.0M (PASS)`;
  $("device-pill-text").textContent = mi.device;
  const cvSum = data.benchmark?.cv_summary;
  if (cvSum?.oof_macro_f1) {
    $("hdr-f1").textContent = cvSum.oof_macro_f1.toFixed(4);
  }

  renderComplianceTab(data);
  renderTaxonomy("");
}

async function runPredict() {
  const queryText = $("query-input").value.trim();
  if (!queryText) return;

  const btn = $("predict-btn");
  btn.disabled = true;
  btn.textContent = "Running...";

  try {
    const payload = {
      query_text: queryText,
      mode: state.mode,
      oos_conf_threshold: Number($("oos-conf-slider").value),
      oos_guard_threshold: Number($("oos-guard-slider").value),
      apply_tail_multipliers: $("tail-multipliers-toggle").checked,
    };

    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.error || "Prediction failed");

    renderPrediction(res);
    state.latencyHistory.unshift({
      query: res.query_text,
      script: res.script_analysis.script,
      total_ms: res.latency.total_ms,
      intent: res.prediction.intent_name,
    });
    if (state.latencyHistory.length > 40) state.latencyHistory.pop();
    renderSessionLatency();
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Classify";
  }
}

function renderPrediction(res) {
  const p = res.prediction;
  const sa = res.script_analysis;
  const lat = res.latency;
  const oos = res.oos_fallback;

  $("script-badge").textContent = `${sa.script} · ${sa.devanagari_pct}% Dev / ${sa.latin_pct}% Lat`;
  $("pred-intent").textContent = p.intent_name;
  $("pred-id").textContent = `Target ${p.target}`;
  $("pred-domain").textContent = `Domain: ${p.domain}`;
  $("pred-conf").textContent = `${(p.confidence * 100).toFixed(1)}% conf`;

  const banner = $("oos-status-banner");
  if (p.is_oos_55 || oos.event !== "none") {
    banner.classList.remove("hidden");
    if (oos.event === "reassigned_from_55_low_conf") {
      banner.textContent = `Class 55 Rescue: Raw argmax was Class 55 (${(p.raw_confidence * 100).toFixed(1)}% < ${(oos.oos_conf_threshold * 100).toFixed(0)}%), reassigned to '${p.intent_name}'.`;
    } else if (oos.event === "routed_to_55_oos_guard") {
      banner.textContent = `OOS Guard Triggered: Top confidence (${(p.raw_confidence * 100).toFixed(1)}%) < ${(oos.oos_guard_threshold * 100).toFixed(0)}%, routed to Class 55 fallback.`;
    } else {
      banner.textContent = `Class 55 (Catch-All / Out-of-Scope) predicted directly at ${(p.confidence * 100).toFixed(1)}% confidence.`;
    }
  } else {
    banner.classList.add("hidden");
  }

  $("trace-xlm-text").textContent = res.transformer_text;
  $("trace-svc-text").textContent = res.xlit_text;
  $("trace-gate").textContent = `${(sa.g_xlm * 100).toFixed(0)}% XLM-R / ${(sa.g_svc * 100).toFixed(0)}% SVC`;
  $("trace-tokens").textContent = `${sa.token_count}`;
  $("trace-entropy").textContent = p.normalized_entropy.toFixed(3);

  $("lat-total").textContent = lat.total_ms.toFixed(2);
  $("lat-qps").textContent = lat.throughput_qps.toFixed(1);
  $("lat-pre").textContent = lat.preprocess_ms.toFixed(2);
  $("lat-xlm").textContent = lat.xlm_ms.toFixed(2);
  $("lat-svc").textContent = lat.svc_ms.toFixed(2);
  $("lat-fus").textContent = lat.fusion_ms.toFixed(2);

  const tot = Math.max(lat.total_ms, 0.01);
  $("bar-pre").style.width = `${Math.max(2, (lat.preprocess_ms / tot) * 100)}%`;
  $("bar-xlm").style.width = `${Math.max(5, (lat.xlm_ms / tot) * 100)}%`;
  $("bar-svc").style.width = `${Math.max(3, (lat.svc_ms / tot) * 100)}%`;
  $("bar-fus").style.width = `${Math.max(2, (lat.fusion_ms / tot) * 100)}%`;

  // Top-5 tabular breakdown
  const top5Wrap = $("top5-container");
  let rowsHtml = `
    <table class="top5-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Intent</th>
          <th>Final</th>
          <th>XLM-R</th>
          <th>SVC</th>
        </tr>
      </thead>
      <tbody>
  `;
  res.top5.forEach((item, idx) => {
    const pct = (item.score_prob * 100).toFixed(1);
    rowsHtml += `
      <tr class="${item.is_final ? "is-winner" : ""}">
        <td class="mono">${idx + 1}</td>
        <td>
          <div><strong>[${item.target}] ${escapeHtml(item.intent_name)}</strong> <span class="muted">(${escapeHtml(item.domain)})</span></div>
          <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
        </td>
        <td class="mono strong">${(item.score_prob * 100).toFixed(2)}%</td>
        <td class="mono muted">${(item.xlm_prob * 100).toFixed(1)}%</td>
        <td class="mono muted">${(item.svc_prob * 100).toFixed(1)}%</td>
      </tr>
    `;
  });
  rowsHtml += `</tbody></table>`;
  top5Wrap.innerHTML = rowsHtml;

  // OOS Inspector
  $("oos-prob-final").textContent = `${(oos.class_55_prob * 100).toFixed(2)}%`;
  $("oos-prob-xlm").textContent = `${(oos.class_55_raw_xlm_prob * 100).toFixed(2)}%`;
  $("oos-prob-svc").textContent = `${(oos.class_55_raw_svc_prob * 100).toFixed(2)}%`;
  $("oos-raw-class").textContent = `[${p.raw_target}] ${p.raw_intent_name}`;

  const badge = $("oos-event-badge");
  badge.className = "tag";
  if (oos.event === "reassigned_from_55_low_conf") {
    badge.classList.add("badge-warn");
    badge.textContent = "Rescued from 55";
    $("oos-explanation").textContent =
      `aavesham_v2 OOS_CONF rule activated: raw argmax was Class 55 at ${(p.raw_confidence * 100).toFixed(1)}% (< ${(oos.oos_conf_threshold * 100).toFixed(0)}%), so prediction fell back to '${p.intent_name}' (${(p.confidence * 100).toFixed(1)}%).`;
  } else if (oos.event === "routed_to_55_oos_guard") {
    badge.classList.add("badge-danger");
    badge.textContent = "Routed to 55 (OOS)";
    $("oos-explanation").textContent =
      `OOS Guard activated: max in-scope probability (${(p.raw_confidence * 100).toFixed(1)}%) fell below the ${(oos.oos_guard_threshold * 100).toFixed(0)}% threshold (entropy ${p.normalized_entropy.toFixed(3)}), routing the utterance to Class 55 fallback.`;
  } else if (p.is_oos_55) {
    badge.classList.add("badge-warn");
    badge.textContent = "Class 55 Direct";
    $("oos-explanation").textContent =
      `Class 55 exceeded the ${(oos.oos_conf_threshold * 100).toFixed(0)}% OOS_CONF threshold directly (${(p.confidence * 100).toFixed(1)}%).`;
  } else {
    badge.classList.add("badge-success");
    badge.textContent = "In-Scope";
    $("oos-explanation").textContent =
      `'${p.intent_name}' cleared the OOS guard (${(p.confidence * 100).toFixed(1)}% >= ${(oos.oos_guard_threshold * 100).toFixed(0)}%). Class 55 holds ${(oos.class_55_prob * 100).toFixed(2)}% probability.`;
  }
}

async function runMultiScript() {
  const btn = $("run-multiscript-btn");
  btn.disabled = true;
  btn.textContent = "Running...";

  try {
    const variants = [
      { label: "Devanagari (Hindi)", text: $("ms-dev").value },
      { label: "Clean Hinglish", text: $("ms-hing").value },
      { label: "Noisy ASR Hinglish", text: $("ms-noisy").value },
      { label: "Code-Switched Mixed", text: $("ms-mixed").value },
    ];

    const resp = await fetch("/api/compare-scripts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        variants,
        mode: state.mode,
        oos_conf_threshold: Number($("oos-conf-slider").value),
        oos_guard_threshold: Number($("oos-guard-slider").value),
        apply_tail_multipliers: $("tail-multipliers-toggle").checked,
      }),
    });
    const data = await resp.json();

    const banner = $("ms-summary-banner");
    banner.classList.remove("hidden", "partial");
    if (data.agreement_pct < 100) banner.classList.add("partial");
    banner.innerHTML = `
      <div><strong>Consensus:</strong> <span class="mono">[${data.consensus_target}] ${escapeHtml(data.consensus_intent)}</span> (${escapeHtml(data.consensus_domain)})</div>
      <div class="mono">Agreement: <strong>${data.agreement_pct}%</strong> · Avg: <strong>${data.avg_latency_ms} ms</strong></div>
    `;

    const grid = $("ms-results-grid");
    grid.innerHTML = "";
    data.results.forEach((r) => {
      const matched = r.prediction.target === data.consensus_target;
      const card = document.createElement("div");
      card.className = `ms-res-card ${matched ? "match" : "mismatch"}`;
      card.innerHTML = `
        <div class="ms-res-head">
          <strong>${escapeHtml(r.label)}</strong>
          <span class="tag ${matched ? "badge-success" : "badge-warn"}">${matched ? "Match" : "Diverged"}</span>
        </div>
        <p class="ms-res-quote">"${escapeHtml(r.query_text)}"</p>
        <div class="ms-res-meta mono">
          <span class="tag">[${r.prediction.target}] ${escapeHtml(r.prediction.intent_name)}</span>
          <span class="tag">${(r.prediction.confidence * 100).toFixed(1)}%</span>
          <span class="tag">${r.latency.total_ms.toFixed(1)} ms</span>
        </div>
        <div class="ms-res-foot mono">
          script=${r.script_analysis.script} · gate=${(r.script_analysis.g_xlm * 100).toFixed(0)}/${(r.script_analysis.g_svc * 100).toFixed(0)} · xlit="${escapeHtml(r.xlit_text)}"
        </div>
      `;
      grid.appendChild(card);
    });
  } finally {
    btn.disabled = false;
    btn.textContent = "Run 4-Way Check";
  }
}

function renderSessionLatency() {
  const hist = state.latencyHistory;
  $("sess-count").textContent = hist.length;
  if (!hist.length) {
    $("sess-mean").textContent = "—";
    $("sess-p50").textContent = "—";
    $("sess-p95").textContent = "—";
    $("latency-history-list").innerHTML = `<p class="caption">Classify queries in the Live tab to record latency percentiles.</p>`;
    return;
  }

  const vals = hist.map((h) => h.total_ms).sort((a, b) => a - b);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const p50 = vals[Math.floor(0.5 * (vals.length - 1))];
  const p95 = vals[Math.floor(0.95 * (vals.length - 1))];

  $("sess-mean").textContent = `${mean.toFixed(1)} ms`;
  $("sess-p50").textContent = `${p50.toFixed(1)} ms`;
  $("sess-p95").textContent = `${p95.toFixed(1)} ms`;

  const list = $("latency-history-list");
  list.innerHTML = "";
  hist.slice(0, 12).forEach((h) => {
    const row = document.createElement("div");
    row.className = "hist-row";
    row.innerHTML = `
      <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:55%;">
        <span class="mono">[${h.script}]</span> ${escapeHtml(h.query)}
      </span>
      <span class="mono muted">${escapeHtml(h.intent)}</span>
      <span class="mono strong">${h.total_ms.toFixed(1)} ms</span>
    `;
    list.appendChild(row);
  });
}

function renderComplianceTab(data) {
  const mi = data.model_info;
  const xlmPct = (mi.xlm_params / mi.param_limit) * 100;
  const svcPct = (mi.svc_params / mi.param_limit) * 100;

  $("param-audit-content").innerHTML = `
    <p class="caption">Verified against the 300,000,000 (300M) active parameter ceiling.</p>
    <div class="audit-meter-wrap">
      <div class="audit-meter">
        <div class="audit-xlm" style="width:${xlmPct.toFixed(2)}%" title="XLM-RoBERTa"></div>
        <div class="audit-svc" style="width:${svcPct.toFixed(2)}%" title="Calibrated LinearSVC"></div>
      </div>
      <div class="latency-stats mono">
        <span><i class="sw" style="background:#09090b"></i>${mi.base_model}: ${(mi.xlm_params / 1e6).toFixed(2)}M (${xlmPct.toFixed(1)}%)</span>
        <span><i class="sw" style="background:#2563eb"></i>LinearSVC (${mi.svc_vocab_size.toLocaleString()} feats): ${(mi.svc_params / 1e6).toFixed(2)}M (${svcPct.toFixed(2)}%)</span>
      </div>
    </div>
    <table class="audit-table">
      <thead><tr><th>Component</th><th>Parameters</th><th>Share of 300M</th></tr></thead>
      <tbody>
        <tr><td><code>${mi.base_model}</code></td><td class="mono">${mi.xlm_params.toLocaleString()}</td><td class="mono">${xlmPct.toFixed(2)}%</td></tr>
        <tr><td>Calibrated <code>LinearSVC</code> (${mi.svc_vocab_size.toLocaleString()} × 56 + 56)</td><td class="mono">${mi.svc_params.toLocaleString()}</td><td class="mono">${svcPct.toFixed(2)}%</td></tr>
        <tr><td><strong>Total Active</strong></td><td class="mono"><strong>${mi.total_active_params.toLocaleString()}</strong></td><td class="mono"><strong>${(xlmPct + svcPct).toFixed(2)}%</strong></td></tr>
        <tr><td class="muted">Remaining Headroom</td><td class="mono" style="color:#15803d">+${mi.headroom_params.toLocaleString()}</td><td class="mono" style="color:#15803d">${mi.headroom_pct}% free</td></tr>
      </tbody>
    </table>
  `;

  const cvFeats = data.benchmark?.cv_feature_comparison || [];
  const sbWrap = $("script-benchmark-bars");
  sbWrap.innerHTML = "";
  cvFeats.forEach((row) => {
    const el = document.createElement("div");
    el.className = "bench-bar-group";
    el.innerHTML = `
      <div class="bench-bar-head">
        <span><strong>${escapeHtml(row.method)}</strong></span>
        <span class="mono">OOF: <strong>${row.oof_macro_f1.toFixed(4)}</strong></span>
      </div>
      <div class="bench-track"><div class="bench-fill-ens" style="width:${(row.oof_macro_f1 * 100).toFixed(1)}%"></div></div>
    `;
    sbWrap.appendChild(el);
  });

  const baselines = data.benchmark?.baselines || [];
  let tbl = `<table class="audit-table"><thead><tr><th>Model</th><th>Mean CV F1</th><th>OOF F1</th><th>Latency</th><th>Params</th></tr></thead><tbody>`;
  baselines.forEach((b) => {
    tbl += `<tr><td>${escapeHtml(b.model)}</td><td class="mono">${b.mean_macro_f1.toFixed(4)}</td><td class="mono">${b.oof_macro_f1.toFixed(4)}</td><td class="mono">${b.mean_latency_ms} ms</td><td class="mono">${escapeHtml(b.parameters)}</td></tr>`;
  });
  tbl += `</tbody></table>`;
  $("baseline-table-wrap").innerHTML = tbl;
}

function renderTaxonomy(filterText) {
  const classes = state.systemStats?.classes || [];
  const q = filterText.trim().toLowerCase();
  const filtered = classes.filter((c) => {
    if (!q) return true;
    return (
      String(c.target) === q ||
      c.intent_name.toLowerCase().includes(q) ||
      c.domain.toLowerCase().includes(q) ||
      c.samples.some((s) => s.toLowerCase().includes(q))
    );
  });

  const grid = $("taxonomy-grid");
  grid.innerHTML = "";
  filtered.forEach((c) => {
    const card = document.createElement("div");
    card.className = "tax-card";
    card.innerHTML = `
      <div class="tax-top">
        <strong class="mono">[${c.target}] ${escapeHtml(c.intent_name)}</strong>
        <span class="tag">${escapeHtml(c.domain)}</span>
      </div>
      <div class="micro-note mono">n=${c.train_samples} · purity=${Number(c.purity).toFixed(1)}%</div>
      ${c.extra_note ? `<div class="micro-note" style="color:#b45309;">${escapeHtml(c.extra_note)}</div>` : ""}
      <div class="tax-samples"></div>
    `;
    const sampleBox = card.querySelector(".tax-samples");
    c.samples.forEach((s) => {
      const b = document.createElement("button");
      b.className = "tax-sample-btn";
      b.textContent = s;
      b.addEventListener("click", () => {
        $("query-input").value = s;
        document.querySelector('.tab[data-tab="live"]').click();
        runPredict();
      });
      sampleBox.appendChild(b);
    });
    grid.appendChild(card);
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initModeToggle();
  initSlidersAndPresets();
  renderSessionLatency();
  await fetchSystemStats();
  await runPredict();
  await runMultiScript();
});
