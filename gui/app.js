/* Name That LLM: a guided, interactive look at the model.
   Reads window.NTL (data.js) and window.NTL_PASSAGES (passages.js); see build_data.py. */
(function () {
  "use strict";
  const D = window.NTL;
  if (!D) {
    document.querySelector("main").innerHTML =
      "<h2>No data yet</h2><p>Run <code>uv run python gui/build_data.py</code> from the project root, then reload.</p>";
    return;
  }
  const PASSAGES = window.NTL_PASSAGES || [];
  const N = D.n, K = D.labels.length, L = D.labels, J = D.order;
  const FAMS = [...new Set(D.families)];
  const FAM_NAME = { claude: "Claude", gemini: "Gemini", glm: "GLM", gpt: "GPT", kimi: "Kimi", qwen: "Qwen" };
  const famName = (f) => FAM_NAME[f] || f;
  const color = (j) => `var(--f-${D.families[j]})`;

  // ------------------------------------------------------------------ small helpers
  const $ = (id) => document.getElementById(id);
  function el(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k === "style") e.style.cssText = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of [].concat(kids || [])) if (c != null && c !== false) e.append(c);
    return e;
  }
  const fill = (id, kids) => { const n = $(id); n.replaceChildren(...[].concat(kids).filter((k) => k != null && k !== false)); return n; };
  const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
  const pts = (x) => (x >= 0 ? "+" : "−") + Math.abs(x * 100).toFixed(1);
  const chip = (j) => el("span", { class: "chip", style: `--c:${color(j)}`, text: L[j] });
  const mark = (ok) => el("span", { class: ok ? "ok" : "no", text: ok ? "✓" : "✗" });
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const list = (a) => (a.length > 1 ? a.slice(0, -1).join(", ") + " and " + a[a.length - 1] : a[0]);

  // ------------------------------------------------------------------ the blend, in the browser
  function logSoftmax(flat, scale) {
    const out = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      let mx = -Infinity, s = 0;
      for (let j = 0; j < K; j++) mx = Math.max(mx, flat[i * K + j] * scale);
      for (let j = 0; j < K; j++) s += Math.exp(flat[i * K + j] * scale - mx);
      const lse = mx + Math.log(s);
      for (let j = 0; j < K; j++) out[i * K + j] = flat[i * K + j] * scale - lse;
    }
    return out;
  }
  function normalise(logp) {
    const p = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      let mx = -Infinity, s = 0;
      for (let j = 0; j < K; j++) mx = Math.max(mx, logp[i * K + j]);
      for (let j = 0; j < K; j++) s += p[i * K + j] = Math.exp(logp[i * K + j] - mx);
      for (let j = 0; j < K; j++) p[i * K + j] /= s;
    }
    return p;
  }
  function sinkhorn(proba, strength) {
    if (strength <= 0) return proba;
    const q = proba.map((v) => v + 1e-12), col = new Float64Array(K);
    for (let it = 0; it < 50; it++) {
      col.fill(0);
      for (let i = 0; i < N; i++) for (let j = 0; j < K; j++) col[j] += q[i * K + j];
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let j = 0; j < K; j++) s += q[i * K + j] = (q[i * K + j] / col[j]) * (N / K);
        for (let j = 0; j < K; j++) q[i * K + j] /= s;
      }
    }
    const logp = new Float64Array(N * K);
    for (let x = 0; x < N * K; x++) logp[x] = (1 - strength) * Math.log(proba[x] + 1e-12) + strength * Math.log(q[x]);
    return normalise(logp);
  }
  const LS = {};
  for (const c of J) if (c !== "svm") LS[c] = logSoftmax(D.scores[c], 1);
  function argmax(p) {
    const out = new Int32Array(N);
    for (let i = 0; i < N; i++) {
      let b = 0;
      for (let j = 1; j < K; j++) if (p[i * K + j] > p[i * K + b]) b = j;
      out[i] = b;
    }
    return out;
  }
  /** w: weight per judge (for the style counter, a temperature) plus strength. */
  function blend(w) {
    const logp = w.svm > 0 ? logSoftmax(D.scores.svm, w.svm) : new Float64Array(N * K);
    for (const c of J) if (c !== "svm" && w[c] > 0) for (let x = 0; x < N * K; x++) logp[x] += w[c] * LS[c][x];
    const raw = normalise(logp), proba = sinkhorn(raw, w.strength);
    return { raw, proba, rawPred: argmax(raw), pred: argmax(proba) };
  }
  function score(pred) {
    const hit = new Float64Array(K), sup = new Float64Array(K);
    for (let i = 0; i < N; i++) { sup[D.y[i]]++; if (pred[i] === D.y[i]) hit[D.y[i]]++; }
    const recall = Array.from(hit, (h, j) => h / sup[j]);
    return { bal: recall.reduce((a, b) => a + b, 0) / K, recall };
  }
  const NEWEST = { ...D.candidate, strength: D.strength }, BOARD = { ...D.submitted, strength: D.strength };
  const CUR = blend(NEWEST), curScore = score(CUR.pred);
  const boardScore = score(blend(BOARD).pred);
  const JP = { svm: normalise(logSoftmax(D.scores.svm, NEWEST.svm)) };
  for (const c of J) if (c !== "svm") JP[c] = normalise(LS[c]);
  const JPRED = Object.fromEntries(J.map((c) => [c, argmax(JP[c])]));
  function top(P, i, n = 3) {
    return Array.from({ length: K }, (_, j) => [j, P[i * K + j]]).sort((a, b) => b[1] - a[1]).slice(0, n);
  }
  /** Top guesses as labelled bars; the true author is bold, and added if it is not in the top. */
  function guesses(P, i) {
    const y = D.y[i], rows = top(P, i);
    if (!rows.some(([j]) => j === y)) rows.push([y, P[i * K + y]]);
    return rows.map(([j, p]) => el("div", { class: "guess" + (j === y ? " truth" : "") }, [
      el("span", { class: "name", style: `--c:${color(j)}`, text: L[j] + (j === y ? " ✓" : ""), title: L[j] }),
      el("div", { class: "meter" }, el("i", { style: `width:${Math.max(1, p * 100)}%;--c:${color(j)}` })),
      el("span", { class: "pct", text: p < 0.005 ? "<1%" : pct(p) }),
    ]));
  }

  // ------------------------------------------------------------------ header + tabs
  fill("scoreline", [
    [pct(1 / K), "blind guess"], [pct(curScore.bal), "own test, 5 judges"], ["61.1%", "leaderboard, 4 judges"],
  ].map(([n, t]) => el("div", {}, [el("b", { text: n }), el("span", { text: t })])));
  const TABS = [["play", "Play"], ["how", "How it decides"], ["spots", "Blind spots"], ["mix", "Mix it"], ["next", "What next"]];
  function show(id) {
    for (const [t] of TABS) $("v-" + t).hidden = t !== id;
    fill("tabs", TABS.map(([t, name], n) => el("button", { type: "button", role: "tab", class: t === id ? "on" : "", "aria-selected": t === id, onclick: () => show(t) },
      [el("i", { text: n + 1 }), el("span", { text: name })])));
    if (id === "how") drawHow();
    if (location.hash.slice(1) !== id) history.replaceState(null, "", "#" + id);
    window.scrollTo(0, 0);
  }

  // ------------------------------------------------------------------ 1. play
  const fair = PASSAGES.filter((p) => !p.extra).sort(() => Math.random() - 0.5);
  const textOf = new Map(PASSAGES.map((p) => [p.i, p.text]));
  const play = { pos: 0, guess: null, clues: false, n: 0, you: 0, model: 0, exact: 0 };
  const CLUE_RE = { em_dash: /—/g, curly_quote: /[‘’“”]/g, straight_quote: /["']/g, semicolon: /;/g, colon: /:/g, parens: /\(/g };
  function cluesOf(text) {
    const out = [];
    for (const [f, re] of Object.entries(CLUE_RE)) {
      const n = (text.match(re) || []).length;
      if (n) out.push([`${D.clues[f].name}: ${n}`, D.clues[f].top]);
    }
    const words = text.split(/\s+/).filter(Boolean).length;
    const wps = words / Math.max(1, text.trim().split(/(?<=[.!?])\s+/).filter(Boolean).length);
    if (wps >= 24) out.push([`long sentences: ${Math.round(wps)} words each`, D.clues.words_per_sentence.top]);
    return out;
  }
  function passageNode(text, marks) {
    if (!marks) return [text];
    return text.split(/([—;:()‘’“”"'])/).map((part, n) => (n % 2 ? el("mark", { text: part }) : part));
  }
  function drawPlay() {
    if (!fair.length) {
      fill("play-text", "The passages are not bundled. Run gui/build_data.py to create passages.js, then reload.").className = "passage empty";
      $("play-clues").hidden = true;
      return;
    }
    const p = fair[play.pos % fair.length], y = D.y[p.i], fam = D.families[y], done = play.guess != null;
    fill("play-text", passageNode(p.text, play.clues));
    $("play-meta").textContent = `${D.words[p.i]} words`;
    $("play-clues").textContent = play.clues ? "Hide clues" : "Show clues";
    const clues = play.clues ? cluesOf(p.text) : [];
    fill("play-cluelist", !play.clues ? [] : clues.length
      ? clues.map(([what, who]) => el("div", { class: "clue" }, [el("b", { text: what }), ` · most typical of ${list(who)}`]))
      : el("div", { class: "clue", text: "No punctuation clues here. The wording has to do the work." }));
    fill("play-buttons", FAMS.map((f) => el("button", {
      type: "button", style: `--c:var(--f-${f})`, disabled: done,
      class: (play.guess === f ? "picked " : "") + (done && f === fam ? "truth" : ""),
      onclick: () => { play.guess = f; play.n++; play.you += f === fam; play.model += D.families[CUR.pred[p.i]] === fam; play.exact += CUR.pred[p.i] === y; drawPlay(); },
    }, [el("b", { text: famName(f) + (done && f === fam ? " ✓" : "") }), el("span", { text: `${D.families.filter((x) => x === f).length} model${D.families.filter((x) => x === f).length > 1 ? "s" : ""}` })])));
    if (!done) { fill("play-reveal", []); }
    else {
      const m = CUR.pred[p.i], youOk = play.guess === fam, mFam = D.families[m] === fam;
      fill("play-reveal", el("div", { class: "reveal" }, [
        el("div", { class: "answer" }, ["Written by ", chip(y), ` from the ${famName(fam)} family.`]),
        el("div", { class: "duo" }, [
          el("div", {}, [el("small", { text: "You" }), el("div", { class: "big" }, [mark(youOk), youOk ? " Right family" : ` You said ${famName(play.guess)}`])]),
          el("div", {}, [el("small", { text: "The model" }), el("div", { class: "big" }, [mark(m === y), m === y ? " Exactly right" : mFam ? " Right family, wrong model" : " Wrong family"]),
            m !== y ? el("div", { style: "margin-top:4px" }, ["It said ", chip(m)]) : null]),
        ]),
        el("div", { class: "votes" }, J.map((c) => el("div", { class: "vote" }, [
          el("span", { text: D.judges[c].name }), el("span", {}, chip(JPRED[c][p.i])), mark(JPRED[c][p.i] === y)]))),
        el("div", { class: "actions" }, [
          el("button", { class: "primary", type: "button", text: "Next passage →", onclick: () => { play.pos++; play.guess = null; drawPlay(); window.scrollTo(0, 0); } }),
          el("button", { type: "button", text: "See how it decided", onclick: () => { how.row = p.i; how.step = 1; show("how"); } }),
        ]),
      ]));
    }
    $("play-score").hidden = !play.n;
    fill("play-score", [
      [`${play.you} / ${play.n}`, "you, right family"], [`${play.model} / ${play.n}`, "model, right family"], [`${play.exact} / ${play.n}`, "model, exact author"],
    ].map(([n, t]) => el("div", {}, [el("b", { text: n }), el("span", { text: t })])));
  }
  $("play-clues").addEventListener("click", () => { play.clues = !play.clues; drawPlay(); });

  // ------------------------------------------------------------------ 2. how it decides
  const how = { row: fair.length ? fair[0].i : 0, step: 1 };
  const STEPS = ["Read", "Guess", "Combine", "Even out"];
  const changed = (() => { let n = 0; for (let i = 0; i < N; i++) n += CUR.pred[i] !== CUR.rawPred[i]; return n; })();
  function judgeCard(c, body, extra) {
    const j = D.judges[c];
    return el("div", { class: "judge" }, [
      el("header", {}, [el("b", {}, [j.name, c === "q14" ? el("span", { class: "tag", text: "new" }) : null]), el("span", { text: extra })]),
      el("p", { text: j.role }), ...[].concat(body || []),
    ]);
  }
  function drawHow() {
    const i = how.row, y = D.y[i], s = how.step, text = textOf.get(i);
    fill("how-rail", STEPS.map((name, n) => el("button", { type: "button", class: n + 1 === s ? "on" : n + 1 < s ? "done" : "", onclick: () => { how.step = n + 1; drawHow(); } },
      [el("b", { text: `Step ${n + 1}` }), el("span", { text: name })])));
    const box = fill("how-text", text ? [text] : ["Passage text is not bundled (run gui/build_data.py). The scores below are real."]);
    box.className = "passage small" + (text ? "" : " empty");
    const truth = el("p", { class: "muted", style: "margin:0 0 12px" }, ["True author: ", chip(y), `  ·  ${D.words[i]} words`]);
    let title, say, body;
    if (s === 1) {
      title = "Five judges read the passage";
      say = "No judge knows the answer. Each one learned from passages whose author was known, then gives its own opinion.";
      body = el("div", { class: "judges" }, J.map((c) => judgeCard(c,
        el("p", { class: "muted", style: "margin:0", text: D.judges[c].kind === "svm" ? "Looks only at the surface: spelling, punctuation, word choice." : `Reads the text itself. Practised on ${D.judges[c].pool_k},000 extra passages.` }),
        `right ${pct(D.judges[c].oof)} alone`)));
    } else if (s === 2) {
      title = "Each judge names its top suspects";
      say = "A longer bar means the judge is more sure. The true author is in bold with a tick.";
      body = el("div", { class: "judges" }, J.map((c) => judgeCard(c, guesses(JP[c], i), JPRED[c][i] === y ? "✓ right" : "✗ wrong")));
    } else if (s === 3) {
      title = "The opinions are combined";
      say = "Stronger judges get more say. The confidences are multiplied, so an author stays likely only if the judges broadly agree.";
      const maxW = Math.max(...J.map((c) => NEWEST[c]));
      body = el("div", { class: "judges" }, [
        el("div", { class: "judge" }, [el("header", {}, el("b", { text: "How much each judge counts" })),
          ...J.map((c) => el("div", { class: "guess" }, [el("span", { class: "name", style: "--c:var(--muted)", text: D.judges[c].name }),
            el("div", { class: "meter" }, el("i", { style: `width:${(NEWEST[c] / maxW) * 100}%` })), el("span", { class: "pct", text: "×" + NEWEST[c] })]))]),
        el("div", { class: "judge final" }, [el("header", {}, [el("b", { text: "Combined opinion" }), el("span", { text: CUR.rawPred[i] === y ? "✓ right" : "✗ wrong" })]), ...guesses(CUR.raw, i)]),
      ]);
    } else {
      title = "The answers are evened out";
      say = `Each of the 18 authors should be named about equally often. Where the model leans on one author too much, its least certain answers are nudged to others. This changed ${changed} of ${N.toLocaleString()} answers.`;
      const moved = CUR.rawPred[i] !== CUR.pred[i];
      body = [el("div", { class: "judges" }, [
        el("div", { class: "judge" }, [el("header", {}, el("b", { text: "Before" })), ...guesses(CUR.raw, i)]),
        el("div", { class: "judge final" }, [el("header", {}, el("b", { text: "After evening out" })), ...guesses(CUR.proba, i)]),
      ]),
      el("div", { class: "verdict" }, [mark(CUR.pred[i] === y), " Final answer: ", chip(CUR.pred[i]),
        CUR.pred[i] === y ? " Correct." : " The author was ", CUR.pred[i] === y ? null : chip(y), CUR.pred[i] === y ? null : ".",
        moved ? " Evening out changed the answer here." : " Evening out did not change this one."])];
    }
    fill("how-body", [truth, el("p", { class: "step-title", text: title }), el("p", { class: "step-say", text: say }), ...[].concat(body)]);
    $("how-back").disabled = s === 1;
    $("how-next").textContent = s === 4 ? "See the blind spots →" : "Next →";
  }
  $("how-back").addEventListener("click", () => { how.step--; drawHow(); });
  $("how-next").addEventListener("click", () => { if (how.step === 4) show("spots"); else { how.step++; drawHow(); } });
  $("how-other").addEventListener("click", () => { how.row = fair.length ? pick(fair).i : Math.floor(Math.random() * N); how.step = Math.max(how.step, 2); drawHow(); });
  const w = NEWEST, lo = Math.min(...J.filter((c) => c !== "svm").map((c) => D.judges[c].pool_k)), hi = Math.max(...J.filter((c) => c !== "svm").map((c) => D.judges[c].pool_k));
  fill("how-hood", [
    "Style counter: TF-IDF of character 1–5-grams and word 1–2-grams, plus 16 punctuation and style rates, classified by a linear SVM.",
    "Language-model judges: the first 24 layers of Qwen3-4B, Qwen3-8B, Qwen3-14B and Mistral-7B, with LoRA adapters (r=16, alpha=32) on every linear layer. The mean hidden state at 8 depths feeds one linear layer with 18 outputs.",
    `Training: one pass over ${lo}k–${hi}k passages from the community Open Pool, then two passes over the ${N.toLocaleString()} competition passages. Six copies of each judge (five folds plus one on all data) are averaged.`,
    `Combining: log_softmax(${w.svm}·svm) + ${w.q4}·log_softmax(4B) + ${w.q8}·log_softmax(8B) + ${w.m7}·log_softmax(Mistral) + ${w.q14}·log_softmax(14B).`,
    `Evening out: Sinkhorn class balancing at strength ${D.strength}.`,
    `Own test: every passage is scored by copies that never saw its prompt (5 folds grouped by prompt, ${D.n_prompts} prompts).`,
  ].map((t) => el("li", { text: t })));

  // ------------------------------------------------------------------ 3. blind spots
  const recall = curScore.recall;
  const CONF = L.map(() => new Array(K).fill(0));
  for (let i = 0; i < N; i++) CONF[D.y[i]][CUR.pred[i]]++;
  const spots = { sort: "family", sel: recall.indexOf(Math.min(...recall)) };
  function sharedClues(a, b) {
    return Object.values(D.clues).filter((c) => c.top.includes(L[a]) && c.top.includes(L[b])).map((c) => c.name.replace(/ \(.*/, ""));
  }
  function tile(j) {
    return el("button", { type: "button", class: "tile" + (j === spots.sel ? " on" : "") + (recall[j] < 0.5 ? " low" : ""), style: `--c:${color(j)};--w:${recall[j] * 100}%`,
      onclick: () => { spots.sel = j; drawSpots(); } }, [el("b", { text: pct(recall[j]) }), el("span", { text: L[j], title: L[j] })]);
  }
  function drawSpots() {
    fill("spots-sort", [["family", "By family"], ["hard", "Hardest first"]].map(([k, t]) =>
      el("button", { type: "button", class: spots.sort === k ? "on" : "", text: t, onclick: () => { spots.sort = k; drawSpots(); } })));
    const all = L.map((_, j) => j);
    fill("lineup", spots.sort === "hard"
      ? el("div", { class: "tiles" }, all.sort((a, b) => recall[a] - recall[b]).map(tile))
      : FAMS.flatMap((f) => [el("div", { class: "fam-head", style: `--c:var(--f-${f})`, text: famName(f) }),
        el("div", { class: "tiles" }, all.filter((j) => D.families[j] === f).map(tile))]));
    const j = spots.sel, row = CONF[j], total = row.reduce((a, b) => a + b, 0);
    const others = row.map((v, c) => [c, v]).filter(([c, v]) => c !== j && v > 0).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const misses = PASSAGES.filter((p) => D.y[p.i] === j && CUR.pred[p.i] !== j);
    let why = "";
    if (others.length) {
      const [o] = others[0], shared = sharedClues(j, o);
      why = D.families[o] === D.families[j]
        ? `${L[o]} is a sibling from the same maker, so the two write alike.`
        : `${L[o]} comes from a different maker` + (shared.length ? `, but both are among the heaviest users of ${list(shared)}.` : ", so something in their style overlaps.");
    }
    fill("spot-detail", [
      el("h3", {}, chip(j)),
      el("p", { class: "bigline" }, [el("b", { text: pct(recall[j]) }), ` caught · ${row[j]} of ${total} passages`]),
      others.length ? el("h4", { text: "When it gets away, the model says" }) : null,
      ...others.map(([c, v]) => el("div", { class: "guess" }, [el("span", { class: "name", style: `--c:${color(c)}`, text: L[c], title: L[c] }),
        el("div", { class: "meter" }, el("i", { style: `width:${(v / total) * 100}%;--c:${color(c)}` })), el("span", { class: "pct", text: pct(v / total) })])),
      why ? el("p", { class: "why", text: why }) : null,
      el("div", { class: "actions" }, el("button", { type: "button", disabled: !misses.length, text: misses.length ? "Show me a miss →" : "No miss in the sample",
        onclick: () => { how.row = pick(misses).i; how.step = 2; show("how"); } })),
    ]);
  }
  const S = D.stats, first = D.length[0], last = D.length[D.length.length - 1];
  fill("facts", [
    el("div", { class: "fact" }, [el("div", { class: "n", text: `${Math.round((S.errors_in_family / S.errors) * 10)} in 10` }),
      el("p", { text: "mistakes are sibling mix-ups: the right family, but the wrong model inside it." })]),
    el("div", { class: "fact" }, [el("div", { class: "n", text: `${pct(first.acc)} → ${pct(last.acc)}` }),
      el("p", { text: `Short passages are much harder. Under ${first.hi} words against ${last.lo} or more.` }),
      el("div", { class: "lenbars" }, D.length.map((d) => el("div", { style: `height:${d.acc * 100}%`, title: `${d.lo}–${d.hi > 500 ? "274" : d.hi - 1} words: ${pct(d.acc)} right (${d.n} passages)` }, el("span", { text: Math.round(d.acc * 100) })))),
      el("div", { class: "lenaxis" }, [el("span", { text: "← shorter" }), el("span", { text: "% right, by length" }), el("span", { text: "longer →" })])]),
    el("div", { class: "fact" }, [el("div", { class: "n", text: `1 in ${Math.round(1 / (1 - S.sure_acc))}` }),
      el("p", { text: `of its near-certain answers is wrong. On ${S.sure_n.toLocaleString()} passages it is ${pct(S.sure_conf)} sure, but right only ${pct(S.sure_acc)} of the time.` })]),
  ]);

  // ------------------------------------------------------------------ 4. mixer
  const best = J.reduce((a, c) => (D.judges[c].oof > D.judges[a].oof ? c : a));
  const zero = Object.fromEntries(J.map((c) => [c, 0]));
  const PRESETS = [
    ["On the leaderboard", BOARD], ["Newest, with 14B", NEWEST],
    [`${D.judges[best].name} alone`, { ...zero, [best]: 1, strength: D.strength }],
    ["Equal say", { ...Object.fromEntries(J.map((c) => [c, 1])), strength: D.strength }],
  ];
  let mix = { ...NEWEST };
  const same = (a, b) => [...J, "strength"].every((k) => a[k] === b[k]);
  function drawMix(full) {
    fill("mix-presets", PRESETS.map(([name, p]) => el("button", { type: "button", class: same(mix, p) ? "on" : "", text: name, onclick: () => { mix = { ...p }; drawMix(true); } })));
    if (full) {
      fill("mix-sliders", [...J, "strength"].map((c) => {
        const j = D.judges[c], out = el("output", { text: c === "strength" ? pct(mix[c]) : "×" + mix[c].toFixed(2) });
        const box = el("div", { class: "slider" + (mix[c] === 0 ? " off" : "") }, [
          el("div", { class: "top" }, [el("div", {}, [el("b", { text: j ? j.name : "Evening out" }), c === "q14" ? el("span", { class: "tag", text: "new" }) : null,
            el("span", { text: "  " + (j ? j.role : "how hard answers are balanced") })]), out]),
          el("input", { type: "range", min: 0, max: c === "strength" ? 1 : c === "svm" ? 4 : 3, step: c === "strength" ? 0.05 : 0.25, value: mix[c], "aria-label": j ? j.name : "Evening out",
            oninput: (e) => { mix[c] = +e.target.value; out.textContent = c === "strength" ? pct(mix[c]) : "×" + mix[c].toFixed(2); box.classList.toggle("off", mix[c] === 0); drawMix(false); } }),
        ]);
        return box;
      }));
    }
    if (!J.some((c) => mix[c] > 0)) { fill("mix-dial", el("p", { class: "muted", text: "Give at least one judge some say." })); fill("mix-movers", []); return; }
    const s = score(blend(mix).pred), d = s.bal - boardScore.bal, g = (v) => Math.max(0, Math.min(100, ((v - 0.4) / 0.3) * 100));
    fill("mix-dial", [
      el("div", { class: "n", text: pct(s.bal, 1) }),
      el("div", { class: "d" }, Math.abs(d) < 5e-4 ? "Same as the leaderboard blend" : [el("span", { class: d > 0 ? "ok" : "no", text: `${pts(d)} points` }), " against the leaderboard blend"]),
      el("div", { class: "gauge" }, [el("i", { style: `width:${g(s.bal)}%` }), el("u", { style: `left:${g(boardScore.bal)}%` }, el("span", { text: "leaderboard blend" })),
        el("em", { style: "left:0", text: "40%" }), el("em", { style: "right:0", text: "70%" })]),
      el("div", { class: "muted", text: `Share of authors named correctly on ${N.toLocaleString()} unseen passages` }),
    ]);
    const moves = L.map((_, j) => [j, s.recall[j] - boardScore.recall[j]]).filter(([, v]) => Math.abs(v) >= 0.01).sort((a, b) => b[1] - a[1]);
    const up = moves.filter(([, v]) => v > 0).slice(0, 4), down = moves.filter(([, v]) => v < 0).slice(-4).reverse();
    const row = (title, items) => items.length ? [el("h4", { text: title }), el("div", {}, items.map(([j, v]) => el("span", { class: "chip", style: `--c:${color(j)}`, text: `${L[j]} ${pts(v)}` })))] : [];
    fill("mix-movers", [...row("Caught more often than on the leaderboard", up), ...row("Caught less often", down)]);
  }

  // ------------------------------------------------------------------ 5. what next
  const JOURNEY = [
    [0.414, "Style counter alone", "Counting characters and punctuation, with extra training passages."],
    [0.537, "+ Qwen3-4B", "The first language-model judge joins."],
    [0.544, "+ Qwen3-8B", "A second, larger language model."],
    [0.563, "More practice for the 4B", "Retrained on 96,000 extra passages instead of 36,000."],
    [0.569, "More practice for the 8B", "Retrained on 88,000 extra passages."],
    [0.589, "Six copies of the 4B", "186,000 extra passages; five fold copies plus one on all data."],
    [0.607, "Six copies of the 8B too", "Older, weaker judges removed. Fewer but better judges."],
    [0.611, "+ Mistral-7B", "A judge from a different model family. This is the current leaderboard score."],
    [null, "+ Qwen3-14B", `Trained, not submitted yet. On our own test: ${pct(S.candidate, 1)} against ${pct(S.submitted, 1)}.`],
  ];
  let jSel = JOURNEY.length - 2;
  function drawJourney() {
    const W = 640, H = 150, X = (n) => 24 + (n / (JOURNEY.length - 1)) * (W - 48), Y = (v) => 18 + (1 - (v - 0.38) / 0.27) * (H - 44);
    const ns = "http://www.w3.org/2000/svg", s = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; return e; };
    const g = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Leaderboard score after each submission" });
    const done = JOURNEY.filter((j) => j[0] != null), lastY = Y(done[done.length - 1][0]);
    g.append(s("polyline", { fill: "none", stroke: "var(--ink2)", "stroke-width": 2, points: done.map((j, n) => `${X(n)},${Y(j[0])}`).join(" ") }));
    g.append(s("line", { x1: X(done.length - 1), y1: lastY, x2: X(done.length), y2: lastY - 16, stroke: "var(--muted)", "stroke-width": 2, "stroke-dasharray": "4 4" }));
    JOURNEY.forEach(([v, name], n) => {
      const cy = v == null ? lastY - 16 : Y(v), on = n === jSel;
      const c = s("circle", { class: "pt", cx: X(n), cy, r: on ? 8 : 6, fill: v == null ? "var(--card)" : on ? "var(--accent)" : "var(--ink2)", stroke: on ? "var(--accent)" : v == null ? "var(--muted)" : "var(--card)", "stroke-width": 2, tabindex: 0, role: "button", "aria-label": name });
      c.addEventListener("click", () => { jSel = n; drawJourney(); });
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { jSel = n; drawJourney(); } });
      g.append(c, s("text", { x: X(n), y: cy - 13, "text-anchor": "middle", style: on ? "fill:var(--ink);font-weight:700" : "" }, v == null ? "next" : (v * 100).toFixed(1)));
    });
    g.append(s("text", { x: 24, y: H - 4 }, "first submission"), s("text", { x: W - 24, y: H - 4, "text-anchor": "end" }, "leaderboard score, %"));
    const [v, name, what] = JOURNEY[jSel];
    fill("journey", [g, el("p", { class: "cap" }, [el("b", { text: name + (v == null ? "" : ` · ${(v * 100).toFixed(1)}%`) }), el("br"), what])]);
  }
  const idx = (m) => L.indexOf(m), r = (m) => pct(recall[idx(m)]);
  const RECS = [
    { n: `${pts(S.candidate - S.submitted)}`, u: "points, own test", t: "Submit the five-judge blend", s: "The new 14B judge is trained but not on the leaderboard yet.",
      why: `On our own test the blend with the 14B judge scores ${pct(S.candidate, 1)} against ${pct(S.submitted, 1)}. It wins in ${Math.round(S.cand_wins * 1000)} of 1,000 resamples, by ${pts(S.cand_delta_lo)} to ${pts(S.cand_delta_hi)} points.`,
      steps: ["The notebook is built: notebooks/kaggle_submission_lora_4bfull_8bfull_mistral7bfull_14bfull.ipynb.", "Watch the 9-hour limit. The 14B needs both T4s and runs six copies. If it is too slow, use fewer copies of it.", "Expect the leaderboard to read 2–3 points below our own test, as it did before (63.6% against 61.1%)."], tags: ["Effort: low", "GPU: one submission run"] },
    { n: pts(S.cand_without_q4 - S.candidate), u: "points if removed", t: "Retire the 4B judge", s: "With the 14B on board, the smallest language model no longer adds anything.",
      why: `Without the 4B the five-judge blend scores ${pct(S.cand_without_q4, 1)} against ${pct(S.candidate, 1)} with it. Try it in the mixer: set Qwen3-4B to zero.`,
      steps: ["Drop its six copies from the submission. That frees run time for the 14B.", "Keep its training outputs. It costs nothing to bring back."], tags: ["Effort: low", "GPU: none"], go: ["mix", "Open the mixer"] },
    { n: r("glm-5.3"), u: "of glm-5.3 caught", t: "Untangle claude-fable-5 from the GLM pair", s: "The three hardest suspects are confused with each other across families.",
      why: `Only ${r("glm-5.3")} of glm-5.3, ${r("claude-fable-5")} of claude-fable-5 and ${r("glm-5.3-flash")} of glm-5.3-flash passages are caught. All three are the heaviest users of em dashes, so surface style cannot separate them.`,
      steps: ["Read a few misses in Blind spots and look for what does differ: topics, openings, sentence rhythm.", "Check the Open Pool rows for these three labels by source. One mislabelled source would cause exactly this.", "If the labels are clean, train a small three-way judge on these classes only, and let it decide when the blend's top two are among them."], tags: ["Effort: low to start", "GPU: little"], go: ["spots", "Open Blind spots", "glm-5.3"] },
    { n: `1 in ${Math.round(1 / (1 - S.sure_acc))}`, u: "sure answers wrong", t: "Make it less sure before evening out", s: "Evening out works on confidence, and the confidence is inflated.",
      why: `On ${S.sure_n.toLocaleString()} passages the blend is ${pct(S.sure_conf)} sure but right ${pct(S.sure_acc)} of the time. Evening out can only move answers the model admits to being unsure about.`,
      steps: ["Divide the combined log-probabilities by one temperature fitted on the own-test answers.", "postprocess.py already simulates test sets with uneven author mixes, but only for the style counter. Run it on the blend to pick temperature and strength together."], tags: ["Effort: low", "GPU: none"] },
    { n: `${Math.round((S.errors_in_family / S.errors) * 10)} in 10`, u: "mistakes are siblings", t: "Teach it to tell siblings apart", s: "It usually finds the family. Picking the model inside it is the hard part.",
      why: `${S.errors_in_family} of ${S.errors} mistakes stay inside the right family. If the family were always right, the score would be ${pct(S.family_fixed, 1)}. The pool has only ${D.pool.counts[idx("gemini-3.7-flash")].toLocaleString()} gemini-3.7-flash and ${D.pool.counts[idx("qwen3.8-max")].toLocaleString()} qwen3.8-max passages.`,
      steps: ["The rules allow generating data through the APIs. Ask each sibling the same new prompts, so only the model differs.", "For gemini-3.7-flash use the Vertex AI route only. Other routes answer with 3.8 and would mislabel the data.", "Match the competition format: one plain paragraph, 36 to 274 words."], tags: ["Effort: medium", "Cost: API calls"] },
    { n: pct(first.acc), u: `right under ${first.hi} words`, t: "Give short passages special treatment", s: `Long passages reach ${pct(last.acc)}. Half of the passages are 101 words or shorter.`,
      why: "A short passage gives every judge less to go on, so their opinions are noisier.",
      steps: ["Test whether short passages prefer different judge weights. If so, weight by length.", "Average more copies or seeds for short passages, which reduces noise.", "Others report that training on cut-down copies did not help, so skip that."], tags: ["Effort: medium", "GPU: little"] },
    { n: `±${(S.boot_sd * 100).toFixed(1)}`, u: "points of noise", t: "Ignore gains smaller than a point", s: "Small changes cannot be told apart from luck, and big judges are expensive.",
      why: `Our own-test score moves by about ${(S.boot_sd * 100).toFixed(1)} points from resampling alone. A learned combiner scores ${pct(S.stacked, 1)}, worse than the simple weights. The 14B judge took ${D.judges.q14.hours} GPU-hours, nearly a full week of Kaggle quota.`,
      steps: ["Keep the simple weighted blend.", "Spend a submission only on a gain of about half a point or more on all 2,699 passages.", "Try the cheap changes above before training another large judge."], tags: ["Effort: none"] },
  ];
  let open = 0;
  function drawRecs() {
    fill("recs", RECS.map((x, n) => el("div", { class: "rec" + (n === open ? " open" : "") }, [
      el("button", { type: "button", "aria-expanded": n === open, onclick: () => { open = open === n ? -1 : n; drawRecs(); } }, [
        el("div", { class: "num" }, [x.n, el("small", { text: x.u })]),
        el("div", {}, [el("div", { class: "t", text: `${n + 1}. ${x.t}` }), el("div", { class: "s", text: x.s })]),
        el("span", { class: "chev", text: "›" })]),
      el("div", { class: "body" }, [el("p", { text: x.why }), el("ol", {}, x.steps.map((t) => el("li", { text: t }))),
        ...x.tags.map((t) => el("span", { class: "pill", text: t })),
        x.go ? el("div", { class: "actions" }, el("button", { type: "button", text: x.go[1] + " →", onclick: () => { if (x.go[2]) { spots.sel = idx(x.go[2]); drawSpots(); } show(x.go[0]); } })) : null]),
    ])));
  }

  drawPlay(); drawSpots(); drawMix(true); drawJourney(); drawRecs();
  const fromHash = () => { const h = location.hash.slice(1); show(TABS.some(([t]) => t === h) ? h : "play"); };
  window.addEventListener("hashchange", fromHash);
  fromHash();
})();
