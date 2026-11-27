/**
 * app.js — MarkovPredict Frontend Logic
 * Handles navigation, API calls, and dynamic UI rendering.
 */

// ═══════════════════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════════════════

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.section;

    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

    btn.classList.add('active');
    document.getElementById(`section-${target}`).classList.add('active');
  });
});

// ═══════════════════════════════════════════════════
//  LOADING STATE
// ═══════════════════════════════════════════════════

function showLoader() {
  document.getElementById('loadingOverlay').classList.add('active');
}
function hideLoader() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

// ═══════════════════════════════════════════════════
//  API HELPER
// ═══════════════════════════════════════════════════

async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return res.json();
}

async function apiGet(url) {
  const res = await fetch(url);
  return res.json();
}

// ═══════════════════════════════════════════════════
//  STARTUP — load hero stats
// ═══════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', async () => {
  await loadHeroStats();
});

async function loadHeroStats() {
  try {
    const data = await apiGet('/stats');
    if (data.corpus_stats) {
      const cs = data.corpus_stats;
      animateNumber('hs-vocab',    cs.vocab_size);
      animateNumber('hs-bigrams',  cs.unique_bigrams);
      animateNumber('hs-trigrams', cs.unique_trigrams);
      animateNumber('hs-tokens',   cs.total_tokens);
    }
  } catch (e) {
    console.warn('Stats load failed:', e);
  }
}

function animateNumber(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  let current = 0;
  const step = Math.ceil(target / 40);
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = current.toLocaleString();
    if (current >= target) clearInterval(timer);
  }, 30);
}

// ═══════════════════════════════════════════════════
//  PREDICTOR
// ═══════════════════════════════════════════════════

function setContext(text) {
  document.getElementById('predictInput').value = text;
  doPrediction();
}

function clearPredict() {
  document.getElementById('predictInput').value = '';
  document.getElementById('predEmpty').style.display = 'flex';
  document.getElementById('predCards').style.display = 'none';
  document.getElementById('contextInfo').style.display = 'none';
}

// Allow Enter key (Ctrl+Enter for newline)
document.getElementById('predictInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    doPrediction();
  }
});

async function doPrediction() {
  const context = document.getElementById('predictInput').value.trim();
  if (!context) return;

  showLoader();
  try {
    const data = await apiPost('/predict', { context });
    hideLoader();

    if (data.error) {
      showError('predCards', data.error);
      return;
    }

    renderPredictions(data.predictions);

    // Show context words used
    const ciWords = document.getElementById('ciWords');
    if (data.predictions.length > 0) {
      const p = data.predictions[0];
      ciWords.innerHTML = '';
      [p.context_w1, p.context_w2].forEach(w => {
        if (w && w !== '<START>') {
          const span = document.createElement('span');
          span.className = 'ci-word';
          span.textContent = w;
          ciWords.appendChild(span);
        }
      });
      document.getElementById('contextInfo').style.display = 'flex';
    }
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

function renderPredictions(preds) {
  const container = document.getElementById('predCards');
  const empty     = document.getElementById('predEmpty');

  if (!preds || preds.length === 0) {
    empty.style.display = 'flex';
    container.style.display = 'none';
    return;
  }

  empty.style.display = 'none';
  container.style.display = 'flex';
  container.innerHTML = '';

  const maxProb = preds[0].probability;

  preds.forEach(p => {
    const barWidth = maxProb > 0 ? (p.probability / maxProb * 100).toFixed(1) : 0;
    const card = document.createElement('div');
    card.className = 'pred-card';
    card.innerHTML = `
      <div class="pred-rank rank-${p.rank}">#${p.rank}</div>
      <div>
        <div class="pred-word">${p.word}</div>
      </div>
      <div class="pred-details">
        <div class="pred-prob">
          P = <strong>${p.probability.toFixed(6)}</strong>
          &nbsp;·&nbsp; ${p.percent.toFixed(3)}%
        </div>
        <div class="pred-bar-wrap">
          <div class="pred-bar bar-${p.rank}" style="width: ${barWidth}%"></div>
        </div>
        <div class="pred-counts">
          Bigram count: ${p.bigram_count} &nbsp;·&nbsp; Trigram count: ${p.trigram_count}
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════
//  GENERATOR
// ═══════════════════════════════════════════════════

async function doGenerate() {
  const seed      = document.getElementById('seedInput').value.trim();
  const max_words = parseInt(document.getElementById('maxWordsSlider').value);

  showLoader();
  try {
    const data = await apiPost('/generate', { seed, max_words });
    hideLoader();

    const container = document.getElementById('genSentences');
    const empty     = document.getElementById('genEmpty');

    if (!data.sentences || data.sentences.length === 0) {
      empty.style.display = 'flex';
      container.style.display = 'none';
      return;
    }

    empty.style.display = 'none';
    container.style.display = 'block';
    container.innerHTML = '';

    data.sentences.forEach((sent, i) => {
      const div = document.createElement('div');
      div.className = 'gen-sentence';
      div.innerHTML = `
        <div class="gen-label">Generated Sentence ${i + 1}</div>
        ${sent}
      `;
      container.appendChild(div);
    });
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

// ═══════════════════════════════════════════════════
//  ANALYSIS — STATS
// ═══════════════════════════════════════════════════

async function loadStats() {
  showLoader();
  try {
    const data = await apiGet('/stats');
    hideLoader();

    if (data.corpus_stats) {
      renderCorpusStats(data.corpus_stats);
      renderSparsity(data.corpus_stats);
    }
    if (data.top_bigrams) {
      renderTopBigrams(data.top_bigrams);
    }
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

function renderCorpusStats(cs) {
  const el = document.getElementById('corpusStats');
  const items = [
    { name: 'Total Tokens',    val: cs.total_tokens.toLocaleString() },
    { name: 'Vocabulary Size', val: cs.vocab_size.toLocaleString() },
    { name: 'Unique Bigrams',  val: cs.unique_bigrams.toLocaleString() },
    { name: 'Unique Trigrams', val: cs.unique_trigrams.toLocaleString() },
    { name: 'Total Bigrams',   val: cs.total_bigrams.toLocaleString() },
    { name: 'Total Trigrams',  val: cs.total_trigrams.toLocaleString() },
  ];
  el.innerHTML = items.map(i => `
    <div class="stat-item">
      <div class="stat-name">${i.name}</div>
      <div class="stat-val">${i.val}</div>
    </div>
  `).join('');
}

function renderSparsity(cs) {
  const el = document.getElementById('sparsityBars');
  el.innerHTML = `
    <div class="sp-row">
      <div class="sp-label">
        <span>Bigram Sparsity</span>
        <span class="sp-val">${cs.bigram_sparsity}%</span>
      </div>
      <div class="sp-bar-bg">
        <div class="sp-bar sp-bigram" style="width: ${cs.bigram_sparsity}%"></div>
      </div>
    </div>
    <div class="sp-row">
      <div class="sp-label">
        <span>Trigram Sparsity</span>
        <span class="sp-val">${cs.trigram_sparsity}%</span>
      </div>
      <div class="sp-bar-bg">
        <div class="sp-bar sp-trigram" style="width: ${Math.min(cs.trigram_sparsity, 100)}%"></div>
      </div>
    </div>
  `;
}

function renderTopBigrams(bigrams) {
  const el = document.getElementById('topBigrams');
  const max = bigrams[0]?.count || 1;
  el.innerHTML = bigrams.map(b => `
    <div class="bigram-item">
      <span class="bigram-text">${b.bigram}</span>
      <div class="bigram-bar-bg">
        <div class="bigram-bar-fill" style="width: ${(b.count / max * 100).toFixed(1)}%"></div>
      </div>
      <span class="bigram-count">×${b.count}</span>
    </div>
  `).join('');
}

// ─── Accuracy ───

async function loadAccuracy() {
  showLoader();
  try {
    const data = await apiPost('/accuracy', {});
    hideLoader();

    const el = document.getElementById('accuracyResult');
    el.innerHTML = `
      <div class="acc-item">
        <div class="acc-pct acc-t1">${data.top1_accuracy}%</div>
        <div class="acc-label">Top-1 Accuracy</div>
        <div class="acc-detail">${data.top1_correct} / ${data.total_predictions}</div>
      </div>
      <div class="acc-item">
        <div class="acc-pct acc-t3">${data.top3_accuracy}%</div>
        <div class="acc-label">Top-3 Accuracy</div>
        <div class="acc-detail">${data.top3_correct} / ${data.total_predictions}</div>
      </div>
    `;
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

// ─── Perplexity ───

async function loadPerplexity() {
  showLoader();
  try {
    const data = await apiPost('/perplexity', {});
    hideLoader();

    const el = document.getElementById('perplexityResult');
    el.innerHTML = `
      <div class="pp-item">
        <div class="pp-val">${data.bigram_perplexity}</div>
        <div class="pp-label">Bigram Perplexity</div>
      </div>
      <div class="pp-item">
        <div class="pp-val">${data.trigram_perplexity}</div>
        <div class="pp-label">Trigram Perplexity</div>
      </div>
      <div class="pp-explain">${data.explanation}</div>
    `;
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

// ─── Benchmark ───

async function loadBenchmark() {
  showLoader();
  try {
    const data = await apiGet('/benchmark');
    hideLoader();

    const el = document.getElementById('benchmarkResult');
    const hm = data.hashmap;
    const tr = data.trie;

    const fasterBuild = hm.build_ms < tr.build_ms ? '⚡ Faster' : '';
    const fasterQuery = hm.query_ms < tr.query_ms ? '⚡ Faster' : '';
    const trFasterBuild = tr.build_ms < hm.build_ms ? '⚡ Faster' : '';
    const trFasterQuery = tr.query_ms < hm.query_ms ? '⚡ Faster' : '';

    el.innerHTML = `
      <div class="bench-ds bench-hmap">
        <div class="bench-title">🗂️ <span>Nested Hash Map</span></div>
        <div class="bench-row"><span class="bench-k">Build Time</span><span class="bench-v">${hm.build_ms} ms ${fasterBuild}</span></div>
        <div class="bench-row"><span class="bench-k">Query Time</span><span class="bench-v">${hm.query_ms} ms ${fasterQuery}</span></div>
        <div class="bench-row"><span class="bench-k">Total Entries</span><span class="bench-v">${hm.entries.toLocaleString()}</span></div>
        <div class="bench-row"><span class="bench-k">Lookup Complexity</span><span class="bench-v">O(1) avg</span></div>
      </div>
      <div class="bench-ds bench-trie">
        <div class="bench-title">🌲 <span>Trie (Prefix Tree)</span></div>
        <div class="bench-row"><span class="bench-k">Build Time</span><span class="bench-v">${tr.build_ms} ms ${trFasterBuild}</span></div>
        <div class="bench-row"><span class="bench-k">Query Time</span><span class="bench-v">${tr.query_ms} ms ${trFasterQuery}</span></div>
        <div class="bench-row"><span class="bench-k">Total Entries</span><span class="bench-v">${tr.entries.toLocaleString()}</span></div>
        <div class="bench-row"><span class="bench-k">Lookup Complexity</span><span class="bench-v">O(m) per key</span></div>
      </div>
    `;
  } catch (e) {
    hideLoader();
    console.error(e);
  }
}

// ═══════════════════════════════════════════════════
//  UTILITY
// ═══════════════════════════════════════════════════

function showError(containerId, msg) {
  const el = document.getElementById(containerId);
  if (el) {
    el.innerHTML = `<div style="color:#f87171;padding:16px;font-size:14px;">⚠️ ${msg}</div>`;
    el.style.display = 'block';
  }
}
