# WordNext AI
## Predictive Text and Next-Word Prediction Using Markov Chain
### B.Tech Computer Science — Probability & Statistics Project

---

## Abstract

WordNext AI is a full-stack web application that demonstrates statistical language modeling using Markov Chains. The system predicts the next word in a sequence using unigram, bigram, and trigram models with Maximum Likelihood Estimation and Laplace smoothing. It includes a live web predictor, model analytics, and a performance comparison of three storage backends: SQLite (DBMS), Python dict (Hash Map), and a manually implemented Trie.

---

## Problem Statement

Modern keyboard apps suggest the next word as you type. How does this work mathematically? This project builds and evaluates the core statistical model — a Markov Chain language model — from scratch, with no ML libraries, demonstrating the full pipeline from raw text to live predictions.

---

## Objectives

1. Implement unigram, bigram, and trigram language models from scratch.
2. Apply Maximum Likelihood Estimation and Laplace Smoothing.
3. Build a backoff strategy (Trigram → Bigram → Unigram).
4. Evaluate models using Top-1 accuracy, Top-3 accuracy, and Perplexity.
5. Compare n-gram storage using SQLite, Hash Map, and Trie.
6. Build a professional interactive web UI.
7. Run experiments demonstrating effect of dataset size and model order.

---

## Technology Stack

| Component  | Technology |
|------------|-----------|
| Backend    | Python 3.11, Flask 3.0 |
| Frontend   | HTML5, CSS3, Vanilla JavaScript |
| Database   | SQLite (built into Python) |
| DSA        | Python dict (HashMap), Manual Trie |
| Charts     | Chart.js 4.4 |
| Fonts      | Inter, JetBrains Mono (Google Fonts) |
| Deploy     | GitHub + Render + Gunicorn |
| Tests      | pytest (111 tests, all passing) |

---

## System Architecture

```
User Browser
     │
     ▼
Flask App (app.py)
     │
     ├── markov.py         ← Core language model
     │       ├── preprocess_text()
     │       ├── count_ngrams()
     │       └── MarkovModel
     │               ├── unigram_prob()
     │               ├── bigram_prob()
     │               ├── trigram_prob()
     │               ├── predict_top3()
     │               ├── generate_sentence()
     │               └── perplexity()
     │
     ├── evaluation.py     ← Accuracy & experiments
     │       ├── evaluate_accuracy()
     │       ├── compare_models()
     │       └── run_dataset_size_experiment()
     │
     └── storage.py        ← DBMS & DSA backends
             ├── SQLiteStore
             ├── HashMapStore
             ├── TrieStore
             └── benchmark_all()
```

---

## Markov Chain Theory

### Markov Property
The fundamental assumption: the next word depends ONLY on the previous n-1 words, not the entire history.

```
P(wₙ | w₁, w₂, ..., wₙ₋₁) ≈ P(wₙ | wₙ₋₂, wₙ₋₁)   [trigram]
```

### Mathematical Formulas

**Unigram MLE:**
```
P(w) = count(w) / N
```

**Bigram MLE:**
```
P(w₂ | w₁) = count(w₁, w₂) / count(w₁)
```

**Trigram MLE:**
```
P(w₃ | w₁, w₂) = count(w₁, w₂, w₃) / count(w₁, w₂)
```

**Laplace Smoothed Bigram (Add-k, k=1):**
```
P(w₂ | w₁) = (count(w₁,w₂) + k) / (count(w₁) + k × |V|)
```

**Perplexity:**
```
PP(W) = 2^( -(1/N) × Σᵢ log₂ P(wᵢ | context) )
```

Lower perplexity = better model.

### Backoff Strategy
```
1. Try trigram: P(w₃ | w₁, w₂) — if training data exists for context
2. Fall back to bigram: P(w₃ | w₂) — if trigram context is sparse
3. Fall back to unigram: P(w₃) — always has a prediction
```

---

## Data Structures

### Hash Map (Python dict)
- Storage: `{(w1, w2): {w3: count}}`
- Lookup: **O(1)** average
- Worst case: O(N) on hash collision (very rare)

### Trie (Prefix Tree — manually implemented)
- Each path root→leaf = one n-gram prefix
- Lookup: **O(m)** where m = prefix length (1 for bigrams, 2 for trigrams)
- Supports prefix search natively
- More memory due to per-node overhead

### SQLite (B-Tree index)
- Tables: `unigrams`, `bigrams`, `trigrams`
- Indexed on (w1) for bigrams and (w1,w2) for trigrams
- Lookup: **O(log N)** via B-Tree index
- File-based = persistent across restarts
- Slower than in-memory due to I/O

---

## DBMS Implementation

SQLite schema:
```sql
CREATE TABLE bigrams (
    w1    TEXT NOT NULL,
    w2    TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (w1, w2)
);
CREATE INDEX idx_bigrams_w1 ON bigrams (w1);

CREATE TABLE trigrams (
    w1 TEXT, w2 TEXT, w3 TEXT,
    count INTEGER,
    PRIMARY KEY (w1, w2, w3)
);
CREATE INDEX idx_trigrams_w1w2 ON trigrams (w1, w2);
```

---

## Algorithm Complexity

| Operation          | Structure | Time (Avg) | Time (Worst) | Space |
|--------------------|-----------|-----------|-------------|-------|
| Unigram count      | Counter   | O(N)       | O(N)         | O(V)  |
| Bigram build       | dict      | O(N)       | O(N)         | O(V²) |
| Trigram build      | dict      | O(N)       | O(N)         | O(V³) |
| HashMap lookup     | dict      | O(1)       | O(N)         | O(entries) |
| Trie lookup        | Trie      | O(m)       | O(m)         | O(N·m) |
| SQLite (indexed)   | B-Tree    | O(log N)   | O(log N)     | O(N)  |
| Top-3 prediction   | Sort      | O(C log C) | O(C log C)   | O(C)  |

N=tokens, V=vocab size, m=prefix length, C=candidate count.

---

## Experimental Methodology

Models trained on 10%, 25%, 50%, 75%, and 100% of corpus.

Metrics measured at each size:
- Vocabulary size
- Unique bigram count
- Bigram sparsity %
- Unigram / Bigram / Trigram Top-1 accuracy
- Bigram / Trigram Top-3 accuracy
- Perplexity (all three orders)

**All results are computed live from actual data — no hardcoded numbers.**

---

## Limitations

1. **Small corpus**: The bundled corpus is ~90 sentences. Accuracy will be lower than large-scale models.
2. **Add-1 smoothing**: Over-smooths on large vocabularies. Advanced: Kneser-Ney smoothing.
3. **No morphology**: "run" and "running" treated as different words.
4. **SQLite on Render free tier**: SQLite file is ephemeral (resets on restart). For production, use PostgreSQL.
5. **No semantic understanding**: Purely statistical — no meaning, context, or grammar rules.

---

## Future Scope

- Implement Kneser-Ney smoothing
- Add Stupid Backoff with lambda weights
- Use Project Gutenberg for large-scale corpus
- Implement word embeddings (Word2Vec)
- Compare with neural n-gram model

---

## Installation and Local Run

```bash
# 1. Clone or download project
cd "WordNext-AI"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open browser
# http://localhost:5000

# 6. Run tests
python -m pytest tests/ -v
```

---

## Deployment to GitHub + Render (Free Live URL)

```bash
# Step 1: Initialize git in WordNext-AI folder
git init
git add .
git commit -m "Initial commit: WordNext AI"

# Step 2: Push to GitHub
# Create new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/wordnext-ai.git
git branch -M main
git push -u origin main

# Step 3: Deploy on Render
# 1. Go to https://render.com
# 2. Click "New" → "Web Service"
# 3. Connect your GitHub repo
# 4. Settings:
#    Build Command: pip install -r requirements.txt
#    Start Command: gunicorn app:app
# 5. Click "Create Web Service"
# → Your app goes live at https://wordnext-ai.onrender.com
```

**Note on SQLite on Render:** Render's free tier has an ephemeral filesystem. The SQLite database (wordnext.db) will reset on each restart. This is fine for demonstration — the model retrains from corpus.txt on startup. For production use, migrate to PostgreSQL (available on Render).

---

## Project Structure

```
WordNext-AI/
├── app.py              ← Flask app + all API routes
├── markov.py           ← Markov Chain engine (MLE, smoothing, backoff)
├── evaluation.py       ← Accuracy, perplexity, experiments
├── storage.py          ← SQLite + HashMap + Trie
├── requirements.txt    ← Flask + Gunicorn
├── Procfile            ← Render/Heroku deployment
├── render.yaml         ← Render configuration
├── .gitignore
├── README.md
│
├── data/
│   └── corpus.txt      ← Training corpus (~90 sentences)
│
├── templates/
│   ├── base.html       ← Sidebar layout
│   ├── index.html      ← Dashboard
│   ├── predictor.html  ← Live predictor
│   ├── analytics.html  ← Model analytics
│   └── comparison.html ← DBMS vs DSA
│
├── static/
│   ├── css/style.css   ← Liquid Glass dark theme
│   └── js/script.js    ← Chart.js + API helpers
│
└── tests/
    ├── test_markov.py      ← 70 tests
    ├── test_storage.py     ← 28 tests
    └── test_evaluation.py  ← 25 tests  (Total: 111 tests)
```

---

*B.Tech Computer Science · Probability & Statistics · Markov Chain NLP Project*
