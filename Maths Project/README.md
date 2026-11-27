# 🔮 Markov Chain Next-Word Predictor
### Project 6 — Probability & Statistics · BTech Computer Science

A full-stack web application that predicts the next word in a sentence using **Markov Chain language models** — the same technique used in phone keyboards.

---

## 📸 Features

- **Next-Word Prediction** — Top-3 predictions with probabilities (bigram + trigram + backoff)
- **Sentence Generation** — Weighted random sampling using transition probabilities
- **Statistical Analysis** — Corpus stats, sparsity problem visualization
- **Model Evaluation** — Top-1 & Top-3 accuracy + Perplexity (bigram vs trigram)
- **DSA Benchmark** — HashMap vs Trie query speed comparison
- **Theory Reference** — All probability & statistics concepts explained

---

## 🧠 Concepts Applied

| Concept | Description |
|--------|-------------|
| Markov Property | P(wₙ \| w₁..wₙ₋₁) ≈ P(wₙ \| wₙ₋₁) |
| MLE | Estimate probabilities from counts |
| Laplace Smoothing | Avoids zero probabilities for unseen pairs |
| Backoff Strategy | Trigram → Bigram → Unigram |
| Perplexity | Model quality metric |
| Data Sparsity | Why smoothing is necessary |
| Trie vs HashMap | DSA comparison for n-gram storage |

---

## 🚀 Run Locally

```bash
# 1. Clone or download the project
cd "Maths Project"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open in browser
# http://localhost:5000
```

---

## 🌐 Deploy on Render (Free — GitHub Live)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click Deploy — your app goes live! 🎉

---

## 📁 Project Structure

```
Maths Project/
├── app.py              ← Flask backend (all API routes)
├── markov_model.py     ← Core Markov Chain + Statistics engine
├── data_store.py       ← Trie + HashMap (DSA component)
├── corpus/
│   └── sample_corpus.txt  ← Training data
├── templates/
│   └── index.html      ← Main UI (Jinja2 template)
├── static/
│   ├── css/style.css   ← Dark theme styling
│   └── js/app.js       ← Frontend logic
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Tech Stack

- **Backend:** Python 3.11, Flask 3.0
- **Frontend:** Vanilla JS, CSS3 (no frameworks — pure clean code)
- **Fonts:** Inter + JetBrains Mono (Google Fonts)
- **Deploy:** Render / Railway / Any WSGI host

---

*BTech First Year — Probability & Statistics Project*
