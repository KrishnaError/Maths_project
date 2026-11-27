"""
app.py
======
Flask Web Application — Project 6: Markov Chain Next-Word Predictor

Routes:
  GET  /              → Main UI
  POST /train         → Train model on uploaded/default corpus
  POST /predict       → Get top-3 next-word predictions
  GET  /generate      → Generate a sample sentence
  GET  /stats         → Corpus & model statistics
  GET  /accuracy      → Top-3 accuracy evaluation
  GET  /benchmark     → HashMap vs Trie speed comparison
  GET  /matrix        → Transition probability matrix
  GET  /perplexity    → Perplexity scores (bigram vs trigram)
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from markov_model import MarkovModel

app = Flask(__name__)

# ── Global model instance ──────────────────────────────
model = MarkovModel(smoothing=1.0)
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "sample_corpus.txt")

# ── Auto-train on startup ──────────────────────────────
def load_and_train():
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    model.fit(text)
    print(f"[INFO] Model trained. Vocab size: {model.vocab_size}")

load_and_train()


# ══════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['POST'])
def train():
    """Train model on custom text submitted via form or use default corpus."""
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        # reload default corpus
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            text = f.read()

    model.fit(text)
    return jsonify({
        "success": True,
        "message": f"Model trained successfully!",
        "stats":   model.corpus_stats
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Return top-3 next-word predictions for given context."""
    data    = request.get_json()
    context = data.get('context', '').strip()

    if not context:
        return jsonify({"error": "Please provide some text context."}), 400

    if not model.is_trained:
        return jsonify({"error": "Model not trained yet."}), 400

    predictions = model.predict_top3(context)
    return jsonify({
        "context":     context,
        "predictions": predictions
    })


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a sample sentence."""
    data     = request.get_json()
    seed     = data.get('seed', '')
    max_words = int(data.get('max_words', 20))

    sentences = [model.generate_sentence(seed, max_words) for _ in range(3)]
    return jsonify({"sentences": sentences, "seed": seed})


@app.route('/stats', methods=['GET'])
def stats():
    """Return corpus and model statistics."""
    if not model.is_trained:
        return jsonify({"error": "Model not trained."}), 400

    top_bigrams = model.get_top_bigrams(10)
    top_unigrams = [
        {"word": w, "count": c}
        for w, c in model.unigram_counts.most_common(15)
        if w not in ('<START>', '<END>')
    ][:10]

    return jsonify({
        "corpus_stats":  model.corpus_stats,
        "top_bigrams":   [{"bigram": b, "count": c} for b, c in top_bigrams],
        "top_unigrams":  top_unigrams
    })


@app.route('/accuracy', methods=['POST'])
def accuracy():
    """Evaluate top-1 and top-3 accuracy on provided test text."""
    data      = request.get_json()
    test_text = data.get('test_text', '').strip()

    if not test_text:
        # Use last 20% of default corpus as held-out test
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()
        lines = full_text.strip().split('\n')
        split  = int(len(lines) * 0.8)
        test_text = '\n'.join(lines[split:])

    result = model.evaluate_accuracy(test_text)
    return jsonify(result)


@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Compare HashMap vs Trie query speed."""
    result = model.run_benchmark()
    return jsonify(result)


@app.route('/matrix', methods=['GET'])
def matrix():
    """Return transition probability matrix for visualization."""
    result = model.get_transition_matrix(top_n=8)
    return jsonify(result)


@app.route('/perplexity', methods=['POST'])
def perplexity():
    """Compute perplexity under bigram and trigram models."""
    data      = request.get_json()
    test_text = data.get('test_text', '').strip()

    if not test_text:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()
        lines     = full_text.strip().split('\n')
        split     = int(len(lines) * 0.8)
        test_text = '\n'.join(lines[split:])

    bigram_pp  = model.perplexity(test_text, order=2)
    trigram_pp = model.perplexity(test_text, order=3)

    return jsonify({
        "bigram_perplexity":  bigram_pp,
        "trigram_perplexity": trigram_pp,
        "explanation": (
            "Lower perplexity means the model predicts the text better. "
            "Trigrams usually have lower perplexity on seen data but suffer "
            "more from sparsity (data sparsity problem)."
        )
    })


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    # For GitHub Pages / Render / Railway — read PORT from environment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
