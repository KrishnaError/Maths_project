"""
app.py  —  WordNext AI
=======================
Flask Web Application

Routes:
  GET  /                 → Dashboard
  GET  /predictor        → Live Next-Word Predictor
  GET  /analytics        → Model Analytics
  GET  /comparison       → DBMS vs DSA Comparison

  API endpoints (JSON):
  POST /api/train        → Train model on corpus
  POST /api/add_text     → Add user text to corpus and retrain
  POST /api/predict      → Live next-word prediction
  POST /api/generate     → Sentence generation
  GET  /api/stats        → Corpus & model statistics
  GET  /api/analytics    → Full model comparison data
  GET  /api/benchmark    → DBMS vs DSA benchmark
  POST /api/experiment   → Dataset-size experiment
"""

import os
import json
import time
from flask import Flask, render_template, request, jsonify

from markov     import MarkovModel
from evaluation import evaluate_accuracy, compare_models, run_dataset_size_experiment, estimate_model_size
from storage    import initialize_stores, benchmark_all, get_sqlite_store, get_hashmap_store, get_trie_store

# ════════════════════════════════════════════════════════════
#  APP SETUP
# ════════════════════════════════════════════════════════════

app = Flask(__name__)

CORPUS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'corpus.txt')

# Global model instance — trained once at startup, can be re-trained via API
_model: MarkovModel = None


def get_model() -> MarkovModel:
    """Return the global trained model, training it if necessary."""
    global _model
    if _model is None or not _model.is_trained:
        _model = _train_on_default_corpus()
    return _model


def _train_on_default_corpus() -> MarkovModel:
    """Train a fresh model on the bundled sample corpus."""
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    m = MarkovModel(smoothing_k=1.0)
    m.train(text, test_ratio=0.2)
    initialize_stores(m.counts)
    return m


# Train on startup so the first request is fast
print('[WordNext AI] Training model on startup...')
try:
    _model = _train_on_default_corpus()
    print(f'[WordNext AI] Ready. Vocab size: {_model.vocab_size}')
except Exception as e:
    print(f'[WordNext AI] WARNING: Startup training failed: {e}')


# ════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ════════════════════════════════════════════════════════════

@app.route('/')
def dashboard():
    """Main dashboard — project overview and stats."""
    return render_template('index.html')


@app.route('/predictor')
def predictor():
    """Live next-word prediction page."""
    return render_template('predictor.html')


@app.route('/analytics')
def analytics():
    """Model analytics and accuracy page."""
    return render_template('analytics.html')


@app.route('/comparison')
def comparison():
    """DBMS vs DSA performance comparison page."""
    return render_template('comparison.html')


# ════════════════════════════════════════════════════════════
#  API: TRAIN
# ════════════════════════════════════════════════════════════

@app.route('/api/train', methods=['POST'])
def api_train():
    """
    Train the model on custom uploaded text or reset to default corpus.

    Request JSON:
        { "text": "optional custom corpus text..." }

    Response JSON:
        { "success": true, "stats": {...} }
    """
    global _model

    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        try:
            with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            return jsonify({'success': False, 'error': 'Default corpus not found.'}), 500

    try:
        _model = MarkovModel(smoothing_k=1.0)
        _model.train(text, test_ratio=0.2)
        initialize_stores(_model.counts)
        # Clear analytics/experiment cache so fresh results load
        for attr in ('_analytics_cache', '_analytics_model_id', '_experiment_cache'):
            if hasattr(app, attr):
                delattr(app, attr)
        return jsonify({'success': True, 'stats': _model.corpus_stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: ADD USER TEXT  (append user sentences + retrain)
# ════════════════════════════════════════════════════════════

@app.route('/api/add_text', methods=['POST'])
def api_add_text():
    """
    Append the user's custom text to the corpus file and retrain the model.

    This lets the user type their own sentences so that words like "I am"
    appear in training data and get better predictions.

    Request JSON:
        { "text": "I am learning statistics and probability this semester." }

    Response JSON:
        { "success": true, "message": "...", "stats": {...} }
    """
    global _model

    data      = request.get_json(silent=True) or {}
    user_text = data.get('text', '').strip()

    if not user_text:
        return jsonify({'success': False, 'error': 'No text provided.'}), 400

    if len(user_text) < 10:
        return jsonify({'success': False, 'error': 'Text too short. Write at least one full sentence.'}), 400

    try:
        # Append user text to the corpus file
        with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
            # Ensure it starts on a new line
            f.write('\n' + user_text.strip())

        # Retrain on the now-larger corpus
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()

        _model = MarkovModel(smoothing_k=1.0)
        _model.train(full_text, test_ratio=0.2)
        initialize_stores(_model.counts)

        # Clear caches
        for attr in ('_analytics_cache', '_analytics_model_id', '_experiment_cache'):
            if hasattr(app, attr):
                delattr(app, attr)

        return jsonify({
            'success': True,
            'message': f'Text added! Model retrained on updated corpus.',
            'chars_added': len(user_text),
            'stats': _model.corpus_stats,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: PREDICT  (called live while typing — fast!)
# ════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Return top-3 next-word predictions for a given context.

    Request JSON:
        { "context": "the probability", "model_order": "trigram" }

    Response JSON:
        { "predictions": [...], "context": "..." }
    """
    model = get_model()
    data  = request.get_json(silent=True) or {}

    context     = data.get('context', '').strip()
    model_order = data.get('model_order', 'trigram')

    if model_order not in ('unigram', 'bigram', 'trigram'):
        return jsonify({'error': 'Invalid model_order. Use unigram, bigram, or trigram.'}), 400

    if not context:
        return jsonify({'predictions': [], 'context': ''})

    preds = model.predict_top3(context, model_order=model_order)
    return jsonify({'predictions': preds, 'context': context})


# ════════════════════════════════════════════════════════════
#  API: GENERATE SENTENCE
# ════════════════════════════════════════════════════════════

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """
    Generate sample sentences using the Markov model.

    Request JSON:
        { "seed": "statistics", "max_words": 20, "model_order": "trigram", "count": 3 }

    Response JSON:
        { "sentences": ["...", "...", "..."] }
    """
    model = get_model()
    data  = request.get_json(silent=True) or {}

    seed        = data.get('seed', '')
    max_words   = min(int(data.get('max_words', 20)), 50)
    model_order = data.get('model_order', 'trigram')
    count       = min(int(data.get('count', 3)), 5)

    sentences = [model.generate_sentence(seed, max_words, model_order)
                 for _ in range(count)]

    return jsonify({'sentences': sentences, 'seed': seed})


# ════════════════════════════════════════════════════════════
#  API: STATISTICS
# ════════════════════════════════════════════════════════════

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """
    Return corpus statistics, top n-grams, and model info.

    Response JSON:
        { "corpus_stats": {...}, "top_unigrams": [...], "top_bigrams": [...], ... }
    """
    model = get_model()

    top_unigrams = [
        {'word': w, 'count': c}
        for w, c in model.get_top_ngrams(15, 'unigram')
        if w not in ('<START>', '<END>')
    ][:10]

    top_bigrams  = [{'ngram': t, 'count': c}
                    for t, c in model.get_top_ngrams(10, 'bigram')]
    top_trigrams = [{'ngram': t, 'count': c}
                    for t, c in model.get_top_ngrams(10, 'trigram')]

    matrix       = model.get_transition_matrix(top_n=7)
    model_size   = estimate_model_size(model)

    return jsonify({
        'corpus_stats':  model.corpus_stats,
        'top_unigrams':  top_unigrams,
        'top_bigrams':   top_bigrams,
        'top_trigrams':  top_trigrams,
        'transition_matrix': matrix,
        'model_size_kb': model_size,
    })


# ════════════════════════════════════════════════════════════
#  API: ANALYTICS
# ════════════════════════════════════════════════════════════

@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    """
    Return full model comparison (unigram vs bigram vs trigram).
    Cached after first call so repeated visits are instant.
    """
    model = get_model()

    # Simple in-process cache — recompute only when model changes
    if not hasattr(app, '_analytics_cache') or app._analytics_model_id != id(model):
        app._analytics_cache   = compare_models(model)
        app._analytics_model_id = id(model)

    return jsonify({
        'model_comparison':  app._analytics_cache,
        'transition_matrix': model.get_transition_matrix(top_n=7),
        'model_size_kb':     estimate_model_size(model),
        'corpus_stats':      model.corpus_stats,
    })


# ════════════════════════════════════════════════════════════
#  API: DATASET-SIZE EXPERIMENT
# ════════════════════════════════════════════════════════════

@app.route('/api/experiment', methods=['GET'])
def api_experiment():
    """
    Run dataset-size experiments. Cached after first call.
    """
    if hasattr(app, '_experiment_cache'):
        return jsonify({'experiments': app._experiment_cache})
    try:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        results = run_dataset_size_experiment(text)
        app._experiment_cache = results
        return jsonify({'experiments': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: DBMS vs DSA BENCHMARK
# ════════════════════════════════════════════════════════════

@app.route('/api/benchmark', methods=['GET'])
def api_benchmark():
    """
    Benchmark SQLite vs HashMap vs Trie and return timing results.

    Returns real measured performance — no hardcoded values.
    """
    model = get_model()
    try:
        results = benchmark_all(model.counts, n_query_trials=20)
        return jsonify({'benchmark': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: SQLITE LIVE VIEWER
# ════════════════════════════════════════════════════════════

@app.route('/api/db_view', methods=['GET'])
def api_db_view():
    """
    Return actual rows from the SQLite database for the live viewer.
    Shows unigrams, bigrams, trigrams tables with real data.
    """
    model = get_model()
    db    = get_sqlite_store()

    try:
        cur = db.conn.cursor()

        # Top 15 unigrams by count
        cur.execute('''
            SELECT word, count FROM unigrams
            WHERE word NOT IN ('<START>', '<END>')
            ORDER BY count DESC LIMIT 15
        ''')
        unigrams = [{'word': r[0], 'count': r[1]} for r in cur.fetchall()]

        # Top 15 bigrams by count
        cur.execute('''
            SELECT w1, w2, count FROM bigrams
            WHERE w1 NOT IN ('<START>','<END>')
              AND w2 NOT IN ('<START>','<END>')
            ORDER BY count DESC LIMIT 15
        ''')
        bigrams = [{'w1': r[0], 'w2': r[1], 'count': r[2]} for r in cur.fetchall()]

        # Top 15 trigrams by count
        cur.execute('''
            SELECT w1, w2, w3, count FROM trigrams
            WHERE w1 NOT IN ('<START>','<END>')
              AND w2 NOT IN ('<START>','<END>')
              AND w3 NOT IN ('<START>','<END>')
            ORDER BY count DESC LIMIT 15
        ''')
        trigrams = [{'w1': r[0], 'w2': r[1], 'w3': r[2], 'count': r[3]}
                    for r in cur.fetchall()]

        # Table sizes
        sizes = db.get_table_sizes()

        # Sample SQL queries for display
        sample_queries = [
            {
                'label':  'Find all words after "the"',
                'sql':    "SELECT w2, count FROM bigrams WHERE w1 = 'the' ORDER BY count DESC LIMIT 5;",
                'result': []
            },
            {
                'label':  'Find all words after "machine learning"',
                'sql':    "SELECT w3, count FROM trigrams WHERE w1='machine' AND w2='learning' ORDER BY count DESC;",
                'result': []
            },
            {
                'label':  'Count total unique bigrams',
                'sql':    "SELECT COUNT(*) as total_bigrams FROM bigrams;",
                'result': []
            },
        ]

        # Execute sample queries
        cur.execute("SELECT w2, count FROM bigrams WHERE w1='the' ORDER BY count DESC LIMIT 5")
        sample_queries[0]['result'] = [{'w2': r[0], 'count': r[1]} for r in cur.fetchall()]

        cur.execute("SELECT w3, count FROM trigrams WHERE w1='machine' AND w2='learning' ORDER BY count DESC")
        sample_queries[1]['result'] = [{'w3': r[0], 'count': r[1]} for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*) FROM bigrams")
        sample_queries[2]['result'] = [{'total_bigrams': cur.fetchone()[0]}]

        return jsonify({
            'unigrams':      unigrams,
            'bigrams':       bigrams,
            'trigrams':      trigrams,
            'table_sizes':   sizes,
            'sample_queries': sample_queries,
            'db_path':       'wordnext.db',
            'schema': {
                'unigrams': 'CREATE TABLE unigrams (word TEXT PRIMARY KEY, count INTEGER)',
                'bigrams':  'CREATE TABLE bigrams (w1 TEXT, w2 TEXT, count INTEGER, PRIMARY KEY(w1,w2))',
                'trigrams': 'CREATE TABLE trigrams (w1 TEXT, w2 TEXT, w3 TEXT, count INTEGER, PRIMARY KEY(w1,w2,w3))',
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
#  API: PERPLEXITY
# ════════════════════════════════════════════════════════════

@app.route('/api/perplexity', methods=['GET'])
def api_perplexity():
    """Return perplexity scores for all three model orders."""
    model = get_model()
    return jsonify({
        'unigram':  model.perplexity(model_order='unigram'),
        'bigram':   model.perplexity(model_order='bigram'),
        'trigram':  model.perplexity(model_order='trigram'),
        'explanation': (
            'Lower perplexity = better model. '
            'Trigrams have lower perplexity on training-domain text '
            'but suffer more from sparsity on unseen text.'
        ),
    })


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
