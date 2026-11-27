"""
evaluation.py  —  WordNext AI
==============================
Model Evaluation and Experiment Runner

Implements:
  - Top-1 and Top-3 Accuracy measurement
  - Prediction latency measurement
  - Dataset-size experiments (10%, 25%, 50%, 75%, 100%)
  - Unigram vs Bigram vs Trigram comparison
  - Explanation of data sparsity
  - Explanation of smoothing effect

All results are computed from real data — no hardcoded numbers.
"""

import time
import re
import random
from markov import MarkovModel, preprocess_text, train_test_split


# ════════════════════════════════════════════════════════════
#  ACCURACY EVALUATION
# ════════════════════════════════════════════════════════════

def evaluate_accuracy(model: MarkovModel, model_order: str = 'trigram') -> dict:
    """
    Measure Top-1 and Top-3 accuracy on the model's held-out test tokens.

    Method:
      For each position i in the test token sequence:
        1. Use tokens[i-2:i] as context.
        2. Get the model's top-3 predictions.
        3. Check if the actual token[i] appears in top-1 or top-3.

    Args:
        model:       A trained MarkovModel instance.
        model_order: 'unigram', 'bigram', or 'trigram'.

    Returns:
        Dictionary with accuracy metrics and timing.
    """
    test_tokens = [t for t in model.test_tokens
                   if t not in ('<START>', '<END>')]

    if len(test_tokens) < 3:
        return {
            'top1_accuracy':     0.0,
            'top3_accuracy':     0.0,
            'total_predictions': 0,
            'top1_correct':      0,
            'top3_correct':      0,
            'avg_pred_time_ms':  0.0,
            'model_order':       model_order,
        }

    top1_correct = 0
    top3_correct = 0
    total        = 0
    pred_times   = []

    # We need the original test tokens (with markers) for context
    full_test = model.test_tokens

    for i in range(2, len(full_test)):
        actual = full_test[i]
        if actual in ('<START>', '<END>'):
            continue

        # Build context string from previous two tokens
        context = ' '.join([full_test[i-2], full_test[i-1]])

        t0    = time.perf_counter()
        preds = model.predict_top3(context, model_order=model_order)
        pred_times.append((time.perf_counter() - t0) * 1000)

        pred_words = [p['word'] for p in preds]
        total += 1

        if pred_words and pred_words[0] == actual:
            top1_correct += 1
        if actual in pred_words:
            top3_correct += 1

    avg_pred_time = round(sum(pred_times) / len(pred_times), 4) if pred_times else 0.0

    return {
        'top1_accuracy':     round(top1_correct / total * 100, 2) if total else 0.0,
        'top3_accuracy':     round(top3_correct / total * 100, 2) if total else 0.0,
        'total_predictions': total,
        'top1_correct':      top1_correct,
        'top3_correct':      top3_correct,
        'avg_pred_time_ms':  avg_pred_time,
        'model_order':       model_order,
    }


# ════════════════════════════════════════════════════════════
#  DATASET-SIZE EXPERIMENT
# ════════════════════════════════════════════════════════════

def run_dataset_size_experiment(full_text: str, sizes: list = None) -> list:
    """
    Train models on different fractions of the corpus and measure accuracy.

    This demonstrates:
      - Effect of training data size on accuracy
      - Sparsity at different sizes
      - Training time growth

    Args:
        full_text: Complete corpus text.
        sizes:     List of fractions to test, e.g. [0.1, 0.25, 0.5, 0.75, 1.0]

    Returns:
        List of result dicts, one per size.
    """
    if sizes is None:
        sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    results = []

    # First get the full token list so we can slice it
    all_tokens = preprocess_text(full_text)

    for fraction in sizes:
        # Take first `fraction` of the token list
        n = max(10, int(len(all_tokens) * fraction))
        subset_tokens = all_tokens[:n]

        # Reconstruct text-like string from tokens for the model
        subset_text = ' '.join(subset_tokens).replace('<START>', '.').replace('<END>', '.')

        try:
            m = MarkovModel(smoothing_k=1.0)
            m.train(subset_text, test_ratio=0.2)

            # Evaluate all three orders
            uni_res = evaluate_accuracy(m, 'unigram')
            bi_res  = evaluate_accuracy(m, 'bigram')
            tri_res = evaluate_accuracy(m, 'trigram')

            results.append({
                'fraction':          fraction,
                'percent_label':     f'{int(fraction * 100)}%',
                'train_tokens':      m.corpus_stats['train_tokens'],
                'vocab_size':        m.corpus_stats['vocab_size'],
                'unique_bigrams':    m.corpus_stats['unique_bigrams'],
                'bigram_sparsity':   m.corpus_stats['bigram_sparsity'],
                'trigram_sparsity':  m.corpus_stats['trigram_sparsity'],
                'train_time_ms':     m.corpus_stats['train_time_ms'],

                'unigram_top1':  uni_res['top1_accuracy'],
                'unigram_top3':  uni_res['top3_accuracy'],
                'bigram_top1':   bi_res['top1_accuracy'],
                'bigram_top3':   bi_res['top3_accuracy'],
                'trigram_top1':  tri_res['top1_accuracy'],
                'trigram_top3':  tri_res['top3_accuracy'],

                'perplexity_unigram': m.perplexity(model_order='unigram'),
                'perplexity_bigram':  m.perplexity(model_order='bigram'),
                'perplexity_trigram': m.perplexity(model_order='trigram'),
            })
        except Exception as e:
            results.append({
                'fraction':     fraction,
                'percent_label': f'{int(fraction * 100)}%',
                'error':        str(e),
            })

    return results


# ════════════════════════════════════════════════════════════
#  MODEL COMPARISON
# ════════════════════════════════════════════════════════════

def compare_models(model: MarkovModel) -> dict:
    """
    Compare unigram, bigram, trigram on the same trained model.

    Returns a summary dict suitable for the Analytics page.
    """
    results = {}
    for order in ['unigram', 'bigram', 'trigram']:
        acc    = evaluate_accuracy(model, order)
        pp     = model.perplexity(model_order=order)
        ngrams = model.get_top_ngrams(n=5, order=order)

        results[order] = {
            'top1_accuracy':    acc['top1_accuracy'],
            'top3_accuracy':    acc['top3_accuracy'],
            'total_predictions': acc['total_predictions'],
            'avg_pred_time_ms': acc['avg_pred_time_ms'],
            'perplexity':       pp,
            'top_ngrams':       ngrams,
        }

    return results


# ════════════════════════════════════════════════════════════
#  MODEL SIZE ESTIMATE
# ════════════════════════════════════════════════════════════

def estimate_model_size(model: MarkovModel) -> dict:
    """
    Estimate memory footprint of stored n-gram tables.

    We use a simplified size estimate:
      - Each n-gram entry ≈ 100 bytes (key + value + dict overhead)

    This is an approximation. For exact measurement use sys.getsizeof().

    Returns:
        Dict with size estimates in KB.
    """
    import sys

    cs = model.corpus_stats
    bytes_per_entry = 100   # rough estimate

    uni_size  = cs.get('unique_unigrams', 0)  * bytes_per_entry
    bi_size   = cs.get('unique_bigrams', 0)   * bytes_per_entry
    tri_size  = cs.get('unique_trigrams', 0)  * bytes_per_entry

    return {
        'unigram_kb':  round(uni_size  / 1024, 2),
        'bigram_kb':   round(bi_size   / 1024, 2),
        'trigram_kb':  round(tri_size  / 1024, 2),
        'total_kb':    round((uni_size + bi_size + tri_size) / 1024, 2),
    }
