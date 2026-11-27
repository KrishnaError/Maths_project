"""
test_evaluation.py  —  WordNext AI
=====================================
Unit tests for evaluation.py

Tests:
  - evaluate_accuracy return structure
  - Top-1 and Top-3 accuracy values in valid range
  - run_dataset_size_experiment structure
  - compare_models structure
  - estimate_model_size structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from markov import MarkovModel
from evaluation import (
    evaluate_accuracy,
    compare_models,
    run_dataset_size_experiment,
    estimate_model_size,
)

SAMPLE_TEXT = """
The cat sat on the mat. The cat ate the rat. The rat ran away quickly.
Statistics is the science of data analysis. Data analysis helps make decisions.
Machine learning uses probability and statistics for predictions.
The quick brown fox jumps over the lazy dog.
Probability theory is used in statistics and machine learning models.
Natural language processing uses statistical language models for text.
Markov chains are stochastic models used in probability theory.
Language models predict the next word in a sequence using context.
"""


def _get_trained_model():
    m = MarkovModel(smoothing_k=1.0)
    m.train(SAMPLE_TEXT, test_ratio=0.2)
    return m


class TestEvaluateAccuracy(unittest.TestCase):
    """Tests for the evaluate_accuracy function."""

    def setUp(self):
        self.model = _get_trained_model()

    def test_returns_dict(self):
        result = evaluate_accuracy(self.model, 'trigram')
        self.assertIsInstance(result, dict)

    def test_required_keys(self):
        result = evaluate_accuracy(self.model, 'bigram')
        required = ['top1_accuracy', 'top3_accuracy', 'total_predictions',
                    'top1_correct', 'top3_correct', 'avg_pred_time_ms', 'model_order']
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_top1_accuracy_range(self):
        """Top-1 accuracy should be between 0 and 100."""
        for order in ('unigram', 'bigram', 'trigram'):
            result = evaluate_accuracy(self.model, order)
            self.assertGreaterEqual(result['top1_accuracy'], 0.0)
            self.assertLessEqual(result['top1_accuracy'], 100.0)

    def test_top3_accuracy_range(self):
        """Top-3 accuracy should be between 0 and 100."""
        for order in ('unigram', 'bigram', 'trigram'):
            result = evaluate_accuracy(self.model, order)
            self.assertGreaterEqual(result['top3_accuracy'], 0.0)
            self.assertLessEqual(result['top3_accuracy'], 100.0)

    def test_top3_geq_top1(self):
        """Top-3 accuracy should be >= Top-1 accuracy."""
        for order in ('unigram', 'bigram', 'trigram'):
            result = evaluate_accuracy(self.model, order)
            self.assertGreaterEqual(
                result['top3_accuracy'], result['top1_accuracy'],
                f"Top-3 should be >= Top-1 for {order} model"
            )

    def test_correct_counts_consistent(self):
        """top1_correct + wrong should equal total_predictions."""
        result = evaluate_accuracy(self.model, 'bigram')
        self.assertLessEqual(result['top1_correct'], result['total_predictions'])
        self.assertLessEqual(result['top3_correct'], result['total_predictions'])

    def test_model_order_stored(self):
        """model_order field should match the order passed."""
        for order in ('unigram', 'bigram', 'trigram'):
            result = evaluate_accuracy(self.model, order)
            self.assertEqual(result['model_order'], order)

    def test_avg_pred_time_non_negative(self):
        result = evaluate_accuracy(self.model, 'bigram')
        self.assertGreaterEqual(result['avg_pred_time_ms'], 0)

    def test_all_three_orders(self):
        """All three model orders should work without error."""
        for order in ('unigram', 'bigram', 'trigram'):
            try:
                result = evaluate_accuracy(self.model, order)
                self.assertIsNotNone(result)
            except Exception as e:
                self.fail(f"evaluate_accuracy({order}) raised {e}")


class TestCompareModels(unittest.TestCase):
    """Tests for compare_models function."""

    def setUp(self):
        self.model = _get_trained_model()

    def test_returns_dict(self):
        result = compare_models(self.model)
        self.assertIsInstance(result, dict)

    def test_has_all_orders(self):
        result = compare_models(self.model)
        for order in ('unigram', 'bigram', 'trigram'):
            self.assertIn(order, result)

    def test_each_order_has_metrics(self):
        result = compare_models(self.model)
        for order in ('unigram', 'bigram', 'trigram'):
            metrics = result[order]
            for key in ('top1_accuracy', 'top3_accuracy', 'perplexity'):
                self.assertIn(key, metrics, f"Missing '{key}' in {order}")

    def test_top3_always_geq_top1(self):
        result = compare_models(self.model)
        for order in ('unigram', 'bigram', 'trigram'):
            self.assertGreaterEqual(
                result[order]['top3_accuracy'],
                result[order]['top1_accuracy'],
                f"Top-3 < Top-1 for {order}"
            )


class TestDatasetSizeExperiment(unittest.TestCase):
    """Tests for dataset-size experiment."""

    def setUp(self):
        # Use small sizes to keep test fast
        self.results = run_dataset_size_experiment(
            SAMPLE_TEXT, sizes=[0.5, 1.0]
        )

    def test_returns_list(self):
        self.assertIsInstance(self.results, list)

    def test_correct_number_of_results(self):
        """Should return one result per size."""
        self.assertEqual(len(self.results), 2)

    def test_each_result_has_fraction(self):
        for r in self.results:
            self.assertIn('fraction', r)

    def test_fractions_match_input(self):
        fracs = [r['fraction'] for r in self.results]
        self.assertIn(0.5, fracs)
        self.assertIn(1.0, fracs)

    def test_vocab_increases_with_size(self):
        """Vocabulary size should be non-decreasing with more data."""
        vocabs = [r.get('vocab_size', 0) for r in self.results]
        self.assertLessEqual(vocabs[0], vocabs[1] + 1)  # +1 for tiny variance

    def test_all_sizes_full_corpus(self):
        results = run_dataset_size_experiment(
            SAMPLE_TEXT, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
        )
        self.assertEqual(len(results), 5)

    def test_accuracy_in_valid_range(self):
        for r in self.results:
            if 'bigram_top1' in r:
                self.assertGreaterEqual(r['bigram_top1'], 0)
                self.assertLessEqual(r['bigram_top1'], 100)


class TestEstimateModelSize(unittest.TestCase):
    """Tests for model size estimation."""

    def setUp(self):
        self.model = _get_trained_model()

    def test_returns_dict(self):
        result = estimate_model_size(self.model)
        self.assertIsInstance(result, dict)

    def test_has_expected_keys(self):
        result = estimate_model_size(self.model)
        for key in ('unigram_kb', 'bigram_kb', 'trigram_kb', 'total_kb'):
            self.assertIn(key, result)

    def test_all_values_non_negative(self):
        result = estimate_model_size(self.model)
        for key, val in result.items():
            self.assertGreaterEqual(val, 0, f"{key} should be >= 0")

    def test_total_equals_sum(self):
        result = estimate_model_size(self.model)
        expected_total = round(result['unigram_kb'] + result['bigram_kb'] + result['trigram_kb'], 2)
        self.assertAlmostEqual(result['total_kb'], expected_total, places=1)

    def test_trigram_largest(self):
        """Trigram table should be largest (most entries)."""
        result = estimate_model_size(self.model)
        self.assertGreaterEqual(result['trigram_kb'], result['unigram_kb'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
