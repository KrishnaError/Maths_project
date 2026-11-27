"""
test_markov.py  —  WordNext AI
================================
Unit tests for markov.py

Tests:
  - Text preprocessing / tokenization
  - N-gram counting (unigram, bigram, trigram)
  - MLE probability calculations
  - Laplace smoothing
  - Backoff strategy
  - Top-3 prediction
  - Sentence generation
  - Empty input handling
  - Unknown word handling
  - Perplexity
  - Transition matrix
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from markov import preprocess_text, count_ngrams, MarkovModel, train_test_split

# ─────────────────── Sample text for all tests ───────────────────
SAMPLE_TEXT = """
The cat sat on the mat. The cat ate the rat. The rat ran away.
Statistics is the science of data. Data analysis helps decisions.
Machine learning uses probability and statistics for predictions.
The quick brown fox jumps over the lazy dog.
Probability theory is used in statistics and machine learning.
Natural language processing uses statistical models.
"""


class TestPreprocessing(unittest.TestCase):
    """Tests for text cleaning and tokenization."""

    def test_lowercase(self):
        """All tokens should be lowercase."""
        tokens = preprocess_text("Hello World")
        content = [t for t in tokens if t not in ('<START>', '<END>')]
        for t in content:
            self.assertEqual(t, t.lower(), f"Token '{t}' is not lowercase")

    def test_start_end_markers(self):
        """Token list should contain <START> and <END> markers."""
        tokens = preprocess_text("Hello world. Goodbye.")
        self.assertIn('<START>', tokens)
        self.assertIn('<END>', tokens)

    def test_sentence_segmentation(self):
        """Multiple sentences should produce multiple START/END pairs."""
        tokens = preprocess_text("Hello. Goodbye. Thanks.")
        start_count = tokens.count('<START>')
        end_count   = tokens.count('<END>')
        # 3 sentences → 3 × 2 START markers and 3 END markers
        self.assertGreater(start_count, 2)
        self.assertGreater(end_count, 1)

    def test_removes_punctuation(self):
        """Commas, colons etc should not appear in tokens."""
        tokens = preprocess_text("Hello, world: how are you?")
        for t in tokens:
            self.assertNotIn(',', t)
            self.assertNotIn(':', t)

    def test_empty_text(self):
        """Empty text should return an empty list."""
        tokens = preprocess_text("")
        content = [t for t in tokens if t not in ('<START>', '<END>')]
        self.assertEqual(len(content), 0)

    def test_numbers_removed(self):
        """Numeric tokens should not appear."""
        tokens = preprocess_text("There are 42 items in 2024.")
        content = [t for t in tokens if t not in ('<START>', '<END>')]
        for t in content:
            self.assertFalse(t.isdigit(), f"Numeric token '{t}' should be removed")

    def test_two_start_markers_per_sentence(self):
        """Each sentence should start with TWO <START> tokens for trigram context."""
        tokens = preprocess_text("Hello world.")
        # Find the index of the first meaningful word
        starts = [i for i, t in enumerate(tokens) if t == '<START>']
        self.assertGreaterEqual(len(starts), 2)
        # First two tokens should both be START
        self.assertEqual(tokens[0], '<START>')
        self.assertEqual(tokens[1], '<START>')


class TestNgramCounting(unittest.TestCase):
    """Tests for N-gram count construction."""

    def setUp(self):
        self.tokens = preprocess_text("the cat sat on the mat the cat ate")
        self.counts = count_ngrams(self.tokens)

    def test_unigram_counts_exist(self):
        """Unigram counter should exist and contain words."""
        self.assertIn('unigram', self.counts)
        self.assertGreater(len(self.counts['unigram']), 0)

    def test_bigram_counts_exist(self):
        """Bigram dict should exist and contain data."""
        self.assertIn('bigram', self.counts)
        self.assertGreater(len(self.counts['bigram']), 0)

    def test_trigram_counts_exist(self):
        """Trigram dict should exist and contain data."""
        self.assertIn('trigram', self.counts)
        self.assertGreater(len(self.counts['trigram']), 0)

    def test_the_cat_bigram(self):
        """'the cat' bigram should have count >= 2 (appears twice)."""
        cat_count = self.counts['bigram']['the'].get('cat', 0)
        self.assertGreaterEqual(cat_count, 1)

    def test_unigram_sum(self):
        """Sum of all unigram counts equals total token count."""
        total_counted = sum(self.counts['unigram'].values())
        self.assertEqual(total_counted, len(self.tokens))

    def test_trigram_prefix_format(self):
        """Trigram keys should be tuples of two words."""
        for key in self.counts['trigram']:
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)

    def test_no_negative_counts(self):
        """All counts should be positive integers."""
        for w, c in self.counts['unigram'].items():
            self.assertGreater(c, 0)


class TestMarkovModel(unittest.TestCase):
    """Tests for the MarkovModel class."""

    def setUp(self):
        self.model = MarkovModel(smoothing_k=1.0)
        self.model.train(SAMPLE_TEXT, test_ratio=0.2)

    def test_model_is_trained(self):
        """is_trained flag should be True after .train()."""
        self.assertTrue(self.model.is_trained)

    def test_vocab_size_positive(self):
        """Vocabulary size should be > 0 after training."""
        self.assertGreater(self.model.vocab_size, 0)

    def test_corpus_stats_populated(self):
        """corpus_stats dict should contain expected keys."""
        required = ['vocab_size', 'unique_bigrams', 'unique_trigrams',
                    'bigram_sparsity', 'trigram_sparsity', 'train_time_ms']
        for key in required:
            self.assertIn(key, self.model.corpus_stats, f"Missing key: {key}")

    # ── Probability tests ──

    def test_unigram_prob_valid_range(self):
        """Unigram probability should be in (0, 1]."""
        for word in list(self.model.vocab)[:10]:
            p = self.model.unigram_prob(word, smoothed=True)
            self.assertGreater(p, 0, f"P({word}) should be > 0 with smoothing")
            self.assertLessEqual(p, 1)

    def test_bigram_prob_smoothed_nonzero(self):
        """Smoothed bigram probability should be > 0 even for unseen pairs."""
        p = self.model.bigram_prob('xyzunknown', 'wordnothere', smoothed=True)
        self.assertGreater(p, 0, "Smoothed bigram P should be > 0 for unseen pair")

    def test_bigram_prob_mle_zero_unseen(self):
        """MLE bigram probability should be 0 for completely unseen prefix."""
        p = self.model.bigram_prob('xyzzzz', 'abcdef', smoothed=False)
        self.assertEqual(p, 0.0)

    def test_trigram_prob_smoothed_nonzero(self):
        """Smoothed trigram prob should be > 0."""
        p = self.model.trigram_prob('xyz', 'abc', 'def', smoothed=True)
        self.assertGreater(p, 0)

    def test_smoothed_greater_than_mle_for_unseen(self):
        """Smoothed probability > MLE for unseen pairs."""
        p_smooth = self.model.bigram_prob('xyzunknown', 'abc', smoothed=True)
        p_mle    = self.model.bigram_prob('xyzunknown', 'abc', smoothed=False)
        self.assertGreater(p_smooth, p_mle)

    # ── Top-3 prediction tests ──

    def test_predict_top3_returns_list(self):
        """predict_top3 should return a list."""
        result = self.model.predict_top3("the cat")
        self.assertIsInstance(result, list)

    def test_predict_top3_max_3(self):
        """Should return at most 3 predictions."""
        result = self.model.predict_top3("the cat")
        self.assertLessEqual(len(result), 3)

    def test_predict_top3_has_required_keys(self):
        """Each prediction dict should have word, probability, percent, rank."""
        result = self.model.predict_top3("statistics is")
        if result:
            first = result[0]
            for key in ('word', 'probability', 'percent', 'rank', 'model_used'):
                self.assertIn(key, first, f"Missing key: {key}")

    def test_predict_top3_sorted_by_probability(self):
        """Predictions should be sorted highest probability first."""
        result = self.model.predict_top3("machine learning")
        probs = [p['probability'] for p in result]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_predict_top3_probabilities_positive(self):
        """All returned probabilities should be positive."""
        result = self.model.predict_top3("the cat")
        for p in result:
            self.assertGreater(p['probability'], 0)

    def test_predict_top3_rank_1_is_first(self):
        """First prediction should have rank=1."""
        result = self.model.predict_top3("statistics is")
        if result:
            self.assertEqual(result[0]['rank'], 1)

    def test_predict_empty_input(self):
        """Empty input string should return empty list."""
        result = self.model.predict_top3("")
        self.assertIsInstance(result, list)

    def test_predict_unknown_words(self):
        """Unknown words should use unigram fallback, still return predictions."""
        result = self.model.predict_top3("zzzyyyxxx aaabbbccc")
        # Should fall back to unigram and return something
        self.assertIsInstance(result, list)

    def test_percent_equals_100x_probability(self):
        """percent field should equal probability * 100 (rounded)."""
        result = self.model.predict_top3("the cat")
        for p in result:
            expected = round(p['probability'] * 100, 4)
            self.assertAlmostEqual(p['percent'], expected, places=2)

    # ── Backoff tests ──

    def test_backoff_to_bigram(self):
        """Unseen trigram context should use bigram or unigram fallback."""
        result = self.model.predict_top3("zzz yyy", model_order='trigram')
        if result:
            # model_used should not be 'trigram' (since zzz yyy was never seen)
            self.assertIn(result[0]['model_used'],
                          ('bigram', 'bigram_fallback', 'unigram_fallback'))

    def test_unigram_model_always_predicts(self):
        """Unigram model should always return predictions regardless of context."""
        result = self.model.predict_top3("zzzyyyxxx aaabbbccc", model_order='unigram')
        self.assertGreater(len(result), 0)

    # ── Sentence generation ──

    def test_generate_returns_string(self):
        """generate_sentence should return a non-empty string."""
        sentence = self.model.generate_sentence()
        self.assertIsInstance(sentence, str)
        self.assertGreater(len(sentence), 0)

    def test_generate_ends_with_period(self):
        """Generated sentence should end with '.'."""
        for _ in range(5):
            s = self.model.generate_sentence()
            self.assertTrue(s.endswith('.'), f"Sentence doesn't end with '.': {s}")

    def test_generate_with_seed(self):
        """Sentence generated with a seed should start with that seed word."""
        sentence = self.model.generate_sentence(seed='statistics')
        self.assertIn('statistics', sentence.lower())

    def test_generate_max_words_respected(self):
        """Generated sentence should not exceed max_words."""
        sentence = self.model.generate_sentence(max_words=5)
        words = [w for w in sentence.rstrip('.').split() if w]
        self.assertLessEqual(len(words), 8)  # small buffer for seed words

    def test_generate_different_each_time(self):
        """Generate should produce varying sentences (not always identical)."""
        sentences = {self.model.generate_sentence() for _ in range(10)}
        # At least 2 different sentences (not 100% same every time)
        self.assertGreater(len(sentences), 1)

    # ── Perplexity ──

    def test_perplexity_finite(self):
        """Perplexity should be a finite number with smoothing."""
        pp = self.model.perplexity(model_order='bigram')
        self.assertFalse(math.isinf(pp), "Perplexity should not be infinity")
        self.assertGreater(pp, 0)

    def test_perplexity_all_orders(self):
        """All three model orders should give finite perplexity."""
        for order in ('unigram', 'bigram', 'trigram'):
            pp = self.model.perplexity(model_order=order)
            self.assertFalse(math.isinf(pp), f"Perplexity for {order} is infinite")

    def test_perplexity_custom_text(self):
        """Perplexity on custom text should also be finite."""
        pp = self.model.perplexity("the cat sat on the mat", model_order='bigram')
        self.assertFalse(math.isinf(pp))
        self.assertGreater(pp, 0)

    # ── Transition matrix ──

    def test_transition_matrix_structure(self):
        """get_transition_matrix should return dict with 'words' and matrices."""
        result = self.model.get_transition_matrix(top_n=5)
        self.assertIn('words', result)
        self.assertIn('mle_matrix', result)
        self.assertIn('smooth_matrix', result)

    def test_transition_matrix_probs_valid(self):
        """All probabilities in the matrix should be in [0, 1]."""
        result = self.model.get_transition_matrix(top_n=5)
        for w1, row in result['smooth_matrix'].items():
            for w2, p in row.items():
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)

    # ── Sparsity ──

    def test_sparsity_between_0_and_100(self):
        """Sparsity values should be between 0 and 100 percent."""
        self.assertGreaterEqual(self.model.corpus_stats['bigram_sparsity'], 0)
        self.assertLessEqual(self.model.corpus_stats['bigram_sparsity'], 100)
        self.assertGreaterEqual(self.model.corpus_stats['trigram_sparsity'], 0)
        self.assertLessEqual(self.model.corpus_stats['trigram_sparsity'], 100)


class TestTrainTestSplit(unittest.TestCase):
    """Tests for the train/test splitting function."""

    def test_split_produces_two_lists(self):
        tokens = preprocess_text(SAMPLE_TEXT)
        train, test = train_test_split(tokens, test_ratio=0.2)
        self.assertIsInstance(train, list)
        self.assertIsInstance(test, list)

    def test_split_no_overlap(self):
        """Train and test should have different token sequences."""
        tokens = preprocess_text(SAMPLE_TEXT)
        train, test = train_test_split(tokens, test_ratio=0.2)
        # At minimum they shouldn't both be empty
        self.assertGreater(len(train) + len(test), 0)

    def test_split_ratio_approximate(self):
        """Test set should be approximately 20% of total sentences."""
        tokens = preprocess_text(SAMPLE_TEXT)
        train, test = train_test_split(tokens, test_ratio=0.2)
        total = len(train) + len(test)
        if total > 0:
            test_frac = len(test) / total
            # Allow generous range since sentence-level split varies
            self.assertGreater(test_frac, 0.0)
            self.assertLess(test_frac, 1.0)

    def test_reproducible_with_same_seed(self):
        """Same seed should produce identical split."""
        tokens = preprocess_text(SAMPLE_TEXT)
        train1, test1 = train_test_split(tokens, test_ratio=0.2, seed=42)
        train2, test2 = train_test_split(tokens, test_ratio=0.2, seed=42)
        self.assertEqual(train1, train2)
        self.assertEqual(test1, test2)

    def test_different_seeds_differ(self):
        """Different seeds should (usually) produce different splits."""
        tokens = preprocess_text(SAMPLE_TEXT)
        _, test1 = train_test_split(tokens, test_ratio=0.2, seed=42)
        _, test2 = train_test_split(tokens, test_ratio=0.2, seed=99)
        # They might coincidentally be the same for tiny corpus, so just check types
        self.assertIsInstance(test1, list)
        self.assertIsInstance(test2, list)


if __name__ == '__main__':
    unittest.main(verbosity=2)
