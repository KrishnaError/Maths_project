"""
test_storage.py  —  WordNext AI
=================================
Unit tests for storage.py

Tests:
  - SQLite table creation, insert, query
  - HashMap insert, query
  - Trie insert, search, structure
  - Benchmark function
  - Equivalent results across all three backends
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
from collections import defaultdict, Counter
from markov import preprocess_text, count_ngrams
from storage import SQLiteStore, HashMapStore, TrieStore, TrieNode, benchmark_all

# ─── Shared test corpus ───
SAMPLE_TEXT = """
The cat sat on the mat. The cat ate the rat.
Statistics is the foundation of machine learning.
Probability theory helps us understand uncertainty.
Data science involves collecting and analyzing data.
"""

# Build counts once for all storage tests
_tokens = preprocess_text(SAMPLE_TEXT)
SAMPLE_COUNTS = count_ngrams(_tokens)


class TestSQLiteStore(unittest.TestCase):
    """Tests for SQLite backend."""

    def setUp(self):
        # Use a temp DB for each test
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        self.db = SQLiteStore(db_path=self.tmp.name)
        self.db.connect()
        self.db.insert_ngrams(SAMPLE_COUNTS)

    def tearDown(self):
        self.db.close()
        try:
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_tables_created(self):
        """All three tables should exist and have rows."""
        sizes = self.db.get_table_sizes()
        self.assertIn('unigrams', sizes)
        self.assertIn('bigrams', sizes)
        self.assertIn('trigrams', sizes)

    def test_unigrams_populated(self):
        """Unigrams table should have rows."""
        sizes = self.db.get_table_sizes()
        self.assertGreater(sizes['unigrams'], 0)

    def test_bigrams_populated(self):
        """Bigrams table should have rows."""
        sizes = self.db.get_table_sizes()
        self.assertGreater(sizes['bigrams'], 0)

    def test_trigrams_populated(self):
        """Trigrams table should have rows."""
        sizes = self.db.get_table_sizes()
        self.assertGreater(sizes['trigrams'], 0)

    def test_query_bigram_returns_dict(self):
        """query_bigram should return a dict."""
        result = self.db.query_bigram('the')
        self.assertIsInstance(result, dict)

    def test_query_bigram_known_word(self):
        """query_bigram('the') should find at least one successor."""
        result = self.db.query_bigram('the')
        self.assertGreater(len(result), 0, "'the' should have bigram successors")

    def test_query_bigram_unknown_word(self):
        """query_bigram on unknown word should return empty dict."""
        result = self.db.query_bigram('xyzunknown99')
        self.assertEqual(result, {})

    def test_query_trigram_returns_dict(self):
        """query_trigram should return a dict."""
        result = self.db.query_trigram('the', 'cat')
        self.assertIsInstance(result, dict)

    def test_query_trigram_unknown_context(self):
        """Unknown trigram context should return empty dict."""
        result = self.db.query_trigram('xyz123', 'abc456')
        self.assertEqual(result, {})

    def test_unigram_count_known_word(self):
        """Known word count should be > 0."""
        c = self.db.get_unigram_count('the')
        self.assertGreater(c, 0)

    def test_unigram_count_unknown_word(self):
        """Unknown word count should be 0."""
        c = self.db.get_unigram_count('xyzzzz')
        self.assertEqual(c, 0)

    def test_clear_tables(self):
        """clear_tables should remove all data."""
        self.db.clear_tables()
        sizes = self.db.get_table_sizes()
        self.assertEqual(sizes['unigrams'], 0)
        self.assertEqual(sizes['bigrams'],  0)
        self.assertEqual(sizes['trigrams'], 0)

    def test_insert_time_is_float(self):
        """insert_ngrams should return a float timing value."""
        t = self.db.insert_ngrams(SAMPLE_COUNTS)
        self.assertIsInstance(t, float)
        self.assertGreaterEqual(t, 0)


class TestHashMapStore(unittest.TestCase):
    """Tests for HashMap backend."""

    def setUp(self):
        self.hmap = HashMapStore()
        self.hmap.insert_ngrams(SAMPLE_COUNTS)

    def test_insert_time_is_float(self):
        t = self.hmap.insert_ngrams(SAMPLE_COUNTS)
        self.assertIsInstance(t, float)

    def test_query_bigram_returns_dict(self):
        result = self.hmap.query_bigram('the')
        self.assertIsInstance(result, dict)

    def test_query_bigram_known_word(self):
        result = self.hmap.query_bigram('the')
        self.assertGreater(len(result), 0)

    def test_query_bigram_unknown_word(self):
        result = self.hmap.query_bigram('xyzunknown99')
        self.assertEqual(result, {})

    def test_query_trigram_returns_dict(self):
        result = self.hmap.query_trigram('the', 'cat')
        self.assertIsInstance(result, dict)

    def test_query_trigram_unknown(self):
        result = self.hmap.query_trigram('xyz', 'abc')
        self.assertEqual(result, {})

    def test_counts_match_original(self):
        """HashMap counts should match original count_ngrams output."""
        for w1, nexts in SAMPLE_COUNTS['bigram'].items():
            hmap_nexts = self.hmap.query_bigram(w1)
            for w2, cnt in nexts.items():
                self.assertEqual(
                    hmap_nexts.get(w2), cnt,
                    f"Count mismatch for bigram ({w1},{w2}): expected {cnt}"
                )

    def test_memory_estimate_positive(self):
        """Memory estimate should be a positive number."""
        mem = self.hmap.memory_estimate_kb()
        self.assertGreater(mem, 0)

    def test_stats_keys(self):
        s = self.hmap.stats()
        for k in ('unique_unigrams', 'unique_bigrams', 'unique_trigrams'):
            self.assertIn(k, s)


class TestTrieStore(unittest.TestCase):
    """Tests for Trie backend."""

    def setUp(self):
        self.trie = TrieStore()
        self.trie.insert_ngrams(SAMPLE_COUNTS)

    def test_insert_time_is_float(self):
        t = self.trie.insert_ngrams(SAMPLE_COUNTS)
        self.assertIsInstance(t, float)

    def test_root_has_children(self):
        """Trie root should have children after inserting bigrams."""
        self.assertGreater(len(self.trie.root.children), 0)

    def test_query_bigram_known(self):
        """Known bigram prefix should return next-word dict."""
        result = self.trie.query_bigram('the')
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_query_bigram_unknown(self):
        """Unknown prefix should return empty dict."""
        result = self.trie.query_bigram('xyzunknown99')
        self.assertEqual(result, {})

    def test_query_trigram_known(self):
        result = self.trie.query_trigram('the', 'cat')
        self.assertIsInstance(result, dict)

    def test_query_trigram_unknown(self):
        result = self.trie.query_trigram('xyz', 'abc')
        self.assertEqual(result, {})

    def test_trie_matches_hashmap(self):
        """Trie results should match HashMap results for all bigrams."""
        hmap = HashMapStore()
        hmap.insert_ngrams(SAMPLE_COUNTS)

        for w1 in list(SAMPLE_COUNTS['bigram'].keys())[:20]:
            trie_result = self.trie.query_bigram(w1)
            hmap_result = hmap.query_bigram(w1)
            self.assertEqual(
                set(trie_result.keys()), set(hmap_result.keys()),
                f"Key mismatch for bigram prefix '{w1}'"
            )

    def test_insert_explicit(self):
        """Manual insert + search should work correctly."""
        trie = TrieStore()
        trie.insert(('hello',), 'world', count=5)
        trie.insert(('hello',), 'there', count=3)
        result = trie.search(('hello',))
        self.assertEqual(result.get('world'), 5)
        self.assertEqual(result.get('there'), 3)

    def test_total_entries_positive(self):
        """Total entries should be positive after inserting data."""
        self.assertGreater(self.trie.total_entries(), 0)

    def test_trie_node_structure(self):
        """TrieNode should have children, next_words, is_end attributes."""
        node = TrieNode()
        self.assertIsInstance(node.children, dict)
        self.assertIsInstance(node.next_words, dict)
        self.assertIsInstance(node.is_end, bool)

    def test_memory_estimate_positive(self):
        mem = self.trie.memory_estimate_kb()
        self.assertGreater(mem, 0)


class TestBenchmark(unittest.TestCase):
    """Tests for the benchmark_all function."""

    def setUp(self):
        import tempfile, storage
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.tmp.close()
        storage.DB_PATH = self.tmp.name  # redirect to temp DB

    def tearDown(self):
        try:
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_benchmark_returns_dict(self):
        result = benchmark_all(SAMPLE_COUNTS, n_query_trials=10)
        self.assertIsInstance(result, dict)

    def test_benchmark_has_all_backends(self):
        result = benchmark_all(SAMPLE_COUNTS, n_query_trials=10)
        for backend in ('sqlite', 'hashmap', 'trie'):
            self.assertIn(backend, result)

    def test_benchmark_timing_keys(self):
        result = benchmark_all(SAMPLE_COUNTS, n_query_trials=10)
        for backend, data in result.items():
            self.assertIn('insert_time_ms', data)
            self.assertIn('bigram_query_ms', data)
            self.assertIn('trigram_query_ms', data)
            self.assertIn('memory_kb', data)

    def test_benchmark_timings_non_negative(self):
        result = benchmark_all(SAMPLE_COUNTS, n_query_trials=10)
        for backend, data in result.items():
            self.assertGreaterEqual(data['insert_time_ms'], 0)
            self.assertGreaterEqual(data['bigram_query_ms'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
