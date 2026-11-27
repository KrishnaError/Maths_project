"""
data_store.py
=============
DBMS & DSA Component — Project 6: Markov Chain Next-Word Predictor

Two data structures to store n-gram counts:
  1. Nested Hash Map  (Python dict of dicts) — O(1) average lookup
  2. Trie             — Tree-based prefix structure

Both store the same data; we compare query speed between them.
"""

import time
from collections import defaultdict


# ─────────────────────────────────────────────
# 1.  NESTED HASH MAP  (simple dict-of-dicts)
# ─────────────────────────────────────────────

class HashMapStore:
    """
    Stores n-gram counts as:
        { prefix_tuple : { next_word : count } }

    Example (bigram):
        ("the", "quick") -> {"brown": 3, "fox": 1}
    """

    def __init__(self):
        # defaultdict so missing keys auto-create inner dict
        self._store: dict = defaultdict(lambda: defaultdict(int))

    def insert(self, prefix: tuple, next_word: str):
        """Increment count for (prefix → next_word)."""
        self._store[prefix][next_word] += 1

    def get_counts(self, prefix: tuple) -> dict:
        """Return {next_word: count} for given prefix, empty dict if missing."""
        return dict(self._store.get(prefix, {}))

    def get_all_prefixes(self):
        return list(self._store.keys())

    def total_entries(self):
        return sum(len(v) for v in self._store.values())


# ─────────────────────────────────────────────
# 2.  TRIE (prefix tree)
# ─────────────────────────────────────────────

class TrieNode:
    """Single node in the Trie."""

    def __init__(self):
        self.children: dict = {}       # word → TrieNode
        self.next_words: dict = {}     # next_word → count  (leaf-level)
        self.is_end: bool = False


class TrieStore:
    """
    Trie where each path root→leaf represents a prefix tuple,
    and the leaf node holds {next_word: count}.

    Example path for prefix ("the", "quick"):
        root ──"the"──> node ──"quick"──> leaf {next_words: {"brown":3}}
    """

    def __init__(self):
        self.root = TrieNode()
        self._entry_count = 0

    def insert(self, prefix: tuple, next_word: str):
        node = self.root
        for word in prefix:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.is_end = True
        if next_word not in node.next_words:
            node.next_words[next_word] = 0
            self._entry_count += 1
        node.next_words[next_word] += 1

    def get_counts(self, prefix: tuple) -> dict:
        node = self.root
        for word in prefix:
            if word not in node.children:
                return {}
            node = node.children[word]
        return dict(node.next_words)

    def total_entries(self):
        return self._entry_count


# ─────────────────────────────────────────────
# 3.  SPEED COMPARISON UTILITY
# ─────────────────────────────────────────────

def benchmark_stores(ngrams: list, queries: list) -> dict:
    """
    Builds both stores from ngrams list, then times queries on both.

    ngrams  : list of (prefix_tuple, next_word) pairs
    queries : list of prefix_tuples to look up

    Returns dict with timing results.
    """
    # ── Build HashMap ──
    hmap = HashMapStore()
    t0 = time.perf_counter()
    for prefix, nw in ngrams:
        hmap.insert(prefix, nw)
    hmap_build = (time.perf_counter() - t0) * 1000  # ms

    # ── Build Trie ──
    trie = TrieStore()
    t0 = time.perf_counter()
    for prefix, nw in ngrams:
        trie.insert(prefix, nw)
    trie_build = (time.perf_counter() - t0) * 1000

    # ── Query HashMap ──
    t0 = time.perf_counter()
    for q in queries:
        hmap.get_counts(q)
    hmap_query = (time.perf_counter() - t0) * 1000

    # ── Query Trie ──
    t0 = time.perf_counter()
    for q in queries:
        trie.get_counts(q)
    trie_query = (time.perf_counter() - t0) * 1000

    return {
        "hashmap": {
            "build_ms": round(hmap_build, 4),
            "query_ms": round(hmap_query, 4),
            "entries":  hmap.total_entries()
        },
        "trie": {
            "build_ms": round(trie_build, 4),
            "query_ms": round(trie_query, 4),
            "entries":  trie.total_entries()
        }
    }
