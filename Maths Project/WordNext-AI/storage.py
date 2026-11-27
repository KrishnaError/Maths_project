"""
storage.py  —  WordNext AI
===========================
DBMS & DSA Component

Three storage backends for N-gram counts:
  1. SQLite   — Relational database with indexes (DBMS)
  2. HashMap  — Python dict of dicts  (DSA: Hash Table)
  3. Trie     — Manually implemented prefix tree  (DSA: Tree)

For each backend we measure:
  - Insertion time
  - Query time
  - Estimated memory usage
  - Scalability across dataset sizes

This module is completely self-contained and does NOT depend on
the Flask app. It can be used or tested independently.
"""

import sqlite3
import time
import os
import sys
from collections import defaultdict


# ════════════════════════════════════════════════════════════
#  DATABASE PATH CONFIGURATION
# ════════════════════════════════════════════════════════════

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wordnext.db')


# ════════════════════════════════════════════════════════════
#  1. SQLITE BACKEND
# ════════════════════════════════════════════════════════════

class SQLiteStore:
    """
    Stores n-gram counts in a SQLite relational database.

    Schema:
        unigrams  (word TEXT PRIMARY KEY, count INTEGER)
        bigrams   (w1 TEXT, w2 TEXT, count INTEGER, PRIMARY KEY(w1,w2))
        trigrams  (w1 TEXT, w2 TEXT, w3 TEXT, count INTEGER, PRIMARY KEY(w1,w2,w3))

    Indexes are created on (w1) for bigrams and (w1,w2) for trigrams
    to speed up prefix lookups.

    Time Complexity (with index):
        Lookup: O(log N) — B-Tree index scan
        Insert: O(log N)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn    = None

    def connect(self):
        """Open or create the database and ensure tables exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')   # faster writes
        self.conn.execute('PRAGMA synchronous=NORMAL') # balance safety/speed
        self._create_tables()
        return self

    def _create_tables(self):
        """Create all tables and indexes if they don't already exist."""
        cur = self.conn.cursor()

        cur.execute('''
            CREATE TABLE IF NOT EXISTS unigrams (
                word  TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            )
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS bigrams (
                w1    TEXT NOT NULL,
                w2    TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (w1, w2)
            )
        ''')
        # Index on w1 for fast "given w1, find all w2" queries
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_bigrams_w1 ON bigrams (w1)
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS trigrams (
                w1    TEXT NOT NULL,
                w2    TEXT NOT NULL,
                w3    TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (w1, w2, w3)
            )
        ''')
        # Index on (w1,w2) for fast trigram prefix lookups
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_trigrams_w1w2 ON trigrams (w1, w2)
        ''')

        self.conn.commit()

    def clear_tables(self):
        """Remove all stored n-gram data (for re-training)."""
        self.conn.execute('DELETE FROM unigrams')
        self.conn.execute('DELETE FROM bigrams')
        self.conn.execute('DELETE FROM trigrams')
        self.conn.commit()

    def insert_ngrams(self, counts: dict) -> float:
        """
        Bulk-insert all n-gram counts into the database.

        Uses INSERT OR REPLACE for idempotency (safe to call twice).

        Args:
            counts: dict from markov.count_ngrams()

        Returns:
            Insertion time in milliseconds.
        """
        t_start = time.perf_counter()
        cur     = self.conn.cursor()

        # Unigrams
        uni_rows = [(w, c) for w, c in counts['unigram'].items()]
        cur.executemany(
            'INSERT OR REPLACE INTO unigrams (word, count) VALUES (?, ?)',
            uni_rows
        )

        # Bigrams
        bi_rows = []
        for w1, nexts in counts['bigram'].items():
            for w2, cnt in nexts.items():
                bi_rows.append((w1, w2, cnt))
        cur.executemany(
            'INSERT OR REPLACE INTO bigrams (w1, w2, count) VALUES (?, ?, ?)',
            bi_rows
        )

        # Trigrams
        tri_rows = []
        for (w1, w2), nexts in counts['trigram'].items():
            for w3, cnt in nexts.items():
                tri_rows.append((w1, w2, w3, cnt))
        cur.executemany(
            'INSERT OR REPLACE INTO trigrams (w1, w2, w3, count) VALUES (?, ?, ?, ?)',
            tri_rows
        )

        self.conn.commit()
        return round((time.perf_counter() - t_start) * 1000, 4)

    def query_bigram(self, w1: str) -> dict:
        """
        Retrieve all bigram successors for a given w1.

        SQL: SELECT w2, count FROM bigrams WHERE w1 = ?

        Args:
            w1: First word.

        Returns:
            {w2: count} dict.
        """
        cur  = self.conn.cursor()
        cur.execute('SELECT w2, count FROM bigrams WHERE w1 = ?', (w1,))
        return {row[0]: row[1] for row in cur.fetchall()}

    def query_trigram(self, w1: str, w2: str) -> dict:
        """
        Retrieve all trigram successors for a given (w1, w2) context.

        SQL: SELECT w3, count FROM trigrams WHERE w1 = ? AND w2 = ?

        Args:
            w1, w2: Context words.

        Returns:
            {w3: count} dict.
        """
        cur = self.conn.cursor()
        cur.execute(
            'SELECT w3, count FROM trigrams WHERE w1 = ? AND w2 = ?',
            (w1, w2)
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_unigram_count(self, word: str) -> int:
        """Return count of a single word. Returns 0 if not found."""
        cur = self.conn.cursor()
        cur.execute('SELECT count FROM unigrams WHERE word = ?', (word,))
        row = cur.fetchone()
        return row[0] if row else 0

    def get_table_sizes(self) -> dict:
        """Return number of rows in each table."""
        cur = self.conn.cursor()
        sizes = {}
        for table in ('unigrams', 'bigrams', 'trigrams'):
            cur.execute(f'SELECT COUNT(*) FROM {table}')
            sizes[table] = cur.fetchone()[0]
        return sizes

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# ════════════════════════════════════════════════════════════
#  2. HASH MAP BACKEND
# ════════════════════════════════════════════════════════════

class HashMapStore:
    """
    Stores n-gram counts using nested Python dicts (hash tables).

    Data layout:
        unigrams:  {word: count}
        bigrams:   {w1: {w2: count}}
        trigrams:  {(w1,w2): {w3: count}}

    Python dict uses a hash table internally:
        Average lookup:  O(1)
        Worst case:      O(N)  (hash collision — very rare)

    Memory:
        Each dict entry uses approximately 50–200 bytes depending on
        string lengths. Total ≈ O(V²) for bigrams.
    """

    def __init__(self):
        self.unigrams: dict = {}
        self.bigrams:  dict = {}   # {w1: {w2: count}}
        self.trigrams: dict = {}   # {(w1,w2): {w3: count}}
        self._insert_time_ms = 0.0

    def insert_ngrams(self, counts: dict) -> float:
        """
        Load all n-gram counts into hash maps.

        This is essentially just copying Python dicts — very fast.

        Args:
            counts: dict from markov.count_ngrams()

        Returns:
            Insertion time in milliseconds.
        """
        t_start = time.perf_counter()

        self.unigrams = dict(counts['unigram'])

        self.bigrams = {}
        for w1, nexts in counts['bigram'].items():
            self.bigrams[w1] = dict(nexts)

        self.trigrams = {}
        for key, nexts in counts['trigram'].items():
            self.trigrams[key] = dict(nexts)

        elapsed = round((time.perf_counter() - t_start) * 1000, 4)
        self._insert_time_ms = elapsed
        return elapsed

    def query_bigram(self, w1: str) -> dict:
        """
        O(1) average — single dict lookup.

        Args:
            w1: Context word.

        Returns:
            {w2: count} or empty dict.
        """
        return self.bigrams.get(w1, {})

    def query_trigram(self, w1: str, w2: str) -> dict:
        """
        O(1) average — single dict lookup by tuple key.

        Args:
            w1, w2: Context words.

        Returns:
            {w3: count} or empty dict.
        """
        return self.trigrams.get((w1, w2), {})

    def get_unigram_count(self, word: str) -> int:
        return self.unigrams.get(word, 0)

    def memory_estimate_kb(self) -> float:
        """Rough memory estimate using sys.getsizeof on dict keys."""
        total_entries = (
            len(self.unigrams) +
            sum(len(v) for v in self.bigrams.values()) +
            sum(len(v) for v in self.trigrams.values())
        )
        return round(total_entries * 100 / 1024, 2)   # ~100 bytes per entry

    def stats(self) -> dict:
        return {
            'unique_unigrams': len(self.unigrams),
            'unique_bigrams':  sum(len(v) for v in self.bigrams.values()),
            'unique_trigrams': sum(len(v) for v in self.trigrams.values()),
        }


# ════════════════════════════════════════════════════════════
#  3. TRIE BACKEND
# ════════════════════════════════════════════════════════════

class TrieNode:
    """
    A single node in the Trie.

    Each node represents one word in the n-gram prefix path.
    children maps the next word to its TrieNode.
    next_words stores counts of words that follow this prefix.
    """
    __slots__ = ('children', 'next_words', 'is_end')

    def __init__(self):
        self.children:  dict = {}   # {next_word: TrieNode}
        self.next_words: dict = {}  # {following_word: count}
        self.is_end: bool = False


class TrieStore:
    """
    Manually implemented Trie for N-gram storage.

    Structure for bigram ("statistics", "helps"):
        root
          └─ "statistics"  [TrieNode, is_end=True]
                next_words: {"helps": 3, "involves": 1}

    Structure for trigram ("data", "science", "involves"):
        root
          └─ "data"
               └─ "science"  [TrieNode, is_end=True]
                     next_words: {"involves": 2}

    Time Complexity:
        Insert:  O(m)  where m = length of prefix (number of words)
        Search:  O(m)
        m is 1 for bigrams, 2 for trigrams — effectively O(1) for us.

    Space Complexity:
        O(N × m) in worst case, but shared prefixes save memory.
    """

    def __init__(self):
        self.root = TrieNode()
        self._entry_count = 0
        self._insert_time_ms = 0.0

    def _get_or_create_node(self, prefix: tuple) -> TrieNode:
        """Walk the trie along prefix, creating nodes as needed."""
        node = self.root
        for word in prefix:
            if word not in node.children:
                node.children[word] = TrieNode()
            node = node.children[word]
        node.is_end = True
        return node

    def insert(self, prefix: tuple, next_word: str, count: int = 1):
        """
        Insert or update a single n-gram entry.

        Args:
            prefix:    Tuple of context words, e.g. ('data', 'science').
            next_word: The word that follows the prefix.
            count:     Observed count.
        """
        node = self._get_or_create_node(prefix)
        if next_word not in node.next_words:
            self._entry_count += 1
        node.next_words[next_word] = node.next_words.get(next_word, 0) + count

    def search(self, prefix: tuple) -> dict:
        """
        Look up all next-word candidates for a given prefix.

        Args:
            prefix: Tuple of context words.

        Returns:
            {next_word: count} or empty dict if prefix not found.

        Time: O(m) where m = len(prefix)
        """
        node = self.root
        for word in prefix:
            if word not in node.children:
                return {}
            node = node.children[word]
        return dict(node.next_words)

    def insert_ngrams(self, counts: dict) -> float:
        """
        Load all n-gram counts into the Trie.

        Args:
            counts: dict from markov.count_ngrams()

        Returns:
            Insertion time in milliseconds.
        """
        t_start = time.perf_counter()

        # Reset
        self.root   = TrieNode()
        self._entry_count = 0

        for w1, nexts in counts['bigram'].items():
            for w2, cnt in nexts.items():
                self.insert((w1,), w2, cnt)

        for (w1, w2), nexts in counts['trigram'].items():
            for w3, cnt in nexts.items():
                self.insert((w1, w2), w3, cnt)

        elapsed = round((time.perf_counter() - t_start) * 1000, 4)
        self._insert_time_ms = elapsed
        return elapsed

    def query_bigram(self, w1: str) -> dict:
        """Retrieve next-word counts for single-word prefix."""
        return self.search((w1,))

    def query_trigram(self, w1: str, w2: str) -> dict:
        """Retrieve next-word counts for two-word prefix."""
        return self.search((w1, w2))

    def memory_estimate_kb(self) -> float:
        """Estimate memory: each trie node ≈ 150 bytes."""
        node_count = self._count_nodes(self.root)
        return round(node_count * 150 / 1024, 2)

    def _count_nodes(self, node: TrieNode) -> int:
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def total_entries(self) -> int:
        return self._entry_count


# ════════════════════════════════════════════════════════════
#  4. PERFORMANCE BENCHMARK
# ════════════════════════════════════════════════════════════

def benchmark_all(counts: dict, n_query_trials: int = 100) -> dict:
    """
    Benchmark SQLite, HashMap, and Trie on:
      - Insertion time
      - Query time (average over n_query_trials)
      - Memory estimate

    Uses equivalent operations on all three backends for fair comparison.
    Results depend on hardware, OS, and dataset — no hardcoded values.

    Args:
        counts:          dict from markov.count_ngrams()
        n_query_trials:  Number of repeated queries to average timing.

    Returns:
        Nested dict with timing and size results for each backend.
    """

    # Collect some query keys from the actual data for realistic testing
    bigram_keys  = list(counts['bigram'].keys())[:n_query_trials]
    trigram_keys = list(counts['trigram'].keys())[:n_query_trials]

    results = {}

    # ── SQLite ──────────────────────────────────────────────
    db = SQLiteStore()
    db.connect()
    db.clear_tables()

    sql_insert_time = db.insert_ngrams(counts)

    # Bigram query timing
    t0 = time.perf_counter()
    for w1 in bigram_keys:
        db.query_bigram(w1)
    sql_bi_query = round((time.perf_counter() - t0) / max(len(bigram_keys), 1) * 1000, 4)

    # Trigram query timing
    t0 = time.perf_counter()
    for (w1, w2) in trigram_keys:
        db.query_trigram(w1, w2)
    sql_tri_query = round((time.perf_counter() - t0) / max(len(trigram_keys), 1) * 1000, 4)

    table_sizes = db.get_table_sizes()

    # Estimate DB file size
    db.close()
    db_size_kb = round(os.path.getsize(DB_PATH) / 1024, 2) if os.path.exists(DB_PATH) else 0.0

    results['sqlite'] = {
        'insert_time_ms': sql_insert_time,
        'bigram_query_ms':  sql_bi_query,
        'trigram_query_ms': sql_tri_query,
        'memory_kb':        db_size_kb,
        'rows_unigrams':    table_sizes['unigrams'],
        'rows_bigrams':     table_sizes['bigrams'],
        'rows_trigrams':    table_sizes['trigrams'],
        'notes': 'File-based DB; includes I/O overhead. Good for persistence and SQL queries.',
    }

    # ── Hash Map ────────────────────────────────────────────
    hmap = HashMapStore()
    hmap_insert_time = hmap.insert_ngrams(counts)

    t0 = time.perf_counter()
    for w1 in bigram_keys:
        hmap.query_bigram(w1)
    hmap_bi_query = round((time.perf_counter() - t0) / max(len(bigram_keys), 1) * 1000, 4)

    t0 = time.perf_counter()
    for (w1, w2) in trigram_keys:
        hmap.query_trigram(w1, w2)
    hmap_tri_query = round((time.perf_counter() - t0) / max(len(trigram_keys), 1) * 1000, 4)

    hmap_stats = hmap.stats()
    results['hashmap'] = {
        'insert_time_ms':   hmap_insert_time,
        'bigram_query_ms':  hmap_bi_query,
        'trigram_query_ms': hmap_tri_query,
        'memory_kb':        hmap.memory_estimate_kb(),
        'rows_unigrams':    hmap_stats['unique_unigrams'],
        'rows_bigrams':     hmap_stats['unique_bigrams'],
        'rows_trigrams':    hmap_stats['unique_trigrams'],
        'notes': 'In-memory Python dict. Fast average O(1) lookup. Lost on restart.',
    }

    # ── Trie ────────────────────────────────────────────────
    trie = TrieStore()
    trie_insert_time = trie.insert_ngrams(counts)

    t0 = time.perf_counter()
    for w1 in bigram_keys:
        trie.query_bigram(w1)
    trie_bi_query = round((time.perf_counter() - t0) / max(len(bigram_keys), 1) * 1000, 4)

    t0 = time.perf_counter()
    for (w1, w2) in trigram_keys:
        trie.query_trigram(w1, w2)
    trie_tri_query = round((time.perf_counter() - t0) / max(len(trigram_keys), 1) * 1000, 4)

    results['trie'] = {
        'insert_time_ms':   trie_insert_time,
        'bigram_query_ms':  trie_bi_query,
        'trigram_query_ms': trie_tri_query,
        'memory_kb':        trie.memory_estimate_kb(),
        'rows_unigrams':    0,   # Trie stores bigrams+trigrams only
        'rows_bigrams':     sum(len(trie.search((w1,))) for w1 in list(counts['bigram'].keys())[:20]),
        'rows_trigrams':    trie.total_entries(),
        'notes': 'Prefix tree. O(m) lookup where m=prefix length. Supports prefix search.',
    }

    return results


# ════════════════════════════════════════════════════════════
#  5. CONVENIENCE: GLOBAL SINGLETON FOR FLASK
# ════════════════════════════════════════════════════════════

# These are used by app.py to avoid re-creating stores on every request.
_sqlite_store  = None
_hashmap_store = None
_trie_store    = None


def get_sqlite_store() -> SQLiteStore:
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore()
        _sqlite_store.connect()
    elif _sqlite_store.conn is None:
        _sqlite_store.connect()
    return _sqlite_store


def get_hashmap_store() -> HashMapStore:
    global _hashmap_store
    if _hashmap_store is None:
        _hashmap_store = HashMapStore()
    return _hashmap_store


def get_trie_store() -> TrieStore:
    global _trie_store
    if _trie_store is None:
        _trie_store = TrieStore()
    return _trie_store


def initialize_stores(counts: dict):
    """Call this after training to populate all three stores."""
    global _sqlite_store, _hashmap_store, _trie_store

    db = get_sqlite_store()
    db.clear_tables()
    db.insert_ngrams(counts)

    hmap = get_hashmap_store()
    hmap.insert_ngrams(counts)

    trie = get_trie_store()
    trie.insert_ngrams(counts)
