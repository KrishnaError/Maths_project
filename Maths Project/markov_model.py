"""
markov_model.py
===============
Core Probability & Statistics Engine — Project 6

Concepts implemented:
  ─ Bigram & Trigram language models
  ─ Maximum Likelihood Estimation (MLE) of transition probabilities
  ─ Laplace (Add-1) smoothing for unseen word pairs
  ─ Backoff strategy: Trigram → Bigram → Unigram
  ─ Top-3 next-word suggestions
  ─ Sentence generation using the Markov property
  ─ Perplexity evaluation on held-out text
  ─ Top-3 accuracy on held-out text
  ─ Sparsity analysis
"""

import re
import math
import random
from collections import defaultdict, Counter
from data_store import HashMapStore, TrieStore, benchmark_stores


# ══════════════════════════════════════════════════════
#  TEXT PRE-PROCESSING
# ══════════════════════════════════════════════════════

def preprocess(text: str) -> list[str]:
    """
    Lowercase, remove punctuation (keep sentence boundaries as tokens),
    return flat list of word tokens with <START> / <END> markers.
    """
    text = text.lower()
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    tokens = []
    for sent in sentences:
        words = re.findall(r"[a-z']+", sent)
        if words:
            tokens += ['<START>', '<START>'] + words + ['<END>']
    return tokens


# ══════════════════════════════════════════════════════
#  MARKOV MODEL CLASS
# ══════════════════════════════════════════════════════

class MarkovModel:
    """
    Builds unigram, bigram, and trigram language models with
    MLE + Laplace smoothing and trigram→bigram→unigram backoff.
    """

    def __init__(self, smoothing: float = 1.0):
        """
        smoothing : Laplace smoothing constant k (default 1 = add-1)
        """
        self.k = smoothing

        # Raw counts
        self.unigram_counts: Counter = Counter()
        self.bigram_counts:  defaultdict = defaultdict(Counter)   # {w1: {w2: count}}
        self.trigram_counts: defaultdict = defaultdict(Counter)   # {(w1,w2): {w3: count}}

        # Vocabulary
        self.vocab: set = set()
        self.vocab_size: int = 0

        # Token list (for sentence generation)
        self.tokens: list = []

        # DSA stores (populated after fit)
        self.hmap_store = HashMapStore()
        self.trie_store = TrieStore()

        self.is_trained = False
        self.corpus_stats = {}

    # ──────────────────────────────────────────────────
    #  FIT
    # ──────────────────────────────────────────────────

    def fit(self, text: str):
        """Train the model on raw text string."""
        self.tokens = preprocess(text)
        self._build_counts(self.tokens)
        self.vocab = set(self.unigram_counts.keys())
        self.vocab_size = len(self.vocab)
        self._populate_stores()
        self._compute_corpus_stats()
        self.is_trained = True
        return self

    def _build_counts(self, tokens: list):
        for i, word in enumerate(tokens):
            self.unigram_counts[word] += 1
            if i > 0:
                self.bigram_counts[tokens[i-1]][word] += 1
            if i > 1:
                self.trigram_counts[(tokens[i-2], tokens[i-1])][word] += 1

    def _populate_stores(self):
        """Fill HashMapStore and TrieStore with bigram & trigram data."""
        for w1, nexts in self.bigram_counts.items():
            for w2, cnt in nexts.items():
                for _ in range(cnt):  # insert count times OR store once and use count
                    pass
                # Efficient: insert once, counts handled internally
                self.hmap_store._store[(w1,)][w2] = cnt
                self.trie_store.insert((w1,), w2)
                # Fix trie count after batch insert
                self.trie_store.root.children[w1].next_words[w2] = cnt

        for (w1, w2), nexts in self.trigram_counts.items():
            for w3, cnt in nexts.items():
                self.hmap_store._store[(w1, w2)][w3] = cnt
                self.trie_store.insert((w1, w2), w3)
                node = self.trie_store.root
                for w in [w1, w2]:
                    node = node.children[w]
                node.next_words[w3] = cnt

    def _compute_corpus_stats(self):
        total_tokens = len([t for t in self.tokens if t not in ('<START>', '<END>')])
        total_bigrams = sum(sum(v.values()) for v in self.bigram_counts.values())
        total_trigrams = sum(sum(v.values()) for v in self.trigram_counts.values())

        # Sparsity = fraction of possible n-grams that are ZERO
        possible_bigrams  = self.vocab_size ** 2
        possible_trigrams = self.vocab_size ** 3
        seen_bigrams  = sum(len(v) for v in self.bigram_counts.values())
        seen_trigrams = sum(len(v) for v in self.trigram_counts.values())

        bigram_sparsity  = 1 - (seen_bigrams  / possible_bigrams)  if possible_bigrams  else 0
        trigram_sparsity = 1 - (seen_trigrams / possible_trigrams) if possible_trigrams else 0

        self.corpus_stats = {
            "total_tokens":      total_tokens,
            "vocab_size":        self.vocab_size,
            "unique_bigrams":    seen_bigrams,
            "unique_trigrams":   seen_trigrams,
            "total_bigrams":     total_bigrams,
            "total_trigrams":    total_trigrams,
            "bigram_sparsity":   round(bigram_sparsity  * 100, 2),
            "trigram_sparsity":  round(trigram_sparsity * 100, 2),
        }

    # ──────────────────────────────────────────────────
    #  PROBABILITY CALCULATIONS (MLE + Laplace)
    # ──────────────────────────────────────────────────

    def bigram_prob(self, w1: str, w2: str) -> float:
        """P(w2 | w1) with Laplace smoothing."""
        num   = self.bigram_counts[w1][w2] + self.k
        denom = self.unigram_counts[w1]    + self.k * self.vocab_size
        return num / denom if denom else 0.0

    def trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        """P(w3 | w1, w2) with Laplace smoothing."""
        num   = self.trigram_counts[(w1, w2)][w3]       + self.k
        denom = self.bigram_counts[w1][w2]               + self.k * self.vocab_size
        return num / denom if denom else 0.0

    def unigram_prob(self, w: str) -> float:
        """P(w) unigram probability."""
        total = sum(self.unigram_counts.values())
        return (self.unigram_counts[w] + self.k) / (total + self.k * self.vocab_size)

    # ──────────────────────────────────────────────────
    #  BACKOFF PREDICTION
    # ──────────────────────────────────────────────────

    def _get_candidates(self, word1: str, word2: str) -> dict:
        """
        Backoff: try trigram, fall back to bigram, then unigram.
        Returns {word: probability} for all vocab words.
        """
        w1 = word1.lower()
        w2 = word2.lower()

        # -- Trigram check --
        if self.trigram_counts.get((w1, w2)):
            nexts = self.trigram_counts[(w1, w2)]
            return {w: self.trigram_prob(w1, w2, w)
                    for w in nexts if w not in ('<END>',)}

        # -- Bigram fallback --
        if self.bigram_counts.get(w2):
            nexts = self.bigram_counts[w2]
            return {w: self.bigram_prob(w2, w)
                    for w in nexts if w not in ('<END>',)}

        # -- Unigram fallback --
        return {w: self.unigram_prob(w)
                for w in self.vocab
                if w not in ('<START>', '<END>')}

    def predict_top3(self, context: str) -> list[dict]:
        """
        Given a text context, return top-3 next-word predictions
        with their probabilities and transition details.
        """
        words = re.findall(r"[a-z'<>]+", context.lower())
        if not words:
            words = ['<START>', '<START>']
        elif len(words) == 1:
            words = ['<START>'] + words

        w1, w2 = words[-2], words[-1]
        candidates = self._get_candidates(w1, w2)

        if not candidates:
            return []

        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:3]

        results = []
        for rank, (word, prob) in enumerate(sorted_cands, 1):
            results.append({
                "rank":        rank,
                "word":        word,
                "probability": round(prob, 6),
                "percent":     round(prob * 100, 3),
                "context_w1":  w1,
                "context_w2":  w2,
                "bigram_count":   self.bigram_counts[w2][word],
                "trigram_count":  self.trigram_counts[(w1, w2)][word],
            })
        return results

    # ──────────────────────────────────────────────────
    #  SENTENCE GENERATION
    # ──────────────────────────────────────────────────

    def generate_sentence(self, seed: str = "", max_words: int = 20) -> str:
        """
        Generate a sample sentence using the Markov property.
        Uses weighted random sampling based on transition probabilities.
        """
        if seed:
            words = re.findall(r"[a-z']+", seed.lower())
            if not words:
                words = ['<START>', '<START>']
            elif len(words) == 1:
                words = ['<START>'] + words
        else:
            words = ['<START>', '<START>']

        sentence_words = [w for w in words if w != '<START>']

        for _ in range(max_words):
            w1 = words[-2] if len(words) >= 2 else '<START>'
            w2 = words[-1]

            candidates = self._get_candidates(w1, w2)
            if not candidates or '<END>' in [w1, w2]:
                break

            # Remove special tokens from generation
            candidates = {w: p for w, p in candidates.items()
                         if w not in ('<START>', '<END>')}
            if not candidates:
                break

            # Weighted random choice (Markov property!)
            total = sum(candidates.values())
            rand  = random.random() * total
            cumulative = 0
            chosen = list(candidates.keys())[0]
            for word, prob in candidates.items():
                cumulative += prob
                if rand <= cumulative:
                    chosen = word
                    break

            sentence_words.append(chosen)
            words.append(chosen)

            if chosen == '<END>':
                break

        return ' '.join(sentence_words).capitalize() + '.'

    # ──────────────────────────────────────────────────
    #  PERPLEXITY
    # ──────────────────────────────────────────────────

    def perplexity(self, test_text: str, order: int = 2) -> float:
        """
        Compute perplexity of test_text under bigram (order=2)
        or trigram (order=3) model.

        Lower perplexity = better model fit.
        """
        tokens = preprocess(test_text)
        if len(tokens) < order:
            return float('inf')

        log_prob_sum = 0.0
        N = 0

        if order == 2:
            for i in range(1, len(tokens)):
                p = self.bigram_prob(tokens[i-1], tokens[i])
                if p > 0:
                    log_prob_sum += math.log2(p)
                    N += 1
        else:  # trigram
            for i in range(2, len(tokens)):
                p = self.trigram_prob(tokens[i-2], tokens[i-1], tokens[i])
                if p > 0:
                    log_prob_sum += math.log2(p)
                    N += 1

        if N == 0:
            return float('inf')

        avg_log_prob = log_prob_sum / N
        return round(2 ** (-avg_log_prob), 4)

    # ──────────────────────────────────────────────────
    #  TOP-3 ACCURACY
    # ──────────────────────────────────────────────────

    def evaluate_accuracy(self, test_text: str) -> dict:
        """
        Evaluate Top-1 and Top-3 accuracy on held-out test text.
        For each word position, check if the actual next word
        appears in our top-1 / top-3 predictions.
        """
        tokens = preprocess(test_text)
        tokens = [t for t in tokens if t not in ('<START>', '<END>')]

        top1_correct = 0
        top3_correct = 0
        total = 0

        for i in range(2, len(tokens)):
            context = ' '.join(tokens[max(0, i-2):i])
            actual  = tokens[i]
            preds   = self.predict_top3(context)
            pred_words = [p['word'] for p in preds]

            if pred_words:
                total += 1
                if pred_words[0] == actual:
                    top1_correct += 1
                if actual in pred_words:
                    top3_correct += 1

        return {
            "total_predictions": total,
            "top1_correct":      top1_correct,
            "top3_correct":      top3_correct,
            "top1_accuracy":     round(top1_correct / total * 100, 2) if total else 0,
            "top3_accuracy":     round(top3_correct / total * 100, 2) if total else 0,
        }

    # ──────────────────────────────────────────────────
    #  DSA BENCHMARK
    # ──────────────────────────────────────────────────

    def run_benchmark(self) -> dict:
        """Compare HashMap vs Trie query speed."""
        all_ngrams = []
        for w1, nexts in self.bigram_counts.items():
            for w2, cnt in nexts.items():
                all_ngrams.extend([(w1,), w2] for _ in range(cnt))

        # Flatten to (prefix, next_word) pairs
        pairs = []
        for w1, nexts in self.bigram_counts.items():
            for w2 in nexts:
                pairs.append(((w1,), w2))
        for (w1, w2), nexts in self.trigram_counts.items():
            for w3 in nexts:
                pairs.append(((w1, w2), w3))

        # Sample queries
        queries = [p[0] for p in pairs[:min(200, len(pairs))]]
        return benchmark_stores(pairs, queries)

    # ──────────────────────────────────────────────────
    #  TRANSITION MATRIX (top words)
    # ──────────────────────────────────────────────────

    def get_transition_matrix(self, top_n: int = 8) -> dict:
        """
        Return a small transition probability matrix for visualization.
        Uses the top_n most frequent non-special words.
        """
        common_words = [
            w for w, _ in self.unigram_counts.most_common(top_n + 10)
            if w not in ('<START>', '<END>', 'a', 'an', 'the', 'is', 'are',
                         'was', 'were', 'in', 'on', 'at', 'to', 'of', 'and',
                         'or', 'but', 'for')
        ][:top_n]

        matrix = {}
        for w1 in common_words:
            matrix[w1] = {}
            for w2 in common_words:
                matrix[w1][w2] = round(self.bigram_prob(w1, w2), 4)

        return {"words": common_words, "matrix": matrix}

    def get_top_bigrams(self, n: int = 10) -> list:
        """Return top-n most frequent bigrams."""
        bigrams = []
        for w1, nexts in self.bigram_counts.items():
            if w1 == '<START>':
                continue
            for w2, cnt in nexts.items():
                if w2 not in ('<END>', '<START>'):
                    bigrams.append((f"{w1} {w2}", cnt))
        return sorted(bigrams, key=lambda x: x[1], reverse=True)[:n]
