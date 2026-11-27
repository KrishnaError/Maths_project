"""
markov.py  —  WordNext AI
=========================
Core Markov Chain Language Model

Concepts implemented (BTech Probability & Statistics):
  - Unigram, Bigram, Trigram models
  - Maximum Likelihood Estimation (MLE)
  - Laplace (Add-k) Smoothing
  - Markov Property
  - Backoff Strategy: Trigram → Bigram → Unigram
  - Top-3 Next-Word Prediction
  - Sentence Generation
  - Perplexity Calculation
  - Transition Probability Matrix
"""

import re
import math
import random
import time
from collections import Counter, defaultdict


# ════════════════════════════════════════════════════════════
#  SECTION 1: TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> list:
    """
    Clean and tokenize raw text into a flat list of word tokens.

    Steps:
      1. Lowercase everything
      2. Split on sentence-ending punctuation to find sentence boundaries
      3. Wrap each sentence with <START> and <END> markers
      4. Extract only alphabetic word tokens

    Why sentence markers?
      The Markov model must learn that some words START sentences
      and others END them. Without markers, "the" after "." looks
      the same as "the" in the middle of a sentence.

    Args:
        text: Raw corpus string.

    Returns:
        List of string tokens including <START> and <END> markers.
    """
    text = text.lower()
    # Split on sentence boundaries  (. ! ?)
    sentences = re.split(r'[.!?]+', text)

    tokens = []
    for sent in sentences:
        # Keep only alphabetic words (removes numbers, punctuation)
        words = re.findall(r"[a-z']+", sent)
        if words:
            # Two START markers for trigram context at sentence beginning
            tokens += ['<START>', '<START>'] + words + ['<END>']

    return tokens


def build_vocabulary(tokens: list) -> set:
    """Return set of all unique tokens (the vocabulary V)."""
    return set(tokens)


def train_test_split(tokens: list, test_ratio: float = 0.2, seed: int = 42):
    """
    Split token list into train and test portions.

    We split at the sentence level (at <START> boundaries) so we
    never cut a sentence in half, keeping linguistic context intact.

    Args:
        tokens:     Full token list.
        test_ratio: Fraction to reserve for testing (default 20%).
        seed:       Random seed for reproducibility.

    Returns:
        (train_tokens, test_tokens) tuple.
    """
    # Find indices where sentences start
    sentence_starts = [i for i, t in enumerate(tokens) if t == '<START>'
                       and (i == 0 or tokens[i-1] == '<END>' or tokens[i-1] == '<START>')]

    # Collect sentence slices
    sentences = []
    seen = set()
    for idx in sentence_starts:
        if idx in seen:
            continue
        # Each sentence is a pair of <START><START>...words...<END>
        # Find the matching <END>
        end = idx
        while end < len(tokens) and tokens[end] != '<END>':
            end += 1
        slice_ = tokens[idx:end+1]
        if len(slice_) > 3:          # skip empty sentences
            sentences.append(slice_)
        seen.update(range(idx, end+1))

    random.seed(seed)
    random.shuffle(sentences)

    split_point = int(len(sentences) * (1 - test_ratio))
    train_sents = sentences[:split_point]
    test_sents  = sentences[split_point:]

    train_tokens = [tok for sent in train_sents for tok in sent]
    test_tokens  = [tok for sent in test_sents  for tok in sent]

    return train_tokens, test_tokens


# ════════════════════════════════════════════════════════════
#  SECTION 2: N-GRAM COUNTING
# ════════════════════════════════════════════════════════════

def count_ngrams(tokens: list) -> dict:
    """
    Build unigram, bigram, and trigram count dictionaries.

    Returns dict with keys:
        'unigram':  Counter  {word: count}
        'bigram':   defaultdict(Counter)  {w1: {w2: count}}
        'trigram':  defaultdict(Counter)  {(w1,w2): {w3: count}}

    Time Complexity:
        O(N) where N = number of tokens
    Space Complexity:
        O(V) for unigrams, O(V²) worst-case for bigrams, O(V³) for trigrams
        In practice much smaller because most combinations never appear.
    """
    unigram  = Counter()
    bigram   = defaultdict(Counter)
    trigram  = defaultdict(Counter)

    for i, word in enumerate(tokens):
        unigram[word] += 1

        if i >= 1:
            bigram[tokens[i-1]][word] += 1

        if i >= 2:
            trigram[(tokens[i-2], tokens[i-1])][word] += 1

    return {
        'unigram': unigram,
        'bigram':  bigram,
        'trigram': trigram,
    }


# ════════════════════════════════════════════════════════════
#  SECTION 3: PROBABILITY CALCULATIONS  (MLE + Laplace)
# ════════════════════════════════════════════════════════════

class MarkovModel:
    """
    Unified Markov Language Model supporting unigram, bigram, trigram.

    Key Mathematical Formulas
    ─────────────────────────
    MLE Unigram:
        P_MLE(w) = count(w) / N

    MLE Bigram:
        P_MLE(w₂|w₁) = count(w₁,w₂) / count(w₁)

    MLE Trigram:
        P_MLE(w₃|w₁,w₂) = count(w₁,w₂,w₃) / count(w₁,w₂)

    Laplace-Smoothed Bigram (Add-k):
        P_smooth(w₂|w₁) = (count(w₁,w₂) + k) / (count(w₁) + k × |V|)

    Laplace-Smoothed Trigram (Add-k):
        P_smooth(w₃|w₁,w₂) = (count(w₁,w₂,w₃) + k) / (count(w₁,w₂) + k × |V|)

    Markov Property:
        The next word depends ONLY on the previous n-1 words, not full history.
    """

    def __init__(self, smoothing_k: float = 1.0):
        """
        Args:
            smoothing_k: Laplace smoothing constant (1.0 = add-one smoothing).
        """
        self.k = smoothing_k

        # Will be populated by .train()
        self.counts       = {}
        self.vocab        = set()
        self.vocab_size   = 0
        self.total_tokens = 0

        self.train_tokens = []
        self.test_tokens  = []
        self.is_trained   = False
        self.corpus_stats = {}
        self.train_time   = 0.0

    # ──────────────────────────────────────────────────────────
    #  TRAIN
    # ──────────────────────────────────────────────────────────

    def train(self, text: str, test_ratio: float = 0.2):
        """
        Full training pipeline:
          preprocess → split → count n-grams → compute stats

        Args:
            text:       Raw corpus text.
            test_ratio: Fraction for held-out test set.
        """
        t_start = time.perf_counter()

        all_tokens = preprocess_text(text)
        self.train_tokens, self.test_tokens = train_test_split(
            all_tokens, test_ratio=test_ratio
        )

        self.counts = count_ngrams(self.train_tokens)
        self.vocab  = build_vocabulary(
            [t for t in self.train_tokens if t not in ('<START>', '<END>')]
        )
        self.vocab_size   = len(self.vocab)
        self.total_tokens = len(
            [t for t in self.train_tokens if t not in ('<START>', '<END>')]
        )

        self.train_time = round((time.perf_counter() - t_start) * 1000, 4)
        self._compute_corpus_stats()
        self.is_trained = True

        return self

    def _compute_corpus_stats(self):
        """Compute sparsity and coverage statistics."""
        ug = self.counts['unigram']
        bg = self.counts['bigram']
        tg = self.counts['trigram']

        unique_bigrams  = sum(len(v) for v in bg.values())
        unique_trigrams = sum(len(v) for v in tg.values())
        total_bigrams   = sum(sum(v.values()) for v in bg.values())
        total_trigrams  = sum(sum(v.values()) for v in tg.values())

        V = self.vocab_size
        possible_bigrams  = V * V
        possible_trigrams = V * V * V

        bigram_sparsity  = round((1 - unique_bigrams  / possible_bigrams)  * 100, 4) if possible_bigrams  else 100.0
        trigram_sparsity = round((1 - unique_trigrams / possible_trigrams) * 100, 6) if possible_trigrams else 100.0

        # Clamp to [0, 100]
        bigram_sparsity  = max(0.0, min(100.0, bigram_sparsity))
        trigram_sparsity = max(0.0, min(100.0, trigram_sparsity))

        self.corpus_stats = {
            'total_tokens':      self.total_tokens,
            'vocab_size':        V,
            'train_tokens':      len(self.train_tokens),
            'test_tokens':       len(self.test_tokens),
            'unique_unigrams':   len(ug),
            'unique_bigrams':    unique_bigrams,
            'unique_trigrams':   unique_trigrams,
            'total_bigrams':     total_bigrams,
            'total_trigrams':    total_trigrams,
            'bigram_sparsity':   bigram_sparsity,
            'trigram_sparsity':  min(100.0, trigram_sparsity),
            'train_time_ms':     self.train_time,
        }

    # ──────────────────────────────────────────────────────────
    #  PROBABILITY METHODS
    # ──────────────────────────────────────────────────────────

    def unigram_prob(self, w: str, smoothed: bool = True) -> float:
        """
        P(w)  — raw MLE or Laplace-smoothed unigram probability.

        MLE:      count(w) / N
        Smoothed: (count(w) + k) / (N + k × |V|)
        """
        ug    = self.counts['unigram']
        total = sum(ug.values())

        if smoothed:
            return (ug[w] + self.k) / (total + self.k * self.vocab_size)
        else:
            return ug[w] / total if total else 0.0

    def bigram_prob(self, w1: str, w2: str, smoothed: bool = True) -> float:
        """
        P(w2 | w1)  — Markov Property (order-1).

        MLE:      count(w1,w2) / count(w1)
        Smoothed: (count(w1,w2) + k) / (count(w1) + k × |V|)
        """
        ug = self.counts['unigram']
        bg = self.counts['bigram']

        num = bg[w1][w2]
        den = ug[w1]

        if smoothed:
            return (num + self.k) / (den + self.k * self.vocab_size) if (den + self.k * self.vocab_size) > 0 else 0.0
        else:
            return num / den if den > 0 else 0.0

    def trigram_prob(self, w1: str, w2: str, w3: str, smoothed: bool = True) -> float:
        """
        P(w3 | w1, w2)  — Markov Property (order-2).

        MLE:      count(w1,w2,w3) / count(w1,w2)
        Smoothed: (count(w1,w2,w3) + k) / (count(w1,w2) + k × |V|)
        """
        bg = self.counts['bigram']
        tg = self.counts['trigram']

        num = tg[(w1, w2)][w3]
        den = bg[w1][w2]

        if smoothed:
            return (num + self.k) / (den + self.k * self.vocab_size) if (den + self.k * self.vocab_size) > 0 else 0.0
        else:
            return num / den if den > 0 else 0.0

    # ──────────────────────────────────────────────────────────
    #  PREDICTION (BACKOFF)
    # ──────────────────────────────────────────────────────────

    def _get_candidates(self, w1: str, w2: str, model_order: str = 'trigram') -> dict:
        """
        Return {word: probability} for all candidate next words.

        Backoff Strategy:
          1. Try trigram  (w1, w2) → w3
          2. If no trigram context found, fall back to bigram  w2 → w3
          3. If no bigram context, fall back to unigram distribution

        This is the Stupid Backoff / Katz Backoff principle.
        """
        bg = self.counts['bigram']
        tg = self.counts['trigram']

        if model_order == 'trigram':
            # Level 1: trigram
            if tg.get((w1, w2)):
                candidates = {w: self.trigram_prob(w1, w2, w, smoothed=True)
                              for w in tg[(w1, w2)]
                              if w not in ('<END>',)}
                if candidates:
                    return candidates, 'trigram'

            # Level 2: bigram fallback
            if bg.get(w2):
                candidates = {w: self.bigram_prob(w2, w, smoothed=True)
                              for w in bg[w2]
                              if w not in ('<END>',)}
                if candidates:
                    return candidates, 'bigram_fallback'

        elif model_order == 'bigram':
            if bg.get(w2):
                candidates = {w: self.bigram_prob(w2, w, smoothed=True)
                              for w in bg[w2]
                              if w not in ('<END>',)}
                if candidates:
                    return candidates, 'bigram'

        # Level 3: unigram fallback
        candidates = {w: self.unigram_prob(w, smoothed=True)
                      for w in self.vocab
                      if w not in ('<START>', '<END>')}
        return candidates, 'unigram_fallback'

    def predict_top3(self, context: str, model_order: str = 'trigram') -> list:
        """
        Predict the top-3 most probable next words given a text context.

        Args:
            context:     Text typed so far (e.g., "the quick brown")
            model_order: 'unigram', 'bigram', or 'trigram'

        Returns:
            List of dicts with rank, word, probability, percent, model_used.
        """
        t_start = time.perf_counter()

        words = re.findall(r"[a-z'<>]+", context.lower())
        words = [w for w in words]  # keep all tokens for context

        if model_order == 'unigram':
            # Unigram: just return top words by frequency
            ug   = self.counts['unigram']
            cands = {w: self.unigram_prob(w, smoothed=True)
                     for w in self.vocab if w not in ('<START>', '<END>')}
            model_used = 'unigram'
        else:
            # Pad context if too short
            if len(words) == 0:
                w1, w2 = '<START>', '<START>'
            elif len(words) == 1:
                w1, w2 = '<START>', words[-1]
            else:
                w1, w2 = words[-2], words[-1]

            cands, model_used = self._get_candidates(w1, w2, model_order)

        # Sort and take top 3
        sorted_cands = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:3]

        pred_time = round((time.perf_counter() - t_start) * 1000, 4)

        results = []
        for rank, (word, prob) in enumerate(sorted_cands, 1):
            results.append({
                'rank':        rank,
                'word':        word,
                'probability': round(prob, 8),
                'percent':     round(prob * 100, 4),
                'model_used':  model_used,
                'pred_time_ms': pred_time,
            })

        return results

    # ──────────────────────────────────────────────────────────
    #  SENTENCE GENERATION
    # ──────────────────────────────────────────────────────────

    def generate_sentence(self, seed: str = '', max_words: int = 20,
                          model_order: str = 'trigram') -> str:
        """
        Generate a new sentence using weighted random sampling.

        The Markov Property is applied at every step:
          - Choose next word based ONLY on previous 1-2 words.
          - Use transition probabilities as weights.

        Args:
            seed:        Optional starting word(s).
            max_words:   Maximum words to generate.
            model_order: 'unigram', 'bigram', or 'trigram'.

        Returns:
            Generated sentence string.
        """
        if seed:
            words = re.findall(r"[a-z']+", seed.lower())
            context = ['<START>', '<START>'] + words
        else:
            context = ['<START>', '<START>']

        output_words = [w for w in context if w != '<START>']

        for _ in range(max_words):
            w1 = context[-2] if len(context) >= 2 else '<START>'
            w2 = context[-1]

            if w2 == '<END>':
                break

            cands, _ = self._get_candidates(w1, w2, model_order)
            # Remove special tokens from generation
            cands = {w: p for w, p in cands.items()
                     if w not in ('<START>', '<END>')}
            if not cands:
                break

            # Weighted random sampling (Markov property in action)
            words_list = list(cands.keys())
            probs_list = list(cands.values())
            total_p    = sum(probs_list)
            probs_norm = [p / total_p for p in probs_list]

            chosen = random.choices(words_list, weights=probs_norm, k=1)[0]
            output_words.append(chosen)
            context.append(chosen)

        if output_words:
            return ' '.join(output_words).capitalize() + '.'
        return 'Could not generate sentence.'

    # ──────────────────────────────────────────────────────────
    #  PERPLEXITY
    # ──────────────────────────────────────────────────────────

    def perplexity(self, text: str = None, model_order: str = 'bigram') -> float:
        """
        Compute perplexity on test set (or given text).

        Formula:
            PP(W) = 2^( -(1/N) × Σ log₂ P(wᵢ | context) )

        Lower perplexity = model is less "confused" = better fit.
        A perplexity of k means the model behaves as if choosing
        uniformly from k words at each step.

        Args:
            text:        Optional custom test text. Uses stored test set if None.
            model_order: 'unigram', 'bigram', or 'trigram'.

        Returns:
            Perplexity as a float. Returns inf if computation fails.
        """
        if text:
            tokens = preprocess_text(text)
        else:
            tokens = self.test_tokens

        if len(tokens) < 2:
            return float('inf')

        log_prob_sum = 0.0
        N = 0

        for i in range(2, len(tokens)):
            w = tokens[i]
            if w in ('<START>',):
                continue

            if model_order == 'unigram':
                p = self.unigram_prob(w, smoothed=True)
            elif model_order == 'bigram':
                p = self.bigram_prob(tokens[i-1], w, smoothed=True)
            else:  # trigram
                p = self.trigram_prob(tokens[i-2], tokens[i-1], w, smoothed=True)

            if p > 0:
                log_prob_sum += math.log2(p)
                N += 1

        if N == 0:
            return float('inf')

        avg_log = log_prob_sum / N
        pp      = 2 ** (-avg_log)
        return round(pp, 4)

    # ──────────────────────────────────────────────────────────
    #  TRANSITION MATRIX
    # ──────────────────────────────────────────────────────────

    def get_transition_matrix(self, top_n: int = 8) -> dict:
        """
        Build a small transition probability matrix for visualization.

        Returns the top_n most common content words and their
        pairwise bigram transition probabilities P(w2 | w1).

        Args:
            top_n: Number of words to include in the matrix.

        Returns:
            {'words': [...], 'mle_matrix': {...}, 'smooth_matrix': {...}}
        """
        # Pick top content words (skip stopwords and markers)
        skip = {'<START>', '<END>', 'a', 'an', 'the', 'is', 'are', 'was',
                'were', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'but',
                'for', 'it', 'this', 'that', 'with', 'by', 'from', 'as'}

        common = [
            w for w, _ in self.counts['unigram'].most_common(top_n + 20)
            if w not in skip
        ][:top_n]

        mle_matrix    = {}
        smooth_matrix = {}

        for w1 in common:
            mle_matrix[w1]    = {}
            smooth_matrix[w1] = {}
            for w2 in common:
                mle_matrix[w1][w2]    = round(self.bigram_prob(w1, w2, smoothed=False), 6)
                smooth_matrix[w1][w2] = round(self.bigram_prob(w1, w2, smoothed=True),  6)

        return {
            'words':         common,
            'mle_matrix':    mle_matrix,
            'smooth_matrix': smooth_matrix,
        }

    # ──────────────────────────────────────────────────────────
    #  TOP NGRAMS
    # ──────────────────────────────────────────────────────────

    def get_top_ngrams(self, n: int = 10, order: str = 'bigram') -> list:
        """
        Return the most frequent n-grams.

        Args:
            n:     How many top n-grams to return.
            order: 'unigram', 'bigram', or 'trigram'.

        Returns:
            List of (ngram_string, count) tuples.
        """
        skip = {'<START>', '<END>'}

        if order == 'unigram':
            return [
                (w, c) for w, c in self.counts['unigram'].most_common(n + 5)
                if w not in skip
            ][:n]

        elif order == 'bigram':
            pairs = []
            for w1, nexts in self.counts['bigram'].items():
                if w1 in skip:
                    continue
                for w2, cnt in nexts.items():
                    if w2 not in skip:
                        pairs.append((f'{w1} {w2}', cnt))
            return sorted(pairs, key=lambda x: x[1], reverse=True)[:n]

        else:  # trigram
            triples = []
            for (w1, w2), nexts in self.counts['trigram'].items():
                if w1 in skip or w2 in skip:
                    continue
                for w3, cnt in nexts.items():
                    if w3 not in skip:
                        triples.append((f'{w1} {w2} {w3}', cnt))
            return sorted(triples, key=lambda x: x[1], reverse=True)[:n]
