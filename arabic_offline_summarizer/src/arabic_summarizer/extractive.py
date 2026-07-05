# src/arabic_summarizer/extractive.py
"""
Extractive Summarization for Arabic
Improved TF-IDF + Position + Centrality scoring
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Common Arabic stopwords (offline list)
ARABIC_STOPWORDS = {
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
    'التي', 'الذي', 'الذين', 'اللاتي', 'أن', 'إن', 'كان', 'كانت', 'لكن',
    'ولكن', 'أو', 'أم', 'ثم', 'حتى', 'إذا', 'لما', 'قد', 'لقد', 'سوف',
    'ما', 'لا', 'لم', 'لن', 'ليس', 'ليست', 'هو', 'هي', 'هم', 'هن', 'أنت',
    'أنا', 'نحن', 'وقد', 'وكان', 'وكانت', 'يكون', 'تكون', 'يكن', 'عند',
    'بعد', 'قبل', 'خلال', 'بين', 'حول', 'تحت', 'فوق', 'ضد', 'نحو',
    'كل', 'بعض', 'جميع', 'غير', 'أي', 'كلا', 'كلما', 'كما',
    'و', 'ف', 'ب', 'ل', 'ك'
}


class TFIDFScorer:
    """
    TF-IDF scorer for Arabic sentences.
    Fully offline implementation.
    """

    def compute_tf(self, sentence_words: List[str]) -> Dict[str, float]:
        """Term frequency for a sentence"""
        if not sentence_words:
            return {}
        word_count = Counter(sentence_words)
        max_count = max(word_count.values()) if word_count else 1
        return {word: count / max_count for word, count in word_count.items()}

    def compute_idf(
        self,
        all_sentences_words: List[List[str]],
        vocab: set
    ) -> Dict[str, float]:
        """Inverse document frequency across sentences"""
        n = len(all_sentences_words)
        if n == 0:
            return {}
        
        idf = {}
        for word in vocab:
            # Count sentences containing this word
            doc_freq = sum(
                1 for words in all_sentences_words if word in set(words)
            )
            if doc_freq > 0:
                idf[word] = math.log((n + 1) / (doc_freq + 1)) + 1.0
            else:
                idf[word] = 1.0
        
        return idf

    def score_sentences_tfidf(
        self,
        sentences: List[str]
    ) -> List[float]:
        """
        Score sentences using TF-IDF.
        
        Returns:
            Normalized TF-IDF scores for each sentence
        """
        # Tokenize and filter stopwords
        all_words_list = []
        for sent in sentences:
            words = [
                w for w in sent.split()
                if w not in ARABIC_STOPWORDS and len(w) > 1
            ]
            all_words_list.append(words)

        # Build vocabulary
        vocab = set(w for words in all_words_list for w in words)
        
        if not vocab:
            return [1.0 / len(sentences)] * len(sentences)

        # Compute IDF
        idf = self.compute_idf(all_words_list, vocab)

        # Compute TF-IDF score for each sentence
        scores = []
        for words in all_words_list:
            if not words:
                scores.append(0.0)
                continue
            tf = self.compute_tf(words)
            score = sum(tf.get(w, 0) * idf.get(w, 1) for w in words)
            scores.append(score / len(words))  # Normalize by sentence length

        # Normalize scores to [0, 1]
        max_score = max(scores) if scores else 1
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores


class ArabicExtractiveSummarizer:
    """
    Multi-feature extractive summarizer for Arabic.
    
    Features:
    - TF-IDF scoring (without stopwords)
    - Position bias (first/last sentences more important)  
    - Sentence length normalization
    - Keyword density bonus
    - Used as: (1) standalone extractor, (2) pre-filter for abstractive model
    """

    def __init__(self):
        self.tfidf = TFIDFScorer()

    def _compute_scores(
        self,
        sentences: List[str]
    ) -> List[float]:
        """
        Multi-feature sentence scoring.
        
        Score = 0.45*tfidf + 0.25*position + 0.20*length + 0.10*keyword
        """
        n = len(sentences)
        if n == 0:
            return []

        # 1. TF-IDF scores
        tfidf_scores = self.tfidf.score_sentences_tfidf(sentences)

        # 2. Position scores
        position_scores = []
        for i in range(n):
            rel_pos = i / max(n - 1, 1)
            if i == 0:
                pos_score = 1.0         # First sentence: highest
            elif i == n - 1:
                pos_score = 0.85        # Last sentence: high
            elif rel_pos <= 0.25:
                pos_score = 0.75        # Early sentences
            elif rel_pos >= 0.75:
                pos_score = 0.55        # Late sentences
            else:
                pos_score = 0.40        # Middle sentences
            position_scores.append(pos_score)

        # 3. Length scores (prefer 8-30 word sentences)
        length_scores = []
        for sent in sentences:
            length = len(sent.split())
            if 8 <= length <= 30:
                len_score = 1.0
            elif 5 <= length < 8:
                len_score = 0.7
            elif 30 < length <= 50:
                len_score = 0.7
            elif length < 5:
                len_score = 0.2
            else:
                len_score = 0.5  # very long sentences
            length_scores.append(len_score)

        # 4. Keyword density (numbers, proper nouns indicators)
        keyword_scores = []
        for sent in sentences:
            words = sent.split()
            if not words:
                keyword_scores.append(0.0)
                continue
            # Heuristic: words with digits, or all-Arabic long words
            keyword_count = sum(
                1 for w in words
                if any(c.isdigit() for c in w) or len(w) >= 6
            )
            keyword_scores.append(min(keyword_count / len(words), 1.0))

        # Combine with weights
        final_scores = []
        weights = (0.45, 0.25, 0.20, 0.10)
        
        for i in range(n):
            score = (
                weights[0] * tfidf_scores[i] +
                weights[1] * position_scores[i] +
                weights[2] * length_scores[i] +
                weights[3] * keyword_scores[i]
            )
            final_scores.append(score)

        return final_scores

    def extract(
        self,
        text: str,
        ratio: float = 0.4,
        sentences: Optional[List[str]] = None,
        preprocessor=None
    ) -> str:
        """
        Extract most important sentences from Arabic text.
        
        Args:
            text: Input Arabic text
            ratio: Fraction of sentences to keep (0.1 - 0.9)
            sentences: Pre-split sentences (optional, saves computation)
            preprocessor: ArabicPreprocessor instance (optional)
            
        Returns:
            Extracted text as string
        """
        # Get sentences
        if sentences is None:
            if preprocessor:
                text = preprocessor.clean(text)
                sentences = preprocessor.split_sentences(text)
            else:
                # Fallback: simple split
                sentences = [
                    s.strip()
                    for s in re.split(r'[.؟!]\s+', text)
                    if s.strip() and len(s.split()) >= 3
                ]

        if not sentences:
            return text

        if len(sentences) <= 3:
            return '. '.join(sentences) + '.'

        # Score sentences
        scores = self._compute_scores(sentences)

        # Select top N sentences
        num_to_select = max(2, int(len(sentences) * ratio))
        num_to_select = min(num_to_select, len(sentences))

        # Get indices of top-scoring sentences
        indexed_scores = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        top_indices = sorted(
            [idx for idx, _ in indexed_scores[:num_to_select]]
        )

        # Reconstruct in original order
        selected = [sentences[i] for i in top_indices]
        
        return '. '.join(s.rstrip('.').strip() for s in selected) + '.'

    def hybrid_prepare(
        self,
        text: str,
        target_ratio: float = 0.2,
        sentences: Optional[List[str]] = None,
        preprocessor=None
    ) -> str:
        """
        Pre-filter text for abstractive model using extraction.
        
        Goal: Reduce input length while keeping key content,
        so abstractive model receives focused input.
        
        Extract ratio is calibrated to be larger than final ratio
        (extractive is rough, abstractive refines).
        """
        # Calculate extraction ratio (3x final, capped at 65%)
        extract_ratio = min(0.65, max(0.35, target_ratio * 3.0))
        
        return self.extract(
            text,
            ratio=extract_ratio,
            sentences=sentences,
            preprocessor=preprocessor
        )