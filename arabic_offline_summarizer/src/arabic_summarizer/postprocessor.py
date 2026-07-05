# src/arabic_summarizer/postprocessor.py
"""
Arabic Summary Post-Processor
Cleans and improves generated summaries
"""

import re
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ArabicPostprocessor:
    """
    Post-processes generated Arabic summaries.

    Handles:
    - Sentence deduplication (Jaccard similarity)
    - Coherence checking
    - Punctuation fixing
    - Output validation
    """

    def __init__(self, config=None):
        self.config = config

        # Arabic connector words that indicate incomplete sentences
        self._connectors = {
            'و', 'أو', 'ثم', 'لكن', 'أن', 'في', 'من', 'على',
            'إلى', 'عن', 'مع', 'كما', 'بينما', 'حيث', 'إذ',
        }

        # Patterns for model artifacts
        self._artifact_patterns = [
            re.compile(r'^(خلاصة|ملخص|باختصار|الخلاصة)[:\s]+', re.IGNORECASE),
            re.compile(r'^(Summary|Abstract)[:\s]+', re.IGNORECASE),
            re.compile(r'\s{2,}', ),           # multiple spaces
            re.compile(r'([.،؛؟!]){2,}'),      # repeated punctuation
        ]

    # ─── Deduplication ────────────────────────────────────────────────────────

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """Jaccard similarity between two word sets"""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def remove_repeated_sentences(
        self,
        text: str,
        similarity_threshold: float = 0.7,
    ) -> str:
        """
        Remove semantically repeated sentences using Jaccard similarity.

        Args:
            text: Input text with potential repetitions
            similarity_threshold: Sentences more similar than this are deduplicated

        Returns:
            Deduplicated text
        """
        # Split into sentences
        raw_sentences = re.split(r'(?<=[.؟!])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if len(sentences) <= 1:
            return text

        unique_sentences: List[str] = []
        seen_word_sets: List[set] = []

        for sentence in sentences:
            words = set(sentence.split())

            # Check similarity against all kept sentences
            is_duplicate = False
            for seen_words in seen_word_sets:
                sim = self._jaccard_similarity(words, seen_words)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sentences.append(sentence)
                seen_word_sets.append(words)

        return ' '.join(unique_sentences)

    # ─── Artifact Removal ─────────────────────────────────────────────────────

    def remove_model_artifacts(self, text: str) -> str:
        """Remove common T5/seq2seq model generation artifacts"""

        # Remove summary prefix labels
        text = re.sub(
            r'^(خلاصة|ملخص|باختصار|الخلاصة|Summary|Abstract)[:\s]+',
            '',
            text,
            flags=re.IGNORECASE,
        )

        # Remove repeated punctuation
        text = re.sub(r'([.،؛؟!])\1+', r'\1', text)

        # Remove incomplete trailing connector
        words = text.rstrip('.').split()
        if words and words[-1] in self._connectors:
            words = words[:-1]
            text = ' '.join(words)

        return text

    # ─── Punctuation & Spacing ────────────────────────────────────────────────

    def fix_sentence_ending(self, text: str) -> str:
        """Ensure text ends with proper punctuation"""
        text = text.rstrip()
        if not text:
            return text

        valid_endings = ('.', '،', '؛', '؟', '!', '۔')
        if not text.endswith(valid_endings):
            text += '.'

        return text

    def fix_spacing_around_punctuation(self, text: str) -> str:
        """Fix spaces around Arabic punctuation"""
        # Remove space BEFORE punctuation
        text = re.sub(r'\s+([.،؛؟!])', r'\1', text)
        # Add space AFTER punctuation (if not at end, not already spaced)
        text = re.sub(r'([.،؛؟!])(?=[^\s\d])', r'\1 ', text)
        # Normalize multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        return text

    # ─── Validation ───────────────────────────────────────────────────────────

    def validate_output(
        self,
        text: str,
        min_words: int = 10,
    ) -> Tuple[bool, str]:
        """
        Validate summary quality.

        Returns:
            (is_valid: bool, reason: str)
        """
        if not text or not text.strip():
            return False, "empty output"

        word_count = len(text.split())
        if word_count < min_words:
            return False, f"too short ({word_count} words, min={min_words})"

        # Check Arabic content ratio
        arabic_chars = len(re.findall(r'[\u0621-\u063A\u0641-\u064A]', text))
        total_chars = max(len(text.replace(' ', '')), 1)
        arabic_ratio = arabic_chars / total_chars

        if arabic_ratio < 0.25:
            return False, f"low Arabic content ({arabic_ratio:.1%})"

        return True, "ok"

    # ─── Main Pipeline ────────────────────────────────────────────────────────

    def process(self, text: str, config=None) -> str:
        """
        Full post-processing pipeline.

        Args:
            text: Raw model output
            config: Optional PostprocessingConfig override

        Returns:
            Cleaned and validated summary
        """
        if not text or not text.strip():
            return ""

        cfg = config or self.config

        # Step 1: Remove model artifacts
        text = self.remove_model_artifacts(text)

        # Step 2: Deduplicate sentences
        remove_reps = getattr(cfg, 'remove_repeated_sentences', True) \
            if cfg else True
        if remove_reps:
            threshold = getattr(cfg, 'sentence_similarity_threshold', 0.7) \
                if cfg else 0.7
            text = self.remove_repeated_sentences(text, threshold)

        # Step 3: Fix punctuation spacing
        fix_sp = getattr(cfg, 'fix_spacing', True) if cfg else True
        if fix_sp:
            text = self.fix_spacing_around_punctuation(text)

        # Step 4: Ensure proper ending
        ensure_end = getattr(cfg, 'ensure_ending_punctuation', True) \
            if cfg else True
        if ensure_end:
            text = self.fix_sentence_ending(text)

        # Step 5: Final strip
        text = text.strip()

        return text