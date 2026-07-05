# src/arabic_summarizer/preprocessor.py
"""
Arabic Text Preprocessor - Production Grade
Handles MSA normalization, cleaning, and sentence segmentation
"""

import re
import unicodedata
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ─── Arabic Unicode Ranges ────────────────────────────────────────────────────
ARABIC_LETTERS = r'\u0621-\u063A\u0641-\u064A'
DIACRITICS = r'\u064B-\u065F\u0670'
TATWEEL = r'\u0640'
ARABIC_PUNCT = r'\u060C\u061B\u061F\u06D4'  # ،؛؟۔
EXTENDED_ARABIC = r'\u0671-\u06FF'

# ─── Regex Patterns ───────────────────────────────────────────────────────────
PATTERNS = {
    'diacritics': re.compile(f'[{DIACRITICS}]'),
    'tatweel': re.compile(f'[{TATWEEL}]'),
    'url': re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    ),
    'email': re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'),
    'multi_space': re.compile(r'[ \t]+'),
    'multi_newline': re.compile(r'\n{3,}'),
    'repeated_punct': re.compile(r'([.،؛؟!])\1+'),
    'non_arabic_chars': re.compile(
        r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
        r'\uFB50-\uFDFF\uFE70-\uFEFF'
        r'a-zA-Z0-9 \t\n.,،؛؟!()[\]{}"\'،\-–—]'
    ),
}

# ─── Sentence Delimiters ──────────────────────────────────────────────────────
SENTENCE_DELIMITERS = re.compile(r'[.؟!]\s+|[،؛]\s+(?=[A-Z\u0621-\u063A])')
HARD_DELIMITERS = re.compile(r'[.؟!۔]+')


class ArabicPreprocessor:
    """
    Production-grade Arabic text preprocessor for summarization.
    
    Handles:
    - Diacritics removal (تشكيل)
    - Arabic letter normalization (الف، ياء، تاء)
    - Tatweel (kashida) removal
    - URL/email cleaning
    - Sentence boundary detection
    - Token-aware truncation
    """

    def __init__(self, config=None):
        self.config = config
        self._setup_normalizers()

    def _setup_normalizers(self):
        """Setup letter normalization maps"""
        
        # Alef normalization: all alef forms → bare alef (ا)
        self.alef_map = str.maketrans({
            'أ': 'ا',  # alef with hamza above
            'إ': 'ا',  # alef with hamza below  
            'آ': 'ا',  # alef with madda
            'ٱ': 'ا',  # alef wasla
            'ٲ': 'ا',  # alef with wavy hamza above
            'ٳ': 'ا',  # alef with wavy hamza below
        })
        
        # Ya normalization: alef maqsura → ya
        self.ya_map = str.maketrans({
            'ى': 'ي',  # alef maqsura → ya
        })
        
        # Teh normalization: teh marbuta → ha
        self.teh_map = str.maketrans({
            'ة': 'ه',  # teh marbuta → ha
        })

    # ─── Core Normalization ───────────────────────────────────────────────────

    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel/harakat)"""
        return PATTERNS['diacritics'].sub('', text)

    def remove_tatweel(self, text: str) -> str:
        """Remove tatweel (kashida stretching character)"""
        return PATTERNS['tatweel'].sub('', text)

    def normalize_alef(self, text: str) -> str:
        """Normalize all alef forms to bare alef"""
        return text.translate(self.alef_map)

    def normalize_ya(self, text: str) -> str:
        """Normalize alef maqsura to ya"""
        return text.translate(self.ya_map)

    def normalize_teh(self, text: str) -> str:
        """Normalize teh marbuta to ha"""
        return text.translate(self.teh_map)

    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization"""
        return unicodedata.normalize('NFC', text)

    # ─── Cleaning ─────────────────────────────────────────────────────────────

    def remove_urls(self, text: str) -> str:
        return PATTERNS['url'].sub(' ', text)

    def remove_emails(self, text: str) -> str:
        return PATTERNS['email'].sub(' ', text)

    def fix_whitespace(self, text: str) -> str:
        """Normalize whitespace (spaces, tabs, newlines)"""
        # Normalize tabs to space
        text = text.replace('\t', ' ')
        # Multiple spaces → single space
        text = PATTERNS['multi_space'].sub(' ', text)
        # Multiple newlines → double newline (paragraph break)
        text = PATTERNS['multi_newline'].sub('\n\n', text)
        return text.strip()

    def fix_punctuation(self, text: str) -> str:
        """Fix repeated punctuation and Arabic punctuation spacing"""
        # Remove repeated punctuation (e.g., ,,, → ,)
        text = PATTERNS['repeated_punct'].sub(r'\1', text)
        # Ensure space after Arabic punctuation
        text = re.sub(r'([،؛؟!])(?!\s)', r'\1 ', text)
        # Ensure space after period if followed by Arabic letter
        text = re.sub(r'\.(?=[' + ARABIC_LETTERS + '])', '. ', text)
        return text

    # ─── Main API ─────────────────────────────────────────────────────────────

    def normalize(self, text: str, config=None) -> str:
        """
        Full normalization pipeline.
        
        Args:
            text: Raw Arabic text
            config: Optional PreprocessingConfig override
            
        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return ""

        cfg = config or self.config

        # Unicode normalization first
        text = self.normalize_unicode(text)

        # Remove diacritics
        if cfg is None or cfg.remove_diacritics:
            text = self.remove_diacritics(text)

        # Remove tatweel
        if cfg is None or cfg.remove_tatweel:
            text = self.remove_tatweel(text)

        # Letter normalizations
        if cfg is None or cfg.normalize_alef:
            text = self.normalize_alef(text)

        if cfg is None or cfg.normalize_ya:
            text = self.normalize_ya(text)

        if cfg is None or cfg.normalize_teh:
            text = self.normalize_teh(text)

        return text

    def clean(self, text: str, config=None) -> str:
        """
        Full cleaning pipeline: normalize + clean noise.
        
        Use this for preprocessing before model input.
        """
        if not text or not text.strip():
            return ""

        cfg = config or self.config

        # Step 1: Normalize
        text = self.normalize(text, config)

        # Step 2: Remove web noise
        if cfg is None or cfg.remove_urls:
            text = self.remove_urls(text)

        if cfg is None or cfg.remove_emails:
            text = self.remove_emails(text)

        # Step 3: Fix punctuation
        text = self.fix_punctuation(text)

        # Step 4: Fix whitespace
        if cfg is None or cfg.normalize_whitespace:
            text = self.fix_whitespace(text)

        return text

    def prepare_for_model(
        self,
        text: str,
        max_chars: int = 4000,
        preserve_structure: bool = True
    ) -> str:
        """
        Prepare text for model input with intelligent truncation.
        
        Args:
            text: Input text
            max_chars: Approximate character limit (proxy for tokens)
            preserve_structure: Try to truncate at sentence boundaries
            
        Returns:
            Cleaned and truncated text
        """
        text = self.clean(text)

        if len(text) <= max_chars:
            return text

        if not preserve_structure:
            return text[:max_chars].strip()

        # Intelligent truncation: break at sentence boundary
        sentences = self.split_sentences(text)
        result_parts = []
        total_chars = 0

        for sent in sentences:
            sent_len = len(sent) + 2  # +2 for ". "
            if total_chars + sent_len > max_chars:
                break
            result_parts.append(sent)
            total_chars += sent_len

        if not result_parts:
            # Fallback: at least return first sentence or truncate
            return text[:max_chars].strip()

        return '. '.join(result_parts) + '.'

    # ─── Sentence Splitting ───────────────────────────────────────────────────

    def split_sentences(
        self,
        text: str,
        min_words: int = 4,
        max_words: int = 100
    ) -> List[str]:
        """
        Split Arabic text into sentences.
        
        Uses a two-pass approach:
        1. Split at hard delimiters (., ؟, !)
        2. Further split long sentences at soft delimiters (،, ؛)
        
        Args:
            text: Input text (should be cleaned first)
            min_words: Minimum words for a valid sentence
            max_words: Maximum words before forced split
            
        Returns:
            List of sentence strings
        """
        if not text:
            return []

        # Pass 1: Split at hard sentence boundaries
        # Handle: period + space, question mark, exclamation
        parts = re.split(r'(?<=[.؟!])\s+', text)

        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            words = part.split()
            
            # Skip very short fragments
            if len(words) < min_words:
                # Append to previous sentence if exists
                if sentences:
                    sentences[-1] = sentences[-1] + ' ' + part
                else:
                    sentences.append(part)
                continue

            # Split very long "sentences" at soft delimiters
            if len(words) > max_words:
                sub_parts = re.split(r'[،؛]\s*', part)
                for sub in sub_parts:
                    sub = sub.strip()
                    sub_words = sub.split()
                    if len(sub_words) >= min_words:
                        sentences.append(sub)
                    elif sentences:
                        sentences[-1] += ' ' + sub
            else:
                sentences.append(part)

        return [s for s in sentences if s.strip()]

    # ─── Statistics ───────────────────────────────────────────────────────────

    def get_text_stats(self, text: str) -> dict:
        """
        Compute text statistics.
        
        Returns:
            Dict with char_count, word_count, sentence_count, etc.
        """
        if not text or not text.strip():
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_words_per_sentence": 0.0,
                "arabic_ratio": 0.0,
            }

        cleaned = self.clean(text)
        sentences = self.split_sentences(cleaned)
        words = cleaned.split()

        # Arabic character ratio
        arabic_chars = len(re.findall(f'[{ARABIC_LETTERS}{EXTENDED_ARABIC}]', text))
        total_chars = max(len(text.replace(' ', '')), 1)
        arabic_ratio = round(arabic_chars / total_chars, 3)

        return {
            "char_count": len(cleaned),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_words_per_sentence": round(
                len(words) / max(len(sentences), 1), 1
            ),
            "arabic_ratio": arabic_ratio,
        }

    def estimate_tokens(self, text: str, chars_per_token: float = 3.5) -> int:
        """
        Estimate token count without loading tokenizer.
        Arabic tokens average ~3-4 characters.
        """
        return int(len(text) / chars_per_token)