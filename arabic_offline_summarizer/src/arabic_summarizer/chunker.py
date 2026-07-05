# src/arabic_summarizer/chunker.py
"""
Text Chunker for Long Arabic Documents
Handles texts that exceed model's context window
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits long texts into overlapping chunks for processing.
    
    Strategy:
    - Split at sentence boundaries
    - Maintain overlap for context continuity
    - Merge chunk summaries intelligently
    """

    def __init__(
        self,
        max_chars_per_chunk: int = 3000,
        overlap_chars: int = 300,
    ):
        self.max_chars = max_chars_per_chunk
        self.overlap_chars = overlap_chars

    def split_into_chunks(
        self,
        sentences: List[str],
        max_chars: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Split list of sentences into chunks.
        
        Args:
            sentences: List of sentences from preprocessor
            max_chars: Override max chars per chunk
            
        Returns:
            List of sentence groups (each group = one chunk)
        """
        max_chars = max_chars or self.max_chars
        
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_len = len(sentence) + 2  # +2 for ". "
            
            if current_length + sent_len > max_chars and current_chunk:
                chunks.append(current_chunk.copy())
                
                # Add overlap: keep last few sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.overlap_chars
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) + 2 for s in overlap_sentences)

            current_chunk.append(sentence)
            current_length += sent_len

        if current_chunk:
            chunks.append(current_chunk)

        logger.debug(f"Split into {len(chunks)} chunks")
        return chunks

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        max_overlap_chars: int
    ) -> List[str]:
        """Get last N sentences within overlap character limit"""
        overlap = []
        total = 0
        
        for sent in reversed(sentences):
            sent_len = len(sent) + 2
            if total + sent_len > max_overlap_chars:
                break
            overlap.insert(0, sent)
            total += sent_len

        return overlap

    def chunk_text(
        self,
        text: str,
        sentences: List[str]
    ) -> List[str]:
        """
        Convert sentence groups to text chunks.
        
        Args:
            text: Original text (unused, kept for API compatibility)
            sentences: Pre-split sentences
            
        Returns:
            List of text strings (one per chunk)
        """
        sentence_groups = self.split_into_chunks(sentences)
        return ['. '.join(group) + '.' for group in sentence_groups]

    def merge_summaries(
        self,
        chunk_summaries: List[str],
        strategy: str = "concatenate"
    ) -> str:
        """
        Merge summaries from multiple chunks.
        
        Strategies:
        - "concatenate": Simple joining with paragraph breaks
        - "extractive": Extract key sentences from merged text
        
        Args:
            chunk_summaries: List of per-chunk summaries
            strategy: Merging strategy
            
        Returns:
            Final merged summary
        """
        if not chunk_summaries:
            return ""

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        if strategy == "concatenate":
            # Join with space, letting postprocessor handle deduplication
            merged = ' '.join(s.rstrip('.').strip() for s in chunk_summaries)
            if not merged.endswith('.'):
                merged += '.'
            return merged

        # Fallback
        return ' '.join(chunk_summaries)

    def should_chunk(self, text: str, threshold_chars: int = 3500) -> bool:
        """Determine if text needs chunking"""
        return len(text) > threshold_chars