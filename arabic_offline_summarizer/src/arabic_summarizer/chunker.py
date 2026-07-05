# src/arabic_summarizer/chunker.py
"""
Text Chunker for Long Arabic Documents
Handles texts that exceed model context window
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits long Arabic texts into overlapping chunks.

    Strategy:
    - Split at sentence boundaries (no mid-sentence cuts)
    - Overlap between chunks for context continuity
    - Merge chunk summaries intelligently

    Example:
        chunker = TextChunker(max_chars_per_chunk=3000, overlap_chars=300)
        chunks = chunker.chunk_text(text, sentences)
        summaries = [model.summarize(c) for c in chunks]
        final = chunker.merge_summaries(summaries)
    """

    def __init__(
        self,
        max_chars_per_chunk: int = 3000,
        overlap_chars: int = 300,
    ):
        self.max_chars = max_chars_per_chunk
        self.overlap_chars = overlap_chars

    def should_chunk(self, text: str, threshold_chars: int = 3500) -> bool:
        """Return True if text exceeds threshold and needs chunking"""
        return len(text) > threshold_chars

    def _get_overlap(
        self,
        sentences: List[str],
        max_chars: int,
    ) -> List[str]:
        """Return last N sentences within max_chars overlap limit"""
        overlap: List[str] = []
        total = 0

        for sent in reversed(sentences):
            length = len(sent) + 2
            if total + length > max_chars:
                break
            overlap.insert(0, sent)
            total += length

        return overlap

    def split_into_groups(
        self,
        sentences: List[str],
        max_chars: Optional[int] = None,
    ) -> List[List[str]]:
        """
        Group sentences into chunks respecting max_chars.

        Args:
            sentences:  Pre-split sentence list
            max_chars:  Override max chars per chunk

        Returns:
            List of sentence groups
        """
        limit = max_chars or self.max_chars

        if not sentences:
            return []

        groups: List[List[str]] = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence) + 2  # +2 for ". "

            if current_len + sent_len > limit and current:
                groups.append(current[:])
                # Start next chunk with overlap
                overlap = self._get_overlap(current, self.overlap_chars)
                current = overlap
                current_len = sum(len(s) + 2 for s in overlap)

            current.append(sentence)
            current_len += sent_len

        if current:
            groups.append(current)

        return groups

    def chunk_text(
        self,
        text: str,
        sentences: List[str],
    ) -> List[str]:
        """
        Convert text + sentences into text chunks.

        Args:
            text:      Original text (used as fallback)
            sentences: Pre-split sentences from preprocessor

        Returns:
            List of text strings ready for model input
        """
        if not sentences:
            return [text] if text else []

        groups = self.split_into_groups(sentences)
        chunks = [". ".join(group).rstrip(".") + "." for group in groups]

        logger.debug(f"Split into {len(chunks)} chunks")
        return chunks

    def merge_summaries(
        self,
        summaries: List[str],
        strategy: str = "concatenate",
    ) -> str:
        """
        Merge per-chunk summaries into final summary.

        Args:
            summaries: List of chunk summaries
            strategy:  "concatenate" (default) - join with space

        Returns:
            Merged summary string
        """
        if not summaries:
            return ""

        if len(summaries) == 1:
            return summaries[0].strip()

        cleaned = [s.rstrip(". ").strip() for s in summaries if s.strip()]

        if not cleaned:
            return ""

        merged = ". ".join(cleaned) + "."
        return merged