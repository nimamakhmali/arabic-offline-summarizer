# src/arabic_summarizer/__init__.py
"""
Arabic Offline Summarizer (AOS)
================================
Fast, accurate, fully offline Arabic text summarization.

Quick start:
    from arabic_summarizer import ArabicSummarizer
    
    summarizer = ArabicSummarizer()
    summary = summarizer.summarize(arabic_text, ratio=0.2)
"""

from .core import ArabicSummarizer, InferenceMode
from .config import (
    SummarizerConfig,
    ModelConfig,
    GenerationConfig,
    PreprocessingConfig,
    PostprocessingConfig,
    HybridConfig,
    DEFAULT_CONFIG,
    FAST_CONFIG,
    QUALITY_CONFIG,
    QURAN_CONFIG,
)
from .preprocessor import ArabicPreprocessor
from .postprocessor import ArabicPostprocessor
from .extractive import ArabicExtractiveSummarizer
from .evaluate import ArabicSummarizationEvaluator

__version__ = "2.0.0"
__author__ = "AOS Team"
__license__ = "Apache-2.0"

__all__ = [
    # Core
    "ArabicSummarizer",
    "InferenceMode",
    # Config
    "SummarizerConfig",
    "ModelConfig",
    "GenerationConfig", 
    "PreprocessingConfig",
    "PostprocessingConfig",
    "HybridConfig",
    "DEFAULT_CONFIG",
    "FAST_CONFIG",
    "QUALITY_CONFIG",
    "QURAN_CONFIG",
    # Sub-modules
    "ArabicPreprocessor",
    "ArabicPostprocessor",
    "ArabicExtractiveSummarizer",
    "ArabicSummarizationEvaluator",
]