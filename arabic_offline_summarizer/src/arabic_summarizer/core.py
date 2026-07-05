# src/arabic_summarizer/core.py
"""
Arabic Offline Summarizer - Core Engine (Final v2.0)
Clean architecture - no lazy imports, proper error handling
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ─── Top-level imports (fail fast, no lazy imports) ──────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. Only extractive mode available.")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

from .config import (
    DEFAULT_CONFIG,
    GenerationConfig,
    HybridConfig,
    SummarizerConfig,
)
from .chunker import TextChunker
from .extractive import ArabicExtractiveSummarizer
from .postprocessor import ArabicPostprocessor
from .preprocessor import ArabicPreprocessor


class InferenceMode(str, Enum):
    ONNX = "onnx"
    TRANSFORMERS = "transformers"
    EXTRACTIVE_ONLY = "extractive_only"


class ModelLoadError(Exception):
    """Raised when model cannot be loaded"""
    pass


class ArabicSummarizer:
    """
    Professional Arabic Text Summarizer - Production Ready.

    Inference priority:
        ONNX (fastest) → Transformers (accurate) → Extractive (fallback)

    Example:
        >>> summarizer = ArabicSummarizer()
        >>> summary = summarizer.summarize(arabic_text, ratio=0.2)

        >>> # With detailed stats
        >>> result = summarizer.summarize(text, return_stats=True)
        >>> print(result["summary"], result["time_seconds"])

        >>> # Batch processing
        >>> results = summarizer.batch_summarize(texts, ratio=0.15)

        >>> # Custom config
        >>> from arabic_summarizer import QURAN_CONFIG
        >>> summarizer = ArabicSummarizer(config=QURAN_CONFIG)
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        config: Optional[SummarizerConfig] = None,
        model_path: Optional[Union[str, Path]] = None,
        force_mode: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize ArabicSummarizer.

        Args:
            config:     SummarizerConfig instance (uses DEFAULT_CONFIG if None)
            model_path: Path to ONNX or Transformers model directory
            force_mode: "onnx" | "transformers" | "extractive_only"
            verbose:    Log progress and status messages
        """
        self.config = config or DEFAULT_CONFIG
        self.verbose = verbose
        self.model_path = Path(model_path) if model_path else None

        # Initialize sub-modules (all pure Python, always available)
        self.preprocessor = ArabicPreprocessor(
            config=self.config.preprocessing
        )
        self.postprocessor = ArabicPostprocessor(
            config=self.config.postprocessing
        )
        self.extractive = ArabicExtractiveSummarizer()
        self.chunker = TextChunker(
            max_chars_per_chunk=self.config.hybrid.chunk_size_tokens * 4,
            overlap_chars=self.config.hybrid.chunk_overlap_tokens * 4,
        )

        # Model state
        self.mode: InferenceMode = InferenceMode.EXTRACTIVE_ONLY
        self.model = None
        self.tokenizer = None

        # Load neural model
        self._initialize(force_mode)

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize(self, force_mode: Optional[str]) -> None:
        """Load model with automatic fallback chain"""

        if force_mode == "extractive_only":
            self.mode = InferenceMode.EXTRACTIVE_ONLY
            self._log("Running in extractive-only mode (no neural model)")
            return

        # Try ONNX
        if force_mode in (None, "onnx") and _ONNX_AVAILABLE and _TORCH_AVAILABLE:
            try:
                self._load_onnx()
                self.mode = InferenceMode.ONNX
                self._log("✓ ONNX model loaded")
                return
            except FileNotFoundError:
                self._log(
                    "ONNX model not found. "
                    "Run: python scripts/export_to_onnx.py",
                    level="warning",
                )
            except Exception as e:
                self._log(f"ONNX load failed: {e}", level="warning")

        # Try Transformers
        if force_mode in (None, "transformers") and _TRANSFORMERS_AVAILABLE:
            try:
                self._load_transformers()
                self.mode = InferenceMode.TRANSFORMERS
                self._log("✓ Transformers model loaded")
                return
            except Exception as e:
                self._log(f"Transformers load failed: {e}", level="warning")

        # Fallback
        self.mode = InferenceMode.EXTRACTIVE_ONLY
        self._log(
            "Neural model unavailable. Using extractive-only fallback.",
            level="warning",
        )

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve ONNX model path from config or default location"""
        if self.model_path and self.model_path.exists():
            return self.model_path

        cfg_path = self.config.model.local_model_path
        if cfg_path and Path(cfg_path).exists():
            return Path(cfg_path)

        # Standard search locations
        candidates = [
            self.config.models_dir / "araT5_summarizer_onnx_quantized",
            self.config.models_dir / "araT5_summarizer_onnx",
            Path("models") / "araT5_summarizer_onnx_quantized",
            Path("models") / "araT5_summarizer_onnx",
        ]

        for candidate in candidates:
            if candidate.exists() and (candidate / "config.json").exists():
                return candidate

        return None

    def _load_onnx(self) -> None:
        """Load quantized ONNX model"""
        path = self._resolve_model_path()
        if path is None:
            raise FileNotFoundError("No ONNX model found")

        self._log(f"Loading ONNX model from: {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
        )

        provider = (
            "CUDAExecutionProvider"
            if self.config.model.device == "cuda"
            else "CPUExecutionProvider"
        )

        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            str(path),
            provider=provider,
            use_io_binding=False,
            local_files_only=True,
        )

    def _load_transformers(self) -> None:
        """Load HuggingFace Transformers model"""
        model_name = self.config.model.primary_model

        # Check local path first
        local = self._resolve_model_path()
        source = str(local) if local else model_name
        local_only = local is not None

        self._log(f"Loading Transformers model: {source}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local_only,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            source,
            local_files_only=local_only,
        )

        # Move to device
        device = self.config.model.device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif device == "auto" and torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.model.eval()

    # ─── Tokenization & Generation ────────────────────────────────────────────

    def _tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize text for model input"""
        return self.tokenizer(
            text,
            max_length=self.config.generation.max_input_tokens,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )

    def _compute_gen_params(
        self,
        input_word_count: int,
        ratio: float,
        max_length: Optional[int],
    ) -> Tuple[int, int]:
        """
        Compute max_new_tokens and min_length.
        Arabic averages ~1.3 words per token for AraT5.
        """
        gen = self.config.generation

        if max_length is not None:
            return (
                max(gen.min_output_tokens, max_length),
                gen.min_output_tokens,
            )

        target_words = int(input_word_count * ratio)
        # Arabic token factor: ~1.3 words per token
        target_tokens = int(target_words * 1.3)

        max_new_tokens = max(
            gen.min_output_tokens,
            min(target_tokens, gen.max_output_tokens),
        )
        min_length = max(
            gen.min_output_tokens,
            int(max_new_tokens * 0.4),
        )

        return max_new_tokens, min_length

    def _generate(
        self,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        min_length: int,
        num_beams: int,
    ) -> str:
        """Run model.generate() - works for both ONNX and Transformers"""
        gen = self.config.generation

        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": gen.do_sample,
            "repetition_penalty": gen.repetition_penalty,
            "no_repeat_ngram_size": gen.no_repeat_ngram_size,
            "length_penalty": gen.length_penalty,
            "early_stopping": gen.early_stopping,
        }

        if gen.do_sample:
            kwargs["temperature"] = gen.temperature
            kwargs["top_p"] = gen.top_p

        # Device handling for Transformers
        if self.mode == InferenceMode.TRANSFORMERS and _TORCH_AVAILABLE:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        ).strip()

    # ─── Pipeline ─────────────────────────────────────────────────────────────

    def _pipeline(
        self,
        text: str,
        ratio: float,
        max_length: Optional[int],
        hybrid: bool,
        num_beams: int,
    ) -> str:
        """Core pipeline: preprocess → extract → generate → postprocess"""

        # 1. Preprocess
        max_chars = self.config.generation.max_input_tokens * 4
        cleaned = self.preprocessor.prepare_for_model(
            text, max_chars=max_chars
        )

        if not cleaned:
            return ""

        # 2. Split into sentences
        sentences = self.preprocessor.split_sentences(cleaned)

        # 3. Check if chunking needed
        if (
            self.config.hybrid.enable_chunking
            and self.chunker.should_chunk(cleaned, threshold_chars=3500)
        ):
            return self._pipeline_chunked(
                cleaned, sentences, ratio, max_length, hybrid, num_beams
            )

        # 4. Extractive pre-filter (Hybrid)
        input_word_count = len(cleaned.split())
        threshold = self.config.hybrid.long_text_threshold_words

        if hybrid and self.config.hybrid.enabled and input_word_count > threshold:
            try:
                pre_filtered = self.extractive.hybrid_prepare(
                    cleaned,
                    target_ratio=ratio,
                    sentences=sentences,
                    preprocessor=self.preprocessor,
                )
                if pre_filtered and len(pre_filtered.split()) >= 20:
                    cleaned = pre_filtered
                    self._log(
                        f"Hybrid: {input_word_count} → "
                        f"{len(cleaned.split())} words"
                    )
            except Exception as e:
                self._log(f"Hybrid pre-filter error: {e}", level="warning")

        # 5. Compute generation parameters
        max_new_tokens, min_length = self._compute_gen_params(
            len(cleaned.split()), ratio, max_length
        )

        # 6. Tokenize & generate
        inputs = self._tokenize(cleaned)
        raw = self._generate(inputs, max_new_tokens, min_length, num_beams)

        # 7. Postprocess
        return self.postprocessor.process(raw, self.config.postprocessing)

    def _pipeline_chunked(
        self,
        text: str,
        sentences: List[str],
        ratio: float,
        max_length: Optional[int],
        hybrid: bool,
        num_beams: int,
    ) -> str:
        """Handle long texts via chunking"""
        self._log(f"Chunking long text ({len(text.split())} words)")

        chunks = self.chunker.chunk_text(text, sentences)
        chunk_summaries = []

        for idx, chunk in enumerate(chunks, 1):
            self._log(f"  Processing chunk {idx}/{len(chunks)}")
            try:
                summary = self._pipeline(
                    chunk,
                    ratio=min(ratio * 1.5, 0.5),
                    max_length=None,
                    hybrid=False,
                    num_beams=max(2, num_beams - 1),
                )
                if summary:
                    chunk_summaries.append(summary)
            except Exception as e:
                self._log(f"  Chunk {idx} failed: {e}", level="warning")

        if not chunk_summaries:
            return self.extractive.extract(text, ratio=ratio)

        merged = self.chunker.merge_summaries(chunk_summaries)

        # Final compression pass
        if len(merged.split()) > 150:
            try:
                merged = self._pipeline(
                    merged,
                    ratio=ratio,
                    max_length=max_length,
                    hybrid=False,
                    num_beams=num_beams,
                )
            except Exception:
                pass

        return self.postprocessor.process(merged, self.config.postprocessing)

    # ─── Public API ───────────────────────────────────────────────────────────

    def summarize(
        self,
        text: str,
        ratio: Optional[float] = None,
        max_length: Optional[int] = None,
        hybrid: Optional[bool] = None,
        num_beams: Optional[int] = None,
        return_stats: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Summarize Arabic text.

        Args:
            text:         Arabic text (MSA recommended)
            ratio:        Summary length fraction 0.10-0.30 (default: 0.20)
            max_length:   Hard token limit (overrides ratio)
            hybrid:       Use extractive pre-filter (default: True)
            num_beams:    Beam search width 2-6 (default: 4)
            return_stats: Return dict with metrics

        Returns:
            str: summary text
            dict: {summary, time_seconds, input_words, output_words,
                   compression_ratio, mode, version} if return_stats=True

        Raises:
            ValueError: if text is empty
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        start = time.time()

        # Clamp ratio to valid range
        gen = self.config.generation
        ratio = max(
            gen.min_ratio,
            min(ratio or gen.default_ratio, gen.max_ratio),
        )
        hybrid = hybrid if hybrid is not None else self.config.hybrid.enabled
        num_beams = num_beams or gen.num_beams

        # Stats before processing
        input_stats = self.preprocessor.get_text_stats(text)

        # Handle very short texts
        if input_stats["word_count"] < 20:
            summary = text.strip()
            elapsed = time.time() - start
            if return_stats:
                return {
                    "summary": summary,
                    "time_seconds": round(elapsed, 2),
                    "input_words": input_stats["word_count"],
                    "output_words": len(summary.split()),
                    "compression_ratio": 1.0,
                    "mode": self.mode.value,
                    "version": self.VERSION,
                    "note": "text too short to summarize",
                }
            return summary

        # Run pipeline
        try:
            if self.mode == InferenceMode.EXTRACTIVE_ONLY:
                raw = self.extractive.extract(text, ratio=ratio)
                summary = self.postprocessor.process(
                    raw, self.config.postprocessing
                )
            else:
                summary = self._pipeline(
                    text, ratio, max_length, hybrid, num_beams
                )
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Emergency fallback
            raw = self.extractive.extract(text, ratio=ratio)
            summary = self.postprocessor.process(
                raw, self.config.postprocessing
            )

        if not summary:
            summary = text[: min(len(text), 500)]

        elapsed = time.time() - start
        self._log(f"Done in {elapsed:.2f}s [{self.mode.value}]")

        if not return_stats:
            return summary

        output_stats = self.preprocessor.get_text_stats(summary)
        return {
            "summary": summary,
            "time_seconds": round(elapsed, 2),
            "input_words": input_stats["word_count"],
            "input_sentences": input_stats["sentence_count"],
            "output_words": output_stats["word_count"],
            "output_sentences": output_stats["sentence_count"],
            "compression_ratio": round(
                output_stats["word_count"]
                / max(input_stats["word_count"], 1),
                4,
            ),
            "mode": self.mode.value,
            "model": self.config.model.primary_model,
            "version": self.VERSION,
        }

    def batch_summarize(
        self,
        texts: List[str],
        ratio: float = 0.20,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Summarize a list of texts.

        Args:
            texts:         List of Arabic texts
            ratio:         Summary ratio for all texts
            show_progress: Print progress indicator
            **kwargs:      Additional args for summarize()

        Returns:
            List of result dicts (always includes stats)
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts, 1):
            if show_progress:
                print(f"\r  ⏳ {i}/{total}", end="", flush=True)
            try:
                result = self.summarize(
                    text,
                    ratio=ratio,
                    return_stats=True,
                    **kwargs,
                )
            except Exception as e:
                result = {
                    "summary": "",
                    "error": str(e),
                    "index": i - 1,
                }
            results.append(result)

        if show_progress:
            print(f"\r  ✅ {total}/{total} texts processed")

        return results

    def warmup(self) -> float:
        """
        Warm up model with a sample inference.
        Call once after initialization for consistent timing.

        Returns:
            Warmup time in seconds
        """
        sample = (
            "الذكاء الاصطناعي هو محاكاة للذكاء البشري في الآلات. "
            "يشمل تعلم الآلة والتعلم العميق ومعالجة اللغة الطبيعية. "
            "وتتزايد تطبيقاته في مجالات الطب والتعليم والصناعة."
        )
        start = time.time()
        self.summarize(sample, ratio=0.3, num_beams=2)
        elapsed = time.time() - start
        self._log(f"Warmup completed in {elapsed:.2f}s")
        return elapsed

    def get_info(self) -> Dict[str, Any]:
        """Return system and model information"""
        import platform

        info: Dict[str, Any] = {
            "version": self.VERSION,
            "mode": self.mode.value,
            "model": self.config.model.primary_model,
            "device": self.config.model.device,
            "hybrid_enabled": self.config.hybrid.enabled,
            "max_input_tokens": self.config.generation.max_input_tokens,
            "default_ratio": self.config.generation.default_ratio,
            "python": platform.python_version(),
            "platform": platform.system(),
        }

        if _TORCH_AVAILABLE:
            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()

        if _TRANSFORMERS_AVAILABLE:
            import transformers
            info["transformers"] = transformers.__version__

        if _ONNX_AVAILABLE:
            import onnxruntime
            info["onnxruntime"] = onnxruntime.__version__

        return info

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info") -> None:
        if not self.verbose:
            return
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def __repr__(self) -> str:
        return (
            f"ArabicSummarizer("
            f"mode={self.mode.value}, "
            f"version={self.VERSION})"
        )