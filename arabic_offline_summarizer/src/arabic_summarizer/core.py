# src/arabic_summarizer/core.py
"""
Arabic Offline Summarizer - Core Engine (v2.0)

Architecture:
  Input → Preprocess → [Extractive Filter] → Abstractive Model → Postprocess → Output

Inference modes:
  1. ONNX (preferred): Fast, quantized, CPU-optimized
  2. Transformers (fallback): Always works, needs internet on first run
"""

import logging
import time
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class InferenceMode(str, Enum):
    ONNX = "onnx"
    TRANSFORMERS = "transformers"
    EXTRACTIVE_ONLY = "extractive_only"  # Emergency fallback


class ArabicSummarizer:
    """
    Professional Arabic Text Summarizer.
    
    Usage:
        summarizer = ArabicSummarizer()
        summary = summarizer.summarize(arabic_text, ratio=0.2)
        
        # With stats
        result = summarizer.summarize(text, return_stats=True)
        print(result["summary"], result["time_seconds"])
        
        # Batch
        summaries = summarizer.batch_summarize(texts)
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        config=None,
        model_path: Optional[Union[str, Path]] = None,
        force_mode: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize summarizer.
        
        Args:
            config: SummarizerConfig instance (uses DEFAULT_CONFIG if None)
            model_path: Path to local ONNX model (auto-detected if None)
            force_mode: "onnx" | "transformers" | "extractive_only"
            verbose: Enable logging
        """
        from .config import DEFAULT_CONFIG
        from .preprocessor import ArabicPreprocessor
        from .postprocessor import ArabicPostprocessor
        from .extractive import ArabicExtractiveSummarizer
        from .chunker import TextChunker

        self.config = config or DEFAULT_CONFIG
        self.verbose = verbose

        # Sub-modules (all offline)
        self.preprocessor = ArabicPreprocessor(config=self.config.preprocessing)
        self.postprocessor = ArabicPostprocessor(config=self.config.postprocessing)
        self.extractive = ArabicExtractiveSummarizer()
        self.chunker = TextChunker(
            max_chars_per_chunk=self.config.hybrid.chunk_size_tokens * 4,  # ~chars
            overlap_chars=self.config.hybrid.chunk_overlap_tokens * 4,
        )

        # Model state
        self.mode: Optional[InferenceMode] = None
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

        # Load model
        self._initialize_model(model_path, force_mode)

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize_model(self, model_path, force_mode):
        """Initialize model with automatic fallback"""
        
        if force_mode == "extractive_only":
            self.mode = InferenceMode.EXTRACTIVE_ONLY
            self._log("✓ Running in extractive-only mode (no neural model)")
            return

        # Try ONNX first
        if force_mode in (None, "onnx"):
            try:
                self._load_onnx_model(model_path)
                self.mode = InferenceMode.ONNX
                self._model_loaded = True
                self._log(f"✓ ONNX model loaded successfully")
                return
            except Exception as e:
                self._log(f"⚠ ONNX failed: {e}", level="warning")

        # Fallback: Transformers
        if force_mode in (None, "transformers"):
            try:
                self._load_transformers_model()
                self.mode = InferenceMode.TRANSFORMERS
                self._model_loaded = True
                self._log(f"✓ Transformers model loaded successfully")
                return
            except Exception as e:
                self._log(f"⚠ Transformers failed: {e}", level="warning")

        # Emergency fallback: extractive only
        self._log(
            "⚠ Neural model unavailable. Using extractive-only mode.",
            level="warning"
        )
        self.mode = InferenceMode.EXTRACTIVE_ONLY

    def _load_onnx_model(self, model_path: Optional[Union[str, Path]]):
        """Load ONNX quantized model"""
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        # Resolve path
        if model_path is None:
            model_path = self.config.model.local_model_path
        if model_path is None:
            # Auto-detect in standard location
            model_path = (
                self.config.models_dir / "araT5_summarizer_onnx_quantized"
            )

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at: {model_path}\n"
                f"Run: python scripts/export_to_onnx.py"
            )

        self._log(f"Loading ONNX model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True  # Force offline
        )

        provider = (
            "CUDAExecutionProvider"
            if self.config.model.device == "cuda"
            else "CPUExecutionProvider"
        )

        self.model = ORTModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            provider=provider,
            use_io_binding=False,
            local_files_only=True
        )

    def _load_transformers_model(self):
        """Load HuggingFace Transformers model"""
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = self.config.model.primary_model
        self._log(f"Loading Transformers model: {model_name}")
        
        # Note: First load requires internet; subsequent loads are cached
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype_map = {
            "float16": "torch.float16",
            "bfloat16": "torch.bfloat16",
            "float32": None,
        }
        torch_kwargs = {}
        if self.config.model.torch_dtype == "float16" and torch.cuda.is_available():
            import torch
            torch_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            **torch_kwargs
        )

        device = self.config.model.device
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        elif device == "auto":
            import torch
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

        self.model.eval()

    # ─── Inference ────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> Dict:
        """Tokenize text for model input"""
        return self.tokenizer(
            text,
            max_length=self.config.generation.max_input_tokens,
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

    def _compute_max_new_tokens(
        self,
        input_word_count: int,
        ratio: float,
        max_length: Optional[int]
    ) -> Tuple[int, int]:
        """
        Compute max_new_tokens and min_length for generation.
        
        Arabic tokens average ~1.2 words per token.
        """
        if max_length is not None:
            return max_length, self.config.generation.min_output_tokens

        # Target word count
        target_words = int(input_word_count * ratio)
        # Convert to tokens (Arabic: ~1.2 words/token)
        target_tokens = int(target_words * 1.2)
        
        max_new_tokens = max(
            self.config.generation.min_output_tokens,
            min(target_tokens, self.config.generation.max_output_tokens)
        )
        min_length = max(
            self.config.generation.min_output_tokens,
            int(max_new_tokens * 0.5)
        )

        return max_new_tokens, min_length

    def _generate(
        self,
        inputs: Dict,
        max_new_tokens: int,
        min_length: int,
        num_beams: int,
    ) -> str:
        """
        Run model generation (works for both ONNX and Transformers).
        """
        import torch
        
        gen_config = self.config.generation
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "num_beams": num_beams,
            "do_sample": gen_config.do_sample,
            "repetition_penalty": gen_config.repetition_penalty,
            "no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
            "length_penalty": gen_config.length_penalty,
            "early_stopping": gen_config.early_stopping,
        }

        if gen_config.do_sample:
            generation_kwargs["temperature"] = gen_config.temperature
            generation_kwargs["top_p"] = gen_config.top_p

        # Move to device if needed (Transformers mode only)
        if self.mode == InferenceMode.TRANSFORMERS:
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ─── Pipeline ─────────────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        text: str,
        ratio: float,
        max_length: Optional[int],
        hybrid: bool,
        num_beams: int,
    ) -> str:
        """
        Core summarization pipeline.
        
        Steps:
        1. Preprocess (normalize + clean)
        2. [Optional] Extractive pre-filter for long texts
        3. Tokenize
        4. Generate summary
        5. Postprocess
        """

        # ── Step 1: Preprocess ─────────────────────────────────────────────
        cleaned = self.preprocessor.prepare_for_model(
            text,
            max_chars=self.config.generation.max_input_tokens * 4,  # ~chars
        )

        # ── Step 2: Check if chunking needed ───────────────────────────────
        sentences = self.preprocessor.split_sentences(cleaned)
        
        if (
            self.config.hybrid.enable_chunking
            and self.chunker.should_chunk(cleaned, threshold_chars=3500)
        ):
            return self._chunked_summarize(
                cleaned, sentences, ratio, max_length, hybrid, num_beams
            )

        # ── Step 3: Extractive pre-filter (Hybrid mode) ────────────────────
        input_words = len(cleaned.split())
        
        if (
            hybrid
            and self.config.hybrid.enabled
            and input_words > self.config.hybrid.long_text_threshold_words
        ):
            try:
                cleaned = self.extractive.hybrid_prepare(
                    cleaned,
                    target_ratio=ratio,
                    sentences=sentences,
                    preprocessor=self.preprocessor,
                )
                self._log(
                    f"Extractive pre-filter: {input_words} → {len(cleaned.split())} words"
                )
            except Exception as e:
                self._log(f"Extractive pre-filter failed: {e}", level="warning")

        # ── Step 4: Compute generation params ─────────────────────────────
        input_word_count = len(cleaned.split())
        max_new_tokens, min_length = self._compute_max_new_tokens(
            input_word_count, ratio, max_length
        )

        # ── Step 5: Generate ───────────────────────────────────────────────
        inputs = self._tokenize(cleaned)
        raw_summary = self._generate(inputs, max_new_tokens, min_length, num_beams)

        # ── Step 6: Postprocess ────────────────────────────────────────────
        summary = self.postprocessor.process(raw_summary, self.config.postprocessing)

        return summary

    def _chunked_summarize(
        self,
        text: str,
        sentences: List[str],
        ratio: float,
        max_length: Optional[int],
        hybrid: bool,
        num_beams: int,
    ) -> str:
        """Handle very long texts through chunked processing"""
        self._log(f"Long text detected. Using chunked summarization.")
        
        chunks = self.chunker.chunk_text(text, sentences)
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            self._log(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                # Each chunk: use higher ratio (we'll merge and compress again)
                chunk_summary = self._run_pipeline(
                    chunk,
                    ratio=ratio * 1.5,  # More content per chunk
                    max_length=None,
                    hybrid=False,  # Already chunked, no need for extractive
                    num_beams=max(2, num_beams - 1),  # Faster for chunks
                )
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
            except Exception as e:
                self._log(f"Chunk {i+1} failed: {e}", level="warning")

        if not chunk_summaries:
            # Fallback to extractive
            return self.extractive.extract(text, ratio=ratio)

        # Merge chunk summaries
        merged = self.chunker.merge_summaries(chunk_summaries)
        
        # Final pass: summarize the merged summaries
        if len(merged.split()) > 200:
            try:
                final = self._run_pipeline(
                    merged,
                    ratio=ratio,
                    max_length=max_length,
                    hybrid=False,
                    num_beams=num_beams,
                )
                return final
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
            text: Arabic text to summarize (MSA preferred)
            ratio: Summary length as fraction of original (0.10-0.30)
            max_length: Hard limit on output tokens (overrides ratio)
            hybrid: Use extractive pre-filtering (default: True)
            num_beams: Beam search width (2-6, higher = better but slower)
            return_stats: Return dict with stats instead of just string
            
        Returns:
            str: Summary text (if return_stats=False)
            dict: {"summary", "time_seconds", "input_words", 
                   "output_words", "compression_ratio", "mode"} (if return_stats=True)
            
        Raises:
            ValueError: If text is empty or too short
        """
        start_time = time.time()

        # Validate input
        if not text or not text.strip():
            raise ValueError("Input text is empty")

        if len(text.split()) < 20:
            # Text too short to summarize - return as-is
            summary = text.strip()
            if return_stats:
                return {
                    "summary": summary,
                    "time_seconds": 0.0,
                    "input_words": len(text.split()),
                    "output_words": len(summary.split()),
                    "compression_ratio": 1.0,
                    "mode": self.mode.value,
                    "note": "text too short to summarize"
                }
            return summary

        # Resolve parameters
        ratio = max(
            self.config.generation.min_ratio,
            min(ratio or self.config.generation.default_ratio,
                self.config.generation.max_ratio)
        )
        hybrid = hybrid if hybrid is not None else self.config.hybrid.enabled
        num_beams = num_beams or self.config.generation.num_beams

        # Get input stats before processing
        input_stats = self.preprocessor.get_text_stats(text)

        # ── Run pipeline ───────────────────────────────────────────────────
        try:
            if self.mode == InferenceMode.EXTRACTIVE_ONLY:
                # No neural model: pure extractive
                summary = self.extractive.extract(text, ratio=ratio)
                summary = self.postprocessor.process(
                    summary, self.config.postprocessing
                )
            else:
                summary = self._run_pipeline(
                    text, ratio, max_length, hybrid, num_beams
                )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Emergency fallback
            summary = self.extractive.extract(text, ratio=ratio)
            summary = self.postprocessor.process(
                summary, self.config.postprocessing
            )

        elapsed = time.time() - start_time
        self._log(f"Summarized in {elapsed:.2f}s ({self.mode.value} mode)")

        if not return_stats:
            return summary

        # Compute stats
        output_stats = self.preprocessor.get_text_stats(summary)
        compression = round(
            output_stats["word_count"] / max(input_stats["word_count"], 1), 3
        )

        return {
            "summary": summary,
            "time_seconds": round(elapsed, 2),
            "input_words": input_stats["word_count"],
            "input_sentences": input_stats["sentence_count"],
            "output_words": output_stats["word_count"],
            "output_sentences": output_stats["sentence_count"],
            "compression_ratio": compression,
            "mode": self.mode.value,
            "model": self.config.model.primary_model,
            "version": self.VERSION,
        }

    def batch_summarize(
        self,
        texts: List[str],
        ratio: float = 0.20,
        show_progress: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Summarize multiple texts.
        
        Args:
            texts: List of Arabic texts
            ratio: Summary ratio (applied to all)
            show_progress: Print progress
            **kwargs: Additional args passed to summarize()
            
        Returns:
            List of result dicts (always returns stats)
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts, 1):
            if show_progress:
                print(f"Processing {i}/{total}...", end="\r")
            try:
                result = self.summarize(text, ratio=ratio, return_stats=True, **kwargs)
            except Exception as e:
                result = {
                    "summary": "",
                    "error": str(e),
                    "input_index": i - 1
                }
            results.append(result)

        if show_progress:
            print(f"✓ Processed {total} texts")

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        try:
            import torch
            torch_version = torch.__version__
        except:
            torch_version = "not installed"

        return {
            "version": self.VERSION,
            "mode": self.mode.value if self.mode else "not initialized",
            "model": self.config.model.primary_model,
            "device": self.config.model.device,
            "max_input_tokens": self.config.generation.max_input_tokens,
            "hybrid_enabled": self.config.hybrid.enabled,
            "python_version": platform.python_version(),
            "torch_version": torch_version,
        }

    def warmup(self, text: str = None) -> float:
        """
        Warm up the model with a sample inference.
        Returns warmup time in seconds.
        """
        if text is None:
            text = (
                "في عام ألفين وعشرين، شهد العالم تحولات كبيرة في مجال "
                "التكنولوجيا والذكاء الاصطناعي. وقد أثرت هذه التحولات "
                "على جميع مجالات الحياة الاقتصادية والاجتماعية والعلمية."
            )
        
        start = time.time()
        self.summarize(text, ratio=0.3, num_beams=2)
        elapsed = time.time() - start
        self._log(f"Warmup completed in {elapsed:.2f}s")
        return elapsed

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info"):
        if not self.verbose:
            return
        if level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)