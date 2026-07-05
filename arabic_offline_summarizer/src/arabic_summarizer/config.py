# src/arabic_summarizer/config.py
"""
Configuration Management for Arabic Offline Summarizer
Supports environment variables and YAML config files
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os
import json


@dataclass
class ModelConfig:
    """Model-specific configuration"""
    
    # Primary model (best for Arabic summarization)
    primary_model: str = "UBC-NLP/AraT5v2-base-1024"
    
    # Fine-tuned alternatives (better accuracy, less control)
    finetuned_models: List[str] = field(default_factory=lambda: [
        "malmarjeh/t5-arabic-text-summarization",
        "csebuetnlp/mT5_multilingual_XLSum",
    ])
    
    # Local model path (for offline/ONNX mode)
    local_model_path: Optional[Path] = None
    
    # Device
    device: str = "cpu"  # "cpu" | "cuda" | "auto"
    
    # Torch dtype
    torch_dtype: str = "float32"  # "float32" | "float16" | "bfloat16"


@dataclass  
class GenerationConfig:
    """Text generation parameters"""
    
    # Length control
    max_input_tokens: int = 1024
    max_output_tokens: int = 400
    min_output_tokens: int = 30
    
    # Default summary ratio (10-30%)
    default_ratio: float = 0.20
    min_ratio: float = 0.10
    max_ratio: float = 0.30
    
    # Beam search
    num_beams: int = 4
    early_stopping: bool = True
    
    # Sampling (disabled by default for consistency)
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    
    # Quality control
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0


@dataclass
class PreprocessingConfig:
    """Arabic text preprocessing settings"""
    
    remove_diacritics: bool = True      # حذف تشکیل
    normalize_alef: bool = True          # نرمال‌سازی الف
    normalize_ya: bool = True            # نرمال‌سازی یاء
    normalize_teh: bool = True           # نرمال‌سازی تاء مربوطه
    remove_tatweel: bool = True          # حذف کشیده
    remove_urls: bool = True             # حذف لینک‌ها
    remove_emails: bool = True           # حذف ایمیل‌ها
    normalize_whitespace: bool = True    # نرمال‌سازی فاصله
    
    # Sentence splitting
    min_sentence_length: int = 5        # حداقل طول جمله (کلمه)
    max_sentence_length: int = 100      # حداکثر طول جمله (کلمه)


@dataclass
class PostprocessingConfig:
    """Output post-processing settings"""
    
    # Repetition removal (improved algorithm)
    remove_repeated_sentences: bool = True
    sentence_similarity_threshold: float = 0.7
    
    # Punctuation
    ensure_ending_punctuation: bool = True
    fix_spacing: bool = True
    
    # Quality filters
    min_output_words: int = 10
    
    
@dataclass
class HybridConfig:
    """Hybrid summarization settings"""
    
    enabled: bool = True
    
    # Threshold: texts longer than this use extractive pre-selection
    long_text_threshold_words: int = 300
    
    # How much to keep in extractive phase
    extractive_keep_ratio: float = 0.60
    
    # Chunking for very long texts
    enable_chunking: bool = True
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100


@dataclass
class SummarizerConfig:
    """Master configuration - combines all sub-configs"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    
    # Runtime
    verbose: bool = True
    cache_model: bool = True
    
    # Paths
    models_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "models"
    )
    
    def to_dict(self) -> dict:
        """Serialize config to dict"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SummarizerConfig":
        """Load config from dict"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_json(cls, path: Path) -> "SummarizerConfig":
        """Load config from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, path: Path):
        """Save config to JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)


# Singleton default config
DEFAULT_CONFIG = SummarizerConfig()


# Preset configs for different use cases
FAST_CONFIG = SummarizerConfig()
FAST_CONFIG.generation.num_beams = 2
FAST_CONFIG.generation.max_output_tokens = 250
FAST_CONFIG.hybrid.enabled = True

QUALITY_CONFIG = SummarizerConfig()
QUALITY_CONFIG.generation.num_beams = 6
QUALITY_CONFIG.generation.repetition_penalty = 1.4
QUALITY_CONFIG.generation.length_penalty = 1.2

QURAN_CONFIG = SummarizerConfig()
QURAN_CONFIG.preprocessing.remove_diacritics = False  # حفظ تشکیل برای قرآن
QURAN_CONFIG.generation.repetition_penalty = 1.5
QURAN_CONFIG.generation.default_ratio = 0.15