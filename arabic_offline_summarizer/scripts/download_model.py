# scripts/download_model.py
#!/usr/bin/env python3
"""
Model Downloader for Arabic Offline Summarizer
Downloads and caches model for offline use.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --model malmarjeh/t5-arabic-text-summarization
    python scripts/download_model.py --list
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Available Models ─────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "araT5-finetuned": {
        "id": "malmarjeh/t5-arabic-text-summarization",
        "description": "AraT5 fine-tuned on Arabic summarization (recommended)",
        "size_gb": 0.9,
        "quality": "★★★★★",
        "speed": "★★★★☆",
        "recommended": True,
    },
    "araT5v2-base": {
        "id": "UBC-NLP/AraT5v2-base-1024",
        "description": "AraT5v2 base model (needs fine-tuning for best results)",
        "size_gb": 0.9,
        "quality": "★★★★☆",
        "speed": "★★★★☆",
        "recommended": False,
    },
    "mt5-multilingual": {
        "id": "csebuetnlp/mT5_multilingual_XLSum",
        "description": "Multilingual mT5 (supports Arabic + 44 other languages)",
        "size_gb": 1.2,
        "quality": "★★★☆☆",
        "speed": "★★★☆☆",
        "recommended": False,
    },
}

DEFAULT_MODEL_KEY = "araT5-finetuned"


def list_models():
    """Display available models"""
    print("\n" + "="*65)
    print("  📦 مدل‌های موجود برای خلاصه‌سازی عربی")
    print("="*65)

    for key, info in AVAILABLE_MODELS.items():
        marker = " ← پیشنهادی" if info["recommended"] else ""
        print(f"\n  [{key}]{marker}")
        print(f"    ID:          {info['id']}")
        print(f"    توضیح:       {info['description']}")
        print(f"    حجم:         ~{info['size_gb']} GB")
        print(f"    کیفیت:       {info['quality']}")
        print(f"    سرعت:        {info['speed']}")

    print("\n" + "="*65)
    print(f"  استفاده: python scripts/download_model.py --model-key araT5-finetuned")
    print("="*65 + "\n")


def download_model(
    model_id: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """
    Download model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier
        output_dir: Directory to save model
        force: Re-download even if exists

    Returns:
        Path to downloaded model
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    output_dir = Path(output_dir)

    # Check if already downloaded
    if output_dir.exists() and not force:
        config_file = output_dir / "config.json"
        if config_file.exists():
            logger.info(f"✓ Model already exists at: {output_dir}")
            logger.info("Use --force to re-download")
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading model: {model_id}")
    logger.info(f"Destination: {output_dir}")
    logger.info("This may take several minutes depending on your connection...")

    start_time = time.time()

    # Download tokenizer
    logger.info("Step 1/2: Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(output_dir / ".cache"),
        )
        tokenizer.save_pretrained(str(output_dir))
        logger.info("✓ Tokenizer downloaded")
    except Exception as e:
        logger.error(f"Tokenizer download failed: {e}")
        raise

    # Download model weights
    logger.info("Step 2/2: Downloading model weights...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            cache_dir=str(output_dir / ".cache"),
        )
        model.save_pretrained(str(output_dir))
        logger.info("✓ Model weights downloaded")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise

    # Cleanup cache
    cache_dir = output_dir / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    elapsed = time.time() - start_time

    # Save metadata
    metadata = {
        "model_id": model_id,
        "download_time_seconds": round(elapsed, 1),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "format": "transformers",
    }
    with open(output_dir / "aos_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Measure size
    total_size = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    ) / (1024 ** 3)

    logger.info(f"✓ Download complete!")
    logger.info(f"  Total size: {total_size:.2f} GB")
    logger.info(f"  Time: {elapsed:.0f}s")
    logger.info(f"  Location: {output_dir}")

    return output_dir


def verify_model(model_dir: Path) -> bool:
    """
    Verify downloaded model works correctly.
    Runs a quick inference test.
    """
    logger.info(f"Verifying model at: {model_dir}")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        model.eval()

        # Quick inference test
        test_text = (
            "الذكاء الاصطناعي هو محاكاة للذكاء البشري في الآلات. "
            "يُستخدم في مجالات عديدة منها معالجة اللغة الطبيعية والرؤية الحاسوبية."
        )

        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=2,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded and len(decoded.strip()) > 0:
            logger.info(f"✓ Model verification passed")
            logger.info(f"  Test output: {decoded[:80]}...")
            return True
        else:
            logger.error("Model produced empty output")
            return False

    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False


def create_model_info_file(model_dir: Path, model_id: str):
    """Create a human-readable model info file"""
    info_path = model_dir / "MODEL_INFO.txt"

    content = f"""Arabic Offline Summarizer - Model Information
============================================

Model ID:     {model_id}
Format:       HuggingFace Transformers (PyTorch)
Location:     {model_dir.absolute()}
Downloaded:   {time.strftime("%Y-%m-%d %H:%M:%S")}

Usage:
    from arabic_summarizer import ArabicSummarizer
    summarizer = ArabicSummarizer(
        model_path="{model_dir.absolute()}"
    )
    summary = summarizer.summarize(arabic_text)

For ONNX optimization (faster inference):
    python scripts/export_to_onnx.py --input "{model_dir.absolute()}"
"""

    with open(info_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Model info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Arabic Summarization Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_model.py
  python scripts/download_model.py --model-key araT5-finetuned
  python scripts/download_model.py --model-id UBC-NLP/AraT5v2-base-1024
  python scripts/download_model.py --list
  python scripts/download_model.py --verify-only
        """,
    )

    parser.add_argument(
        "--model-key",
        default=DEFAULT_MODEL_KEY,
        choices=list(AVAILABLE_MODELS.keys()),
        help=f"Model preset key (default: {DEFAULT_MODEL_KEY})",
    )
    parser.add_argument(
        "--model-id",
        help="Custom HuggingFace model ID (overrides --model-key)",
    )
    parser.add_argument(
        "--output",
        default="models/transformers_model",
        help="Output directory (default: models/transformers_model)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip model verification after download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing model (no download)",
    )

    args = parser.parse_args()

    # List models
    if args.list:
        list_models()
        return

    # Resolve model ID
    if args.model_id:
        model_id = args.model_id
    else:
        model_info = AVAILABLE_MODELS[args.model_key]
        model_id = model_info["id"]
        logger.info(f"Selected model: {args.model_key} ({model_id})")

    output_dir = Path(args.output)

    # Verify only
    if args.verify_only:
        success = verify_model(output_dir)
        sys.exit(0 if success else 1)

    # Download
    try:
        download_model(model_id, output_dir, force=args.force)
        create_model_info_file(output_dir, model_id)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # Verify
    if not args.no_verify:
        success = verify_model(output_dir)
        if not success:
            logger.warning("Verification failed - model may not work correctly")
            sys.exit(1)

    print(f"\n✅ مدل آماده است!")
    print(f"   مسیر: {output_dir.absolute()}")
    print(f"\n   اجرای دمو: python demo/app.py")
    print(f"   تبدیل به ONNX: python scripts/export_to_onnx.py\n")


if __name__ == "__main__":
    main()