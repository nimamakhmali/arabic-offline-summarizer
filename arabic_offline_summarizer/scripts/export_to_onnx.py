# scripts/export_to_onnx.py
#!/usr/bin/env python3
"""
AraT5v2 → ONNX Export + Quantization Script

Usage:
    python scripts/export_to_onnx.py
    python scripts/export_to_onnx.py --model malmarjeh/t5-arabic-text-summarization
    python scripts/export_to_onnx.py --no-quantize --output models/my_model
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Best model for Arabic summarization
DEFAULT_MODEL = "malmarjeh/t5-arabic-text-summarization"

# Alternative: base model (needs fine-tuning)
BASE_MODEL = "UBC-NLP/AraT5v2-base-1024"


def check_dependencies():
    """Check that required packages are installed"""
    missing = []
    
    try:
        import optimum
    except ImportError:
        missing.append("optimum[onnxruntime]")
    
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def export_to_onnx(model_name: str, output_dir: Path) -> Path:
    """
    Export HuggingFace model to ONNX format.
    
    Uses optimum's new API (v1.14+).
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    logger.info(f"📥 Exporting model: {model_name}")
    logger.info(f"📁 Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and export in one step (optimum handles conversion)
    logger.info("Converting to ONNX (this may take 5-15 minutes)...")
    
    # New API: use export=True
    try:
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name,
            export=True,  # New API in optimum >= 1.14
        )
    except TypeError:
        # Older API
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            model_name,
            from_transformers=True,
        )

    logger.info("Saving ONNX model...")
    ort_model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    # Check exported files
    onnx_files = list(output_dir.glob("*.onnx"))
    logger.info(f"✓ Exported {len(onnx_files)} ONNX files: {[f.name for f in onnx_files]}")

    return output_dir


def quantize_model(model_dir: Path, output_dir: Path) -> Path:
    """
    Apply INT8 dynamic quantization to ONNX model.
    
    Dynamic quantization is safest for seq2seq models.
    """
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    logger.info("⚡ Applying INT8 dynamic quantization...")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create quantizer for each ONNX component
        # T5 has: encoder_model.onnx, decoder_model.onnx, decoder_with_past_model.onnx
        onnx_files = list(model_dir.glob("*.onnx"))
        
        quantization_config = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            per_channel=False,
        )

        for onnx_file in onnx_files:
            logger.info(f"Quantizing: {onnx_file.name}")
            try:
                quantizer = ORTQuantizer.from_pretrained(
                    model_dir,
                    file_name=onnx_file.name,
                )
                quantizer.quantize(
                    save_dir=output_dir,
                    quantization_config=quantization_config,
                )
            except Exception as e:
                logger.warning(f"Failed to quantize {onnx_file.name}: {e}")
                # Copy original if quantization fails
                shutil.copy(onnx_file, output_dir / onnx_file.name)

        # Copy tokenizer and config files
        for ext in ["*.json", "*.txt", "*.model", "*.sentencepiece"]:
            for f in model_dir.glob(ext):
                if not (output_dir / f.name).exists():
                    shutil.copy(f, output_dir / f.name)

        logger.info(f"✓ Quantized model saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        logger.info("Copying unquantized model as fallback...")
        if output_dir != model_dir:
            shutil.copytree(model_dir, output_dir, dirs_exist_ok=True)

    return output_dir


def measure_model_size(model_dir: Path) -> dict:
    """Measure model file sizes"""
    onnx_files = list(model_dir.glob("*.onnx"))
    sizes = {}
    total = 0
    
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        sizes[f.name] = f"{size_mb:.1f} MB"
        total += size_mb
    
    sizes["TOTAL"] = f"{total:.1f} MB"
    return sizes


def benchmark_model(model_dir: Path, sample_text: str = None) -> dict:
    """Quick benchmark of the exported model"""
    import time
    
    if sample_text is None:
        sample_text = (
            "في السنوات الأخيرة شهد العالم تطوراً كبيراً في مجال الذكاء الاصطناعي. "
            "أصبحت تقنيات معالجة اللغة الطبيعية أكثر تقدماً وكفاءة. "
            "وتُستخدم هذه التقنيات في مجالات متعددة كالترجمة الآلية والخلاصة التلقائية "
            "والتصنيف والإجابة على الأسئلة. ويُعدّ مجال خلاصة النصوص العربية من "
            "المجالات التي تشهد اهتماماً متزايداً من الباحثين والمطورين."
        )

    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        import torch

        logger.info("Running benchmark...")
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = ORTModelForSeq2SeqLM.from_pretrained(
            str(model_dir),
            provider="CPUExecutionProvider"
        )

        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Warmup
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, num_beams=2)

        # Benchmark
        times = []
        for _ in range(3):
            start = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    num_beams=4,
                    repetition_penalty=1.3,
                )
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)

        return {
            "avg_inference_time": f"{avg_time:.2f}s",
            "min_time": f"{min(times):.2f}s",
            "max_time": f"{max(times):.2f}s",
            "sample_output_length": len(summary.split()),
            "meets_target": avg_time <= 15.0,
        }

    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Export AraT5 model to ONNX format for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output",
        default="models/araT5_summarizer_onnx",
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--quantized-output",
        default="models/araT5_summarizer_onnx_quantized",
        help="Output directory for quantized model"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization step"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after export"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export (only quantize existing model)"
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    output_dir = Path(args.output)
    quantized_dir = Path(args.quantized_output)

    print("\n" + "="*60)
    print("  AraT5 → ONNX Export Pipeline")
    print("="*60)
    print(f"  Model: {args.model}")
    print(f"  ONNX output: {output_dir}")
    print(f"  Quantized output: {quantized_dir}")
    print(f"  Quantize: {not args.no_quantize}")
    print("="*60 + "\n")

    # Step 1: Export to ONNX
    if not args.skip_export:
        export_to_onnx(args.model, output_dir)
        
        sizes = measure_model_size(output_dir)
        logger.info(f"ONNX model sizes: {sizes}")
    else:
        logger.info(f"Skipping export, using existing model at: {output_dir}")

    # Step 2: Quantize
    final_dir = output_dir
    if not args.no_quantize:
        final_dir = quantize_model(output_dir, quantized_dir)
        
        sizes = measure_model_size(final_dir)
        logger.info(f"Quantized model sizes: {sizes}")

    # Step 3: Benchmark
    if args.benchmark:
        logger.info("Running benchmark...")
        bench_results = benchmark_model(final_dir)
        
        print("\n" + "="*40)
        print("  Benchmark Results")
        print("="*40)
        for k, v in bench_results.items():
            print(f"  {k}: {v}")
        print("="*40)

    print(f"\n✅ Done! Model ready at: {final_dir}")
    print(f"\nTo use the model:")
    print(f"  from arabic_summarizer import ArabicSummarizer")
    print(f"  summarizer = ArabicSummarizer(model_path='{final_dir}')")


if __name__ == "__main__":
    main()