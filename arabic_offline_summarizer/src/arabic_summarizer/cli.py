# src/arabic_summarizer/cli.py
"""
Command Line Interface for Arabic Offline Summarizer
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s"
)


def main_summarize():
    """
    CLI entry point: aos-summarize
    Usage: aos-summarize --text "النص العربي" --ratio 0.2
    """
    parser = argparse.ArgumentParser(
        prog="aos-summarize",
        description="خلاصه‌سازی متن عربی از خط فرمان",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  aos-summarize --text "النص العربي هنا..."
  aos-summarize --file input.txt --output summary.txt --ratio 0.15
  aos-summarize --text "النص..." --mode extractive_only --ratio 0.2
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="متن عربی برای خلاصه‌سازی"
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="مسیر فایل متنی (UTF-8)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="ذخیره خلاصه در فایل (اختیاری)"
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.20,
        help="نسبت خلاصه 0.10 تا 0.30 (پیش‌فرض: 0.20)"
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "transformers", "extractive_only"],
        default=None,
        help="حالت اجرا (پیش‌فرض: خودکار)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="مسیر مدل محلی"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="نمایش آمار تفصیلی"
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="غیرفعال کردن حالت ترکیبی"
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=4,
        choices=range(2, 7),
        metavar="2-6",
        help="تعداد beam در جستجو (پیش‌فرض: 4)"
    )

    args = parser.parse_args()

    # Validate ratio
    if not (0.05 <= args.ratio <= 0.50):
        parser.error("نسبت باید بین 0.05 و 0.50 باشد")

    # Read input
    if args.file:
        if not args.file.exists():
            print(f"❌ فایل پیدا نشد: {args.file}", file=sys.stderr)
            sys.exit(1)
        try:
            text = args.file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"❌ خطا در خواندن فایل. مطمئن شوید UTF-8 است.", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    if not text or not text.strip():
        print("❌ متن خالی است.", file=sys.stderr)
        sys.exit(1)

    # Load summarizer
    try:
        # Add src to path if running directly
        src_path = Path(__file__).parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from arabic_summarizer import ArabicSummarizer

        summarizer = ArabicSummarizer(
            model_path=args.model_path,
            force_mode=args.mode,
            verbose=False,
        )
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}", file=sys.stderr)
        sys.exit(1)

    # Summarize
    try:
        result = summarizer.summarize(
            text=text,
            ratio=args.ratio,
            hybrid=not args.no_hybrid,
            num_beams=args.beams,
            return_stats=True,
        )
    except Exception as e:
        print(f"❌ خطا در خلاصه‌سازی: {e}", file=sys.stderr)
        sys.exit(1)

    summary = result["summary"]

    # Output
    if args.output:
        args.output.write_text(summary, encoding="utf-8")
        print(f"✅ خلاصه ذخیره شد: {args.output}")
    else:
        print(summary)

    # Stats
    if args.stats:
        print(f"\n--- آمار ---", file=sys.stderr)
        print(f"ورودی:    {result['input_words']} کلمه", file=sys.stderr)
        print(f"خروجی:    {result['output_words']} کلمه", file=sys.stderr)
        print(f"نسبت:     {result['compression_ratio']:.1%}", file=sys.stderr)
        print(f"زمان:     {result['time_seconds']}s", file=sys.stderr)
        print(f"حالت:     {result['mode'].upper()}", file=sys.stderr)


def main_demo():
    """
    CLI entry point: aos-demo
    Launches the Gradio web interface
    """
    import subprocess

    demo_path = Path(__file__).parent.parent.parent / "demo" / "app.py"

    if not demo_path.exists():
        print(f"❌ فایل دمو پیدا نشد: {demo_path}", file=sys.stderr)
        sys.exit(1)

    print("🚀 در حال راه‌اندازی رابط گرافیکی...")
    print("   آدرس: http://localhost:7860")
    print("   برای خروج: Ctrl+C\n")

    try:
        subprocess.run([sys.executable, str(demo_path)], check=True)
    except KeyboardInterrupt:
        print("\nبرنامه بسته شد.")
    except subprocess.CalledProcessError as e:
        print(f"❌ خطا: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main_summarize()