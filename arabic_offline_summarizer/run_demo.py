# run_demo.py
#!/usr/bin/env python3
"""
Quick Start Script for Arabic Offline Summarizer
Tests the system and provides interactive summarization

Usage:
    python run_demo.py                    # Full demo
    python run_demo.py --quick            # Quick test only
    python run_demo.py --text "النص..."  # Summarize specific text
    python run_demo.py --benchmark        # Run benchmark
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)


# ─── Test Texts ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "خبر (News)",
        "domain": "news",
        "text": (
            "أعلنت منظمة الصحة العالمية عن إطلاق مبادرة عالمية جديدة تهدف إلى "
            "تعزيز التغطية الصحية الشاملة في الدول النامية. وتشمل هذه المبادرة "
            "توفير اللقاحات الأساسية والرعاية الصحية الأولية لما يزيد على مليار "
            "شخص حول العالم. وأكد المدير العام للمنظمة أن هذه الخطوة تمثل نقلة "
            "نوعية في مسيرة تحقيق أهداف التنمية المستدامة المتعلقة بالصحة. "
            "وستعمل المنظمة بالتعاون مع الحكومات والقطاع الخاص لضمان وصول "
            "الخدمات الصحية إلى الفئات الأكثر احتياجاً في المناطق النائية."
        ),
        "expected_ratio": (0.10, 0.30),
    },
    {
        "name": "علمي (Scientific)",
        "domain": "scientific",
        "text": (
            "يُعدّ الذكاء الاصطناعي من أبرز الثورات التكنولوجية في القرن الحادي "
            "والعشرين، إذ يُحاكي القدرات المعرفية البشرية من خلال خوارزميات معقدة. "
            "ويشمل الذكاء الاصطناعي مجالات متعددة كتعلم الآلة والتعلم العميق "
            "ومعالجة اللغة الطبيعية والرؤية الحاسوبية والروبوتيات. وتتجلى "
            "تطبيقاته في مختلف القطاعات كالطب والتعليم والصناعة والنقل. "
            "ولا يزال الباحثون يسعون إلى تطوير نماذج أكثر كفاءة وأقل استهلاكاً "
            "للطاقة، مع الحرص على معالجة الإشكاليات الأخلاقية المرتبطة "
            "باستخدام هذه التقنيات كمسائل الخصوصية والتحيز الخوارزمي."
        ),
        "expected_ratio": (0.10, 0.30),
    },
    {
        "name": "وثيقة رسمية (Official Document)",
        "domain": "official",
        "text": (
            "قرار رقم 2024/أ المتعلق بتنظيم قطاع التقنية والاتصالات: "
            "استناداً إلى أحكام القانون رقم 12 لسنة 2020، وبناءً على التوصيات "
            "الفنية الصادرة عن اللجنة المتخصصة، تقرر ما يلي: "
            "أولاً: إلزام جميع مزودي خدمات الإنترنت بتوفير سرعات لا تقل عن "
            "مئة ميغابت في الثانية لجميع المشتركين في المناطق الحضرية. "
            "ثانياً: تخصيص ميزانية خاصة قدرها خمسة مليارات ريال لتطوير "
            "البنية التحتية الرقمية في المناطق الريفية خلال الفترة 2024-2026. "
            "ثالثاً: إنشاء هيئة تنظيمية مستقلة تتولى الإشراف على تطبيق هذه "
            "القرارات وضمان الامتثال لمعايير الجودة المحددة."
        ),
        "expected_ratio": (0.10, 0.30),
    },
]


def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════╗
║     📖  Arabic Offline Summarizer (AOS) v2.0            ║
║     خلاصه‌ساز هوشمند متون عربی - نسخه تجاری            ║
╠══════════════════════════════════════════════════════════╣
║  🔌 کاملاً آفلاین  |  ⚡ سریع  |  🎯 دقیق              ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_system():
    """Check system requirements"""
    print("🔍 بررسی سیستم...")

    checks = []

    # Python version
    import sys
    py_version = sys.version_info
    py_ok = py_version >= (3, 10)
    checks.append((
        f"Python {py_version.major}.{py_version.minor}",
        py_ok,
        "نیاز: 3.10+"
    ))

    # PyTorch
    try:
        import torch
        checks.append((f"PyTorch {torch.__version__}", True, ""))
    except ImportError:
        checks.append(("PyTorch", False, "pip install torch"))

    # Transformers
    try:
        import transformers
        checks.append((f"Transformers {transformers.__version__}", True, ""))
    except ImportError:
        checks.append(("Transformers", False, "pip install transformers"))

    # Optimum (optional)
    try:
        import optimum
        checks.append((f"Optimum {optimum.__version__}", True, "(ONNX support)"))
    except ImportError:
        checks.append(("Optimum", None, "(اختیاری - برای ONNX)"))

    # pyarabic
    try:
        import pyarabic
        checks.append(("pyarabic", True, ""))
    except ImportError:
        checks.append(("pyarabic", False, "pip install pyarabic"))

    # Print results
    all_ok = True
    for name, status, note in checks:
        if status is True:
            icon = "  ✅"
        elif status is False:
            icon = "  ❌"
            all_ok = False
        else:
            icon = "  ⚠️"

        note_str = f"  ({note})" if note else ""
        print(f"{icon} {name}{note_str}")

    print()
    return all_ok


def run_quick_test(summarizer) -> bool:
    """Run a quick functionality test"""
    print("⚡ تست سریع...")

    test_text = (
        "في السنوات الأخيرة شهد العالم تطوراً كبيراً في مجال الذكاء الاصطناعي. "
        "أصبحت تقنيات معالجة اللغة الطبيعية أكثر تقدماً وكفاءة من ذي قبل. "
        "وتُستخدم هذه التقنيات في مجالات متعددة كالترجمة الآلية والخلاصة التلقائية."
    )

    try:
        result = summarizer.summarize(test_text, ratio=0.3, return_stats=True)
        summary = result["summary"]
        elapsed = result["time_seconds"]
        mode = result["mode"]

        print(f"  ✅ تست موفق!")
        print(f"  ⏱ زمان: {elapsed}s")
        print(f"  🔧 حالت: {mode.upper()}")
        print(f"  📝 خروجی: {summary[:80]}...")
        print()
        return True

    except Exception as e:
        print(f"  ❌ تست ناموفق: {e}")
        return False


def run_full_demo(summarizer):
    """Run full demo on all test cases"""
    print("="*60)
    print("  🚀 اجرای دمو کامل")
    print("="*60)

    from arabic_summarizer import ArabicSummarizationEvaluator
    evaluator = ArabicSummarizationEvaluator()

    all_results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['name']}")
        print("-" * 50)
        print(f"📄 متن اصلی ({len(case['text'].split())} کلمه):")
        print(f"   {case['text'][:120]}...")

        start = time.time()
        result = summarizer.summarize(
            case["text"],
            ratio=0.20,
            return_stats=True,
        )
        elapsed = time.time() - start

        summary = result["summary"]
        input_words = result["input_words"]
        output_words = result["output_words"]
        ratio = result["compression_ratio"]

        print(f"\n✨ خلاصه ({output_words} کلمه | {ratio:.1%} از اصلی):")
        print(f"   {summary}")

        # Quick evaluation
        eval_result = evaluator.evaluate_single(case["text"], summary)
        quality = eval_result["quality_score"]
        fluency = eval_result["fluency_score"]

        print(f"\n📊 ارزیابی:")
        print(f"   کیفیت: {quality}/100 | روانی: {fluency:.2f} | زمان: {elapsed:.2f}s")

        # Check target ratio
        min_r, max_r = case["expected_ratio"]
        in_target = min_r <= ratio <= max_r
        print(f"   بازه هدف (۱۰-۳۰٪): {'✅' if in_target else '❌'}")

        all_results.append({
            "case": case["name"],
            "quality": quality,
            "ratio": ratio,
            "time": elapsed,
            "in_target": in_target,
        })

    # Summary table
    print("\n" + "="*60)
    print("  📈 جمع‌بندی نتایج")
    print("="*60)
    print(f"  {'دامنه':<25} {'کیفیت':>8} {'نسبت':>8} {'زمان':>8} {'هدف':>6}")
    print("  " + "-"*55)

    for r in all_results:
        target_icon = "✅" if r["in_target"] else "❌"
        print(
            f"  {r['case']:<25} "
            f"{r['quality']:>7.1f} "
            f"{r['ratio']:>7.1%} "
            f"{r['time']:>7.2f}s "
            f"{target_icon:>6}"
        )

    avg_quality = sum(r["quality"] for r in all_results) / len(all_results)
    avg_time = sum(r["time"] for r in all_results) / len(all_results)
    all_in_target = all(r["in_target"] for r in all_results)

    print("  " + "-"*55)
    print(f"  {'میانگین':<25} {avg_quality:>7.1f}  {'':>8} {avg_time:>7.2f}s")
    print(f"\n  🎯 همه در بازه هدف: {'✅' if all_in_target else '❌'}")
    print("="*60)


def run_benchmark(summarizer):
    """Run performance benchmark"""
    print("\n📊 در حال اجرای بنچمارک...")

    from arabic_summarizer import ArabicSummarizationEvaluator
    evaluator = ArabicSummarizationEvaluator()

    report = evaluator.benchmark_summarizer(
        summarizer=summarizer,
        test_cases=TEST_CASES,
        verbose=True,
    )

    return report


def interactive_mode(summarizer):
    """Interactive summarization loop"""
    print("\n🎤 حالت تعاملی - متن عربی وارد کنید (quit برای خروج)")
    print("-" * 50)

    while True:
        try:
            print("\nمتن (یا 'quit'): ", end="", flush=True)
            text = input().strip()

            if text.lower() in ("quit", "exit", "q", "خروج"):
                print("خداحافظ! | مع السلامة!")
                break

            if not text:
                continue

            print("نسبت (پیش‌فرض 0.2): ", end="", flush=True)
            ratio_input = input().strip()
            ratio = float(ratio_input) if ratio_input else 0.20
            ratio = max(0.10, min(0.30, ratio))

            print("\n⏳ در حال خلاصه‌سازی...")
            result = summarizer.summarize(text, ratio=ratio, return_stats=True)

            print(f"\n✨ خلاصه:")
            print(f"   {result['summary']}")
            print(f"\n📊 {result['input_words']} → {result['output_words']} کلمه | "
                  f"{result['compression_ratio']:.1%} | {result['time_seconds']}s")

        except KeyboardInterrupt:
            print("\n\nخداحافظ!")
            break
        except ValueError:
            print("نسبت نامعتبر - از 0.20 استفاده می‌شود")
        except Exception as e:
            print(f"خطا: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Arabic Offline Summarizer - Quick Demo"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test only"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Summarize specific Arabic text"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode"
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "transformers", "extractive_only"],
        default=None,
        help="Force inference mode"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip system check"
    )

    args = parser.parse_args()

    print_banner()

    # System check
    if not args.no_check:
        ok = check_system()
        if not ok:
            print("⚠️ برخی وابستگی‌ها نصب نیستند.")
            print("   اجرا: pip install -r requirements.txt\n")

    # Load summarizer
    print("🔄 بارگذاری مدل...")
    try:
        from arabic_summarizer import ArabicSummarizer
        summarizer = ArabicSummarizer(
            force_mode=args.mode,
            verbose=False,
        )
        info = summarizer.get_info()
        print(f"✅ مدل آماده | حالت: {info['mode'].upper()} | نسخه: {info['version']}\n")
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        sys.exit(1)

    # Route to appropriate action
    if args.text:
        # Single text summarization
        print(f"📄 متن ({len(args.text.split())} کلمه):")
        print(f"   {args.text}\n")
        result = summarizer.summarize(args.text, ratio=0.2, return_stats=True)
        print(f"✨ خلاصه:")
        print(f"   {result['summary']}")
        print(f"\n📊 {result['input_words']} → {result['output_words']} کلمه | "
              f"{result['compression_ratio']:.1%} | {result['time_seconds']}s")

    elif args.benchmark:
        run_benchmark(summarizer)

    elif args.interactive:
        interactive_mode(summarizer)

    elif args.quick:
        run_quick_test(summarizer)

    else:
        # Full demo
        run_quick_test(summarizer)
        run_full_demo(summarizer)

        print("\n💡 دستورات دیگر:")
        print("   python run_demo.py --interactive    (حالت تعاملی)")
        print("   python run_demo.py --benchmark      (بنچمارک)")
        print("   python demo/app.py                  (رابط گرافیکی)")


if __name__ == "__main__":
    main()