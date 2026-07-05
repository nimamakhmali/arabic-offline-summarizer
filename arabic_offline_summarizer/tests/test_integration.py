# tests/test_integration.py
"""
Integration Tests - Full Pipeline
Tests the complete system end-to-end in extractive_only mode
(no neural model required → runs in CI without GPU/downloads)
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def summarizer():
    """Single summarizer instance for all integration tests"""
    from arabic_summarizer import ArabicSummarizer
    return ArabicSummarizer(force_mode="extractive_only", verbose=False)


@pytest.fixture(scope="module")
def evaluator():
    from arabic_summarizer import ArabicSummarizationEvaluator
    return ArabicSummarizationEvaluator(use_library_rouge=False)


# ─── Test Data ────────────────────────────────────────────────────────────────

NEWS_TEXT = (
    "أعلنت منظمة الصحة العالمية عن إطلاق مبادرة عالمية جديدة تهدف إلى "
    "تعزيز التغطية الصحية الشاملة في الدول النامية. وتشمل هذه المبادرة "
    "توفير اللقاحات الأساسية والرعاية الصحية الأولية لما يزيد على مليار "
    "شخص حول العالم. وأكد المدير العام للمنظمة أن هذه الخطوة تمثل نقلة "
    "نوعية في مسيرة تحقيق أهداف التنمية المستدامة المتعلقة بالصحة. "
    "وستعمل المنظمة بالتعاون مع الحكومات والقطاع الخاص ومنظمات المجتمع "
    "المدني لضمان وصول الخدمات الصحية إلى الفئات الأكثر احتياجاً."
)

SCIENCE_TEXT = (
    "يُعدّ الذكاء الاصطناعي من أبرز الثورات التكنولوجية في القرن الحادي "
    "والعشرين، إذ يُحاكي القدرات المعرفية البشرية من خلال خوارزميات معقدة. "
    "ويشمل الذكاء الاصطناعي مجالات متعددة كتعلم الآلة والتعلم العميق "
    "ومعالجة اللغة الطبيعية والرؤية الحاسوبية والروبوتيات. "
    "وتتجلى تطبيقاته في مختلف القطاعات كالطب والتعليم والصناعة والنقل."
)

LONG_TEXT = (NEWS_TEXT + " " + SCIENCE_TEXT) * 3

DIACRITICS_TEXT = (
    "في السَّنَوَاتِ الأَخِيرَةِ شَهِدَ العَالَمُ تَطَوُّراً كَبِيراً "
    "فِي مَجَالِ الذَّكَاءِ الاصطِنَاعِيِّ وَمُعَالَجَةِ اللُّغَةِ الطَّبِيعِيَّةِ. "
    "وَقَدْ أَصْبَحَتْ هَذِهِ التَّقْنِيَاتُ أَكْثَرَ تَقَدُّماً وَكَفَاءَةً مِنْ ذِي قَبْلُ. "
    "وَتُسْتَخْدَمُ فِي مَجَالاتٍ مُتَعَدِّدَةٍ كَالتَّرْجَمَةِ الآلِيَّةِ وَالخُلاصَةِ التِّلْقَائِيَّةِ."
)


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestEndToEnd:
    """Full pipeline integration tests"""

    def test_news_summarization(self, summarizer):
        result = summarizer.summarize(NEWS_TEXT, ratio=0.25, return_stats=True)

        assert isinstance(result["summary"], str)
        assert len(result["summary"].strip()) > 0
        assert result["input_words"] > 0
        assert result["output_words"] > 0
        assert result["output_words"] < result["input_words"]
        assert result["mode"] == "extractive_only"

    def test_science_summarization(self, summarizer):
        result = summarizer.summarize(SCIENCE_TEXT, ratio=0.25, return_stats=True)
        assert isinstance(result["summary"], str)
        assert result["compression_ratio"] <= 1.0

    def test_diacritics_handled(self, summarizer):
        """Text with tashkeel should be handled correctly"""
        result = summarizer.summarize(
            DIACRITICS_TEXT, ratio=0.3, return_stats=True
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_ratio_10_percent(self, summarizer):
        result = summarizer.summarize(
            NEWS_TEXT * 3, ratio=0.10, return_stats=True
        )
        assert result["compression_ratio"] <= 0.50  # At least some compression

    def test_ratio_30_percent(self, summarizer):
        result = summarizer.summarize(NEWS_TEXT, ratio=0.30, return_stats=True)
        assert isinstance(result["summary"], str)

    def test_long_text_chunking(self, summarizer):
        """Long text should be handled without error"""
        result = summarizer.summarize(LONG_TEXT, ratio=0.15, return_stats=True)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_batch_summarize(self, summarizer):
        texts = [NEWS_TEXT, SCIENCE_TEXT, DIACRITICS_TEXT]
        results = summarizer.batch_summarize(
            texts, ratio=0.25, show_progress=False
        )

        assert len(results) == 3
        for result in results:
            assert "summary" in result
            # No errors
            assert "error" not in result

    def test_return_stats_structure(self, summarizer):
        result = summarizer.summarize(NEWS_TEXT, return_stats=True)
        required_keys = [
            "summary", "time_seconds", "input_words",
            "output_words", "compression_ratio", "mode", "version",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_performance_under_15s(self, summarizer):
        """Must complete within 15 seconds (project requirement)"""
        start = time.time()
        summarizer.summarize(NEWS_TEXT, ratio=0.20)
        elapsed = time.time() - start
        assert elapsed <= 15.0, f"Too slow: {elapsed:.2f}s (max: 15s)"

    def test_output_arabic_content(self, summarizer):
        """Output should contain Arabic characters"""
        import re
        summary = summarizer.summarize(NEWS_TEXT, ratio=0.25)
        arabic_chars = re.findall(r"[\u0621-\u063A\u0641-\u064A]", summary)
        assert len(arabic_chars) > 0, "Summary contains no Arabic text"

    def test_output_ends_with_punctuation(self, summarizer):
        """Output should end with proper punctuation"""
        summary = summarizer.summarize(NEWS_TEXT, ratio=0.20)
        valid = (".", "،", "؛", "؟", "!")
        assert summary.rstrip().endswith(valid), (
            f"Summary doesn't end with punctuation: '{summary[-10:]}'"
        )

    def test_empty_text_raises(self, summarizer):
        with pytest.raises((ValueError, Exception)):
            summarizer.summarize("")

    def test_none_text_raises(self, summarizer):
        with pytest.raises((ValueError, Exception, AttributeError)):
            summarizer.summarize(None)

    def test_get_info_structure(self, summarizer):
        info = summarizer.get_info()
        assert "version" in info
        assert "mode" in info
        assert "model" in info
        assert info["version"] == "2.0.0"

    def test_repr(self, summarizer):
        text = repr(summarizer)
        assert "ArabicSummarizer" in text
        assert "extractive_only" in text


class TestDomainSpecific:
    """Tests for specific use cases mentioned in project requirements"""

    def test_quran_text(self, summarizer):
        """Quran/religious text should be handled"""
        quran_text = (
            "قال الله تعالى في كتابه الكريم: إن الله يأمر بالعدل والإحسان "
            "وإيتاء ذي القربى وينهى عن الفحشاء والمنكر والبغي يعظكم لعلكم "
            "تذكرون. وقد أكد علماء التفسير أن هذه الآية الكريمة تجمع في "
            "ألفاظ قليلة أسس الشريعة الإسلامية كلها. فالعدل هو إعطاء كل "
            "ذي حق حقه. والإحسان هو التفضل على الناس بما يزيد على الواجب."
        )
        result = summarizer.summarize(quran_text, ratio=0.20, return_stats=True)
        assert result["output_words"] > 0
        assert result["compression_ratio"] < 1.0

    def test_official_document(self, summarizer):
        """Official document text"""
        doc = (
            "قرار رقم 2024/أ المتعلق بتنظيم قطاع التقنية والاتصالات: "
            "استناداً إلى أحكام القانون رقم 12 لسنة 2020، وبناءً على "
            "التوصيات الفنية الصادرة عن اللجنة المتخصصة، تقرر ما يلي: "
            "أولاً: إلزام جميع مزودي خدمات الإنترنت بتوفير سرعات لا تقل "
            "عن مئة ميغابت في الثانية لجميع المشتركين في المناطق الحضرية. "
            "ثانياً: تخصيص ميزانية خاصة قدرها خمسة مليارات ريال لتطوير "
            "البنية التحتية الرقمية في المناطق الريفية خلال الفترة 2024-2026."
        )
        result = summarizer.summarize(doc, ratio=0.20, return_stats=True)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_news_article(self, summarizer):
        result = summarizer.summarize(NEWS_TEXT, ratio=0.20, return_stats=True)
        # Should produce reasonable compression
        assert 0.05 <= result["compression_ratio"] <= 0.80

    def test_scientific_paper(self, summarizer):
        result = summarizer.summarize(
            SCIENCE_TEXT, ratio=0.20, return_stats=True
        )
        assert isinstance(result["summary"], str)


class TestEvaluationIntegration:
    """Integration tests for evaluator"""

    def test_full_evaluation_pipeline(self, summarizer, evaluator):
        """Test complete evaluation pipeline"""
        result = summarizer.summarize(NEWS_TEXT, ratio=0.20, return_stats=True)
        summary = result["summary"]

        eval_result = evaluator.evaluate_single(
            original=NEWS_TEXT,
            generated=summary,
        )

        assert "quality_score" in eval_result
        assert "compression_ratio" in eval_result
        assert "fluency_score" in eval_result
        assert 0.0 <= eval_result["quality_score"] <= 100.0
        assert 0.0 <= eval_result["compression_ratio"] <= 1.0

    def test_benchmark_pipeline(self, summarizer, evaluator):
        """Test benchmark produces valid report"""
        test_cases = [
            {"text": NEWS_TEXT},
            {"text": SCIENCE_TEXT},
        ]

        report = evaluator.benchmark_summarizer(
            summarizer=summarizer,
            test_cases=test_cases,
            verbose=False,
        )

        assert "aggregates" in report
        assert "timing" in report
        assert report["total_cases"] == 2
        assert report["timing"]["mean_seconds"] <= 15.0

    def test_rouge_scores_valid(self, evaluator):
        """ROUGE scores should be in valid range"""
        reference = "الذكاء الاصطناعي يتطور بسرعة في مجالات متعددة."
        generated = "تطور الذكاء الاصطناعي في المجالات المختلفة بسرعة كبيرة."

        scores = evaluator.compute_rouge([generated], [reference])

        for metric in ["rouge1", "rouge2", "rougeL"]:
            assert metric in scores
            assert 0.0 <= scores[metric] <= 1.0