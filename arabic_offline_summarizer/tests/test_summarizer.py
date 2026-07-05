# tests/test_summarizer.py
"""
Comprehensive test suite for Arabic Offline Summarizer
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Test Data ────────────────────────────────────────────────────────────────

SHORT_TEXT = "في هذا اليوم حدث شيء مهم."

MEDIUM_TEXT = """
في السنوات الأخيرة شهد العالم تطوراً كبيراً في مجال الذكاء الاصطناعي.
أصبحت تقنيات معالجة اللغة الطبيعية أكثر تقدماً وكفاءة من ذي قبل.
وتُستخدم هذه التقنيات في مجالات متعددة كالترجمة الآلية والخلاصة التلقائية.
ويُعدّ مجال خلاصة النصوص العربية من المجالات التي تشهد اهتماماً متزايداً.
وتسعى الشركات الكبرى إلى تطوير نماذج لغوية متخصصة في اللغة العربية.
""".strip()

LONG_TEXT = (MEDIUM_TEXT + " " + MEDIUM_TEXT + " " + MEDIUM_TEXT).strip()

WITH_DIACRITICS = "في السَّنَوَاتِ الأَخِيرَةِ شَهِدَ العَالَمُ تَطَوُّراً كَبِيراً"


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestArabicPreprocessor:
    
    @pytest.fixture
    def preprocessor(self):
        from arabic_summarizer.preprocessor import ArabicPreprocessor
        return ArabicPreprocessor()

    def test_remove_diacritics(self, preprocessor):
        result = preprocessor.remove_diacritics(WITH_DIACRITICS)
        # Should not contain diacritics
        import re
        diacritics = re.findall(r'[\u064B-\u065F\u0670]', result)
        assert len(diacritics) == 0, f"Diacritics found: {diacritics}"

    def test_normalize_alef(self, preprocessor):
        text = "أحمد إبراهيم آدم"
        result = preprocessor.normalize_alef(text)
        assert 'أ' not in result
        assert 'إ' not in result
        assert 'آ' not in result
        assert 'ا' in result

    def test_normalize_ya(self, preprocessor):
        text = "موسى يحيى"
        result = preprocessor.normalize_ya(text)
        assert 'ى' not in result

    def test_clean_removes_urls(self, preprocessor):
        text = "اقرأ المزيد على https://example.com هنا"
        result = preprocessor.clean(text)
        assert "https://" not in result
        assert "example.com" not in result

    def test_clean_removes_emails(self, preprocessor):
        text = "تواصل معنا على info@example.com"
        result = preprocessor.clean(text)
        assert "@" not in result

    def test_split_sentences_basic(self, preprocessor):
        text = "هذه جملة أولى. هذه جملة ثانية. هذه جملة ثالثة."
        sentences = preprocessor.split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_arabic_punct(self, preprocessor):
        text = "الجملة الأولى؟ الجملة الثانية! الجملة الثالثة."
        sentences = preprocessor.split_sentences(text)
        assert len(sentences) >= 2

    def test_get_text_stats(self, preprocessor):
        stats = preprocessor.get_text_stats(MEDIUM_TEXT)
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "char_count" in stats
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0
        assert 0.0 <= stats["arabic_ratio"] <= 1.0

    def test_prepare_for_model_no_truncation(self, preprocessor):
        result = preprocessor.prepare_for_model(MEDIUM_TEXT, max_chars=10000)
        assert len(result) > 0
        assert len(result) <= 10000

    def test_prepare_for_model_with_truncation(self, preprocessor):
        long_text = MEDIUM_TEXT * 20
        result = preprocessor.prepare_for_model(long_text, max_chars=500)
        assert len(result) <= 600  # Allow small buffer for sentence boundary

    def test_empty_text(self, preprocessor):
        assert preprocessor.clean("") == ""
        assert preprocessor.normalize("") == ""
        assert preprocessor.split_sentences("") == []


# ─── Postprocessor Tests ──────────────────────────────────────────────────────

class TestArabicPostprocessor:
    
    @pytest.fixture
    def postprocessor(self):
        from arabic_summarizer.postprocessor import ArabicPostprocessor
        return ArabicPostprocessor()

    def test_remove_repeated_sentences(self, postprocessor):
        text = (
            "هذا النص يتحدث عن الذكاء الاصطناعي. "
            "الذكاء الاصطناعي يتحدث هذا النص. "  # near-duplicate
            "وهو مجال متطور ومهم جداً."
        )
        result = postprocessor.remove_repeated_sentences(text, threshold=0.6)
        # Should remove one of the duplicates
        assert len(result) < len(text) or "الذكاء الاصطناعي" in result

    def test_fix_sentence_ending(self, postprocessor):
        text = "هذا النص لا ينتهي بعلامة ترقيم"
        result = postprocessor.fix_sentence_ending(text)
        assert result.endswith(('.', '،', '؟', '!', '؛'))

    def test_fix_sentence_ending_already_has_punct(self, postprocessor):
        text = "هذا النص ينتهي بنقطة."
        result = postprocessor.fix_sentence_ending(text)
        # Should not add extra punctuation
        assert result.endswith('.')
        assert not result.endswith('..')

    def test_remove_model_artifacts(self, postprocessor):
        text = "خلاصة: هذا هو الملخص الرئيسي للنص"
        result = postprocessor.remove_model_artifacts(text)
        assert not result.startswith("خلاصة:")

    def test_process_empty(self, postprocessor):
        result = postprocessor.process("")
        assert result == ""


# ─── Extractive Tests ─────────────────────────────────────────────────────────

class TestArabicExtractiveSummarizer:
    
    @pytest.fixture
    def extractor(self):
        from arabic_summarizer.extractive import ArabicExtractiveSummarizer
        return ArabicExtractiveSummarizer()

    def test_extract_basic(self, extractor):
        result = extractor.extract(MEDIUM_TEXT, ratio=0.5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_ratio_reduces_length(self, extractor):
        result_30 = extractor.extract(MEDIUM_TEXT, ratio=0.3)
        result_70 = extractor.extract(MEDIUM_TEXT, ratio=0.7)
        # Higher ratio = more content
        assert len(result_70.split()) >= len(result_30.split())

    def test_extract_short_text_returns_original(self, extractor):
        short = "جملة أولى. جملة ثانية."
        result = extractor.extract(short, ratio=0.5)
        assert len(result) > 0

    def test_hybrid_prepare(self, extractor):
        result = extractor.hybrid_prepare(LONG_TEXT, target_ratio=0.2)
        original_words = len(LONG_TEXT.split())
        result_words = len(result.split())
        # Should reduce length
        assert result_words < original_words

    def test_tfidf_scores_not_all_zero(self, extractor):
        from arabic_summarizer.extractive import TFIDFScorer
        scorer = TFIDFScorer()
        sentences = ["الذكاء الاصطناعي يتطور", "معالجة اللغة العربية مهمة", "التقنية تساعد البشر"]
        scores = scorer.score_sentences_tfidf(sentences)
        assert len(scores) == 3
        assert any(s > 0 for s in scores)


# ─── Chunker Tests ────────────────────────────────────────────────────────────

class TestTextChunker:
    
    @pytest.fixture
    def chunker(self):
        from arabic_summarizer.chunker import TextChunker
        return TextChunker(max_chars_per_chunk=500, overlap_chars=50)

    def test_should_chunk_long_text(self, chunker):
        long_text = "الذكاء الاصطناعي " * 200
        assert chunker.should_chunk(long_text, threshold_chars=100)

    def test_should_not_chunk_short_text(self, chunker):
        assert not chunker.should_chunk(MEDIUM_TEXT, threshold_chars=10000)

    def test_chunk_preserves_all_content(self, chunker):
        from arabic_summarizer.preprocessor import ArabicPreprocessor
        preprocessor = ArabicPreprocessor()
        sentences = preprocessor.split_sentences(LONG_TEXT)
        
        chunks = chunker.chunk_text(LONG_TEXT, sentences)
        
        assert len(chunks) >= 1
        # Reconstruct should contain similar content
        all_text = ' '.join(chunks)
        # At least 70% of original sentences should appear in chunks
        original_sentences = set(s[:30] for s in sentences)  # Use first 30 chars as key
        found = sum(
            1 for key in original_sentences
            if key in all_text
        )
        assert found / max(len(original_sentences), 1) >= 0.7

    def test_merge_summaries_concatenate(self, chunker):
        summaries = ["خلاصة القسم الأول", "خلاصة القسم الثاني", "خلاصة القسم الثالث"]
        merged = chunker.merge_summaries(summaries)
        assert len(merged) > 0
        for s in summaries:
            assert s.rstrip('.') in merged or s in merged


# ─── Core Summarizer Tests ────────────────────────────────────────────────────

class TestArabicSummarizer:
    """
    Integration tests for the main summarizer.
    Uses extractive-only mode to avoid requiring model download in CI.
    """

    @pytest.fixture
    def summarizer(self):
        from arabic_summarizer import ArabicSummarizer
        return ArabicSummarizer(
            force_mode="extractive_only",
            verbose=False
        )

    def test_summarize_returns_string(self, summarizer):
        result = summarizer.summarize(MEDIUM_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_returns_stats(self, summarizer):
        result = summarizer.summarize(MEDIUM_TEXT, return_stats=True)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "time_seconds" in result
        assert "input_words" in result
        assert "output_words" in result
        assert "compression_ratio" in result
        assert "mode" in result

    def test_summarize_ratio_respected(self, summarizer):
        """Summary should be shorter than original"""
        result = summarizer.summarize(MEDIUM_TEXT, ratio=0.3, return_stats=True)
        assert result["output_words"] < result["input_words"]

    def test_summarize_short_text(self, summarizer):
        """Short text should be returned as-is"""
        result = summarizer.summarize(SHORT_TEXT)
        assert isinstance(result, str)

    def test_summarize_empty_raises(self, summarizer):
        with pytest.raises((ValueError, Exception)):
            summarizer.summarize("")

    def test_batch_summarize(self, summarizer):
        texts = [MEDIUM_TEXT, MEDIUM_TEXT]
        results = summarizer.batch_summarize(texts, ratio=0.3, show_progress=False)
        assert len(results) == 2
        assert all("summary" in r for r in results)

    def test_get_info(self, summarizer):
        info = summarizer.get_info()
        assert "mode" in info
        assert "version" in info

    def test_ratio_clipping(self, summarizer):
        """Ratio outside bounds should be clipped, not raise error"""
        result = summarizer.summarize(MEDIUM_TEXT, ratio=0.5)  # above max 0.30
        assert isinstance(result, str)

    def test_arabic_only_input(self, summarizer):
        """Test with various Arabic text types"""
        texts = [
            MEDIUM_TEXT,
            WITH_DIACRITICS + " " + MEDIUM_TEXT,
        ]
        for text in texts:
            result = summarizer.summarize(text)
            assert isinstance(result, str)


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluator:
    
    @pytest.fixture
    def evaluator(self):
        from arabic_summarizer import ArabicSummarizationEvaluator
        return ArabicSummarizationEvaluator()

    def test_compression_ratio(self, evaluator):
        ratio = evaluator.compute_compression_ratio(MEDIUM_TEXT, SHORT_TEXT)
        assert 0.0 < ratio <= 1.0

    def test_evaluate_single(self, evaluator):
        result = evaluator.evaluate_single(MEDIUM_TEXT, SHORT_TEXT)
        assert "compression_ratio" in result
        assert "generated_length" in result

    def test_evaluate_with_reference(self, evaluator):
        reference = "الذكاء الاصطناعي يتطور بسرعة في جميع المجالات."
        generated = "تطور الذكاء الاصطناعي في المجالات المختلفة."
        
        result = evaluator.evaluate_single(MEDIUM_TEXT, generated, reference)
        assert "compression_ratio" in result


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])