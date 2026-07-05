# tests/test_evaluate.py
"""Tests for evaluation module"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


ORIGINAL = (
    "الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات المبرمجة للتفكير كالبشر. "
    "يشمل تعلم الآلة والتعلم العميق ومعالجة اللغة الطبيعية. "
    "يُستخدم في الطب والتعليم والصناعة والنقل وغيرها من المجالات."
)

GENERATED = (
    "الذكاء الاصطناعي محاكاة للذكاء البشري يشمل تعلم الآلة ومعالجة اللغة "
    "ويُستخدم في مجالات متعددة."
)

REFERENCE = (
    "الذكاء الاصطناعي تقنية تحاكي الذكاء البشري وتشمل تعلم الآلة "
    "والتعلم العميق مع تطبيقات في الطب والتعليم."
)


class TestOfflineROUGE:

    @pytest.fixture
    def rouge(self):
        from arabic_summarizer.evaluate import OfflineROUGE
        return OfflineROUGE()

    def test_rouge1_perfect(self, rouge):
        """Identical text should give high ROUGE-1"""
        scores = rouge.compute(GENERATED, GENERATED)
        assert scores["rouge1"]["f1"] > 0.9

    def test_rouge1_range(self, rouge):
        """ROUGE-1 should be between 0 and 1"""
        scores = rouge.compute(GENERATED, REFERENCE)
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= scores["rouge1"][key] <= 1.0

    def test_rouge2_range(self, rouge):
        """ROUGE-2 should be between 0 and 1"""
        scores = rouge.compute(GENERATED, REFERENCE)
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= scores["rouge2"][key] <= 1.0

    def test_rougeL_range(self, rouge):
        """ROUGE-L should be between 0 and 1"""
        scores = rouge.compute(GENERATED, REFERENCE)
        for key in ["precision", "recall", "f1"]:
            assert 0.0 <= scores["rougeL"][key] <= 1.0

    def test_rouge2_le_rouge1(self, rouge):
        """ROUGE-2 should be <= ROUGE-1 (bigrams harder to match)"""
        scores = rouge.compute(GENERATED, REFERENCE)
        assert scores["rouge2"]["f1"] <= scores["rouge1"]["f1"]

    def test_rouge_empty_hypothesis(self, rouge):
        """Empty hypothesis should give 0"""
        scores = rouge.compute("", REFERENCE)
        assert scores["rouge1"]["f1"] == 0.0

    def test_batch_compute(self, rouge):
        """Batch should average correctly"""
        hypotheses = [GENERATED, GENERATED]
        references = [REFERENCE, REFERENCE]
        scores = rouge.compute_batch(hypotheses, references)
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

    def test_batch_single_vs_batch(self, rouge):
        """Single and batch should give same result for 1 pair"""
        single = rouge.compute(GENERATED, REFERENCE)
        batch = rouge.compute_batch([GENERATED], [REFERENCE])
        assert abs(single["rouge1"]["f1"] - batch["rouge1"]["f1"]) < 0.001


class TestQualityMetrics:

    @pytest.fixture
    def metrics(self):
        from arabic_summarizer.evaluate import QualityMetrics
        return QualityMetrics()

    def test_compression_ratio_range(self, metrics):
        ratio = metrics.compression_ratio(ORIGINAL, GENERATED)
        assert 0.0 <= ratio <= 1.0

    def test_compression_ratio_shorter_summary(self, metrics):
        ratio = metrics.compression_ratio(ORIGINAL, GENERATED)
        # Summary should be shorter
        assert ratio < 1.0

    def test_coverage_score_range(self, metrics):
        score = metrics.coverage_score(ORIGINAL, GENERATED)
        assert 0.0 <= score <= 1.0

    def test_novelty_score_range(self, metrics):
        score = metrics.novelty_score(ORIGINAL, GENERATED)
        assert 0.0 <= score <= 1.0

    def test_novelty_extractive_zero(self, metrics):
        """Pure copy should have low novelty"""
        score = metrics.novelty_score(ORIGINAL, ORIGINAL)
        assert score < 0.1

    def test_fluency_score_range(self, metrics):
        score = metrics.fluency_score(GENERATED)
        assert 0.0 <= score <= 1.0

    def test_fluency_empty_text(self, metrics):
        score = metrics.fluency_score("")
        assert score == 0.0

    def test_density_score_range(self, metrics):
        score = metrics.density_score(ORIGINAL, GENERATED)
        assert score >= 0.0


class TestArabicSummarizationEvaluator:

    @pytest.fixture
    def evaluator(self):
        from arabic_summarizer import ArabicSummarizationEvaluator
        return ArabicSummarizationEvaluator(use_library_rouge=False)

    def test_evaluate_single_no_reference(self, evaluator):
        result = evaluator.evaluate_single(ORIGINAL, GENERATED)
        assert "compression_ratio" in result
        assert "coverage_score" in result
        assert "novelty_score" in result
        assert "fluency_score" in result
        assert "quality_score" in result
        assert "assessment" in result

    def test_evaluate_single_with_reference(self, evaluator):
        result = evaluator.evaluate_single(ORIGINAL, GENERATED, REFERENCE)
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result

    def test_quality_score_range(self, evaluator):
        result = evaluator.evaluate_single(ORIGINAL, GENERATED)
        assert 0.0 <= result["quality_score"] <= 100.0

    def test_in_target_ratio(self, evaluator):
        # GENERATED is much shorter than ORIGINAL
        result = evaluator.evaluate_single(ORIGINAL, GENERATED)
        assert "in_target_ratio" in result
        assert isinstance(result["in_target_ratio"], bool)

    def test_assessment_string(self, evaluator):
        result = evaluator.evaluate_single(ORIGINAL, GENERATED)
        assert isinstance(result["assessment"], str)
        assert len(result["assessment"]) > 0

    def test_compute_rouge_list(self, evaluator):
        rouge = evaluator.compute_rouge([GENERATED], [REFERENCE])
        assert "rouge1" in rouge
        assert "rouge2" in rouge
        assert "rougeL" in rouge

    def test_evaluate_batch(self, evaluator):
        originals = [ORIGINAL, ORIGINAL]
        generated = [GENERATED, GENERATED]
        references = [REFERENCE, REFERENCE]

        report = evaluator.evaluate_batch(
            originals, generated, references, verbose=False
        )

        assert "samples" in report
        assert "aggregates" in report
        assert len(report["samples"]) == 2

    def test_batch_aggregates(self, evaluator):
        originals = [ORIGINAL, ORIGINAL]
        generated = [GENERATED, GENERATED]

        report = evaluator.evaluate_batch(
            originals, generated, verbose=False
        )

        agg = report["aggregates"]
        assert "quality_score_mean" in agg
        assert "compression_ratio_mean" in agg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])