# src/arabic_summarizer/evaluate.py
"""
Evaluation Module for Arabic Summarization
Supports ROUGE, compression metrics, and quality scoring - fully offline
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ─── Offline ROUGE Implementation ─────────────────────────────────────────────

class OfflineROUGE:
    """
    Pure-Python ROUGE implementation.
    No external dependencies - works 100% offline.
    Implements ROUGE-1, ROUGE-2, ROUGE-L.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer for Arabic"""
        # Remove punctuation and split
        text = re.sub(r'[.،؛؟!،\-–—"\'()[\]{}]', ' ', text)
        return [w.strip() for w in text.split() if w.strip()]

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list"""
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return Counter(ngrams)

    def _rouge_n(
        self,
        hypothesis: str,
        reference: str,
        n: int,
    ) -> Dict[str, float]:
        """Compute ROUGE-N (precision, recall, f1)"""
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        hyp_ngrams = self._get_ngrams(hyp_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        # Count overlapping n-grams
        overlap = 0
        for ngram, count in hyp_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))

        precision = overlap / max(sum(hyp_ngrams.values()), 1)
        recall = overlap / max(sum(ref_ngrams.values()), 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute Longest Common Subsequence length (space-optimized)"""
        m, n = len(x), len(y)
        # Use two rows to save memory
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(curr[j-1], prev[j])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def _rouge_l(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, float]:
        """Compute ROUGE-L based on LCS"""
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        lcs = self._lcs_length(hyp_tokens, ref_tokens)

        precision = lcs / max(len(hyp_tokens), 1)
        recall = lcs / max(len(ref_tokens), 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def compute(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute all ROUGE metrics.

        Args:
            hypothesis: Generated summary
            reference: Reference (human) summary

        Returns:
            Dict with rouge1, rouge2, rougeL scores
        """
        return {
            "rouge1": self._rouge_n(hypothesis, reference, 1),
            "rouge2": self._rouge_n(hypothesis, reference, 2),
            "rougeL": self._rouge_l(hypothesis, reference),
        }

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute averaged ROUGE over multiple pairs.

        Args:
            hypotheses: List of generated summaries
            references: List of reference summaries

        Returns:
            Averaged ROUGE scores
        """
        if len(hypotheses) != len(references):
            raise ValueError("hypotheses and references must have same length")

        if not hypotheses:
            return {
                "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # Accumulate scores
        accumulated = {
            "rouge1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        }

        n = len(hypotheses)
        for hyp, ref in zip(hypotheses, references):
            scores = self.compute(hyp, ref)
            for metric, values in scores.items():
                for key, val in values.items():
                    accumulated[metric][key] += val / n

        # Round
        for metric in accumulated:
            for key in accumulated[metric]:
                accumulated[metric][key] = round(accumulated[metric][key], 4)

        return accumulated


# ─── Quality Metrics ──────────────────────────────────────────────────────────

class QualityMetrics:
    """Additional quality metrics beyond ROUGE"""

    @staticmethod
    def compression_ratio(original: str, summary: str) -> float:
        """
        Word-level compression ratio.
        1.0 = no compression, 0.0 = empty summary
        Target: 0.10 - 0.30
        """
        orig_words = len(original.split())
        sum_words = len(summary.split())
        if orig_words == 0:
            return 0.0
        return round(sum_words / orig_words, 4)

    @staticmethod
    def coverage_score(original: str, summary: str) -> float:
        """
        What fraction of original's key words appear in summary.
        Measures content coverage (recall-like).
        """
        # Remove Arabic stopwords for meaningful comparison
        stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك',
            'التي', 'الذي', 'أن', 'إن', 'كان', 'لكن', 'أو', 'ثم', 'حتى',
            'ما', 'لا', 'لم', 'لن', 'هو', 'هي', 'هم', 'أنا', 'نحن',
            'كل', 'بعض', 'و', 'ف', 'ب', 'ل',
        }

        orig_words = set(original.split()) - stopwords
        sum_words = set(summary.split()) - stopwords

        if not orig_words:
            return 0.0

        covered = len(orig_words & sum_words)
        return round(covered / len(orig_words), 4)

    @staticmethod
    def novelty_score(original: str, summary: str) -> float:
        """
        What fraction of summary words are NOT in original.
        High novelty = abstractive (good).
        Low novelty = extractive (copied).
        Target: 0.2 - 0.5 for abstractive summarization.
        """
        orig_words = set(original.split())
        sum_words = summary.split()

        if not sum_words:
            return 0.0

        novel = sum(1 for w in sum_words if w not in orig_words)
        return round(novel / len(sum_words), 4)

    @staticmethod
    def density_score(original: str, summary: str) -> float:
        """
        Average length of extracted fragments.
        Higher = more extractive style.
        Lower = more abstractive style.
        """
        orig_words = original.split()
        sum_words = summary.split()

        if not sum_words or not orig_words:
            return 0.0

        # Find consecutive matches (fragments)
        fragments = []
        i = 0
        while i < len(sum_words):
            if sum_words[i] in set(orig_words):
                frag_len = 1
                j = i + 1
                while j < len(sum_words) and sum_words[j] in set(orig_words):
                    frag_len += 1
                    j += 1
                fragments.append(frag_len)
                i = j
            else:
                i += 1

        if not fragments:
            return 0.0

        return round(sum(f**2 for f in fragments) / len(sum_words), 4)

    @staticmethod
    def fluency_score(text: str) -> float:
        """
        Heuristic fluency score for Arabic text.
        Checks: proper endings, average sentence length, no truncation artifacts.
        Returns score in [0, 1].
        """
        if not text:
            return 0.0

        score = 0.0
        checks = 0

        # Check 1: Ends with proper punctuation
        checks += 1
        if text.rstrip().endswith(('.', '؟', '!', '،', '؛')):
            score += 1.0

        # Check 2: Reasonable sentence length
        sentences = re.split(r'[.؟!]\s+', text)
        sentences = [s for s in sentences if s.strip()]
        checks += 1
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 6 <= avg_len <= 35:
                score += 1.0
            elif 4 <= avg_len <= 50:
                score += 0.5

        # Check 3: Arabic content present
        checks += 1
        arabic_ratio = len(re.findall(
            r'[\u0621-\u063A\u0641-\u064A]', text
        )) / max(len(text), 1)
        if arabic_ratio >= 0.4:
            score += 1.0
        elif arabic_ratio >= 0.2:
            score += 0.5

        # Check 4: No obvious repetition (same 4-gram repeated)
        checks += 1
        words = text.split()
        if len(words) >= 4:
            fourgrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
            fourgram_counts = Counter(fourgrams)
            max_repeat = max(fourgram_counts.values()) if fourgram_counts else 1
            if max_repeat <= 2:
                score += 1.0
            elif max_repeat <= 3:
                score += 0.5

        return round(score / checks, 4)


# ─── Main Evaluator ───────────────────────────────────────────────────────────

class ArabicSummarizationEvaluator:
    """
    Comprehensive evaluation for Arabic summarization.

    Computes:
    - ROUGE-1, ROUGE-2, ROUGE-L (offline implementation)
    - Compression ratio, Coverage, Novelty, Density, Fluency
    - Overall quality score

    Usage:
        evaluator = ArabicSummarizationEvaluator()

        # Single evaluation
        result = evaluator.evaluate_single(original, generated, reference)

        # Batch evaluation
        results = evaluator.evaluate_batch(originals, summaries, references)

        # Just ROUGE
        rouge = evaluator.compute_rouge([generated], [reference])
    """

    def __init__(self, use_library_rouge: bool = True):
        """
        Args:
            use_library_rouge: Try to use 'evaluate' library first (more accurate).
                               Falls back to offline implementation automatically.
        """
        self.offline_rouge = OfflineROUGE()
        self.quality = QualityMetrics()
        self._library_rouge = None

        if use_library_rouge:
            self._library_rouge = self._load_library_rouge()

    def _load_library_rouge(self):
        """Try to load HuggingFace evaluate library"""
        try:
            import evaluate as hf_evaluate
            rouge = hf_evaluate.load("rouge")
            logger.info("Using HuggingFace evaluate library for ROUGE")
            return rouge
        except Exception as e:
            logger.info(f"HF evaluate not available ({e}). Using offline ROUGE.")
            return None

    # ─── ROUGE ────────────────────────────────────────────────────────────────

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            predictions: List of generated summaries
            references: List of reference summaries

        Returns:
            Dict with rouge1, rouge2, rougeL (f1 scores)
        """
        if not predictions or not references:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        # Try library first
        if self._library_rouge is not None:
            try:
                result = self._library_rouge.compute(
                    predictions=predictions,
                    references=references,
                    use_stemmer=False,
                    use_aggregator=True,
                )
                return {
                    "rouge1": round(result["rouge1"], 4),
                    "rouge2": round(result["rouge2"], 4),
                    "rougeL": round(result["rougeL"], 4),
                }
            except Exception as e:
                logger.warning(f"Library ROUGE failed: {e}. Using offline.")

        # Fallback: offline implementation
        batch_scores = self.offline_rouge.compute_batch(predictions, references)
        return {
            "rouge1": batch_scores["rouge1"]["f1"],
            "rouge2": batch_scores["rouge2"]["f1"],
            "rougeL": batch_scores["rougeL"]["f1"],
        }

    # ─── Single Evaluation ────────────────────────────────────────────────────

    def evaluate_single(
        self,
        original: str,
        generated: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated summary.

        Args:
            original: Source text
            generated: Generated summary to evaluate
            reference: Human reference summary (optional, for ROUGE)

        Returns:
            Dict with all metrics
        """
        result: Dict[str, Any] = {}

        # Basic stats
        result["original_words"] = len(original.split())
        result["generated_words"] = len(generated.split())

        # Quality metrics (no reference needed)
        result["compression_ratio"] = self.quality.compression_ratio(original, generated)
        result["coverage_score"] = self.quality.coverage_score(original, generated)
        result["novelty_score"] = self.quality.novelty_score(original, generated)
        result["density_score"] = self.quality.density_score(original, generated)
        result["fluency_score"] = self.quality.fluency_score(generated)

        # Check if compression is in target range (10-30%)
        ratio = result["compression_ratio"]
        result["in_target_ratio"] = 0.08 <= ratio <= 0.32

        # ROUGE (requires reference)
        if reference:
            rouge_scores = self.compute_rouge([generated], [reference])
            result.update(rouge_scores)

            # Also compute detailed ROUGE
            detailed = self.offline_rouge.compute(generated, reference)
            result["rouge_detailed"] = {
                k: v for k, v in detailed.items()
            }

        # Overall quality score (0-100)
        result["quality_score"] = self._compute_overall_score(result)

        # Human-readable summary
        result["assessment"] = self._assess(result)

        return result

    # ─── Batch Evaluation ─────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        originals: List[str],
        generated: List[str],
        references: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of summaries.

        Args:
            originals: List of source texts
            generated: List of generated summaries
            references: List of reference summaries (optional)
            verbose: Print progress

        Returns:
            Dict with per-sample results and aggregate statistics
        """
        if len(originals) != len(generated):
            raise ValueError("originals and generated must have the same length")

        results = []
        refs = references or [None] * len(originals)

        for i, (orig, gen, ref) in enumerate(zip(originals, generated, refs)):
            if verbose:
                print(f"Evaluating {i+1}/{len(originals)}...", end="\r")

            result = self.evaluate_single(orig, gen, ref)
            result["index"] = i
            results.append(result)

        if verbose:
            print(f"✓ Evaluated {len(results)} samples")

        # Compute aggregates
        numeric_keys = [
            "compression_ratio", "coverage_score", "novelty_score",
            "density_score", "fluency_score", "quality_score",
        ]
        if references:
            numeric_keys.extend(["rouge1", "rouge2", "rougeL"])

        aggregates: Dict[str, float] = {}
        for key in numeric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregates[f"{key}_mean"] = round(sum(values) / len(values), 4)
                aggregates[f"{key}_min"] = round(min(values), 4)
                aggregates[f"{key}_max"] = round(max(values), 4)

        # In-target ratio percentage
        in_target = sum(1 for r in results if r.get("in_target_ratio", False))
        aggregates["in_target_ratio_pct"] = round(in_target / len(results) * 100, 1)

        return {
            "samples": results,
            "aggregates": aggregates,
            "total_evaluated": len(results),
        }

    # ─── Score Computation ────────────────────────────────────────────────────

    def _compute_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall quality score (0-100).

        Weighted combination of available metrics.
        """
        score = 0.0
        total_weight = 0.0

        # Fluency (always available)
        score += metrics.get("fluency_score", 0.0) * 30
        total_weight += 30

        # Coverage
        score += metrics.get("coverage_score", 0.0) * 25
        total_weight += 25

        # Compression ratio (penalize if outside 10-30%)
        ratio = metrics.get("compression_ratio", 0.0)
        if 0.10 <= ratio <= 0.30:
            ratio_score = 1.0
        elif 0.08 <= ratio <= 0.35:
            ratio_score = 0.7
        else:
            ratio_score = 0.3
        score += ratio_score * 20
        total_weight += 20

        # Novelty (abstractive quality)
        novelty = metrics.get("novelty_score", 0.0)
        novelty_score = 1.0 if 0.15 <= novelty <= 0.60 else 0.5
        score += novelty_score * 15
        total_weight += 15

        # ROUGE-L (if available)
        if "rougeL" in metrics:
            score += metrics["rougeL"] * 10
            total_weight += 10

        return round((score / total_weight) * 100, 1) if total_weight > 0 else 0.0

    def _assess(self, metrics: Dict[str, Any]) -> str:
        """Generate human-readable assessment"""
        qs = metrics.get("quality_score", 0)

        if qs >= 80:
            label = "ممتاز"
        elif qs >= 65:
            label = "جيد جداً"
        elif qs >= 50:
            label = "جيد"
        elif qs >= 35:
            label = "مقبول"
        else:
            label = "يحتاج تحسين"

        issues = []
        ratio = metrics.get("compression_ratio", 0)
        if not (0.10 <= ratio <= 0.30):
            issues.append(f"نسبة الضغط خارج النطاق ({ratio:.1%})")

        if metrics.get("coverage_score", 1) < 0.15:
            issues.append("تغطية محتوى منخفضة")

        if metrics.get("fluency_score", 1) < 0.5:
            issues.append("طلاقة منخفضة")

        assessment = f"{label} (نقاط: {qs}/100)"
        if issues:
            assessment += " | تحذيرات: " + "، ".join(issues)

        return assessment

    # ─── Benchmark ────────────────────────────────────────────────────────────

    def benchmark_summarizer(
        self,
        summarizer,
        test_cases: List[Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Full benchmark of a summarizer on test cases.

        Args:
            summarizer: ArabicSummarizer instance
            test_cases: List of {"text": str, "reference": str (optional)}
            verbose: Print results

        Returns:
            Benchmark report
        """
        import time

        originals = []
        generated_list = []
        references = []
        times = []

        for i, case in enumerate(test_cases):
            text = case["text"]
            ref = case.get("reference")

            if verbose:
                print(f"Benchmarking case {i+1}/{len(test_cases)}...", end="\r")

            start = time.time()
            try:
                result = summarizer.summarize(text, return_stats=True)
                elapsed = time.time() - start
                summary = result["summary"]
            except Exception as e:
                elapsed = time.time() - start
                summary = ""
                logger.error(f"Case {i+1} failed: {e}")

            originals.append(text)
            generated_list.append(summary)
            if ref:
                references.append(ref)
            times.append(elapsed)

        # Evaluate
        eval_refs = references if len(references) == len(originals) else None
        eval_results = self.evaluate_batch(
            originals, generated_list, eval_refs, verbose=False
        )

        # Add timing stats
        eval_results["timing"] = {
            "mean_seconds": round(sum(times) / len(times), 2),
            "min_seconds": round(min(times), 2),
            "max_seconds": round(max(times), 2),
            "meets_5s_target": sum(1 for t in times if t <= 5) / len(times),
            "meets_15s_target": sum(1 for t in times if t <= 15) / len(times),
        }

        eval_results["total_cases"] = len(test_cases)

        if verbose:
            self._print_benchmark_report(eval_results)

        return eval_results

    def _print_benchmark_report(self, report: Dict[str, Any]):
        """Print formatted benchmark report"""
        print("\n" + "="*60)
        print("  📊 گزارش ارزیابی خلاصه‌ساز عربی")
        print("="*60)

        agg = report.get("aggregates", {})

        metrics_display = [
            ("میانگین کیفیت کلی", "quality_score_mean", "/100"),
            ("نسبت فشرده‌سازی", "compression_ratio_mean", ""),
            ("پوشش محتوا", "coverage_score_mean", ""),
            ("نوآوری (انتزاعی)", "novelty_score_mean", ""),
            ("روانی متن", "fluency_score_mean", ""),
        ]

        for label, key, suffix in metrics_display:
            val = agg.get(key, "N/A")
            if isinstance(val, float):
                print(f"  {label}: {val:.4f}{suffix}")

        if "rouge1_mean" in agg:
            print(f"\n  ROUGE-1: {agg.get('rouge1_mean', 0):.4f}")
            print(f"  ROUGE-2: {agg.get('rouge2_mean', 0):.4f}")
            print(f"  ROUGE-L: {agg.get('rougeL_mean', 0):.4f}")

        timing = report.get("timing", {})
        if timing:
            print(f"\n  ⏱ میانگین زمان: {timing.get('mean_seconds')}s")
            print(f"  ⏱ حداکثر زمان: {timing.get('max_seconds')}s")
            meets_15 = timing.get('meets_15s_target', 0)
            print(f"  🎯 درصد زیر ۱۵s: {meets_15:.1%}")

        in_target = agg.get("in_target_ratio_pct", 0)
        print(f"\n  🎯 در بازه هدف (10-30%): {in_target}%")
        print("="*60 + "\n")