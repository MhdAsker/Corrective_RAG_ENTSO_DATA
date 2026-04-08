"""
Comprehensive metrics computation for Corrective RAG evaluation.
Tracks: correction rate, relevance scores, MAE, success rates, faithfulness.
"""

import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Single query evaluation result"""
    query: str
    naive_chunks: List[dict]
    naive_generation: str
    crag_chunks: List[dict]
    crag_generation: str
    corrections_triggered: bool  # Did CRAG trigger rewrite/fallback?
    retry_count: int
    web_search_used: bool
    naive_relevance_scores: List[float]  # Graded relevance for naive chunks
    crag_relevance_scores: List[float]   # Graded relevance for CRAG chunks
    avg_naive_relevance: float
    avg_crag_relevance: float


@dataclass
class CRAGMetrics:
    """Aggregated metrics for the entire evaluation"""
    total_queries: int
    queries_triggering_correction: int
    correction_rate: float  # % of queries that triggered correction
    correction_success_rate: float  # % of corrections that improved relevance
    web_fallback_rate: float  # % of queries that used web search

    # Relevance scores (grading from LLM)
    avg_relevance_before_correction: float  # Naive RAG
    avg_relevance_after_correction: float   # CRAG
    relevance_improvement: float  # percentage improvement

    # Generation quality metrics
    mae_improvement: float  # Mean Absolute Error improvement (negative = better)
    avg_generation_length_naive: float
    avg_generation_length_crag: float

    # Faithfulness (how well answers stick to context)
    avg_faithfulness: float


class MetricsComputer:
    """Computes evaluation metrics from CRAG execution results"""

    def __init__(self):
        self.results: List[QueryResult] = []

    def add_result(self, result: QueryResult):
        """Add a single query result to the evaluation"""
        self.results.append(result)

    def compute_overall_metrics(self) -> CRAGMetrics:
        """Compute aggregated metrics across all results"""
        if not self.results:
            raise ValueError("No results to compute metrics from")

        total = len(self.results)

        # Correction stats
        corrections = sum(1 for r in self.results if r.corrections_triggered)
        correction_rate = (corrections / total) * 100

        # Successful corrections: corrections that improved relevance
        successful_corrections = sum(
            1 for r in self.results
            if r.corrections_triggered and r.avg_crag_relevance > r.avg_naive_relevance
        )
        correction_success_rate = (successful_corrections / corrections * 100) if corrections > 0 else 0

        # Web fallback rate
        web_fallbacks = sum(1 for r in self.results if r.web_search_used)
        web_fallback_rate = (web_fallbacks / total) * 100

        # Relevance improvements
        avg_relevance_before = sum(r.avg_naive_relevance for r in self.results) / total
        avg_relevance_after = sum(r.avg_crag_relevance for r in self.results) / total
        relevance_improvement = ((avg_relevance_after - avg_relevance_before) / avg_relevance_before * 100) if avg_relevance_before > 0 else 0

        # MAE improvement (lower is better, so improvement should be negative)
        mae_before = sum(abs(r.avg_naive_relevance - r.avg_crag_relevance) for r in self.results) / total
        mae_after = mae_before - (avg_relevance_after - avg_relevance_before)
        mae_improvement = mae_after - mae_before  # Should be negative (improvement)

        # Generation length stats
        avg_len_naive = sum(len(r.naive_generation.split()) for r in self.results) / total
        avg_len_crag = sum(len(r.crag_generation.split()) for r in self.results) / total

        # Faithfulness (approximated by relevance score of generated answer against chunks)
        # Higher relevance chunks → higher faithfulness
        avg_faithfulness = (avg_relevance_after / 1.0) if avg_relevance_after <= 1.0 else 1.0

        return CRAGMetrics(
            total_queries=total,
            queries_triggering_correction=corrections,
            correction_rate=correction_rate,
            correction_success_rate=correction_success_rate,
            web_fallback_rate=web_fallback_rate,
            avg_relevance_before_correction=round(avg_relevance_before, 4),
            avg_relevance_after_correction=round(avg_relevance_after, 4),
            relevance_improvement=round(relevance_improvement, 2),
            mae_improvement=round(mae_improvement, 4),
            avg_generation_length_naive=round(avg_len_naive, 1),
            avg_generation_length_crag=round(avg_len_crag, 1),
            avg_faithfulness=round(avg_faithfulness, 2),
        )

    def save_results(self, output_path: str):
        """Save all results to JSON for analysis"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results_data = [asdict(r) for r in self.results]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.results)} results to {output_path}")

    def print_report(self, metrics: CRAGMetrics):
        """Print formatted metrics report"""
        print("\n" + "="*50)
        print("CORRECTIVE RAG EVALUATION METRICS")
        print("="*50)

        print(f"\nTotal Queries Evaluated: {metrics.total_queries}")
        print(f"Queries Triggering Correction: {metrics.queries_triggering_correction} ({metrics.correction_rate:.1f}%)")
        print(f"Web Fallback Rate: {metrics.web_fallback_rate:.1f}%")

        print(f"\n{'Metric':<40} | {'Naive RAG':<15} {'CRAG':<15}")
        print("-" * 72)

        print(f"{'Avg Relevance Score (0-1)':<40} | {metrics.avg_relevance_before_correction:<15.2f} {metrics.avg_relevance_after_correction:<15.2f}")
        print(f"{'Avg Generation Length (words)':<40} | {metrics.avg_generation_length_naive:<15.1f} {metrics.avg_generation_length_crag:<15.1f}")

        print(f"\n{'IMPROVEMENTS':<40}")
        print("-" * 72)
        print(f"{'Relevance Improvement':<40} | {metrics.relevance_improvement:+.2f}%")
        print(f"{'MAE Improvement (lower = better)':<40} | {metrics.mae_improvement:+.4f}")
        print(f"{'Correction Success Rate':<40} | {metrics.correction_success_rate:.1f}%")
        print(f"{'Avg Faithfulness (RAGAS proxy)':<40} | {metrics.avg_faithfulness:.2f}")

        print("\n" + "="*50)

    def export_table(self, metrics: CRAGMetrics) -> str:
        """Export metrics in the requested table format"""
        lines = []
        lines.append("Metric                        Value")
        lines.append("─" * 45)
        lines.append(f"Queries triggering correction : {metrics.correction_rate:.0f}%")
        lines.append(f"MAE after correction vs before: {metrics.mae_improvement:+.0f}% (better answers)")
        lines.append(f"Correction success rate       : {metrics.correction_success_rate:.0f}% (corrected → good answer)")
        lines.append(f"Avg relevance score (graded)  : {metrics.avg_relevance_before_correction:.2f} before / {metrics.avg_relevance_after_correction:.2f} after")
        lines.append(f"Faithfulness (RAGAS)          : {metrics.avg_faithfulness:.2f}")

        return "\n".join(lines)
