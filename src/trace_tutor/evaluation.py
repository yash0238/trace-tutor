"""
Evaluation metrics and protocols for TRACE-Tutor.

Includes retrieval metrics, generation metrics, and ablation comparison.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
import math
from collections import defaultdict

import numpy as np

from trace_tutor.schema import Evidence, Query, Answer, Role


logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    recall_at_5: float
    recall_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    role_coverage_at_3: float
    grade_fit_rate: float
    num_queries: int


@dataclass
class GenerationMetrics:
    """Generation and grounding metrics."""
    claim_support_rate: float
    contradiction_rate: float
    answer_relevance: float
    source_traceability: float
    fallback_rate: float
    num_answers: int


@dataclass
class EfficiencyMetrics:
    """Efficiency and resource metrics."""
    avg_bundles_used: float
    avg_tokens_used: float
    latency_p50: float
    latency_p95: float
    num_queries: int


def recall_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@k: fraction of relevant items in top-k.
    
    Args:
        ranked_ids: Ranked list of retrieved item IDs
        relevant_ids: Set of relevant item IDs
        k: Cut-off rank
    
    Returns:
        Recall value [0, 1]
    """
    if not relevant_ids:
        return float('nan')
    
    retrieved_relevant = set(ranked_ids[:k]) & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)


def ndcg_at_k(ranked_ids: List[str], gains: Dict[str, float], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.
    
    Args:
        ranked_ids: Ranked list of retrieved item IDs
        gains: Dict mapping item IDs to relevance gains (0-3 scale)
        k: Cut-off rank
    
    Returns:
        NDCG value [0, 1]
    """
    def dcg(ids_list: List[str], k: int) -> float:
        return sum(
            (2 ** gains.get(item_id, 0) - 1) / math.log2(idx + 2)
            for idx, item_id in enumerate(ids_list[:k])
        )
    
    if not gains:
        return float('nan')
    
    # Actual DCG
    actual_dcg = dcg(ranked_ids, k)
    
    # Ideal DCG (sort by gains)
    ideal_ranking = sorted(gains.keys(), key=lambda x: -gains[x])
    ideal_dcg = dcg(ideal_ranking, k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_score(ranked_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Mean Reciprocal Rank.
    
    Returns reciprocal rank of first relevant item.
    """
    if not relevant_ids:
        return float('nan')
    
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    
    return 0.0


def role_coverage(
    selected_bundles: List[Evidence],
    required_roles: Set[Role]
) -> float:
    """
    Role coverage: fraction of required roles present.
    
    Args:
        selected_bundles: Selected evidence bundles
        required_roles: Set of required pedagogical roles
    
    Returns:
        Coverage fraction [0, 1]
    """
    if not required_roles:
        return 1.0
    
    covered_roles = set()
    for bundle in selected_bundles:
        covered_roles.update(bundle.roles)
    
    return len(covered_roles & required_roles) / len(required_roles)


def grade_fit_rate(bundles: List[Evidence], target_grade: int) -> float:
    """
    Fraction of bundles matching target grade.
    
    Args:
        bundles: Evidence bundles
        target_grade: Target grade level
    
    Returns:
        Fraction matching grade [0, 1]
    """
    if not bundles:
        return 0.0
    
    matching = sum(1 for b in bundles if b.grade == target_grade)
    return matching / len(bundles)


class RetrievalEvaluator:
    """Evaluate retrieval quality."""
    
    def evaluate(
        self,
        queries: List[Query],
        rankings: List[List[str]],  # Retrieved bundle IDs per query
        gold_labels: Dict[str, Set[str]],  # query_id -> relevant bundle IDs
        gold_gains: Dict[str, Dict[str, float]],  # query_id -> {bundle_id: gain}
        bundles_by_id: Dict[str, Evidence]
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            queries: List of queries
            rankings: Retrieved rankings per query
            gold_labels: Gold relevant bundles per query
            gold_gains: Gold relevance gains per query
            bundles_by_id: Mapping of bundle IDs to Evidence objects
        
        Returns:
            RetrievalMetrics object
        """
        recall_5_scores = []
        recall_10_scores = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        mrr_scores = []
        role_cov_scores = []
        grade_fit_scores = []
        
        for query, ranking in zip(queries, rankings):
            relevant = gold_labels.get(query.id, set())
            gains = gold_gains.get(query.id, {})
            
            # Retrieval metrics
            r5 = recall_at_k(ranking, relevant, k=5)
            r10 = recall_at_k(ranking, relevant, k=10)
            n5 = ndcg_at_k(ranking, gains, k=5)
            n10 = ndcg_at_k(ranking, gains, k=10)
            mrr = mrr_score(ranking, relevant)
            
            if not math.isnan(r5):
                recall_5_scores.append(r5)
            if not math.isnan(r10):
                recall_10_scores.append(r10)
            if not math.isnan(n5):
                ndcg_5_scores.append(n5)
            if not math.isnan(n10):
                ndcg_10_scores.append(n10)
            if not math.isnan(mrr):
                mrr_scores.append(mrr)
            
            # Role coverage (top-3)
            top_3_bundles = [bundles_by_id[bid] for bid in ranking[:3] if bid in bundles_by_id]
            from trace_tutor.selection import get_required_roles
            required = get_required_roles(query)
            cov = role_coverage(top_3_bundles, required)
            role_cov_scores.append(cov)
            
            # Grade fit (top-3)
            gf = grade_fit_rate(top_3_bundles, query.grade)
            grade_fit_scores.append(gf)
        
        return RetrievalMetrics(
            recall_at_5=np.mean(recall_5_scores) if recall_5_scores else 0.0,
            recall_at_10=np.mean(recall_10_scores) if recall_10_scores else 0.0,
            ndcg_at_5=np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0,
            ndcg_at_10=np.mean(ndcg_10_scores) if ndcg_10_scores else 0.0,
            mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
            role_coverage_at_3=np.mean(role_cov_scores),
            grade_fit_rate=np.mean(grade_fit_scores),
            num_queries=len(queries)
        )


class GenerationEvaluator:
    """Evaluate answer generation quality."""
    
    def evaluate(
        self,
        answers: List[Answer],
        gold_answers: Optional[Dict[str, str]] = None
    ) -> GenerationMetrics:
        """
        Evaluate generation performance.
        
        Args:
            answers: Generated answers
            gold_answers: Optional gold reference answers
        
        Returns:
            GenerationMetrics object
        """
        support_scores = []
        contradiction_counts = []
        has_sources = []
        fallbacks = []
        
        for answer in answers:
            support_scores.append(answer.support_score)
            contradiction_counts.append(len(answer.contradictions))
            has_sources.append(1 if answer.evidence_bundle_ids else 0)
            fallbacks.append(1 if answer.status == "fallback" else 0)
        
        # Compute metrics
        claim_support_rate = np.mean(support_scores) if support_scores else 0.0
        contradiction_rate = np.mean([1 if c > 0 else 0 for c in contradiction_counts])
        source_traceability = np.mean(has_sources)
        fallback_rate = np.mean(fallbacks)
        
        # Answer relevance (would require human evaluation or automatic metric)
        answer_relevance = 0.0  # Placeholder
        
        return GenerationMetrics(
            claim_support_rate=claim_support_rate,
            contradiction_rate=contradiction_rate,
            answer_relevance=answer_relevance,
            source_traceability=source_traceability,
            fallback_rate=fallback_rate,
            num_answers=len(answers)
        )


class EfficiencyEvaluator:
    """Evaluate system efficiency."""
    
    def evaluate(
        self,
        answers: List[Answer],
        bundles_by_id: Dict[str, Evidence]
    ) -> EfficiencyMetrics:
        """
        Evaluate efficiency metrics.
        
        Args:
            answers: Generated answers
            bundles_by_id: Evidence bundles
        
        Returns:
            EfficiencyMetrics object
        """
        bundles_used = []
        tokens_used = []
        latencies = []
        
        for answer in answers:
            bundles_used.append(len(answer.evidence_bundle_ids))
            
            # Compute total tokens
            total_tokens = 0
            for bid in answer.evidence_bundle_ids:
                if bid in bundles_by_id:
                    total_tokens += bundles_by_id[bid].token_count()
            tokens_used.append(total_tokens)
            
            latencies.append(answer.total_time)
        
        return EfficiencyMetrics(
            avg_bundles_used=np.mean(bundles_used) if bundles_used else 0.0,
            avg_tokens_used=np.mean(tokens_used) if tokens_used else 0.0,
            latency_p50=np.percentile(latencies, 50) if latencies else 0.0,
            latency_p95=np.percentile(latencies, 95) if latencies else 0.0,
            num_queries=len(answers)
        )


def compare_systems(
    results_a: Dict,
    results_b: Dict,
    name_a: str = "System A",
    name_b: str = "System B"
) -> str:
    """
    Compare two system results and generate report.
    
    Args:
        results_a: Results dict from system A
        results_b: Results dict from system B
        name_a: Name of system A
        name_b: Name of system B
    
    Returns:
        Comparison report string
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"System Comparison: {name_a} vs {name_b}")
    lines.append(f"{'='*60}\n")
    
    # Retrieval metrics
    if "retrieval" in results_a and "retrieval" in results_b:
        lines.append("Retrieval Metrics:")
        lines.append(f"  {'Metric':<20} {name_a:>15} {name_b:>15} {'Diff':>10}")
        lines.append(f"  {'-'*60}")
        
        ret_a = results_a["retrieval"]
        ret_b = results_b["retrieval"]
        
        for key in ["recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10", "mrr"]:
            val_a = ret_a.get(key, 0.0)
            val_b = ret_b.get(key, 0.0)
            diff = val_b - val_a
            sign = "+" if diff > 0 else ""
            lines.append(f"  {key:<20} {val_a:>15.4f} {val_b:>15.4f} {sign}{diff:>9.4f}")
        
        lines.append("")
    
    # Generation metrics
    if "generation" in results_a and "generation" in results_b:
        lines.append("Generation Metrics:")
        lines.append(f"  {'Metric':<20} {name_a:>15} {name_b:>15} {'Diff':>10}")
        lines.append(f"  {'-'*60}")
        
        gen_a = results_a["generation"]
        gen_b = results_b["generation"]
        
        for key in ["claim_support_rate", "fallback_rate"]:
            val_a = gen_a.get(key, 0.0)
            val_b = gen_b.get(key, 0.0)
            diff = val_b - val_a
            sign = "+" if diff > 0 else ""
            lines.append(f"  {key:<20} {val_a:>15.4f} {val_b:>15.4f} {sign}{diff:>9.4f}")
        
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Evaluation module ready.")
    print("Includes retrieval, generation, and efficiency metrics.")
