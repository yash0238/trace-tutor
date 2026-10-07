"""
Run complete TRACE-Tutor experiment with reproducibility.

Executes pipeline, saves results, and generates metrics.
"""

import argparse
import logging
from pathlib import Path
import json
import yaml
from datetime import datetime
from typing import List, Dict
import hashlib

from trace_tutor.tutor import TRACETutor
from trace_tutor.schema import Query, Evidence
from trace_tutor.ingestion import BundleBuilder
from trace_tutor.evaluation import (
    RetrievalEvaluator,
    GenerationEvaluator,
    EfficiencyEvaluator
)


logger = logging.getLogger(__name__)


def load_gold_questions(filepath: str) -> List[Query]:
    """Load gold questions from JSONL."""
    questions = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                query = Query(**data)
                questions.append(query)
    
    logger.info(f"Loaded {len(questions)} gold questions")
    return questions


def compute_corpus_hash(bundles_file: str) -> str:
    """Compute hash of corpus for reproducibility."""
    with open(bundles_file, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Run TRACE-Tutor experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment configuration file"
    )
    parser.add_argument(
        "--gold-questions",
        type=str,
        default="data/gold/questions_pilot.jsonl",
        help="Gold questions file"
    )
    parser.add_argument(
        "--bundles",
        type=str,
        default="data/bundles/corpus_all.jsonl",
        help="Evidence bundles file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: runs/exp_TIMESTAMP)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions for testing"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem
        output_dir = Path(f"runs/{config_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TRACE-Tutor Experiment")
    logger.info(f"{'='*60}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save config copy
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Load gold questions
    questions = load_gold_questions(args.gold_questions)
    
    if args.limit:
        questions = questions[:args.limit]
        logger.info(f"Limited to {args.limit} questions")
    
    # Initialize TRACE-Tutor
    logger.info("Initializing TRACE-Tutor...")
    tutor = TRACETutor(
        config_path=args.config,
        bundles_path=args.bundles,
        graph_path=config.get("prerequisite", {}).get("graph_file")
    )
    
    # Process all questions
    logger.info(f"\nProcessing {len(questions)} questions...")
    
    answers = []
    rankings = []
    
    answers_file = output_dir / "answers.jsonl"
    ranks_file = output_dir / "ranks.jsonl"
    
    with open(answers_file, 'w', encoding='utf-8') as af, \
         open(ranks_file, 'w', encoding='utf-8') as rf:
        
        for idx, query in enumerate(questions, 1):
            if idx % 10 == 0:
                logger.info(f"  Processed {idx}/{len(questions)}")
            
            try:
                # Get answer
                answer = tutor.process_query(query)
                answers.append(answer)
                
                # Save answer
                af.write(answer.model_dump_json() + '\n')
                af.flush()
                
                # Get ranking (for retrieval evaluation)
                # Note: This is simplified - full implementation would capture rankings
                ranking_info = {
                    "query_id": query.id,
                    "retrieved_ids": answer.evidence_bundle_ids,
                    "num_retrieved": len(answer.evidence_bundle_ids)
                }
                rankings.append(answer.evidence_bundle_ids)
                
                rf.write(json.dumps(ranking_info) + '\n')
                rf.flush()
            
            except Exception as e:
                logger.error(f"Error processing query {query.id}: {e}")
                continue
    
    logger.info(f"\n✓ Processed {len(answers)} questions")
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    
    bundles = BundleBuilder.load_bundles(args.bundles)
    bundles_by_id = {b.id: b for b in bundles}
    
    # Generation metrics
    gen_evaluator = GenerationEvaluator()
    gen_metrics = gen_evaluator.evaluate(answers)
    
    # Efficiency metrics
    eff_evaluator = EfficiencyEvaluator()
    eff_metrics = eff_evaluator.evaluate(answers, bundles_by_id)
    
    # Save metrics
    metrics = {
        "generation": {
            "claim_support_rate": gen_metrics.claim_support_rate,
            "contradiction_rate": gen_metrics.contradiction_rate,
            "source_traceability": gen_metrics.source_traceability,
            "fallback_rate": gen_metrics.fallback_rate,
            "num_answers": gen_metrics.num_answers
        },
        "efficiency": {
            "avg_bundles_used": eff_metrics.avg_bundles_used,
            "avg_tokens_used": eff_metrics.avg_tokens_used,
            "latency_p50": eff_metrics.latency_p50,
            "latency_p95": eff_metrics.latency_p95,
            "num_queries": eff_metrics.num_queries
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create manifest
    manifest = {
        "experiment_name": config.get("project", {}).get("name", "trace-tutor"),
        "config_file": args.config,
        "timestamp": datetime.now().isoformat(),
        "corpus_file": args.bundles,
        "corpus_hash": compute_corpus_hash(args.bundles),
        "num_bundles": len(bundles),
        "num_questions": len(questions),
        "gold_questions_file": args.gold_questions,
        "seed": config.get("project", {}).get("seed", 42),
        "models": {
            "dense": config.get("retrieval", {}).get("dense", {}).get("model"),
            "reranker": config.get("retrieval", {}).get("reranker", {}).get("model"),
            "generator": config.get("generation", {}).get("model"),
            "verifier": config.get("verification", {}).get("model")
        },
        "metrics_summary": metrics
    }
    
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Experiment Results")
    logger.info(f"{'='*60}")
    logger.info(f"\nGeneration Metrics:")
    logger.info(f"  Claim Support Rate: {gen_metrics.claim_support_rate:.3f}")
    logger.info(f"  Fallback Rate: {gen_metrics.fallback_rate:.3f}")
    logger.info(f"  Source Traceability: {gen_metrics.source_traceability:.3f}")
    
    logger.info(f"\nEfficiency Metrics:")
    logger.info(f"  Avg Bundles Used: {eff_metrics.avg_bundles_used:.2f}")
    logger.info(f"  Avg Tokens Used: {eff_metrics.avg_tokens_used:.1f}")
    logger.info(f"  Latency P50: {eff_metrics.latency_p50:.3f}s")
    logger.info(f"  Latency P95: {eff_metrics.latency_p95:.3f}s")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - answers.jsonl")
    logger.info(f"  - ranks.jsonl")
    logger.info(f"  - metrics.json")
    logger.info(f"  - manifest.json")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
