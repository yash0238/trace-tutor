"""
Build evidence corpus from downloaded NCERT PDFs.

Extracts content, builds bundles, and validates provenance.
"""

import argparse
import logging
from pathlib import Path
import json
import yaml

from trace_tutor.ingestion import NCERTExtractor, BundleBuilder, validate_bundles
from trace_tutor.graph import create_sample_graph


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build TRACE-Tutor evidence corpus")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/study.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory with downloaded PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bundles",
        help="Output directory for bundles"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing bundles"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    corpus_config = config.get("corpus", {})
    evidence_config = config.get("evidence", {})
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.validate_only:
        logger.info("Validation mode")
        bundle_files = list(output_dir.glob("*.jsonl"))
        
        for bundle_file in bundle_files:
            logger.info(f"Validating {bundle_file}")
            bundles = BundleBuilder.load_bundles(str(bundle_file))
            valid, errors = validate_bundles(bundles)
            
            logger.info(f"  Valid: {len(valid)}/{len(bundles)}")
            if errors:
                for error in errors[:10]:  # Show first 10 errors
                    logger.warning(f"    {error}")
        
        return
    
    # Extract and build bundles
    extractor = NCERTExtractor(output_dir="data/interim")
    builder = BundleBuilder(evidence_config)
    
    subjects = corpus_config.get("subjects", ["EVS"])
    grades = corpus_config.get("grades", [3, 4, 5])
    
    all_bundles = []
    
    for subject in subjects:
        for grade in grades:
            # Find PDF file
            pdf_pattern = f"ncert_{subject.lower()}_grade{grade}.pdf"
            pdf_files = list(input_dir.glob(pdf_pattern))
            
            if not pdf_files:
                logger.warning(f"PDF not found: {pdf_pattern}")
                continue
            
            pdf_path = pdf_files[0]
            source_id = f"ncert_{subject.lower()}_g{grade}"
            
            logger.info(f"\nProcessing: {pdf_path.name}")
            
            # Extract pages
            pages = extractor.extract_pdf(str(pdf_path), source_id)
            extractor.save_pages(pages, source_id)
            
            # Build bundles
            bundles = builder.build_from_pages(
                pages,
                source_id=source_id,
                grade=grade,
                subject=subject.upper()
            )
            
            # Save bundles
            output_file = output_dir / f"{source_id}_bundles.jsonl"
            builder.save_bundles(bundles, str(output_file))
            
            all_bundles.extend(bundles)
            
            logger.info(f"Created {len(bundles)} bundles")
    
    # Validate all bundles
    logger.info(f"\nValidating {len(all_bundles)} total bundles...")
    valid_bundles, errors = validate_bundles(all_bundles)
    
    logger.info(f"Valid bundles: {len(valid_bundles)}/{len(all_bundles)}")
    
    if errors:
        logger.warning(f"Found {len(errors)} validation errors:")
        for error in errors[:20]:
            logger.warning(f"  {error}")
    
    # Save combined corpus
    combined_file = output_dir / "corpus_all.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for bundle in valid_bundles:
            f.write(bundle.model_dump_json() + '\n')
    
    logger.info(f"\nSaved combined corpus to: {combined_file}")
    
    # Create sample prerequisite graph
    logger.info("\nCreating sample prerequisite graph...")
    graph = create_sample_graph()
    graph_file = Path("data/graph/prerequisites_sample.jsonl")
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(graph_file))
    
    logger.info(f"Saved prerequisite graph: {graph_file}")
    logger.info(f"Graph stats: {graph.get_statistics()}")
    
    # Save corpus statistics
    stats = {
        "total_bundles": len(valid_bundles),
        "by_grade": {},
        "by_subject": {},
        "by_role": {},
        "validation_errors": len(errors)
    }
    
    for bundle in valid_bundles:
        stats["by_grade"][bundle.grade] = stats["by_grade"].get(bundle.grade, 0) + 1
        stats["by_subject"][bundle.subject] = stats["by_subject"].get(bundle.subject, 0) + 1
        for role in bundle.roles:
            stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
    
    stats_file = output_dir / "corpus_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nCorpus Statistics:")
    logger.info(f"  Total bundles: {stats['total_bundles']}")
    logger.info(f"  By grade: {stats['by_grade']}")
    logger.info(f"  By subject: {stats['by_subject']}")
    logger.info(f"  By role: {stats['by_role']}")
    
    logger.info("\n✓ Corpus building complete!")
    logger.info("Next steps:")
    logger.info("  1. Review and refine evidence bundles manually")
    logger.info("  2. Create gold questions: python scripts/create_gold_questions.py")
    logger.info("  3. Build retrieval index: python scripts/build_index.py")


if __name__ == "__main__":
    main()
