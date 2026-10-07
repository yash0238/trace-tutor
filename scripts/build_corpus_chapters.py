"""
Build corpus from chapter-wise NCERT PDFs (Math + Science).

Handles the actual directory structure with nested folders and chapter PDFs.
"""

import argparse
import logging
from pathlib import Path
import json
import yaml
import re

from trace_tutor.ingestion import NCERTExtractor, BundleBuilder, validate_bundles
from trace_tutor.graph import create_sample_graph


logger = logging.getLogger(__name__)


# NCERT folder naming conventions
# Format: {subject}{grade}mm{lang}dd or {subject}ecu{lang}dd
# c = Class/Grade 3, d = Grade 4, e = Grade 5
# f = Grade 6, g = Grade 7, h = Grade 8
FOLDER_MAPPING = {
    'cemm1dd': {'subject': 'Mathematics', 'grade': 3},
    'demm1dd': {'subject': 'Mathematics', 'grade': 4},
    'eemm1dd': {'subject': 'Mathematics', 'grade': 5},
    'fecu1dd': {'subject': 'Science', 'grade': 6},
    'gecu1dd': {'subject': 'Science', 'grade': 7},
    'hecu1dd': {'subject': 'Science', 'grade': 8},
}


def discover_chapters(base_dir: Path) -> dict:
    """
    Discover all chapter PDFs in the directory structure.
    
    Returns:
        Dict mapping folder -> list of PDF paths
    """
    chapters = {}
    
    for folder_name, metadata in FOLDER_MAPPING.items():
        folder_path = base_dir / folder_name
        
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue
        
        # Navigate nested structure (e.g., cemm1dd/cemm1dd/)
        nested_path = folder_path / folder_name
        if nested_path.exists():
            pdf_path = nested_path
        else:
            pdf_path = folder_path
        
        # Find all chapter PDFs (excluding cover pages and preliminaries)
        pdf_files = []
        for pdf_file in pdf_path.glob("*.pdf"):
            # Skip preliminaries (ps = prelim sections)
            if 'ps' in pdf_file.stem.lower() or 'cc' in pdf_file.stem.lower():
                continue
            
            # Extract chapter number
            match = re.search(r'(\d+)', pdf_file.stem)
            if match:
                chapter_num = int(match.group(1))
                pdf_files.append((chapter_num, pdf_file))
        
        # Sort by chapter number
        pdf_files.sort()
        
        chapters[folder_name] = {
            'metadata': metadata,
            'pdfs': [pdf for _, pdf in pdf_files]
        }
        
        logger.info(f"Found {len(pdf_files)} chapters in {folder_name}: {metadata}")
    
    return chapters


def main():
    parser = argparse.ArgumentParser(description="Build corpus from chapter-wise PDFs")
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
        help="Directory with chapter PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bundles",
        help="Output directory for bundles"
    )
    parser.add_argument(
        "--max-chapters",
        type=int,
        default=None,
        help="Maximum chapters per grade (for testing)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Building TRACE-Tutor Corpus from Chapter PDFs")
    logger.info(f"{'='*60}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evidence_config = config.get("evidence", {})
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover all chapters
    logger.info("Discovering chapter PDFs...")
    chapters = discover_chapters(input_dir)
    
    if not chapters:
        logger.error("No chapters found! Check directory structure.")
        return
    
    # Print summary
    logger.info("\nDiscovered Corpus:")
    total_chapters = 0
    for folder_name, data in chapters.items():
        meta = data['metadata']
        num_chapters = len(data['pdfs'])
        total_chapters += num_chapters
        logger.info(f"  {meta['subject']} Grade {meta['grade']}: {num_chapters} chapters")
    logger.info(f"Total: {total_chapters} chapters\n")
    
    # Extract and build bundles
    extractor = NCERTExtractor(output_dir="data/interim")
    builder = BundleBuilder(evidence_config)
    
    all_bundles = []
    chapter_mapping = {}  # For tracking which pages belong to which chapters
    
    for folder_name, data in chapters.items():
        metadata = data['metadata']
        subject = metadata['subject']
        grade = metadata['grade']
        source_id = f"ncert_{subject.lower()}_g{grade}"
        
        pdfs_to_process = data['pdfs']
        if args.max_chapters:
            pdfs_to_process = pdfs_to_process[:args.max_chapters]
        
        logger.info(f"\nProcessing: {subject} Grade {grade} ({len(pdfs_to_process)} chapters)")
        
        for chapter_num, pdf_path in enumerate(pdfs_to_process, start=1):
            logger.info(f"  Chapter {chapter_num}: {pdf_path.name}")
            
            try:
                # Extract chapter
                pages = extractor.extract_pdf(str(pdf_path), f"{source_id}_ch{chapter_num}")
                
                # Build bundles with chapter context
                chapter_bundles = builder.build_from_pages(
                    pages,
                    source_id=source_id,
                    grade=grade,
                    subject=subject,
                    chapter_mapping={i: f"ch{chapter_num}" for i in range(1, len(pages) + 1)}
                )
                
                all_bundles.extend(chapter_bundles)
                logger.info(f"    Created {len(chapter_bundles)} bundles")
            
            except Exception as e:
                logger.error(f"    Failed to process {pdf_path.name}: {e}")
                continue
    
    # Validate all bundles
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating {len(all_bundles)} total bundles...")
    valid_bundles, errors = validate_bundles(all_bundles)
    
    logger.info(f"✓ Valid bundles: {len(valid_bundles)}/{len(all_bundles)}")
    
    if errors:
        logger.warning(f"Found {len(errors)} validation errors (showing first 10):")
        for error in errors[:10]:
            logger.warning(f"  {error}")
    
    # Save combined corpus
    combined_file = output_dir / "corpus_all.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for bundle in valid_bundles:
            f.write(bundle.model_dump_json() + '\n')
    
    logger.info(f"\n✓ Saved combined corpus: {combined_file}")
    
    # Create sample prerequisite graph for Math + Science
    logger.info("\nCreating prerequisite graph...")
    graph = create_enhanced_graph()
    graph_file = Path("data/graph/prerequisites.jsonl")
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph.save(str(graph_file))
    
    logger.info(f"✓ Saved graph: {graph_file}")
    logger.info(f"  Stats: {graph.get_statistics()}")
    
    # Save corpus statistics
    stats = {
        "total_bundles": len(valid_bundles),
        "by_grade": {},
        "by_subject": {},
        "by_role": {},
        "by_chapter": {},
        "validation_errors": len(errors),
        "source_chapters": total_chapters
    }
    
    for bundle in valid_bundles:
        stats["by_grade"][bundle.grade] = stats["by_grade"].get(bundle.grade, 0) + 1
        stats["by_subject"][bundle.subject] = stats["by_subject"].get(bundle.subject, 0) + 1
        stats["by_chapter"][bundle.chapter_id] = stats["by_chapter"].get(bundle.chapter_id, 0) + 1
        for role in bundle.roles:
            stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
    
    stats_file = output_dir / "corpus_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Corpus Statistics:")
    logger.info(f"  Total bundles: {stats['total_bundles']}")
    logger.info(f"  Subjects: {list(stats['by_subject'].keys())}")
    logger.info(f"  Grades: {sorted(stats['by_grade'].keys())}")
    logger.info(f"  Chapters: {len(stats['by_chapter'])}")
    logger.info(f"  Roles: {list(stats['by_role'].keys())}")
    logger.info(f"{'='*60}\n")
    
    logger.info("✅ Corpus building complete!")
    logger.info("\nNext steps:")
    logger.info("  1. python scripts/build_index.py")
    logger.info("  2. python app.py  # Launch dashboard")


def create_enhanced_graph():
    """Create prerequisite graph for Math + Science grades 3-8."""
    from trace_tutor.graph import PrerequisiteGraph
    from trace_tutor.schema import Edge
    
    graph = PrerequisiteGraph()
    
    # Mathematics prerequisites (Grade 3 → 4 → 5)
    math_edges = [
        # Grade 3 foundations
        ("numbers_place_value_g3", "large_numbers_g4", "verified", 0.9),
        ("basic_addition_g3", "multi_digit_addition_g4", "verified", 0.95),
        ("basic_multiplication_g3", "multiplication_division_g4", "verified", 0.95),
        ("shapes_2d_g3", "shapes_3d_g4", "verified", 0.8),
        ("measurement_basic_g3", "measurement_conversion_g4", "verified", 0.85),
        
        # Grade 4 → 5
        ("large_numbers_g4", "decimals_g5", "verified", 0.9),
        ("fractions_intro_g4", "fractions_operations_g5", "verified", 0.95),
        ("multiplication_division_g4", "factors_multiples_g5", "verified", 0.9),
        ("shapes_3d_g4", "volume_surface_area_g5", "verified", 0.85),
        ("measurement_conversion_g4", "metric_system_g5", "verified", 0.8),
    ]
    
    # Science prerequisites (Grade 6 → 7 → 8)
    science_edges = [
        # Grade 6 foundations
        ("food_nutrients_g6", "nutrition_digestion_g7", "verified", 0.9),
        ("materials_around_us_g6", "physical_chemical_changes_g7", "verified", 0.85),
        ("living_nonliving_g6", "cell_structure_g8", "verified", 0.9),
        ("motion_measurement_g6", "motion_force_g7", "verified", 0.9),
        ("light_shadows_g6", "light_reflection_g7", "verified", 0.85),
        
        # Grade 7 → 8
        ("nutrition_digestion_g7", "reproduction_g8", "verified", 0.8),
        ("physical_chemical_changes_g7", "chemical_reactions_g8", "verified", 0.9),
        ("motion_force_g7", "force_pressure_g8", "verified", 0.85),
        ("heat_temperature_g7", "heat_transfer_g8", "verified", 0.9),
        ("electricity_circuits_g7", "electricity_effects_g8", "verified", 0.9),
    ]
    
    all_edges = math_edges + science_edges
    
    for prereq, target, status, strength in all_edges:
        edge = Edge(
            prerequisite_lo=prereq,
            target_lo=target,
            review_status=status,
            strength=strength,
            reviewer_id="system_generated"
        )
        graph.add_edge(edge)
    
    return graph


if __name__ == "__main__":
    main()
