"""
Generate all paper-ready visualizations and tables.

Processes experiment results and creates publication-quality figures.
"""

import argparse
import logging
from pathlib import Path
import json

from trace_tutor.visualization import generate_paper_results, ResultVisualizer, TableGenerator


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready results")
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        required=True,
        help="Experiment result directories to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper",
        help="Output directory for paper materials"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats for figures"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("TRACE-Tutor Paper Results Generator")
    logger.info(f"{'='*60}\n")
    
    # Verify experiment directories exist
    valid_dirs = []
    for exp_dir in args.experiment_dirs:
        exp_path = Path(exp_dir)
        if exp_path.exists():
            valid_dirs.append(str(exp_path))
            logger.info(f"Found: {exp_path}")
        else:
            logger.warning(f"Not found: {exp_path}")
    
    if not valid_dirs:
        logger.error("No valid experiment directories found!")
        return
    
    # Generate results
    generate_paper_results(valid_dirs, args.output_dir)
    
    # Create summary report
    output_path = Path(args.output_dir)
    summary_file = output_path / "RESULTS_SUMMARY.md"
    
    with open(summary_file, 'w') as f:
        f.write("# TRACE-Tutor Experiment Results Summary\n\n")
        f.write(f"Generated from {len(valid_dirs)} experiment(s)\n\n")
        
        f.write("## Experiments\n\n")
        for exp_dir in valid_dirs:
            exp_path = Path(exp_dir)
            manifest_file = exp_path / "manifest.json"
            
            if manifest_file.exists():
                with open(manifest_file, 'r') as mf:
                    manifest = json.load(mf)
                    
                    f.write(f"### {manifest.get('experiment_name', exp_path.name)}\n\n")
                    f.write(f"- **Timestamp**: {manifest.get('timestamp', 'unknown')}\n")
                    f.write(f"- **Questions**: {manifest.get('num_questions', 0)}\n")
                    f.write(f"- **Bundles**: {manifest.get('num_bundles', 0)}\n")
                    f.write(f"- **Models**:\n")
                    for model_type, model_name in manifest.get('models', {}).items():
                        f.write(f"  - {model_type}: {model_name}\n")
                    f.write("\n")
        
        f.write("## Generated Materials\n\n")
        f.write("### Figures\n\n")
        f.write("- `figures/ablation_comparison.pdf` - Ablation study bar charts\n")
        f.write("- `figures/metrics_heatmap.pdf` - Performance heatmap across systems\n")
        f.write("- `figures/latency_distribution.pdf` - Latency analysis\n")
        f.write("- `figures/retrieval_curve.pdf` - Retrieval performance curves\n\n")
        
        f.write("### Tables\n\n")
        f.write("- `tables/ablation_results.tex` - LaTeX ablation table\n")
        f.write("- `tables/comparison_table.tex` - Baseline vs TRACE-Tutor comparison\n")
        f.write("- `tables/summary_stats.tex` - Summary statistics\n\n")
        
        f.write("## Usage in LaTeX\n\n")
        f.write("```latex\n")
        f.write("% Include figure\n")
        f.write("\\begin{figure}[t]\n")
        f.write("\\centering\n")
        f.write("\\includegraphics[width=0.8\\linewidth]{figures/ablation_comparison.pdf}\n")
        f.write("\\caption{Ablation study results.}\n")
        f.write("\\label{fig:ablation}\n")
        f.write("\\end{figure}\n\n")
        
        f.write("% Include table\n")
        f.write("\\input{tables/ablation_results.tex}\n")
        f.write("```\n")
    
    logger.info(f"\nSummary report: {summary_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Paper results generation complete!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
