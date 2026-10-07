"""
Visualization and paper-ready result generation.

Creates publication-quality figures, tables, and LaTeX snippets from experimental results.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

logger = logging.getLogger(__name__)

# Set publication-ready style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


class ResultVisualizer:
    """Generate publication-ready visualizations from experiment results."""
    
    def __init__(self, output_dir: str = "paper/figures"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer output: {self.output_dir}")
    
    def plot_ablation_comparison(
        self,
        results: Dict[str, Dict],
        metrics: List[str],
        output_file: str = "ablation_comparison.pdf"
    ):
        """
        Create ablation study comparison chart.
        
        Args:
            results: Dict mapping system names to metric dicts
            metrics: List of metric names to plot
            output_file: Output filename
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
        
        if len(metrics) == 1:
            axes = [axes]
        
        system_names = list(results.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [results[sys].get(metric, 0.0) for sys in system_names]
            
            # Create bar chart
            bars = ax.bar(range(len(system_names)), values, color='steelblue', alpha=0.8)
            
            # Highlight best
            best_idx = np.argmax(values)
            bars[best_idx].set_color('darkgreen')
            bars[best_idx].set_alpha(1.0)
            
            ax.set_xticks(range(len(system_names)))
            ax.set_xticklabels(system_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, min(1.1, max(values) * 1.2))
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ablation comparison: {output_path}")
        return str(output_path)
    
    def plot_metric_heatmap(
        self,
        results: Dict[str, Dict],
        output_file: str = "metrics_heatmap.pdf"
    ):
        """
        Create heatmap of all metrics across systems.
        
        Args:
            results: Dict mapping system names to metric dicts
            output_file: Output filename
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Normalize each column to [0, 1] for visualization
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_norm = df_norm.fillna(0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            df_norm,
            annot=df,  # Show actual values
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Systems')
        ax.set_title('System Performance Across Metrics')
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics heatmap: {output_path}")
        return str(output_path)
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        output_file: str = "latency_distribution.pdf"
    ):
        """
        Plot latency distribution.
        
        Args:
            latencies: List of query latencies
            output_file: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Histogram
        ax1.hist(latencies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.median(latencies), color='red', linestyle='--', 
                   label=f'Median: {np.median(latencies):.2f}s')
        ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
                   label=f'P95: {np.percentile(latencies, 95):.2f}s')
        ax1.set_xlabel('Latency (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Latency Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot
        ax2.boxplot(latencies, vert=True)
        ax2.set_ylabel('Latency (seconds)')
        ax2.set_title('Latency Statistics')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved latency distribution: {output_path}")
        return str(output_path)
    
    def plot_grade_performance(
        self,
        results_by_grade: Dict[int, Dict],
        metric: str = "claim_support_rate",
        output_file: str = "grade_performance.pdf"
    ):
        """
        Plot performance across grade levels.
        
        Args:
            results_by_grade: Dict mapping grades to metric dicts
            metric: Metric to plot
            output_file: Output filename
        """
        grades = sorted(results_by_grade.keys())
        values = [results_by_grade[g].get(metric, 0.0) for g in grades]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(grades, values, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.fill_between(grades, values, alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Grade Level')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Grade')
        ax.set_xticks(grades)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved grade performance: {output_path}")
        return str(output_path)
    
    def plot_retrieval_curve(
        self,
        recall_scores: Dict[int, float],
        ndcg_scores: Dict[int, float],
        output_file: str = "retrieval_curve.pdf"
    ):
        """
        Plot retrieval performance curves (Recall@k and NDCG@k).
        
        Args:
            recall_scores: Dict mapping k to Recall@k
            ndcg_scores: Dict mapping k to NDCG@k
            output_file: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        k_values = sorted(recall_scores.keys())
        recall_vals = [recall_scores[k] for k in k_values]
        ndcg_vals = [ndcg_scores[k] for k in k_values]
        
        ax.plot(k_values, recall_vals, marker='o', label='Recall@k', linewidth=2)
        ax.plot(k_values, ndcg_vals, marker='s', label='NDCG@k', linewidth=2)
        
        ax.set_xlabel('k (Top-k Results)')
        ax.set_ylabel('Score')
        ax.set_title('Retrieval Performance Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        output_path = self.output_dir / output_file
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved retrieval curve: {output_path}")
        return str(output_path)


class TableGenerator:
    """Generate paper-ready LaTeX tables."""
    
    def __init__(self, output_dir: str = "paper/tables"):
        """Initialize table generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Table output: {self.output_dir}")
    
    def generate_ablation_table(
        self,
        results: Dict[str, Dict],
        output_file: str = "ablation_results.tex"
    ) -> str:
        """
        Generate LaTeX table for ablation study.
        
        Args:
            results: Dict mapping system names to metric dicts
            output_file: Output filename
        
        Returns:
            LaTeX table string
        """
        # Get all metrics
        all_metrics = set()
        for metrics in results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)
        
        # Build LaTeX table
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Ablation Study Results}")
        lines.append("\\label{tab:ablation}")
        
        # Column specification
        n_cols = len(all_metrics) + 1
        lines.append(f"\\begin{{tabular}}{{l{'c' * len(all_metrics)}}}")
        lines.append("\\toprule")
        
        # Header
        header = "System & " + " & ".join([m.replace('_', ' ').title() for m in all_metrics]) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")
        
        # Rows
        for system_name, metrics in results.items():
            values = [f"{metrics.get(m, 0.0):.3f}" for m in all_metrics]
            
            # Bold best value in each column
            for i, m in enumerate(all_metrics):
                col_values = [results[s].get(m, 0.0) for s in results.keys()]
                if metrics.get(m, 0.0) == max(col_values):
                    values[i] = f"\\textbf{{{values[i]}}}"
            
            row = system_name.replace('_', '\\_') + " & " + " & ".join(values) + " \\\\"
            lines.append(row)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        latex_table = "\n".join(lines)
        
        # Save to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved LaTeX table: {output_path}")
        return latex_table
    
    def generate_comparison_table(
        self,
        baseline_results: Dict,
        proposed_results: Dict,
        baseline_name: str = "Baseline",
        proposed_name: str = "TRACE-Tutor",
        output_file: str = "comparison_table.tex"
    ) -> str:
        """
        Generate comparison table between baseline and proposed system.
        
        Args:
            baseline_results: Baseline metrics
            proposed_results: Proposed system metrics
            baseline_name: Name of baseline
            proposed_name: Name of proposed system
            output_file: Output filename
        
        Returns:
            LaTeX table string
        """
        metrics = sorted(set(baseline_results.keys()) & set(proposed_results.keys()))
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Comparison: {baseline_name} vs {proposed_name}}}")
        lines.append("\\label{tab:comparison}")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append(f"Metric & {baseline_name} & {proposed_name} & Improvement \\\\"
        lines.append("\\midrule")
        
        for metric in metrics:
            baseline_val = baseline_results.get(metric, 0.0)
            proposed_val = proposed_results.get(metric, 0.0)
            
            improvement = ((proposed_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0.0
            sign = "+" if improvement > 0 else ""
            
            # Bold if improvement > 0
            proposed_str = f"\\textbf{{{proposed_val:.3f}}}" if improvement > 0 else f"{proposed_val:.3f}"
            improvement_str = f"{sign}{improvement:.1f}\\%"
            
            row = f"{metric.replace('_', ' ').title()} & {baseline_val:.3f} & {proposed_str} & {improvement_str} \\\\"
            lines.append(row)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        latex_table = "\n".join(lines)
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved comparison table: {output_path}")
        return latex_table
    
    def generate_summary_statistics(
        self,
        results: Dict,
        output_file: str = "summary_stats.tex"
    ) -> str:
        """
        Generate summary statistics table.
        
        Args:
            results: Results dictionary
            output_file: Output filename
        
        Returns:
            LaTeX table string
        """
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Summary Statistics}")
        lines.append("\\label{tab:summary}")
        lines.append("\\begin{tabular}{lr}")
        lines.append("\\toprule")
        lines.append("Statistic & Value \\\\")
        lines.append("\\midrule")
        
        for key, value in results.items():
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            elif isinstance(value, int):
                value_str = f"{value:,}"
            else:
                value_str = str(value)
            
            lines.append(f"{key.replace('_', ' ').title()} & {value_str} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        latex_table = "\n".join(lines)
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"Saved summary table: {output_path}")
        return latex_table


def generate_paper_results(
    experiment_dirs: List[str],
    output_dir: str = "paper"
):
    """
    Generate all paper-ready visualizations and tables from experiment results.
    
    Args:
        experiment_dirs: List of experiment result directories
        output_dir: Output directory for paper materials
    """
    logger.info("Generating paper-ready results...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = {}
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        
        if not exp_path.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            continue
        
        # Load manifest
        manifest_file = exp_path / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                exp_name = manifest.get("experiment_name", exp_path.name)
        else:
            exp_name = exp_path.name
        
        # Load metrics
        metrics_file = exp_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_results[exp_name] = metrics
    
    if not all_results:
        logger.error("No results found to visualize")
        return
    
    # Initialize generators
    visualizer = ResultVisualizer(output_dir=str(output_path / "figures"))
    table_gen = TableGenerator(output_dir=str(output_path / "tables"))
    
    # Generate visualizations
    logger.info("Creating visualizations...")
    
    # Flatten metrics for comparison
    flat_results = {}
    for name, metrics in all_results.items():
        flat_results[name] = {}
        if "generation" in metrics:
            flat_results[name].update(metrics["generation"])
        if "efficiency" in metrics:
            flat_results[name].update(metrics["efficiency"])
    
    # Ablation comparison
    if len(flat_results) > 1:
        key_metrics = ["claim_support_rate", "fallback_rate", "avg_bundles_used"]
        visualizer.plot_ablation_comparison(
            flat_results,
            [m for m in key_metrics if any(m in r for r in flat_results.values())]
        )
        
        visualizer.plot_metric_heatmap(flat_results)
    
    # Generate tables
    logger.info("Creating tables...")
    
    if len(flat_results) > 1:
        table_gen.generate_ablation_table(flat_results)
        
        # Comparison table (assume first is baseline, last is full)
        system_names = list(flat_results.keys())
        if len(system_names) >= 2:
            table_gen.generate_comparison_table(
                flat_results[system_names[0]],
                flat_results[system_names[-1]],
                baseline_name=system_names[0],
                proposed_name=system_names[-1]
            )
    
    logger.info(f"\n✓ Paper-ready results generated in: {output_path}")
    logger.info(f"  - Figures: {output_path / 'figures'}")
    logger.info(f"  - Tables: {output_path / 'tables'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Visualization module ready.")
    print("Use generate_paper_results() to create publication figures and tables.")
