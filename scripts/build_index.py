"""
Build and save retrieval indices.

Pre-computes embeddings and saves indices for faster experimentation.
"""

import argparse
import logging
from pathlib import Path
import yaml

from trace_tutor.ingestion import BundleBuilder
from trace_tutor.retrieval import HybridRetriever


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/study.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--bundles",
        type=str,
        default="data/bundles/corpus_all.jsonl",
        help="Evidence bundles file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/interim/retriever_index.pkl",
        help="Output index file"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load bundles
    logger.info(f"Loading bundles from: {args.bundles}")
    bundles = BundleBuilder.load_bundles(args.bundles)
    
    logger.info(f"Loaded {len(bundles)} bundles")
    
    # Build retriever
    retrieval_config = config.get("retrieval", {})
    hardware_config = config.get("hardware", {})
    
    logger.info("Building hybrid retriever...")
    logger.info(f"  Dense model: {retrieval_config.get('dense', {}).get('model')}")
    logger.info(f"  Device: {hardware_config.get('device', 'cpu')}")
    
    retriever = HybridRetriever(
        evidence_bundles=bundles,
        dense_model=retrieval_config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=hardware_config.get("cache_dir"),
        device=hardware_config.get("device", "cpu")
    )
    
    # Save index
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving index to: {output_path}")
    retriever.save_index(str(output_path))
    
    logger.info("\n✓ Index building complete!")
    logger.info(f"Index saved to: {output_path}")
    logger.info(f"Index size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
