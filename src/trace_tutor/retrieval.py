"""
Hybrid retrieval: BM25 + Dense + RRF fusion with cross-encoder reranking.

CPU-optimized with caching for efficient solo research.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import pickle

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from trace_tutor.schema import Evidence, Query


logger = logging.getLogger(__name__)


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and dense embeddings.
    
    Optimized for CPU with caching and batched processing.
    """
    
    def __init__(
        self,
        evidence_bundles: List[Evidence],
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            evidence_bundles: List of Evidence objects to index
            dense_model: HuggingFace model for dense retrieval
            cache_dir: Directory for caching embeddings
            device: 'cpu' or 'cuda'
        """
        self.bundles = evidence_bundles
        self.bundle_ids = [b.id for b in evidence_bundles]
        self.id_to_idx = {b.id: idx for idx, b in enumerate(evidence_bundles)}
        
        self.dense_model_name = dense_model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        logger.info(f"Initializing retriever with {len(evidence_bundles)} bundles")
        
        # Build BM25 index
        self._build_bm25()
        
        # Build dense index
        self._build_dense(dense_model)
        
        logger.info("Retriever initialized")
    
    def _build_bm25(self):
        """Build BM25 sparse index."""
        logger.info("Building BM25 index...")
        
        corpus_texts = [b.text for b in self.bundles]
        tokenized_corpus = [tokenize_simple(text) for text in corpus_texts]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("BM25 index built")
    
    def _build_dense(self, model_name: str):
        """Build dense embedding index with caching."""
        logger.info(f"Building dense index with {model_name}...")
        
        # Try to load cached embeddings
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"embeddings_{model_name.replace('/', '_')}.pkl"
            
            if cache_file.exists():
                logger.info(f"Loading cached embeddings from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if cache_data['bundle_ids'] == self.bundle_ids:
                        self.embeddings = cache_data['embeddings']
                        self.encoder = SentenceTransformer(model_name, device=self.device)
                        logger.info("Loaded cached embeddings")
                        return
                    else:
                        logger.info("Cache mismatch, rebuilding...")
        
        # Build embeddings
        self.encoder = SentenceTransformer(model_name, device=self.device)
        
        corpus_texts = [b.text for b in self.bundles]
        self.embeddings = self.encoder.encode(
            corpus_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Cache embeddings
        if cache_file:
            logger.info(f"Caching embeddings to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'bundle_ids': self.bundle_ids,
                    'embeddings': self.embeddings,
                    'model': model_name
                }, f)
        
        logger.info(f"Dense index built: {self.embeddings.shape}")
    
    def search_sparse(
        self,
        query: str,
        allowed_ids: Optional[Set[str]] = None,
        top_k: int = 20
    ) -> List[Tuple[Evidence, float]]:
        """BM25 sparse retrieval."""
        query_tokens = tokenize_simple(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Filter by allowed IDs
        if allowed_ids:
            indices = [i for i, b in enumerate(self.bundles) if b.id in allowed_ids]
        else:
            indices = list(range(len(self.bundles)))
        
        # Sort by score
        scored = [(i, scores[i]) for i in indices]
        scored.sort(key=lambda x: -x[1])
        
        results = [
            (self.bundles[idx], float(score))
            for idx, score in scored[:top_k]
        ]
        
        return results
    
    def search_dense(
        self,
        query: str,
        allowed_ids: Optional[Set[str]] = None,
        top_k: int = 20
    ) -> List[Tuple[Evidence, float]]:
        """Dense semantic retrieval."""
        # Encode query
        query_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        
        # Compute similarities
        scores = self.embeddings @ query_vec
        
        # Filter by allowed IDs
        if allowed_ids:
            indices = [i for i, b in enumerate(self.bundles) if b.id in allowed_ids]
        else:
            indices = list(range(len(self.bundles)))
        
        # Sort by score
        scored = [(i, scores[i]) for i in indices]
        scored.sort(key=lambda x: -x[1])
        
        results = [
            (self.bundles[idx], float(score))
            for idx, score in scored[:top_k]
        ]
        
        return results
    
    def search_hybrid(
        self,
        query: str,
        allowed_ids: Optional[Set[str]] = None,
        top_k_each: int = 20,
        rrf_k: int = 60
    ) -> List[Tuple[Evidence, float]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        
        Args:
            query: Query text
            allowed_ids: Set of allowed bundle IDs (for curriculum filtering)
            top_k_each: Number of results from each retriever
            rrf_k: RRF constant (typically 60)
        
        Returns:
            List of (Evidence, fused_score) tuples
        """
        # Get results from both retrievers
        sparse_results = self.search_sparse(query, allowed_ids, top_k_each)
        dense_results = self.search_dense(query, allowed_ids, top_k_each)
        
        # Build ranked lists
        sparse_ranked = [b.id for b, _ in sparse_results]
        dense_ranked = [b.id for b, _ in dense_results]
        
        # Reciprocal Rank Fusion
        fused_scores = {}
        
        for rank, bundle_id in enumerate(sparse_ranked, start=1):
            fused_scores[bundle_id] = fused_scores.get(bundle_id, 0.0) + 1.0 / (rrf_k + rank)
        
        for rank, bundle_id in enumerate(dense_ranked, start=1):
            fused_scores[bundle_id] = fused_scores.get(bundle_id, 0.0) + 1.0 / (rrf_k + rank)
        
        # Sort by fused score
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: -fused_scores[x])
        
        results = [
            (self.bundles[self.id_to_idx[bid]], fused_scores[bid])
            for bid in sorted_ids
        ]
        
        return results
    
    def filter_by_curriculum(
        self,
        query: Query,
        allow_prerequisites: bool = True,
        verified_ancestors: Optional[Set[str]] = None
    ) -> Set[str]:
        """
        Filter evidence bundles by curriculum constraints.
        
        Returns:
            Set of allowed bundle IDs
        """
        allowed_ids = set()
        
        for bundle in self.bundles:
            # Must match subject
            if bundle.subject != query.subject:
                continue
            
            # Must be at or below target grade
            if bundle.grade == query.grade:
                allowed_ids.add(bundle.id)
            elif bundle.grade < query.grade and allow_prerequisites:
                # Check if prerequisite is verified
                if verified_ancestors and bundle.lo_id:
                    if bundle.lo_id in verified_ancestors:
                        # Check learner mastery
                        mastery = query.mastery.get(bundle.lo_id)
                        if mastery is not None and mastery < 0.5:
                            allowed_ids.add(bundle.id)
                # If no mastery info, don't assume gap
            # Exclude higher grades
        
        return allowed_ids
    
    def save_index(self, filepath: str):
        """Save retriever state for later loading."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'bundle_ids': self.bundle_ids,
            'dense_model': self.dense_model_name,
            'embeddings': self.embeddings,
            'bm25_corpus': [tokenize_simple(b.text) for b in self.bundles]
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved retriever index to {filepath}")
    
    @classmethod
    def load_index(cls, filepath: str, bundles: List[Evidence], device: str = "cpu"):
        """Load pre-built retriever index."""
        logger.info(f"Loading retriever index from {filepath}")
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        retriever = cls.__new__(cls)
        retriever.bundles = bundles
        retriever.bundle_ids = [b.id for b in bundles]
        retriever.id_to_idx = {b.id: idx for idx, b in enumerate(bundles)}
        retriever.device = device
        
        # Restore BM25
        retriever.bm25 = BM25Okapi(state['bm25_corpus'])
        
        # Restore dense
        retriever.dense_model_name = state['dense_model']
        retriever.encoder = SentenceTransformer(state['dense_model'], device=device)
        retriever.embeddings = state['embeddings']
        
        logger.info("Retriever loaded")
        return retriever


class Reranker:
    """
    Cross-encoder reranker for final candidate refinement.
    
    More expensive but more accurate than bi-encoder retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model
            device: 'cpu' or 'cuda'
        """
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Evidence, float]],
        top_k: int = 12
    ) -> List[Tuple[Evidence, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of (Evidence, initial_score) tuples
            top_k: Number of candidates to rerank
        
        Returns:
            Reranked list of (Evidence, reranker_score) tuples
        """
        if not candidates:
            return []
        
        # Take top candidates for reranking (rerankers are expensive)
        subset = candidates[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, bundle.text) for bundle, _ in subset]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Sort by new scores
        reranked = list(zip([b for b, _ in subset], scores))
        reranked.sort(key=lambda x: -float(x[1]))
        
        return [(b, float(s)) for b, s in reranked]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Retrieval module ready. Use through TRACETutor interface.")
