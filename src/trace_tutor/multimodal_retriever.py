"""
Multimodal Retrieval Module

Implements cross-modal retrieval combining text, image, and audio evidence.
Uses weighted fusion to merge results from multiple modality-specific retrievers.
"""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from trace_tutor.schema import Evidence, Query
from trace_tutor.retrieval import HybridRetriever
from trace_tutor.image_understanding import CLIPImageEncoder

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    """
    Cross-modal evidence retriever.
    
    Supports:
    - Text → text (dense + BM25)
    - Text → image (CLIP)
    - Image → image (CLIP)
    - Audio → text (via transcription)
    - Weighted fusion across modalities
    """
    
    def __init__(
        self,
        text_retriever: HybridRetriever,
        clip_encoder: Optional[CLIPImageEncoder] = None,
        modality_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu"
    ):
        """
        Initialize multimodal retriever.
        
        Args:
            text_retriever: Text-based hybrid retriever
            clip_encoder: CLIP encoder for images (optional)
            modality_weights: Weights for fusion {text: 0.5, image: 0.3, audio: 0.2}
            device: 'cpu' or 'cuda'
        """
        self.text_retriever = text_retriever
        self.clip_encoder = clip_encoder or CLIPImageEncoder(device=device)
        
        # Default modality weights (normalized dynamically based on available modalities)
        self.modality_weights = modality_weights or {
            "text": 0.5,
            "image": 0.3,
            "audio": 0.2
        }
        
        logger.info(f"MultimodalRetriever ready (weights: {self.modality_weights})")
    
    def retrieve(
        self,
        query: Query,
        k: int = 20,
        grade_filter: Optional[int] = None,
        subject_filter: Optional[str] = None
    ) -> List[Tuple[Evidence, float]]:
        """
        Retrieve evidence across modalities.
        
        Args:
            query: Query object (may contain text, image_path, audio_path)
            k: Number of results to return
            grade_filter: Filter by grade level
            subject_filter: Filter by subject
            
        Returns:
            List of (Evidence, score) tuples, sorted by relevance
        """
        # Determine active modalities from query
        active_modalities = self._get_active_modalities(query)
        
        if not active_modalities:
            logger.warning("No valid query modalities found")
            return []
        
        # Normalize weights for active modalities only
        active_weights = self._normalize_weights(active_modalities)
        
        logger.info(f"Retrieving with modalities: {list(active_weights.keys())}")
        
        # Retrieve from each modality
        modality_results = {}
        
        if "text" in active_modalities:
            modality_results["text"] = self._retrieve_text(
                query.text, k=k*2, grade=grade_filter, subject=subject_filter
            )
        
        if "image" in active_modalities and query.image_path:
            modality_results["image"] = self._retrieve_by_image(
                query.image_path, k=k*2, grade=grade_filter, subject=subject_filter
            )
        
        if "audio" in active_modalities and query.audio_transcription:
            # Audio queries are transcribed to text
            modality_results["audio"] = self._retrieve_text(
                query.audio_transcription, k=k*2, grade=grade_filter, subject=subject_filter
            )
        
        # Fuse results with weighted scores
        fused_results = self._weighted_fusion(modality_results, active_weights)
        
        # Sort by fused score and return top-k
        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]
    
    def _get_active_modalities(self, query: Query) -> List[str]:
        """Determine which modalities are present in query."""
        active = []
        
        if query.text and len(query.text.strip()) > 0:
            active.append("text")
        
        if query.image_path:
            active.append("image")
        
        if query.audio_transcription:
            active.append("audio")
        
        return active
    
    def _normalize_weights(self, active_modalities: List[str]) -> Dict[str, float]:
        """Normalize weights for active modalities only."""
        active_weights = {m: self.modality_weights.get(m, 0.0) for m in active_modalities}
        total = sum(active_weights.values())
        
        if total == 0:
            # Equal weights if all zero
            return {m: 1.0 / len(active_modalities) for m in active_modalities}
        
        return {m: w / total for m, w in active_weights.items()}
    
    def _retrieve_text(
        self,
        text_query: str,
        k: int,
        grade: Optional[int] = None,
        subject: Optional[str] = None
    ) -> List[Tuple[Evidence, float]]:
        """Retrieve using text query."""
        try:
            results = self.text_retriever.retrieve(
                text_query,
                k=k,
                filters={"grade": grade, "subject": subject} if grade or subject else None
            )
            return results
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []
    
    def _retrieve_by_image(
        self,
        image_path: str,
        k: int,
        grade: Optional[int] = None,
        subject: Optional[str] = None
    ) -> List[Tuple[Evidence, float]]:
        """Retrieve using image query via CLIP."""
        if self.clip_encoder.model is None:
            logger.warning("CLIP encoder not available, skipping image retrieval")
            return []
        
        try:
            # Encode query image
            query_embedding = self.clip_encoder.encode_image(image_path)
            
            # Search visual evidence bundles
            visual_results = []
            
            for evidence in self.text_retriever.evidence_bundles:
                # Filter by grade/subject if specified
                if grade and evidence.grade != grade:
                    continue
                if subject and evidence.subject != subject:
                    continue
                
                # Check if evidence has image
                if evidence.modality != "image" or "clip_embedding" not in evidence.metadata:
                    continue
                
                # Compute similarity
                evidence_embedding = np.array(evidence.metadata["clip_embedding"])
                similarity = self._cosine_similarity(query_embedding, evidence_embedding)
                
                visual_results.append((evidence, float(similarity)))
            
            # Sort by similarity
            visual_results.sort(key=lambda x: x[1], reverse=True)
            return visual_results[:k]
            
        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            return []
    
    def _weighted_fusion(
        self,
        modality_results: Dict[str, List[Tuple[Evidence, float]]],
        weights: Dict[str, float]
    ) -> List[Tuple[Evidence, float]]:
        """
        Fuse results from multiple modalities using weighted scores.
        
        Uses reciprocal rank fusion (RRF) style: combines ranks instead of raw scores.
        """
        # Collect all unique evidence items
        evidence_scores = defaultdict(float)
        evidence_objects = {}
        
        for modality, results in modality_results.items():
            weight = weights.get(modality, 0.0)
            
            for rank, (evidence, score) in enumerate(results, start=1):
                # Use evidence.chunk_id as unique key
                key = evidence.chunk_id
                
                # RRF-style score: 1 / (k + rank)
                # k=60 is standard; weight by modality importance
                rrf_score = weight * (1.0 / (60 + rank))
                
                evidence_scores[key] += rrf_score
                evidence_objects[key] = evidence
        
        # Convert back to list of (Evidence, score) tuples
        fused = [(evidence_objects[key], score) for key, score in evidence_scores.items()]
        
        return fused
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b + 1e-8)


class MMRSelector:
    """
    Maximal Marginal Relevance (MMR) diversity selector.
    
    Prevents retrieving multiple similar/duplicate chunks.
    """
    
    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR selector.
        
        Args:
            lambda_param: Trade-off between relevance and diversity [0, 1]
                         1.0 = pure relevance, 0.0 = pure diversity
        """
        self.lambda_param = lambda_param
    
    def select(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple[Evidence, float]],
        k: int
    ) -> List[Tuple[Evidence, float]]:
        """
        Select diverse evidence using MMR.
        
        Args:
            query_embedding: Query embedding vector
            candidates: List of (Evidence, relevance_score) from retriever
            k: Number of items to select
            
        Returns:
            List of k diverse (Evidence, score) tuples
        """
        if len(candidates) <= k:
            return candidates
        
        selected = []
        candidate_pool = list(candidates)
        
        while len(selected) < k and candidate_pool:
            mmr_scores = []
            
            for evidence, relevance_score in candidate_pool:
                # Get embedding for this evidence
                evidence_emb = self._get_embedding(evidence)
                
                if evidence_emb is None:
                    # Fallback: use relevance score only
                    mmr_score = relevance_score
                else:
                    # Compute similarity to query
                    query_sim = self._cosine_similarity(query_embedding, evidence_emb)
                    
                    # Compute max similarity to already selected items
                    max_selected_sim = 0.0
                    if selected:
                        selected_embs = [self._get_embedding(e) for e, _ in selected]
                        selected_embs = [emb for emb in selected_embs if emb is not None]
                        
                        if selected_embs:
                            similarities = [self._cosine_similarity(evidence_emb, sel_emb) 
                                         for sel_emb in selected_embs]
                            max_selected_sim = max(similarities)
                    
                    # MMR formula
                    mmr_score = (self.lambda_param * query_sim - 
                                (1 - self.lambda_param) * max_selected_sim)
                
                mmr_scores.append((evidence, relevance_score, mmr_score))
            
            # Select item with highest MMR score
            best = max(mmr_scores, key=lambda x: x[2])
            selected.append((best[0], best[1]))  # (Evidence, original_relevance_score)
            
            # Remove from pool
            candidate_pool = [(e, s) for e, s in candidate_pool if e.chunk_id != best[0].chunk_id]
        
        return selected
    
    def _get_embedding(self, evidence: Evidence) -> Optional[np.ndarray]:
        """Extract embedding from evidence metadata."""
        if "clip_embedding" in evidence.metadata:
            return np.array(evidence.metadata["clip_embedding"])
        elif "dense_embedding" in evidence.metadata:
            return np.array(evidence.metadata["dense_embedding"])
        return None
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
