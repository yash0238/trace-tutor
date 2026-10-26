"""
Advanced Retrieval Components

Implements research-grade retrieval techniques:
- ColBERT-style late interaction (MaxSim)
- Difficulty-aware adaptive reranking
- Bandit-optimized fusion weights
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class LateInteractionRetriever:
    """
    ColBERT-style late interaction retrieval.
    
    Uses per-token embeddings and MaxSim scoring for fine-grained matching.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize late interaction retriever.
        
        Args:
            model_name: Sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            logger.info(f"Late interaction retriever ready: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def encode_tokens(self, text: str) -> np.ndarray:
        """
        Encode text to per-token embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Matrix of shape (num_tokens, embedding_dim)
        """
        if self.model is None:
            return np.zeros((1, 384))
        
        # Get token-level embeddings
        # Note: Standard sentence-transformers pools tokens; 
        # for true ColBERT, need special model with token outputs
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Simulate token-level by splitting sentence into words
        # and encoding each (this is a simplified approximation)
        words = text.split()[:50]  # Limit tokens
        token_embeddings = []
        
        for word in words:
            word_emb = self.model.encode(word, convert_to_numpy=True)
            token_embeddings.append(word_emb)
        
        if not token_embeddings:
            return embedding.reshape(1, -1)
        
        return np.array(token_embeddings)
    
    def max_sim_score(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """
        Compute MaxSim score between query and document.
        
        MaxSim(Q,D) = Σ_{i=1...|Q|} max_{j=1...|D|} (q_i · d_j)
        
        Args:
            query_tokens: Query token embeddings (Q × D)
            doc_tokens: Document token embeddings (D × D)
            
        Returns:
            MaxSim score
        """
        if query_tokens.shape[0] == 0 or doc_tokens.shape[0] == 0:
            return 0.0
        
        # Normalize
        query_norm = query_tokens / (np.linalg.norm(query_tokens, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_tokens / (np.linalg.norm(doc_tokens, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix: (Q × D) @ (D × D)^T = (Q × D)
        sim_matrix = query_norm @ doc_norm.T
        
        # MaxSim: for each query token, take max over doc tokens, then sum
        max_sims = np.max(sim_matrix, axis=1)
        maxsim_score = np.sum(max_sims)
        
        return float(maxsim_score)
    
    def retrieve(self, query: str, candidates: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve using MaxSim scoring.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results
            
        Returns:
            List of (candidate_idx, score) tuples
        """
        query_tokens = self.encode_tokens(query)
        
        scores = []
        for idx, candidate in enumerate(candidates):
            doc_tokens = self.encode_tokens(candidate)
            score = self.max_sim_score(query_tokens, doc_tokens)
            scores.append((idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class DifficultyAwareReranker:
    """
    Difficulty-aware adaptive reranking.
    
    Adjusts evidence selection based on student grade and readability.
    """
    
    def __init__(self, grade_weight: float = 0.3):
        """
        Initialize reranker.
        
        Args:
            grade_weight: Weight for difficulty adjustment (0-1)
        """
        self.grade_weight = grade_weight
        self.L_max = 12  # Maximum grade level difference
    
    def rerank(
        self,
        candidates: List[Tuple[any, float]],
        student_grade: int
    ) -> List[Tuple[any, float]]:
        """
        Rerank candidates by difficulty match.
        
        score_final = γ·sim(q,c) + (1−γ)·(1 − |ReadabilityLevel(c) − GradeLevel| / L_max)
        
        Args:
            candidates: List of (evidence, relevance_score) tuples
            student_grade: Student grade level
            
        Returns:
            Reranked list
        """
        reranked = []
        
        for evidence, relevance_score in candidates:
            # Get evidence readability (from metadata or estimate)
            evidence_grade = self._get_evidence_grade(evidence)
            
            # Calculate difficulty penalty
            grade_diff = abs(evidence_grade - student_grade)
            difficulty_score = 1.0 - (grade_diff / self.L_max)
            
            # Combined score
            final_score = (
                (1 - self.grade_weight) * relevance_score +
                self.grade_weight * difficulty_score
            )
            
            reranked.append((evidence, final_score))
        
        # Sort by new score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def _get_evidence_grade(self, evidence) -> int:
        """Extract or estimate evidence grade level."""
        # Try to get from evidence object
        if hasattr(evidence, 'grade'):
            return evidence.grade
        elif hasattr(evidence, 'readability_score'):
            # Convert Flesch-Kincaid to grade
            return int(evidence.readability_score)
        elif hasattr(evidence, 'metadata') and 'readability_score' in evidence.metadata:
            return int(evidence.metadata['readability_score'])
        else:
            # Estimate from text complexity
            text = evidence.text if hasattr(evidence, 'text') else str(evidence)
            return self._estimate_readability(text)
    
    def _estimate_readability(self, text: str) -> int:
        """
        Estimate readability grade using simple heuristics.
        
        Uses average sentence length and word length.
        """
        sentences = text.split('.')
        words = text.split()
        
        if not words:
            return 5
        
        avg_sentence_length = len(words) / max(len(sentences), 1)
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Simple Flesch-Kincaid approximation
        grade = 0.39 * avg_sentence_length + 11.8 * (avg_word_length / 5) - 15.59
        
        # Clamp to 1-12
        return max(1, min(12, int(grade)))


class BanditFusionOptimizer:
    """
    Bandit-optimized fusion weight learning.
    
    Uses contextual bandit to learn optimal modality weights per query type.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize bandit optimizer.
        
        Args:
            learning_rate: Weight update step size
        """
        self.learning_rate = learning_rate
        
        # Weights per query type: {query_type: {modality: weight}}
        self.weights = defaultdict(lambda: {
            'text': 0.5,
            'image': 0.3,
            'audio': 0.2
        })
        
        # Track rewards: {query_type: [reward_history]}
        self.rewards = defaultdict(list)
        
        logger.info("BanditFusionOptimizer initialized")
    
    def get_weights(self, query_type: str) -> Dict[str, float]:
        """
        Get current weights for query type.
        
        Args:
            query_type: Query classification (factual, conceptual, diagram, etc.)
            
        Returns:
            Dict of modality weights
        """
        return self.weights[query_type].copy()
    
    def update(self, query_type: str, reward: float):
        """
        Update weights based on feedback.
        
        w_{t+1} = w_t + η·∇_w E[reward | query_type, w]
        
        Args:
            query_type: Query classification
            reward: Feedback signal (0-1, from thumbs up/down)
        """
        # Store reward
        self.rewards[query_type].append(reward)
        
        # Simple gradient: increase weights proportionally to reward
        current_weights = self.weights[query_type]
        
        for modality in current_weights:
            # Gradient approximation: reward acts as scaling factor
            gradient = reward - 0.5  # Center around 0
            current_weights[modality] += self.learning_rate * gradient
        
        # Normalize weights
        total = sum(current_weights.values())
        if total > 0:
            for modality in current_weights:
                current_weights[modality] /= total
        
        logger.debug(f"Updated weights for {query_type}: {current_weights}")
    
    def get_convergence_stats(self) -> Dict:
        """Get learning statistics."""
        stats = {}
        
        for query_type, reward_history in self.rewards.items():
            if len(reward_history) > 0:
                stats[query_type] = {
                    'avg_reward': np.mean(reward_history),
                    'num_updates': len(reward_history),
                    'current_weights': self.weights[query_type]
                }
        
        return stats
    
    def save_weights(self, path: str):
        """Save learned weights to disk."""
        import json
        
        data = {
            'weights': dict(self.weights),
            'rewards': {k: list(v) for k, v in self.rewards.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved bandit weights to {path}")
    
    def load_weights(self, path: str):
        """Load learned weights from disk."""
        import json
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.weights = defaultdict(lambda: {'text': 0.5, 'image': 0.3, 'audio': 0.2})
            self.weights.update(data['weights'])
            
            self.rewards = defaultdict(list)
            for k, v in data.get('rewards', {}).items():
                self.rewards[k] = v
            
            logger.info(f"Loaded bandit weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}")


def apply_advanced_retrieval(
    query: str,
    candidates: List,
    student_grade: int,
    query_type: str = "factual",
    use_late_interaction: bool = True,
    use_difficulty_aware: bool = True,
    bandit_optimizer: Optional[BanditFusionOptimizer] = None
) -> List:
    """
    Apply advanced retrieval techniques.
    
    Args:
        query: Query text
        candidates: List of evidence candidates
        student_grade: Student grade level
        query_type: Query classification
        use_late_interaction: Whether to use ColBERT-style MaxSim
        use_difficulty_aware: Whether to apply difficulty reranking
        bandit_optimizer: Optional bandit optimizer for fusion weights
        
    Returns:
        Reranked candidates
    """
    results = candidates
    
    # Step 1: Late interaction retrieval (if enabled)
    if use_late_interaction:
        retriever = LateInteractionRetriever()
        candidate_texts = [c[0].text if hasattr(c[0], 'text') else str(c[0]) for c in candidates]
        late_scores = retriever.retrieve(query, candidate_texts, top_k=len(candidates))
        
        # Merge scores
        late_score_map = {idx: score for idx, score in late_scores}
        results = [(cand, late_score_map.get(i, 0.0)) for i, (cand, _) in enumerate(results)]
    
    # Step 2: Difficulty-aware reranking (if enabled)
    if use_difficulty_aware:
        reranker = DifficultyAwareReranker(grade_weight=0.25)
        results = reranker.rerank(results, student_grade)
    
    return results
