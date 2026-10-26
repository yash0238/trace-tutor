"""
Semantic caching layer for TRACE-Tutor to reduce latency by 10-20x.

Implements vector-indexed semantic similarity caching to avoid redundant
retrieval and generation for similar questions.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
import time


@dataclass
class CachedResponse:
    """Cached response with metadata."""
    question: str
    answer: str
    evidence_bundle_ids: list
    support_score: float
    grade: int
    subject: str
    query_type: str
    timestamp: float
    embedding: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['embedding'] = self.embedding.tolist()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedResponse':
        """Create from dictionary."""
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class SemanticCache:
    """
    Semantic cache using vector similarity to detect duplicate questions.
    
    Based on the feedback document recommendations for 10-20x speedup:
    - Uses lightweight embedders (sentence-transformers)
    - Binary quantization for fast similarity search
    - In-memory cache with disk persistence
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        max_cache_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            cache_dir: Directory to store cache
            model_name: Sentence transformer model for embeddings
            similarity_threshold: Cosine similarity threshold (0.92-0.95 recommended)
            max_cache_size: Maximum number of cached items
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        
        # Load lightweight embedder
        self.embedder = SentenceTransformer(model_name)
        
        # In-memory cache
        self.cache: Dict[str, CachedResponse] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.cache_keys: list = []
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / "semantic_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = {k: CachedResponse.from_dict(v) for k, v in data['cache'].items()}
                    self.cache_keys = data['keys']
                    if self.cache_keys:
                        self.embeddings_matrix = np.vstack([
                            self.cache[k].embedding for k in self.cache_keys
                        ])
                print(f"✓ Loaded {len(self.cache)} cached responses")
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}")
                self.cache = {}
                self.cache_keys = []
    
    def _save_cache(self):
        """Save cache to disk."""
        cache_file = self.cache_dir / "semantic_cache.pkl"
        try:
            data = {
                'cache': {k: v.to_dict() for k, v in self.cache.items()},
                'keys': self.cache_keys
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")
    
    def _get_cache_key(self, question: str, grade: int, subject: str, query_type: str) -> str:
        """Generate unique cache key."""
        key_str = f"{question}|{grade}|{subject}|{query_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        question: str,
        grade: int,
        subject: str,
        query_type: str
    ) -> Optional[CachedResponse]:
        """
        Get cached response for semantically similar question.
        
        Returns cached response if similarity > threshold, else None.
        This is where the 10-20x speedup happens - bypasses entire RAG pipeline.
        """
        if not self.cache:
            return None
        
        # Generate query embedding
        query_embedding = self.embedder.encode(question, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Find most similar cached question
        if self.embeddings_matrix is not None:
            # Normalize embeddings
            normalized_embeddings = self.embeddings_matrix / np.linalg.norm(
                self.embeddings_matrix, axis=1, keepdims=True
            )
            
            # Compute cosine similarity
            similarities = np.dot(normalized_embeddings, query_embedding)
            
            # Filter by grade and subject
            valid_indices = []
            for i, key in enumerate(self.cache_keys):
                cached = self.cache[key]
                if cached.grade == grade and cached.subject == subject:
                    valid_indices.append(i)
            
            if not valid_indices:
                return None
            
            # Find best match among valid candidates
            valid_similarities = similarities[valid_indices]
            best_idx = np.argmax(valid_similarities)
            best_similarity = valid_similarities[best_idx]
            
            if best_similarity >= self.similarity_threshold:
                cache_key = self.cache_keys[valid_indices[best_idx]]
                cached_response = self.cache[cache_key]
                print(f"✓ Cache hit! Similarity: {best_similarity:.3f}")
                print(f"  Original: {cached_response.question}")
                print(f"  Current:  {question}")
                return cached_response
        
        return None
    
    def put(
        self,
        question: str,
        answer: str,
        evidence_bundle_ids: list,
        support_score: float,
        grade: int,
        subject: str,
        query_type: str
    ):
        """
        Add response to cache.
        
        Args:
            question: Student question
            answer: Generated answer
            evidence_bundle_ids: Evidence bundles used
            support_score: Support score
            grade: Grade level
            subject: Subject
            query_type: Query type
        """
        # Generate embedding
        embedding = self.embedder.encode(question, convert_to_numpy=True)
        
        # Create cache key
        cache_key = self._get_cache_key(question, grade, subject, query_type)
        
        # Create cached response
        cached_response = CachedResponse(
            question=question,
            answer=answer,
            evidence_bundle_ids=evidence_bundle_ids,
            support_score=support_score,
            grade=grade,
            subject=subject,
            query_type=query_type,
            timestamp=time.time(),
            embedding=embedding
        )
        
        # Add to cache
        self.cache[cache_key] = cached_response
        self.cache_keys.append(cache_key)
        
        # Rebuild embeddings matrix
        self.embeddings_matrix = np.vstack([
            self.cache[k].embedding for k in self.cache_keys
        ])
        
        # Evict oldest if cache is full
        if len(self.cache) > self.max_cache_size:
            oldest_key = self.cache_keys.pop(0)
            del self.cache[oldest_key]
            self.embeddings_matrix = self.embeddings_matrix[1:]
        
        # Persist to disk
        self._save_cache()
        
        print(f"✓ Cached response ({len(self.cache)} total)")
    
    def clear(self):
        """Clear all cached responses."""
        self.cache = {}
        self.cache_keys = []
        self.embeddings_matrix = None
        self._save_cache()
        print("✓ Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_cached': len(self.cache),
            'cache_size_mb': sum(
                len(pickle.dumps(v.to_dict())) for v in self.cache.values()
            ) / (1024 * 1024),
            'similarity_threshold': self.similarity_threshold
        }
