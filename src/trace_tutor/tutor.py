"""
Main TRACE-Tutor interface.

Orchestrates the complete pipeline: retrieval → selection → generation → verification.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import yaml
import time

from trace_tutor.schema import Evidence, Query, Answer
from trace_tutor.ingestion import BundleBuilder
from trace_tutor.retrieval import HybridRetriever, Reranker
from trace_tutor.graph import PrerequisiteGraph
from trace_tutor.selection import PSEvidenceSelector, SelectionConfig
from trace_tutor.generation import GroundedGenerator, GenerationConfig


logger = logging.getLogger(__name__)


class TRACETutor:
    """
    Complete TRACE-Tutor system.
    
    Combines curriculum-grounded retrieval, role-constrained selection,
    and verified answer generation.
    """
    
    def __init__(
        self,
        config_path: str = "configs/study.yaml",
        bundles_path: Optional[str] = None,
        graph_path: Optional[str] = None
    ):
        """
        Initialize TRACE-Tutor.
        
        Args:
            config_path: Path to configuration YAML
            bundles_path: Path to evidence bundles JSONL (optional)
            graph_path: Path to prerequisite graph JSONL (optional)
        """
        logger.info("Initializing TRACE-Tutor...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load evidence bundles
        if bundles_path:
            self.bundles = BundleBuilder.load_bundles(bundles_path)
            logger.info(f"Loaded {len(self.bundles)} evidence bundles")
        else:
            self.bundles = []
            logger.warning("No bundles loaded - run build_corpus.py first")
        
        # Initialize components
        self._init_retriever()
        self._init_graph(graph_path)
        self._init_selector()
        self._init_generator()
        
        logger.info("TRACE-Tutor ready")
    
    def _init_retriever(self):
        """Initialize hybrid retriever."""
        if not self.bundles:
            self.retriever = None
            return
        
        retrieval_config = self.config.get("retrieval", {})
        
        self.retriever = HybridRetriever(
            evidence_bundles=self.bundles,
            dense_model=retrieval_config.get("dense", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_dir=self.config.get("hardware", {}).get("cache_dir"),
            device=self.config.get("hardware", {}).get("device", "cpu")
        )
        
        # Initialize reranker if enabled
        if retrieval_config.get("reranker", {}).get("enabled", True):
            self.reranker = Reranker(
                model_name=retrieval_config["reranker"].get("model", "cross-encoder/ms-marco-MiniLM-L6-v2"),
                device=self.config.get("hardware", {}).get("device", "cpu")
            )
        else:
            self.reranker = None
    
    def _init_graph(self, graph_path: Optional[str]):
        """Initialize prerequisite graph."""
        if graph_path and Path(graph_path).exists():
            self.graph = PrerequisiteGraph.load(graph_path)
        elif self.config.get("prerequisite", {}).get("enabled"):
            graph_file = self.config["prerequisite"].get("graph_file")
            if graph_file and Path(graph_file).exists():
                self.graph = PrerequisiteGraph.load(graph_file)
            else:
                logger.warning("Prerequisite graph not found - using empty graph")
                self.graph = PrerequisiteGraph()
        else:
            self.graph = PrerequisiteGraph()
    
    def _init_selector(self):
        """Initialize PS-Evidence selector."""
        selection_config = self.config.get("selection", {})
        
        self.selector = PSEvidenceSelector(
            config=SelectionConfig(
                max_bundles=selection_config.get("max_bundles", 3),
                token_budget=selection_config.get("token_budget", 420),
                relevance_weight=selection_config.get("weights", {}).get("relevance", 1.0),
                role_coverage_weight=selection_config.get("weights", {}).get("role_coverage", 0.3),
                grade_fit_weight=selection_config.get("weights", {}).get("grade_fit", 0.2),
                prerequisite_value_weight=selection_config.get("weights", {}).get("prerequisite_value", 0.15),
                redundancy_penalty=selection_config.get("weights", {}).get("redundancy_penalty", 0.1),
                length_penalty=selection_config.get("weights", {}).get("length_penalty", 0.15),
                require_role_coverage=selection_config.get("require_role_coverage", True)
            )
        )
    
    def _init_generator(self):
        """Initialize answer generator."""
        gen_config = self.config.get("generation", {})
        
        self.generator = GroundedGenerator(
            config=GenerationConfig(
                provider=gen_config.get("provider", "ollama"),
                model=gen_config.get("model", "phi3:3b-mini-instruct"),
                temperature=gen_config.get("temperature", 0.3),
                max_tokens=gen_config.get("max_tokens", 256),
                top_p=gen_config.get("top_p", 0.9),
                api_key_env=gen_config.get("api_key_env", "LLM_API_KEY"),
                api_model=gen_config.get("api_model", "gpt-3.5-turbo")
            )
        )
    
    def ask(
        self,
        question: str,
        grade: int,
        subject: str = "EVS",
        query_type: str = "conceptual",
        **kwargs
    ) -> Answer:
        """
        Ask a question and get grounded answer.
        
        Args:
            question: Student question
            grade: Target grade level
            subject: Subject area
            query_type: Type of question (conceptual, direct_fact, etc.)
            **kwargs: Additional query parameters
        
        Returns:
            Answer object with complete provenance
        """
        if not self.retriever:
            logger.error("No retriever initialized - load bundles first")
            return self._error_answer("System not initialized")
        
        # Create query object
        query = Query(
            id=f"q_{int(time.time())}",
            text=question,
            grade=grade,
            subject=subject,
            query_type=query_type,
            **kwargs
        )
        
        try:
            return self.process_query(query)
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_answer(f"Processing error: {str(e)}")
    
    def process_query(self, query: Query) -> Answer:
        """
        Process complete query pipeline.
        
        Pipeline:
        1. Filter by curriculum scope
        2. Hybrid retrieval (BM25 + Dense + RRF)
        3. Rerank with cross-encoder
        4. PS-Evidence selection
        5. Grounded generation
        6. (Optional) Verification
        
        Args:
            query: Query object
        
        Returns:
            Answer object
        """
        total_start = time.time()
        
        # Step 1: Curriculum filtering
        logger.debug(f"Processing query: {query.text[:100]}")
        
        verified_ancestors = None
        if self.config.get("prerequisite", {}).get("enabled") and query.target_lo:
            verified_ancestors = self.graph.get_verified_ancestors(query.target_lo)
        
        allowed_ids = self.retriever.filter_by_curriculum(
            query,
            allow_prerequisites=self.config.get("prerequisite", {}).get("enabled", True),
            verified_ancestors=verified_ancestors
        )
        
        if not allowed_ids:
            return self._fallback_answer(query, "No curriculum-appropriate evidence found")
        
        # Step 2: Hybrid retrieval
        fusion_config = self.config.get("retrieval", {}).get("fusion", {})
        candidates = self.retriever.search_hybrid(
            query.text,
            allowed_ids=allowed_ids,
            top_k_each=fusion_config.get("top_candidates", 20),
            rrf_k=fusion_config.get("k", 60)
        )
        
        # Step 3: Rerank
        if self.reranker:
            rerank_config = self.config.get("retrieval", {}).get("reranker", {})
            candidates = self.reranker.rerank(
                query.text,
                candidates,
                top_k=rerank_config.get("top_k", 12)
            )
        
        # Step 4: PS-Evidence selection
        selection_result = self.selector.select(
            query,
            candidates,
            verified_ancestors=verified_ancestors
        )
        
        if selection_result.status != "ok":
            return self._fallback_answer(
                query,
                f"Evidence selection failed: {selection_result.status}"
            )
        
        # Step 5: Generation
        answer = self.generator.generate_answer(query, selection_result.selected_bundles)
        answer.total_time = time.time() - total_start
        
        # Step 6: Verification (if enabled)
        if self.config.get("verification", {}).get("enabled"):
            # TODO: Implement verification
            pass
        
        return answer
    
    def _fallback_answer(self, query: Query, reason: str) -> Answer:
        """Create fallback answer when pipeline cannot produce result."""
        return Answer(
            query_id=query.id,
            answer="I don't have enough information to answer this question.",
            evidence_bundle_ids=[],
            source_pages=[],
            verified=False,
            support_score=0.0,
            model_name="fallback",
            generation_time=0.0,
            total_time=0.0,
            status="fallback",
            fallback_reason=reason
        )
    
    def _error_answer(self, reason: str) -> Answer:
        """Create error answer."""
        return Answer(
            query_id="error",
            answer=f"System error: {reason}",
            evidence_bundle_ids=[],
            source_pages=[],
            verified=False,
            support_score=0.0,
            model_name="error",
            generation_time=0.0,
            total_time=0.0,
            status="abstain",
            fallback_reason=reason
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    try:
        tutor = TRACETutor(config_path="configs/study.yaml")
        
        response = tutor.ask(
            question="Why do plants need sunlight?",
            grade=4,
            subject="EVS"
        )
        
        print(f"Answer: {response.answer}")
        print(f"Sources: {response.source_pages}")
        print(f"Time: {response.total_time:.2f}s")
    
    except FileNotFoundError as e:
        print(f"Configuration or data files not found: {e}")
        print("Run setup scripts first:")
        print("  1. python scripts/download_ncert.py --grades 3 4 5")
        print("  2. python scripts/build_corpus.py")
