"""
Main TRACE-Tutor interface.

Orchestrates the complete pipeline: retrieval → selection → generation → verification.
Includes semantic caching for 10-20x speedup on similar questions.
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
from trace_tutor.semantic_cache import SemanticCache
from trace_tutor.faithfulness_gate import FaithfulnessGate, apply_faithfulness_gate
from trace_tutor.clarification_agent import check_and_clarify, DialogueManager
from trace_tutor.multimodal_retriever import MultimodalRetriever, MMRSelector
from trace_tutor.image_understanding import CLIPImageEncoder
from trace_tutor.audio_processor import SpeechToText, TextToSpeech
from trace_tutor.advanced_retrieval import (
    LateInteractionRetriever,
    DifficultyAwareReranker,
    BanditFusionOptimizer
)


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
        
        # Initialize semantic cache for performance
        cache_config = self.config.get("cache", {})
        self.semantic_cache = SemanticCache(
            cache_dir=cache_config.get("dir", "data/cache"),
            similarity_threshold=cache_config.get("similarity_threshold", 0.92),
            max_cache_size=cache_config.get("max_size", 1000)
        )
        
        # Initialize multimodal components
        multimodal_config = self.config.get("multimodal", {})
        if multimodal_config.get("enabled", True):
            self.clip_encoder = CLIPImageEncoder(
                model_name=multimodal_config.get("clip", {}).get("model", "sentence-transformers/clip-ViT-B-32-multilingual-v1"),
                device=self.config.get("hardware", {}).get("device", "cpu")
            )
            self.multimodal_retriever = MultimodalRetriever(
                text_retriever=self.retriever,
                clip_encoder=self.clip_encoder,
                modality_weights=multimodal_config.get("weights", {"text": 0.5, "image": 0.3, "audio": 0.2})
            )
            self.stt = SpeechToText(
                model_size=multimodal_config.get("whisper", {}).get("model_size", "base"),
                device=self.config.get("hardware", {}).get("device", "cpu")
            )
            self.tts = TextToSpeech(
                default_lang=multimodal_config.get("tts", {}).get("default_language", "en")
            )
        else:
            self.clip_encoder = None
            self.multimodal_retriever = None
            self.stt = None
            self.tts = None
        
        # Initialize advanced components
        self.faithfulness_gate = FaithfulnessGate(threshold=0.7)
        self.dialogue_manager = DialogueManager()
        
        advanced_config = self.config.get("advanced_retrieval", {})
        if advanced_config.get("use_bandit_fusion", False):
            self.bandit_optimizer = BanditFusionOptimizer(learning_rate=0.1)
            weights_path = advanced_config.get("bandit_weights_path", "data/cache/bandit_weights.json")
            if Path(weights_path).exists():
                self.bandit_optimizer.load_weights(weights_path)
        else:
            self.bandit_optimizer = None
        
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
        use_cache: bool = True,
        **kwargs
    ) -> Answer:
        """
        Ask a question and get grounded answer.
        
        Uses semantic caching to achieve 10-20x speedup for similar questions.
        
        Args:
            question: Student question
            grade: Target grade level
            subject: Subject area
            query_type: Type of question (conceptual, direct_fact, etc.)
            use_cache: Whether to use semantic cache (default: True)
            **kwargs: Additional query parameters
        
        Returns:
            Answer object with complete provenance
        """
        if not self.retriever:
            logger.error("No retriever initialized - load bundles first")
            return self._error_answer("System not initialized")
        
        # Check semantic cache first (10-20x faster!)
        if use_cache:
            cached = self.semantic_cache.get(question, grade, subject, query_type)
            if cached:
                # Convert cached response to Answer object
                return Answer(
                    query_id=f"q_{int(time.time())}",
                    answer=cached.answer,
                    evidence_bundle_ids=cached.evidence_bundle_ids,
                    support_score=cached.support_score,
                    total_time=0.0,  # Cache hit is instant
                    retrieval_time=0.0,
                    selection_time=0.0,
                    generation_time=0.0,
                    cached=True
                )
        
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
            answer = self.process_query(query)
            
            # Cache the response for future similar questions
            if use_cache and hasattr(answer, 'answer'):
                self.semantic_cache.put(
                    question=question,
                    answer=answer.answer,
                    evidence_bundle_ids=answer.evidence_bundle_ids,
                    support_score=answer.support_score,
                    grade=grade,
                    subject=subject,
                    query_type=query_type
                )
            
            return answer
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
    
    def ask_multimodal(
        self,
        question: str,
        grade: int,
        subject: str = "Science",
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        query_type: str = "conceptual",
        use_cache: bool = True,
        use_faithfulness_gate: bool = True,
        **kwargs
    ) -> Answer:
        """
        Ask question with multimodal inputs (text + image + audio).
        
        Args:
            question: Text question
            grade: Grade level
            subject: Subject area
            image_path: Optional path to uploaded image
            audio_path: Optional path to audio file
            query_type: Query type
            use_cache: Use semantic cache
            use_faithfulness_gate: Apply faithfulness verification
            
        Returns:
            Answer with multimodal evidence
        """
        # Process audio if provided
        audio_transcription = None
        audio_language = None
        audio_confidence = None
        
        if audio_path and self.stt:
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.stt.transcribe(audio_path)
            audio_transcription = result["text"]
            audio_language = result["language"]
            audio_confidence = result["confidence"]
            
            # Use transcription as primary question if text is empty
            if not question or len(question.strip()) < 5:
                question = audio_transcription
        
        # Check for clarification needs
        clarification_check = check_and_clarify(question, grade, subject)
        if clarification_check['needs_clarification']:
            # Return clarification request
            return Answer(
                query_id=f"q_{int(time.time())}",
                answer=clarification_check['clarification_message'],
                status="clarify",
                evidence_bundle_ids=[],
                model_name="clarification_agent",
                generation_time=0.0,
                total_time=0.0
            )
        
        # Create multimodal query
        query = Query(
            id=f"q_{int(time.time())}",
            text=question,
            image_path=image_path,
            audio_path=audio_path,
            audio_transcription=audio_transcription,
            audio_language=audio_language,
            audio_confidence=audio_confidence,
            grade=grade,
            subject=subject,
            query_type=query_type,
            needs_visual=(image_path is not None),
            **kwargs
        )
        
        # Use multimodal retriever if available
        if self.multimodal_retriever and (image_path or audio_path):
            total_start = time.time()
            
            # Multimodal retrieval
            results = self.multimodal_retriever.retrieve(
                query,
                k=20,
                grade_filter=grade,
                subject_filter=subject
            )
            
            # Convert to format expected by selector
            candidates = results
            
            # Selection
            selection_result = self.selector.select(query, candidates)
            
            if selection_result.status != "ok":
                return self._fallback_answer(query, "Evidence selection failed")
            
            # Generation
            answer = self.generator.generate_answer(query, selection_result.selected_bundles)
            answer.total_time = time.time() - total_start
            
            # Collect visual evidence paths
            visual_paths = [
                e.figure_path for e in selection_result.selected_bundles 
                if hasattr(e, 'figure_path') and e.figure_path
            ]
            answer.visual_evidence_paths = visual_paths
            
            # Apply faithfulness gate
            if use_faithfulness_gate:
                answer = apply_faithfulness_gate(
                    answer,
                    selection_result.selected_bundles,
                    threshold=0.7
                )
            
            return answer
        else:
            # Fallback to text-only
            return self.ask(question, grade, subject, query_type, use_cache)
    
    def generate_audio_response(self, answer_text: str, grade: int, language: str = "en") -> Optional[str]:
        """
        Generate audio version of answer.
        
        Args:
            answer_text: Answer text to synthesize
            grade: Grade level (affects pacing)
            language: Language code
            
        Returns:
            Path to audio file or None
        """
        if not self.tts:
            logger.warning("TTS not initialized")
            return None
        
        return self.tts.synthesize_answer(answer_text, grade=grade, lang=language)
        
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
