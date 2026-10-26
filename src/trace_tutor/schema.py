"""
Core data schemas for TRACE-Tutor.

Defines Evidence bundles, Queries, Prerequisite edges, and Answers
with full validation and provenance tracking.
"""

from typing import Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


# Type aliases
Role = Literal["concept", "example", "visual", "exercise", "prerequisite", "misconception"]
QueryType = Literal[
    "direct_fact",
    "conceptual",
    "diagram",
    "worked_problem",
    "misconception",
    "prerequisite_gap"
]
ReviewStatus = Literal["verified", "proposed", "rejected", "pending"]
Language = Literal["en", "hi", "mr"]  # English, Hindi, Marathi


class Evidence(BaseModel):
    """
    Curriculum Evidence Bundle: the atomic unit of instructional content.
    
    Preserves text, visual references, exercises, solutions, learning outcomes,
    prerequisites, and complete provenance metadata.
    """
    
    # Identity
    id: str = Field(
        ...,
        description="Unique bundle identifier (e.g., 'evs_g4_c2_p09_a01')"
    )
    source_id: str = Field(
        ...,
        description="Source document identifier (e.g., 'ncert_evs_grade4_2024')"
    )
    
    # Provenance
    page: int = Field(..., ge=1, description="Source page number")
    chapter_id: str = Field(..., description="Chapter identifier")
    section_id: Optional[str] = Field(None, description="Section within chapter")
    
    # Curriculum metadata
    grade: int = Field(..., ge=1, le=12, description="Target grade level")
    subject: str = Field(..., description="Subject (e.g., 'EVS', 'Science')")
    lo_id: Optional[str] = Field(None, description="Learning outcome identifier")
    
    # Pedagogical role
    roles: set[Role] = Field(
        ...,
        description="Instructional roles this bundle serves"
    )
    
    # Content
    text: str = Field(..., min_length=10, description="Main text content")
    source_atom_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Source span identifiers for deduplication"
    )
    
    # Visual content
    figure_id: Optional[str] = Field(None, description="Associated figure identifier")
    figure_caption: Optional[str] = Field(None, description="Figure caption text")
    figure_path: Optional[str] = Field(None, description="Path to extracted figure image")
    alt_text: Optional[str] = Field(None, description="Accessibility alt-text")
    
    # Multimodal metadata
    modality: str = Field(default="text", description="Content modality: text, image, table, audio")
    metadata: dict = Field(default_factory=dict, description="Modality-specific metadata (embeddings, etc.)")
    
    # Exercise content
    exercise_text: Optional[str] = Field(None, description="Exercise question")
    exercise_solution: Optional[str] = Field(None, description="Verified solution")
    
    # Readability
    readability_score: Optional[float] = Field(
        None,
        description="Flesch-Kincaid Grade Level or similar"
    )
    
    # Multilingual
    language: Language = Field(default="en", description="Content language")
    translation_status: Optional[str] = Field(None, description="Translation quality status")
    
    # Rights and quality
    rights_status: str = Field(
        default="authorized_local_only",
        description="Usage rights status"
    )
    quality_checked: bool = Field(default=False, description="Manual QA completed")
    
    # Timestamps
    created_at: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Bundle creation timestamp"
    )
    
    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        return v
    
    @model_validator(mode="after")
    def check_provenance(self):
        """Ensure complete provenance and role assignment."""
        if not self.roles:
            raise ValueError("At least one role must be assigned")
        if not self.source_atom_ids:
            raise ValueError("source_atom_ids cannot be empty")
        # Optional: warn but don't fail if figure has no caption
        # if self.figure_id and not self.figure_caption:
        #     raise ValueError("Figures must have captions")
        return self
    
    def token_count(self, tokenizer=None) -> int:
        """Estimate token count (rough approximation)."""
        if tokenizer:
            return len(tokenizer.encode(self.text))
        # Rough heuristic: ~0.75 tokens per word
        return int(len(self.text.split()) * 0.75)


class Query(BaseModel):
    """
    Student query with curriculum context and learner state.
    Supports multimodal input: text, image, and audio.
    """
    
    # Identity
    id: str = Field(..., description="Query identifier")
    text: str = Field(..., min_length=5, description="Question text")
    
    # Multimodal inputs
    image_path: Optional[str] = Field(None, description="Path to uploaded image (for visual queries)")
    audio_path: Optional[str] = Field(None, description="Path to audio file (for voice queries)")
    audio_transcription: Optional[str] = Field(None, description="Transcribed text from audio")
    audio_language: Optional[str] = Field(None, description="Detected audio language")
    audio_confidence: Optional[float] = Field(None, description="Transcription confidence score")
    
    # Curriculum context
    grade: int = Field(..., ge=1, le=12, description="Student grade level")
    subject: str = Field(..., description="Subject area")
    chapter_id: Optional[str] = Field(None, description="Current chapter context")
    
    # Query classification
    target_lo: Optional[str] = Field(None, description="Target learning outcome")
    query_type: QueryType = Field(
        default="direct_fact",
        description="Question type for role policy"
    )
    needs_visual: bool = Field(
        default=False,
        description="Requires diagram/figure evidence"
    )
    
    # Learner state
    mastery: dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="LO mastery estimates (0-1 scale, None=unknown)"
    )
    
    # Processing metadata
    language: Language = Field(default="en", description="Query language")
    ambiguous: bool = Field(default=False, description="Requires clarification")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator("mastery")
    @classmethod
    def validate_mastery(cls, v: dict) -> dict:
        """Ensure mastery values are in valid range."""
        for lo_id, score in v.items():
            if score is not None and not (0 <= score <= 1):
                raise ValueError(f"Mastery score for {lo_id} must be between 0 and 1")
        return v


class Edge(BaseModel):
    """
    Prerequisite relationship between learning outcomes.
    
    Must be expert-reviewed before use in production.
    """
    
    prerequisite_lo: str = Field(..., description="Prerequisite LO identifier")
    target_lo: str = Field(..., description="Target LO identifier")
    
    # Review metadata
    review_status: ReviewStatus = Field(
        default="pending",
        description="Expert review status"
    )
    reviewer_id: Optional[str] = Field(None, description="Reviewer identifier")
    review_date: Optional[datetime] = Field(None, description="Review timestamp")
    
    # Relationship metadata
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Prerequisite strength (0=weak, 1=strong)"
    )
    justification: Optional[str] = Field(
        None,
        description="Rationale for prerequisite relationship"
    )
    
    @model_validator(mode="after")
    def no_self_loops(self):
        """Prevent LO from being its own prerequisite."""
        if self.prerequisite_lo == self.target_lo:
            raise ValueError("Learning outcome cannot be its own prerequisite")
        return self


class Answer(BaseModel):
    """
    Generated answer with complete provenance and verification.
    """
    
    # Query reference
    query_id: str = Field(..., description="Source query identifier")
    
    # Answer content
    answer: str = Field(..., description="Generated answer text")
    explanation: Optional[str] = Field(None, description="Additional explanation")
    next_step: Optional[str] = Field(None, description="Suggested next action")
    
    # Multimodal output
    audio_path: Optional[str] = Field(None, description="Path to TTS audio file")
    audio_language: Optional[str] = Field(None, description="Audio output language")
    visual_evidence_paths: list[str] = Field(default_factory=list, description="Paths to visual evidence")
    
    # Evidence provenance
    evidence_bundle_ids: list[str] = Field(
        ...,
        description="IDs of evidence bundles used"
    )
    source_pages: list[int] = Field(
        default_factory=list,
        description="Source page numbers"
    )
    
    # Verification
    verified: bool = Field(default=False, description="Passed verification")
    support_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Claim support confidence"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims that lack support"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Contradictory statements"
    )
    
    # Rendering
    mode: str = Field(default="standard", description="Rendering mode")
    language: Language = Field(default="en", description="Answer language")
    
    # Metadata
    model_name: str = Field(..., description="Generator model identifier")
    model_version: Optional[str] = Field(None, description="Model version")
    generation_time: float = Field(..., description="Generation time in seconds")
    total_time: float = Field(..., description="Total pipeline time in seconds")
    retrieval_time: float = Field(default=0.0, description="Retrieval time in seconds")
    selection_time: float = Field(default=0.0, description="Selection time in seconds")
    cached: bool = Field(default=False, description="Whether this was a cache hit")
    
    # Status
    status: Literal["answer", "fallback", "clarify", "abstain"] = Field(
        default="answer",
        description="Response status"
    )
    fallback_reason: Optional[str] = Field(None, description="Why fallback was triggered")
    
    timestamp: datetime = Field(default_factory=datetime.now)


class AnnotationLabel(BaseModel):
    """
    Human annotation for evaluation and training.
    """
    
    query_id: str
    evidence_id: str
    
    # Relevance judgment
    relevance: int = Field(..., ge=0, le=3, description="0=irrelevant, 3=essential")
    
    # Role coverage
    satisfies_roles: set[Role] = Field(default_factory=set)
    
    # Quality judgments
    factually_correct: Optional[bool] = None
    grade_appropriate: Optional[bool] = None
    pedagogically_complete: Optional[bool] = None
    
    # Annotator
    annotator_id: str
    annotation_time: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None


class ExperimentConfig(BaseModel):
    """
    Complete configuration for reproducible experiment.
    """
    
    experiment_name: str
    config_file: str
    seed: int = 42
    
    # Corpus
    corpus_snapshot: str  # Hash or version
    num_bundles: int
    num_questions: int
    
    # Models
    dense_model: str
    reranker_model: Optional[str] = None
    generator_model: str
    verifier_model: Optional[str] = None
    
    # Hyperparameters
    max_bundles: int = 3
    token_budget: int = 420
    fusion_k: int = 60
    verification_threshold: float = 0.75
    
    # Ablation settings
    use_bundles: bool = True
    use_roles: bool = True
    use_prerequisites: bool = True
    use_verification: bool = True
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "experiment_name": "baseline_b1",
                "config_file": "configs/baseline_hybrid.yaml",
                "seed": 42,
                "corpus_snapshot": "ncert_evs_g345_20260105",
                "num_bundles": 2000,
                "num_questions": 300,
                "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                "generator_model": "phi3:3b-mini-instruct",
                "max_bundles": 3,
            }
        }
