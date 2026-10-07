"""
TRACE-Tutor Production API Server

FastAPI-based REST API for enterprise deployment.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
from datetime import datetime
import logging

from trace_tutor.tutor import TRACETutor
from trace_tutor.schema import Answer


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="TRACE-Tutor API",
    description="Curriculum-Grounded AI Tutoring System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global tutor instance
tutor = None


# API Models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Student question")
    grade: int = Field(..., ge=3, le=8, description="Student grade level")
    subject: str = Field(..., description="Subject (Mathematics or Science)")
    query_type: str = Field(default="conceptual", description="Question type")
    needs_visual: bool = Field(default=False, description="Requires diagrams")
    student_id: Optional[str] = Field(None, description="Student identifier (pseudonymous)")


class AnswerResponse(BaseModel):
    query_id: str
    answer: str
    explanation: Optional[str] = None
    evidence_bundle_ids: List[str]
    source_pages: List[int]
    verified: bool
    support_score: float
    status: str
    generation_time: float
    total_time: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    bundles_loaded: int
    graph_edges: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize TRACE-Tutor on startup."""
    global tutor
    
    try:
        logger.info("Initializing TRACE-Tutor...")
        tutor = TRACETutor(
            config_path="configs/study.yaml",
            bundles_path="data/bundles/corpus_all.jsonl",
            graph_path="data/graph/prerequisites.jsonl"
        )
        logger.info(f"✓ Loaded {len(tutor.bundles)} evidence bundles")
        logger.info(f"✓ Loaded {tutor.graph.graph.number_of_edges()} prerequisite edges")
        logger.info("✓ TRACE-Tutor API ready")
    
    except Exception as e:
        logger.error(f"Failed to initialize TRACE-Tutor: {e}")
        tutor = None


# API Endpoints
@app.get("/", tags=["System"])
async def root():
    """Root endpoint."""
    return {
        "name": "TRACE-Tutor API",
        "version": "1.0.0",
        "status": "operational" if tutor else "initializing",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if not tutor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0,  # TODO: Track actual uptime
        bundles_loaded=len(tutor.bundles),
        graph_edges=tutor.graph.graph.number_of_edges()
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Tutoring"])
async def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Ask a question and get a grounded answer.
    
    This endpoint:
    1. Retrieves relevant curriculum evidence
    2. Selects minimal pedagogically-complete set
    3. Generates grounded answer
    4. Verifies claim support
    5. Returns answer with complete provenance
    """
    if not tutor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    try:
        # Get answer
        response = tutor.ask(
            question=request.question,
            grade=request.grade,
            subject=request.subject,
            query_type=request.query_type
        )
        
        # Convert to API response
        return AnswerResponse(
            query_id=response.query_id,
            answer=response.answer,
            explanation=response.explanation,
            evidence_bundle_ids=response.evidence_bundle_ids,
            source_pages=response.source_pages,
            verified=response.verified,
            support_score=response.support_score,
            status=response.status,
            generation_time=response.generation_time,
            total_time=response.total_time,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subjects", tags=["Metadata"])
async def get_subjects():
    """Get available subjects."""
    if not tutor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    subjects = set(b.subject for b in tutor.bundles)
    return {"subjects": list(subjects)}


@app.get("/grades", tags=["Metadata"])
async def get_grades():
    """Get available grade levels."""
    if not tutor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    grades = sorted(set(b.grade for b in tutor.bundles))
    return {"grades": grades}


@app.get("/stats", tags=["Metadata"])
async def get_statistics():
    """Get system statistics."""
    if not tutor:
        raise HTTPException(status_code=503, detail="System not ready")
    
    # Compute statistics
    subjects = {}
    grades = {}
    roles = {}
    
    for bundle in tutor.bundles:
        subjects[bundle.subject] = subjects.get(bundle.subject, 0) + 1
        grades[bundle.grade] = grades.get(bundle.grade, 0) + 1
        for role in bundle.roles:
            roles[role] = roles.get(role, 0) + 1
    
    return {
        "total_bundles": len(tutor.bundles),
        "by_subject": subjects,
        "by_grade": grades,
        "by_role": roles,
        "graph_statistics": tutor.graph.get_statistics()
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
