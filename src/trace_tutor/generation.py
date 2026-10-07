"""
Grounded answer generation with local LLM support.

Supports both local models (Ollama, llama.cpp) and API-based generation
with evidence-limited prompting.
"""

import logging
import time
from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
import os

from trace_tutor.schema import Evidence, Query, Answer


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""
    provider: str = "ollama"  # "ollama", "openai", "anthropic", "local"
    model: str = "phi3:3b-mini-instruct"
    temperature: float = 0.3
    max_tokens: int = 256
    top_p: float = 0.9
    
    # API settings
    api_key_env: str = "LLM_API_KEY"
    api_model: str = "gpt-3.5-turbo"
    
    # Prompt settings
    include_source_ids: bool = True
    include_grade: bool = True
    include_language: bool = True


class Generator(Protocol):
    """Protocol for answer generators."""
    
    def generate(
        self,
        question: str,
        evidence: List[Evidence],
        grade: int,
        language: str = "en"
    ) -> tuple[str, Dict]:
        """
        Generate answer from evidence.
        
        Returns:
            (answer_text, metadata)
        """
        ...


class OllamaGenerator:
    """
    Generator using Ollama for local model inference.
    
    Ollama provides easy CPU/GPU inference for Llama, Phi, Qwen, etc.
    Install: https://ollama.ai
    """
    
    def __init__(self, config: GenerationConfig):
        """Initialize Ollama generator."""
        self.config = config
        
        try:
            import ollama
            self.client = ollama.Client()
            # Test connection
            self.client.list()
            logger.info(f"Ollama connected, using model: {config.model}")
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            logger.info("Install Ollama from https://ollama.ai and run: ollama pull phi3:3b-mini-instruct")
            raise
    
    def generate(
        self,
        question: str,
        evidence: List[Evidence],
        grade: int,
        language: str = "en"
    ) -> tuple[str, Dict]:
        """Generate answer using Ollama."""
        import ollama
        
        # Build prompt
        prompt = self._build_prompt(question, evidence, grade, language)
        
        start_time = time.time()
        
        try:
            response = self.client.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                }
            )
            
            answer = response['response'].strip()
            generation_time = time.time() - start_time
            
            metadata = {
                "model": self.config.model,
                "provider": "ollama",
                "generation_time": generation_time,
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0)
            }
            
            return answer, metadata
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"[Generation error: {str(e)}]", {"error": str(e)}
    
    def _build_prompt(
        self,
        question: str,
        evidence: List[Evidence],
        grade: int,
        language: str
    ) -> str:
        """Build evidence-grounded prompt."""
        prompt_parts = []
        
        # System instruction
        prompt_parts.append(
            "You are a grade-aligned tutor. Use ONLY the supplied evidence below. "
            "Do not invent facts, examples, or sources. "
            "If the evidence is insufficient, say so clearly. "
            "Answer briefly and appropriately for the grade level."
        )
        
        # Grade context
        prompt_parts.append(f"\nTarget grade: {grade}")
        prompt_parts.append(f"Language: {language}")
        
        # Evidence
        prompt_parts.append("\n**Evidence:**")
        for idx, bundle in enumerate(evidence, 1):
            prompt_parts.append(f"\n[Evidence {idx}] (Source: Page {bundle.page})")
            prompt_parts.append(bundle.text[:400])  # Truncate if very long
            
            if bundle.figure_caption:
                prompt_parts.append(f"[Figure: {bundle.figure_caption}]")
        
        # Question
        prompt_parts.append(f"\n**Question:** {question}")
        
        # Output format
        prompt_parts.append(
            "\n**Instructions:** Provide a brief answer based only on the evidence above. "
            "Include one example if helpful. Keep language simple for the grade level."
        )
        
        return "\n".join(prompt_parts)


class OpenAIGenerator:
    """Generator using OpenAI API."""
    
    def __init__(self, config: GenerationConfig):
        """Initialize OpenAI generator."""
        self.config = config
        
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {config.api_key_env}")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI initialized, model: {config.api_model}")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
    
    def generate(
        self,
        question: str,
        evidence: List[Evidence],
        grade: int,
        language: str = "en"
    ) -> tuple[str, Dict]:
        """Generate answer using OpenAI API."""
        messages = self._build_messages(question, evidence, grade, language)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.api_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            answer = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            
            metadata = {
                "model": self.config.api_model,
                "provider": "openai",
                "generation_time": generation_time,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return answer, metadata
        
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return f"[Generation error: {str(e)}]", {"error": str(e)}
    
    def _build_messages(
        self,
        question: str,
        evidence: List[Evidence],
        grade: int,
        language: str
    ) -> List[Dict]:
        """Build messages for chat API."""
        system_msg = (
            "You are a grade-aligned tutor. Use ONLY the supplied evidence. "
            "Do not invent facts. If evidence is insufficient, say so clearly."
        )
        
        evidence_text = "\n\n".join([
            f"[Evidence {i+1}] (Page {b.page})\n{b.text[:400]}"
            for i, b in enumerate(evidence)
        ])
        
        user_msg = (
            f"Grade: {grade}\n"
            f"Language: {language}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Question: {question}\n\n"
            f"Provide a brief, grade-appropriate answer based only on the evidence."
        )
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]


class GroundedGenerator:
    """
    Main interface for grounded answer generation.
    
    Handles provider selection and evidence formatting.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize generator with config."""
        self.config = config or GenerationConfig()
        
        # Initialize appropriate backend
        if self.config.provider == "ollama":
            self.backend = OllamaGenerator(self.config)
        elif self.config.provider == "openai":
            self.backend = OpenAIGenerator(self.config)
        elif self.config.provider == "anthropic":
            # TODO: Implement AnthropicGenerator
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def generate_answer(
        self,
        query: Query,
        evidence_bundles: List[Evidence]
    ) -> Answer:
        """
        Generate complete Answer object from query and evidence.
        
        Args:
            query: Student query
            evidence_bundles: Selected evidence bundles
        
        Returns:
            Answer object with generation and provenance
        """
        if not evidence_bundles:
            return Answer(
                query_id=query.id,
                answer="I don't have enough information to answer this question.",
                evidence_bundle_ids=[],
                source_pages=[],
                verified=False,
                support_score=0.0,
                model_name=self.config.model,
                generation_time=0.0,
                total_time=0.0,
                status="fallback",
                fallback_reason="no_evidence"
            )
        
        total_start = time.time()
        
        # Generate answer
        answer_text, metadata = self.backend.generate(
            question=query.text,
            evidence=evidence_bundles,
            grade=query.grade,
            language=query.language
        )
        
        total_time = time.time() - total_start
        
        # Extract provenance
        bundle_ids = [b.id for b in evidence_bundles]
        source_pages = sorted(set(b.page for b in evidence_bundles))
        
        return Answer(
            query_id=query.id,
            answer=answer_text,
            evidence_bundle_ids=bundle_ids,
            source_pages=source_pages,
            verified=False,  # Will be set by verifier
            support_score=0.0,  # Will be set by verifier
            model_name=metadata.get("model", self.config.model),
            model_version=metadata.get("version"),
            generation_time=metadata.get("generation_time", 0.0),
            total_time=total_time,
            mode="standard",
            language=query.language,
            status="answer"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Generation module ready.")
    print("Supported providers: ollama (default), openai")
    print("Set LLM_API_KEY environment variable for API providers.")
