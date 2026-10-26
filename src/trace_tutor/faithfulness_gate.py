"""
Faithfulness Gate Module

Verifies that generated answers are supported by retrieved evidence.
Decomposes answers into atomic claims and checks entailment.
"""

import logging
from typing import List, Dict, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class FaithfulnessGate:
    """
    Verify answer faithfulness to evidence.
    
    Prevents hallucinations by checking claim support.
    """
    
    def __init__(self, threshold: float = 0.7, use_llm: bool = False):
        """
        Initialize faithfulness gate.
        
        Args:
            threshold: Minimum support score to pass (0-1)
            use_llm: Whether to use LLM for claim verification (more accurate but slower)
        """
        self.threshold = threshold
        self.use_llm = use_llm
        
        logger.info(f"FaithfulnessGate initialized (threshold={threshold}, use_llm={use_llm})")
    
    def verify(self, answer: str, evidence_list: List[str]) -> Dict:
        """
        Verify answer against evidence.
        
        Args:
            answer: Generated answer text
            evidence_list: List of evidence texts
            
        Returns:
            Dict with {
                'passed': bool,
                'support_score': float,
                'claims': List[str],
                'supported_claims': List[str],
                'unsupported_claims': List[str]
            }
        """
        # Decompose answer into atomic claims
        claims = self._decompose_claims(answer)
        
        if not claims:
            logger.warning("No claims extracted from answer")
            return {
                'passed': True,  # Empty answer is technically faithful
                'support_score': 1.0,
                'claims': [],
                'supported_claims': [],
                'unsupported_claims': []
            }
        
        # Combine all evidence
        combined_evidence = " ".join(evidence_list)
        
        # Check each claim
        supported = []
        unsupported = []
        
        for claim in claims:
            if self._is_supported(claim, combined_evidence):
                supported.append(claim)
            else:
                unsupported.append(claim)
        
        # Calculate support score
        support_score = len(supported) / len(claims) if claims else 1.0
        
        result = {
            'passed': support_score >= self.threshold,
            'support_score': support_score,
            'claims': claims,
            'supported_claims': supported,
            'unsupported_claims': unsupported
        }
        
        logger.info(f"Faithfulness check: {len(supported)}/{len(claims)} claims supported (score={support_score:.2f})")
        
        return result
    
    def _decompose_claims(self, answer: str) -> List[str]:
        """
        Decompose answer into atomic claims.
        
        Uses simple sentence splitting. For better results, use LLM.
        """
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', answer)
        
        # Clean and filter
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Filter very short fragments
                claims.append(sent)
        
        return claims
    
    def _is_supported(self, claim: str, evidence: str) -> bool:
        """
        Check if claim is supported by evidence.
        
        Uses keyword overlap heuristic. For better results, use NLI model.
        """
        if self.use_llm:
            return self._check_support_llm(claim, evidence)
        else:
            return self._check_support_heuristic(claim, evidence)
    
    def _check_support_heuristic(self, claim: str, evidence: str) -> bool:
        """
        Heuristic-based support check using keyword overlap.
        """
        # Extract keywords from claim (nouns, numbers, key terms)
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract important words (simple: length > 4, not common words)
        stop_words = {'the', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'what', 'when', 'where'}
        
        claim_words = [w for w in re.findall(r'\b\w+\b', claim_lower) 
                      if len(w) > 3 and w not in stop_words]
        
        if not claim_words:
            return True  # Empty claim is supported
        
        # Check overlap
        matches = sum(1 for word in claim_words if word in evidence_lower)
        overlap_ratio = matches / len(claim_words)
        
        # Require at least 60% of keywords present
        return overlap_ratio >= 0.6
    
    def _check_support_llm(self, claim: str, evidence: str) -> bool:
        """
        LLM-based support check (more accurate).
        
        Uses simple prompt to check entailment.
        """
        # This would use an LLM API call
        # For now, fallback to heuristic
        logger.warning("LLM-based verification not implemented, using heuristic")
        return self._check_support_heuristic(claim, evidence)
    
    def generate_fallback(self, unsupported_claims: List[str]) -> str:
        """
        Generate fallback message when faithfulness check fails.
        
        Args:
            unsupported_claims: List of claims that lack support
            
        Returns:
            Fallback message for user
        """
        fallback = (
            "I'm not completely sure about all the details in your question. "
            "Let's ask your teacher for help to make sure you get the right answer. "
        )
        
        if len(unsupported_claims) > 0:
            fallback += f"(I couldn't verify {len(unsupported_claims)} part(s) of the answer from the textbook.)"
        
        return fallback


class AnswerValidator:
    """
    Complete answer validation pipeline.
    
    Combines faithfulness, safety, and quality checks.
    """
    
    def __init__(self, faithfulness_threshold: float = 0.7):
        """
        Initialize validator.
        
        Args:
            faithfulness_threshold: Minimum faithfulness score
        """
        self.faithfulness_gate = FaithfulnessGate(threshold=faithfulness_threshold)
    
    def validate(self, answer: str, evidence_list: List[str], grade: int) -> Dict:
        """
        Validate answer.
        
        Args:
            answer: Generated answer
            evidence_list: Supporting evidence texts
            grade: Student grade level
            
        Returns:
            Validation result with status and feedback
        """
        result = {
            'valid': True,
            'confidence': 1.0,
            'issues': [],
            'fallback': None
        }
        
        # Check faithfulness
        faithfulness_result = self.faithfulness_gate.verify(answer, evidence_list)
        
        if not faithfulness_result['passed']:
            result['valid'] = False
            result['confidence'] = faithfulness_result['support_score']
            result['issues'].append(f"Low faithfulness: {faithfulness_result['support_score']:.0%}")
            result['fallback'] = self.faithfulness_gate.generate_fallback(
                faithfulness_result['unsupported_claims']
            )
        
        # Check length (grade-appropriate)
        word_count = len(answer.split())
        max_words = {1: 50, 2: 75, 3: 100, 4: 125}.get(grade, 150)
        
        if word_count > max_words * 1.5:
            result['issues'].append(f"Answer too long for grade {grade}")
        
        # Check for hedging language (signs of uncertainty)
        hedges = ['maybe', 'possibly', 'might be', 'could be', 'not sure', 'I think']
        hedge_count = sum(1 for hedge in hedges if hedge in answer.lower())
        
        if hedge_count > 2:
            result['issues'].append("Answer contains uncertain language")
            result['confidence'] *= 0.9
        
        return result


def apply_faithfulness_gate(answer_obj, evidence_bundles: List, threshold: float = 0.7):
    """
    Apply faithfulness gate to Answer object.
    
    Updates answer object with verification results.
    
    Args:
        answer_obj: Answer object to validate
        evidence_bundles: List of Evidence objects
        threshold: Minimum support threshold
    """
    gate = FaithfulnessGate(threshold=threshold)
    
    # Extract evidence texts
    evidence_texts = [e.text for e in evidence_bundles]
    
    # Verify
    result = gate.verify(answer_obj.answer, evidence_texts)
    
    # Update answer object
    answer_obj.verified = result['passed']
    answer_obj.support_score = result['support_score']
    answer_obj.unsupported_claims = result['unsupported_claims']
    
    # If failed, update status
    if not result['passed']:
        answer_obj.status = "fallback"
        answer_obj.fallback_reason = f"Low faithfulness ({result['support_score']:.0%})"
    
    return answer_obj
