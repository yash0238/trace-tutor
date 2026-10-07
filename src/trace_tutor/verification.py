"""
Claim support verification using NLI (Natural Language Inference).

Checks whether generated answer claims are supported by the evidence.
"""

import logging
from typing import List, Tuple, Dict, Optional
import re

from trace_tutor.schema import Evidence, Answer


logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extract atomic claims from generated answers."""
    
    def __init__(self, method: str = "sentence"):
        """
        Initialize claim extractor.
        
        Args:
            method: Extraction method ("sentence" for simple splitting)
        """
        self.method = method
    
    def extract_claims(self, text: str, min_length: int = 10) -> List[str]:
        """
        Extract atomic claims from text.
        
        Simple implementation: split by sentences.
        More sophisticated: use dependency parsing or LLM-based extraction.
        
        Args:
            text: Answer text
            min_length: Minimum claim length in characters
        
        Returns:
            List of claim strings
        """
        if self.method == "sentence":
            # Split on sentence boundaries
            sentences = re.split(r'[.!?]+', text)
            claims = [
                s.strip()
                for s in sentences
                if len(s.strip()) >= min_length
            ]
            return claims
        
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")


class NLIVerifier:
    """
    Natural Language Inference verifier.
    
    Uses a pre-trained NLI model to check if evidence supports claims.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-mnli",
        device: str = "cpu",
        threshold: float = 0.75
    ):
        """
        Initialize NLI verifier.
        
        Args:
            model_name: HuggingFace NLI model
            device: 'cpu' or 'cuda'
            threshold: Confidence threshold for "entailment"
        """
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        
        logger.info(f"Loading NLI model: {model_name}")
        
        try:
            from transformers import pipeline
            self.nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if device == "cuda" else -1
            )
            logger.info("NLI model loaded")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            self.nli_pipeline = None
    
    def check_support(
        self,
        claim: str,
        evidence_texts: List[str]
    ) -> Tuple[float, str]:
        """
        Check if evidence supports a claim.
        
        Args:
            claim: Claim to verify
            evidence_texts: List of evidence texts
        
        Returns:
            (support_score, label) where label is "supported", "contradicted", or "neutral"
        """
        if not self.nli_pipeline:
            return 0.5, "unknown"
        
        max_entailment = 0.0
        max_contradiction = 0.0
        
        for evidence_text in evidence_texts:
            # Truncate long evidence
            premise = evidence_text[:512]
            hypothesis = claim
            
            try:
                result = self.nli_pipeline(
                    f"{premise} [SEP] {hypothesis}",
                    truncation=True,
                    max_length=512
                )[0]
                
                label = result['label'].lower()
                score = result['score']
                
                if 'entail' in label:
                    max_entailment = max(max_entailment, score)
                elif 'contradict' in label:
                    max_contradiction = max(max_contradiction, score)
            
            except Exception as e:
                logger.warning(f"NLI inference failed: {e}")
                continue
        
        # Determine final label
        if max_entailment >= self.threshold:
            return max_entailment, "supported"
        elif max_contradiction >= self.threshold:
            return max_contradiction, "contradicted"
        else:
            return max(max_entailment, 0.0), "neutral"
    
    def verify_answer(
        self,
        answer: Answer,
        evidence_bundles: List[Evidence]
    ) -> Dict:
        """
        Verify complete answer against evidence.
        
        Args:
            answer: Generated answer
            evidence_bundles: Evidence bundles used
        
        Returns:
            Verification result dictionary
        """
        # Extract claims
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(answer.answer)
        
        if not claims:
            return {
                "verified": False,
                "support_score": 0.0,
                "num_claims": 0,
                "supported_claims": [],
                "unsupported_claims": [],
                "contradictions": [],
                "reason": "no_claims_extracted"
            }
        
        # Get evidence texts
        evidence_texts = [b.text for b in evidence_bundles]
        
        # Check each claim
        supported = []
        unsupported = []
        contradicted = []
        scores = []
        
        for claim in claims:
            score, label = self.check_support(claim, evidence_texts)
            scores.append(score)
            
            if label == "supported":
                supported.append(claim)
            elif label == "contradicted":
                contradicted.append(claim)
            else:
                unsupported.append(claim)
        
        # Compute overall support score
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine if answer is verified
        verified = (
            len(contradicted) == 0 and
            len(supported) >= len(claims) * 0.7 and  # At least 70% supported
            overall_score >= self.threshold * 0.8  # Slightly relaxed threshold
        )
        
        return {
            "verified": verified,
            "support_score": overall_score,
            "num_claims": len(claims),
            "num_supported": len(supported),
            "num_unsupported": len(unsupported),
            "num_contradicted": len(contradicted),
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "contradictions": contradicted,
            "threshold": self.threshold
        }


class Verifier:
    """
    High-level verifier interface.
    
    Coordinates claim extraction and NLI verification.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-mnli",
        device: str = "cpu",
        threshold: float = 0.75
    ):
        """Initialize verifier."""
        self.nli_verifier = NLIVerifier(
            model_name=model_name,
            device=device,
            threshold=threshold
        )
        self.threshold = threshold
    
    def verify(
        self,
        answer: Answer,
        evidence_bundles: List[Evidence]
    ) -> Answer:
        """
        Verify answer and update Answer object.
        
        Args:
            answer: Generated answer
            evidence_bundles: Evidence used
        
        Returns:
            Updated answer with verification results
        """
        result = self.nli_verifier.verify_answer(answer, evidence_bundles)
        
        # Update answer object
        answer.verified = result["verified"]
        answer.support_score = result["support_score"]
        answer.unsupported_claims = result.get("unsupported_claims", [])
        answer.contradictions = result.get("contradictions", [])
        
        # Update status if verification fails
        if not result["verified"]:
            if result.get("num_contradicted", 0) > 0:
                answer.status = "fallback"
                answer.fallback_reason = "contradiction_detected"
            elif result.get("support_score", 0.0) < self.threshold:
                answer.status = "fallback"
                answer.fallback_reason = "low_support"
        
        return answer
    
    def calibrate(
        self,
        dev_data: List[Tuple[Answer, List[Evidence], bool]]
    ) -> float:
        """
        Calibrate threshold on development data.
        
        Args:
            dev_data: List of (answer, evidence, is_correct) tuples
        
        Returns:
            Optimal threshold
        """
        # Try different thresholds
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        best_threshold = 0.75
        best_f1 = 0.0
        
        for threshold in thresholds:
            self.nli_verifier.threshold = threshold
            
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0
            
            for answer, evidence, is_correct in dev_data:
                result = self.nli_verifier.verify_answer(answer, evidence)
                verified = result["verified"]
                
                if verified and is_correct:
                    true_pos += 1
                elif verified and not is_correct:
                    false_pos += 1
                elif not verified and not is_correct:
                    true_neg += 1
                else:
                    false_neg += 1
            
            # Compute F1
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.info(f"Threshold {threshold}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold} (F1={best_f1:.3f})")
        self.threshold = best_threshold
        self.nli_verifier.threshold = best_threshold
        
        return best_threshold


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Verification module ready.")
    print("Uses NLI model for claim support checking.")
