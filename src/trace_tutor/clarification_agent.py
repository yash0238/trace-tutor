"""
Clarification Agent Module

Handles ambiguous queries by asking follow-up questions.
Implements the ask_clarify action from the reasoning-agentic retrieval spec.
"""

import logging
from typing import List, Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class ClarificationAgent:
    """
    Agent that detects ambiguous queries and asks clarifying questions.
    
    Implements policy: retrieve | ask_clarify | answer | stop
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize clarification agent.
        
        Args:
            confidence_threshold: Below this, ask for clarification
        """
        self.confidence_threshold = confidence_threshold
        
        # Ambiguity patterns
        self.ambiguity_indicators = [
            r'\b(it|this|that|the thing)\b',  # Vague references
            r'\b(stuff|things)\b',  # Vague nouns
            r'^(what|which|how)\s+\w+\s*\??$',  # Very short questions
        ]
        
        logger.info(f"ClarificationAgent ready (threshold={confidence_threshold})")
    
    def analyze_query(self, query_text: str, grade: int, subject: str) -> Dict:
        """
        Analyze query for ambiguity.
        
        Args:
            query_text: Student question
            grade: Grade level
            subject: Subject area
            
        Returns:
            Dict with {
                'is_ambiguous': bool,
                'confidence': float,
                'ambiguity_type': str,
                'suggested_clarifications': List[str]
            }
        """
        result = {
            'is_ambiguous': False,
            'confidence': 1.0,
            'ambiguity_type': None,
            'suggested_clarifications': []
        }
        
        query_lower = query_text.lower().strip()
        
        # Check 1: Very short query
        word_count = len(query_text.split())
        if word_count < 3:
            result['is_ambiguous'] = True
            result['confidence'] = 0.3
            result['ambiguity_type'] = 'too_vague'
            result['suggested_clarifications'] = [
                f"Can you be more specific about what you want to know in {subject}?",
                "What exactly would you like to learn about?"
            ]
            return result
        
        # Check 2: Vague references
        for pattern in self.ambiguity_indicators:
            if re.search(pattern, query_lower):
                result['is_ambiguous'] = True
                result['confidence'] = 0.5
                result['ambiguity_type'] = 'vague_reference'
                
                # Generate context-specific clarifications
                result['suggested_clarifications'] = self._generate_clarifications(
                    query_text, subject, grade
                )
                return result
        
        # Check 3: Multiple possible interpretations
        multi_meaning_keywords = self._detect_multi_meaning(query_lower, subject)
        if multi_meaning_keywords:
            result['is_ambiguous'] = True
            result['confidence'] = 0.4
            result['ambiguity_type'] = 'multiple_meanings'
            result['suggested_clarifications'] = [
                f"Do you mean {meaning}?" for meaning in multi_meaning_keywords
            ]
            return result
        
        # Query seems clear
        result['confidence'] = 0.9
        return result
    
    def _generate_clarifications(self, query: str, subject: str, grade: int) -> List[str]:
        """Generate context-specific clarification questions."""
        clarifications = []
        
        # Subject-specific templates
        if 'science' in subject.lower():
            if 'how' in query.lower():
                clarifications.append("Do you want to know how the process works step-by-step?")
                clarifications.append("Or do you want to know why it happens?")
            elif 'what' in query.lower():
                clarifications.append("Are you asking about the definition?")
                clarifications.append("Or are you asking about examples?")
        
        elif 'math' in subject.lower():
            clarifications.append("Do you need help understanding the concept?")
            clarifications.append("Or do you need help solving a problem?")
        
        else:
            clarifications.append("Can you provide more details about what you want to learn?")
            clarifications.append(f"Which part of {subject} is your question about?")
        
        return clarifications[:3]  # Limit to 3
    
    def _detect_multi_meaning(self, query: str, subject: str) -> List[str]:
        """Detect words/phrases with multiple meanings in context."""
        multi_meaning_terms = {
            'plant': ['living plant organism', 'factory/plant facility'],
            'cell': ['biological cell', 'phone cell', 'prison cell'],
            'wave': ['ocean wave', 'light wave', 'sound wave', 'hand gesture'],
            'bank': ['river bank', 'financial bank'],
            'state': ['physical state (solid/liquid)', 'government state', 'condition'],
            'current': ['water/ocean current', 'electrical current', 'present time'],
        }
        
        found_terms = []
        for term, meanings in multi_meaning_terms.items():
            if term in query.lower():
                # Filter meanings relevant to subject
                relevant = self._filter_by_subject(meanings, subject)
                if len(relevant) > 1:
                    found_terms.extend(relevant[:2])
        
        return found_terms[:3]
    
    def _filter_by_subject(self, meanings: List[str], subject: str) -> List[str]:
        """Filter meanings relevant to subject."""
        subject_lower = subject.lower()
        
        if 'science' in subject_lower or 'biology' in subject_lower:
            return [m for m in meanings if any(kw in m.lower() 
                   for kw in ['cell', 'plant', 'wave', 'organism', 'biological'])]
        elif 'physics' in subject_lower:
            return [m for m in meanings if any(kw in m.lower() 
                   for kw in ['wave', 'current', 'electrical', 'light'])]
        else:
            return meanings
    
    def should_clarify(self, analysis: Dict) -> bool:
        """
        Decide whether to ask for clarification.
        
        Args:
            analysis: Result from analyze_query
            
        Returns:
            True if should ask clarification
        """
        return (analysis['is_ambiguous'] and 
                analysis['confidence'] < self.confidence_threshold)
    
    def format_clarification_response(self, analysis: Dict, grade: int) -> str:
        """
        Format clarification question for user.
        
        Args:
            analysis: Query analysis result
            grade: Grade level (affects language complexity)
            
        Returns:
            Formatted clarification message
        """
        if grade <= 2:
            intro = "I want to help you! Can you tell me more? "
        elif grade <= 4:
            intro = "I need a bit more information to help you better. "
        else:
            intro = "To give you the best answer, I need to understand your question better. "
        
        clarifications = analysis['suggested_clarifications']
        
        if not clarifications:
            return intro + "Can you explain what you'd like to know?"
        
        if len(clarifications) == 1:
            return intro + clarifications[0]
        else:
            # Multiple choice format
            response = intro + "\n\n"
            for i, clarification in enumerate(clarifications, 1):
                response += f"{i}. {clarification}\n"
            return response


class DialogueManager:
    """
    Manages multi-turn clarification dialogues.
    
    Tracks conversation history and context.
    """
    
    def __init__(self):
        """Initialize dialogue manager."""
        self.history = []
        self.current_clarification = None
    
    def add_turn(self, role: str, message: str):
        """
        Add a turn to conversation history.
        
        Args:
            role: 'user' or 'agent'
            message: Message text
        """
        self.history.append({
            'role': role,
            'message': message,
            'timestamp': None  # Could add datetime
        })
    
    def needs_clarification(self, query: str, grade: int, subject: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query needs clarification.
        
        Args:
            query: User query
            grade: Grade level
            subject: Subject
            
        Returns:
            (needs_clarification, clarification_message)
        """
        agent = ClarificationAgent()
        analysis = agent.analyze_query(query, grade, subject)
        
        if agent.should_clarify(analysis):
            clarification = agent.format_clarification_response(analysis, grade)
            self.current_clarification = analysis
            return True, clarification
        
        return False, None
    
    def resolve_clarification(self, user_response: str) -> str:
        """
        Process user's clarification response.
        
        Args:
            user_response: User's follow-up
            
        Returns:
            Refined query
        """
        if not self.current_clarification:
            return user_response
        
        # Simple: append to original query
        original_query = self.history[-2]['message'] if len(self.history) >= 2 else ""
        refined = f"{original_query} {user_response}"
        
        self.current_clarification = None
        return refined
    
    def get_context(self) -> str:
        """Get conversation context as string."""
        return "\n".join([f"{turn['role']}: {turn['message']}" for turn in self.history[-5:]])


def check_and_clarify(query_text: str, grade: int, subject: str) -> Dict:
    """
    Main entry point: check if clarification needed.
    
    Args:
        query_text: User query
        grade: Grade level
        subject: Subject
        
    Returns:
        Dict with {
            'needs_clarification': bool,
            'clarification_message': str or None,
            'original_query': str,
            'confidence': float
        }
    """
    agent = ClarificationAgent()
    analysis = agent.analyze_query(query_text, grade, subject)
    
    result = {
        'needs_clarification': agent.should_clarify(analysis),
        'clarification_message': None,
        'original_query': query_text,
        'confidence': analysis['confidence']
    }
    
    if result['needs_clarification']:
        result['clarification_message'] = agent.format_clarification_response(analysis, grade)
    
    return result
