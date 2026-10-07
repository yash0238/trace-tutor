"""
PS-Evidence: Pedagogically Sufficient Evidence Selection.

Constrained optimization for minimal, role-complete evidence sets.
This is the core research contribution of TRACE-Tutor.
"""

import logging
from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations
from dataclasses import dataclass

import numpy as np

from trace_tutor.schema import Evidence, Query, Role


logger = logging.getLogger(__name__)


# Role policies for different question types
ROLE_POLICY: Dict[str, Set[Role]] = {
    "direct_fact": {"concept"},
    "conceptual": {"concept", "example"},
    "diagram": {"concept", "visual"},
    "worked_problem": {"example"},
    "misconception": {"concept", "misconception"},
    "prerequisite_gap": {"concept", "prerequisite"},
}


@dataclass
class SelectionConfig:
    """Configuration for PS-Evidence selector."""
    max_bundles: int = 3
    token_budget: int = 420
    
    # Optimization weights
    relevance_weight: float = 1.0
    role_coverage_weight: float = 0.3
    grade_fit_weight: float = 0.2
    prerequisite_value_weight: float = 0.15
    redundancy_penalty: float = 0.1
    length_penalty: float = 0.15
    
    # Constraints
    require_role_coverage: bool = True
    require_provenance: bool = True


@dataclass
class SelectionResult:
    """Result of evidence selection."""
    status: str  # "ok", "insufficient_evidence", "missing_roles"
    selected_bundles: List[Evidence]
    unmet_roles: List[Role]
    selection_score: float
    metadata: Dict


def get_required_roles(query: Query) -> Set[Role]:
    """Get required pedagogical roles for a query type."""
    return ROLE_POLICY.get(query.query_type, {"concept"})


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    if not scores or len(scores) == 1:
        return [1.0] * len(scores)
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


class PSEvidenceSelector:
    """
    Pedagogically Sufficient Evidence Selector.
    
    Selects minimal evidence sets that:
    - Cover required instructional roles
    - Fit target grade level
    - Minimize redundancy
    - Respect token budget
    - Provide complete provenance
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize PS-Evidence selector.
        
        Args:
            config: Selection configuration
        """
        self.config = config or SelectionConfig()
    
    def select(
        self,
        query: Query,
        candidates: List[Tuple[Evidence, float]],
        verified_ancestors: Optional[Set[str]] = None
    ) -> SelectionResult:
        """
        Select optimal evidence set for query.
        
        Args:
            query: Student query with context
            candidates: Ranked list of (Evidence, relevance_score) tuples
            verified_ancestors: Set of verified prerequisite LOs
        
        Returns:
            SelectionResult with selected bundles and metadata
        """
        if not candidates:
            return SelectionResult(
                status="insufficient_evidence",
                selected_bundles=[],
                unmet_roles=list(get_required_roles(query)),
                selection_score=0.0,
                metadata={"reason": "no_candidates"}
            )
        
        # Get required roles
        required_roles = get_required_roles(query)
        
        # Filter admissible candidates
        admissible = self._filter_admissible(
            candidates,
            query,
            verified_ancestors
        )
        
        if not admissible:
            return SelectionResult(
                status="insufficient_evidence",
                selected_bundles=[],
                unmet_roles=list(required_roles),
                selection_score=0.0,
                metadata={"reason": "no_admissible_candidates"}
            )
        
        # Normalize relevance scores
        relevance_scores = {
            bundle.id: norm_score
            for (bundle, _), norm_score in zip(admissible, normalize_scores([s for _, s in admissible]))
        }
        
        # Search for optimal evidence set
        best_set = self._optimize_set(
            admissible,
            query,
            required_roles,
            relevance_scores
        )
        
        if best_set is None:
            # Find which roles are missing
            available_roles = set()
            for bundle, _ in admissible:
                available_roles.update(bundle.roles)
            unmet = required_roles - available_roles
            
            return SelectionResult(
                status="missing_roles",
                selected_bundles=[],
                unmet_roles=list(unmet),
                selection_score=0.0,
                metadata={"available_roles": list(available_roles)}
            )
        
        score, bundles = best_set
        
        return SelectionResult(
            status="ok",
            selected_bundles=bundles,
            unmet_roles=[],
            selection_score=score,
            metadata={
                "num_candidates": len(candidates),
                "num_admissible": len(admissible),
                "total_tokens": sum(b.token_count() for b in bundles),
                "covered_roles": list(set().union(*(b.roles for b in bundles)))
            }
        )
    
    def _filter_admissible(
        self,
        candidates: List[Tuple[Evidence, float]],
        query: Query,
        verified_ancestors: Optional[Set[str]]
    ) -> List[Tuple[Evidence, float]]:
        """
        Filter candidates to admissible set based on curriculum constraints.
        
        Admissible if:
        - Same subject
        - Same grade OR verified prerequisite with mastery gap
        - Has provenance
        """
        admissible = []
        
        for bundle, score in candidates:
            # Subject match required
            if bundle.subject != query.subject:
                continue
            
            # Grade constraints
            if bundle.grade == query.grade:
                admissible.append((bundle, score))
            elif bundle.grade < query.grade:
                # Check prerequisite licensing
                if verified_ancestors and bundle.lo_id in verified_ancestors:
                    # Check mastery gap
                    mastery = query.mastery.get(bundle.lo_id)
                    if mastery is not None and mastery < 0.5:
                        admissible.append((bundle, score))
                    # Unknown mastery does NOT imply gap (conservative)
            # Higher grades excluded
            
            # Provenance required
            if self.config.require_provenance:
                if not bundle.source_atom_ids or not bundle.source_id:
                    continue
        
        return admissible
    
    def _optimize_set(
        self,
        candidates: List[Tuple[Evidence, float]],
        query: Query,
        required_roles: Set[Role],
        relevance_scores: Dict[str, float]
    ) -> Optional[Tuple[float, List[Evidence]]]:
        """
        Find optimal evidence set using enumerative search.
        
        For small candidate pools (~12) and small K (≤3), enumeration
        is tractable and easier to audit than neural optimization.
        """
        # Take top candidates for enumeration
        top_n = min(12, len(candidates))
        candidate_pool = [b for b, _ in candidates[:top_n]]
        
        best_score = -float('inf')
        best_set = None
        
        # Try all combinations up to max_bundles
        for size in range(1, min(self.config.max_bundles, len(candidate_pool)) + 1):
            for combo in combinations(candidate_pool, size):
                # Check constraints
                if not self._satisfies_constraints(combo, required_roles, query):
                    continue
                
                # Compute score
                score = self._score_set(combo, required_roles, relevance_scores, query)
                
                if score > best_score:
                    best_score = score
                    best_set = (score, list(combo))
        
        return best_set
    
    def _satisfies_constraints(
        self,
        evidence_set: Tuple[Evidence, ...],
        required_roles: Set[Role],
        query: Query
    ) -> bool:
        """Check if evidence set satisfies hard constraints."""
        # Token budget
        total_tokens = sum(b.token_count() for b in evidence_set)
        if total_tokens > self.config.token_budget:
            return False
        
        # Distinct source atoms (avoid counting same content multiple times)
        all_atoms = []
        for bundle in evidence_set:
            all_atoms.extend(bundle.source_atom_ids)
        if len(all_atoms) != len(set(all_atoms)):
            # Has duplicate source atoms
            return False
        
        # Role coverage (if required)
        if self.config.require_role_coverage:
            covered_roles = set().union(*(b.roles for b in evidence_set))
            if not required_roles.issubset(covered_roles):
                return False
        
        return True
    
    def _score_set(
        self,
        evidence_set: Tuple[Evidence, ...],
        required_roles: Set[Role],
        relevance_scores: Dict[str, float],
        query: Query
    ) -> float:
        """
        Compute optimization score for evidence set.
        
        Score = relevance + role_coverage + grade_fit 
                + prerequisite_value - redundancy - length
        """
        cfg = self.config
        
        # Relevance: sum of normalized relevance scores
        relevance = sum(relevance_scores[b.id] for b in evidence_set)
        
        # Role coverage: fraction of required roles covered
        covered_roles = set().union(*(b.roles for b in evidence_set))
        role_coverage = len(covered_roles & required_roles) / max(1, len(required_roles))
        
        # Grade fit: prefer exact grade match
        grade_fit = sum(1.0 if b.grade == query.grade else 0.5 for b in evidence_set) / len(evidence_set)
        
        # Prerequisite value: reward verified prerequisites when needed
        prerequisite_value = 0.0
        if query.query_type == "prerequisite_gap":
            prerequisite_value = sum(1.0 if "prerequisite" in b.roles else 0.0 for b in evidence_set)
        
        # Redundancy: penalize similar content (simplified)
        redundancy = self._compute_redundancy(evidence_set)
        
        # Length: prefer smaller sets
        length_penalty = len(evidence_set) / self.config.max_bundles
        
        # Token penalty: prefer using less of budget
        total_tokens = sum(b.token_count() for b in evidence_set)
        token_penalty = total_tokens / self.config.token_budget
        
        # Weighted combination
        score = (
            cfg.relevance_weight * relevance
            + cfg.role_coverage_weight * role_coverage
            + cfg.grade_fit_weight * grade_fit
            + cfg.prerequisite_value_weight * prerequisite_value
            - cfg.redundancy_penalty * redundancy
            - cfg.length_penalty * (length_penalty + token_penalty) / 2
        )
        
        return score
    
    def _compute_redundancy(self, evidence_set: Tuple[Evidence, ...]) -> float:
        """
        Compute redundancy penalty for evidence set.
        
        Simple heuristic: penalize if bundles are from same page.
        More sophisticated: use embedding similarity (future work).
        """
        if len(evidence_set) <= 1:
            return 0.0
        
        pages = [b.page for b in evidence_set]
        unique_pages = len(set(pages))
        
        # Redundancy increases as unique pages decrease
        redundancy = 1.0 - (unique_pages / len(evidence_set))
        
        return redundancy


def select_top_k_baseline(
    candidates: List[Tuple[Evidence, float]],
    k: int = 3
) -> List[Evidence]:
    """
    Baseline: simple top-k selection (no role constraints).
    
    Used for ablation comparison.
    """
    return [bundle for bundle, _ in candidates[:k]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("PS-Evidence selection module ready.")
    print(f"Role policies: {ROLE_POLICY}")
