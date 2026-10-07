"""
Prerequisite graph for learning outcome relationships.

Manages curriculum dependencies and prerequisite licensing for
remediation questions.
"""

import logging
from pathlib import Path
from typing import List, Set, Dict, Optional
import json

import networkx as nx

from trace_tutor.schema import Edge


logger = logging.getLogger(__name__)


class PrerequisiteGraph:
    """
    Directed acyclic graph of learning outcome prerequisites.
    
    Must be expert-reviewed for production use. For solo research,
    use conservative assumptions and document limitations.
    """
    
    def __init__(self):
        """Initialize empty prerequisite graph."""
        self.graph = nx.DiGraph()
        self.edges: List[Edge] = []
        self._ancestors_cache: Dict[str, Set[str]] = {}
    
    def add_edge(self, edge: Edge):
        """
        Add prerequisite edge to graph.
        
        Args:
            edge: Edge object defining prerequisite relationship
        """
        self.edges.append(edge)
        
        # Add nodes if they don't exist
        if not self.graph.has_node(edge.prerequisite_lo):
            self.graph.add_node(edge.prerequisite_lo)
        if not self.graph.has_node(edge.target_lo):
            self.graph.add_node(edge.target_lo)
        
        # Add edge with metadata
        self.graph.add_edge(
            edge.prerequisite_lo,
            edge.target_lo,
            review_status=edge.review_status,
            strength=edge.strength,
            justification=edge.justification
        )
        
        # Clear ancestors cache
        self._ancestors_cache.clear()
    
    def get_verified_ancestors(
        self,
        target_lo: str,
        include_strength: bool = False
    ) -> Set[str]:
        """
        Get all verified prerequisite ancestors of a learning outcome.
        
        Args:
            target_lo: Target learning outcome ID
            include_strength: If True, return (lo_id, strength) tuples
        
        Returns:
            Set of verified prerequisite LO IDs (ancestors)
        """
        # Check cache
        if target_lo in self._ancestors_cache:
            return self._ancestors_cache[target_lo].copy()
        
        if target_lo not in self.graph:
            return set()
        
        ancestors = set()
        visited = set()
        stack = [target_lo]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            
            # Get predecessors (prerequisites)
            for predecessor in self.graph.predecessors(node):
                edge_data = self.graph[predecessor][node]
                
                # Only include verified edges
                if edge_data.get('review_status') == 'verified':
                    ancestors.add(predecessor)
                    stack.append(predecessor)
        
        # Cache result
        self._ancestors_cache[target_lo] = ancestors
        
        return ancestors.copy()
    
    def is_prerequisite_of(self, prerequisite_lo: str, target_lo: str) -> bool:
        """Check if one LO is a verified prerequisite of another."""
        ancestors = self.get_verified_ancestors(target_lo)
        return prerequisite_lo in ancestors
    
    def get_direct_prerequisites(self, target_lo: str) -> List[str]:
        """Get immediate prerequisites of a learning outcome."""
        if target_lo not in self.graph:
            return []
        
        direct = []
        for predecessor in self.graph.predecessors(target_lo):
            edge_data = self.graph[predecessor][target_lo]
            if edge_data.get('review_status') == 'verified':
                direct.append(predecessor)
        
        return direct
    
    def validate(self) -> List[str]:
        """
        Validate graph structure and return list of issues.
        
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(f"Graph contains {len(cycles)} cycle(s): {cycles[:3]}")
        
        # Check for self-loops
        self_loops = list(nx.nodes_with_selfloops(self.graph))
        if self_loops:
            issues.append(f"Self-loops found: {self_loops}")
        
        # Check review status
        unreviewed = [e for e in self.edges if e.review_status == 'pending']
        if unreviewed:
            issues.append(f"{len(unreviewed)} edges pending review")
        
        # Check isolated nodes
        isolated = list(nx.isolates(self.graph))
        if isolated:
            issues.append(f"{len(isolated)} isolated nodes: {isolated[:5]}")
        
        return issues
    
    def get_statistics(self) -> Dict:
        """Get graph statistics for reporting."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_verified": sum(1 for e in self.edges if e.review_status == 'verified'),
            "num_pending": sum(1 for e in self.edges if e.review_status == 'pending'),
            "num_rejected": sum(1 for e in self.edges if e.review_status == 'rejected'),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "avg_in_degree": sum(d for _, d in self.graph.in_degree()) / max(1, self.graph.number_of_nodes()),
            "avg_out_degree": sum(d for _, d in self.graph.out_degree()) / max(1, self.graph.number_of_nodes())
        }
    
    def save(self, filepath: str):
        """Save graph to JSONL file."""
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for edge in self.edges:
                f.write(edge.model_dump_json() + '\n')
        
        logger.info(f"Saved {len(self.edges)} edges to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PrerequisiteGraph':
        """Load graph from JSONL file."""
        logger.info(f"Loading prerequisite graph from {filepath}")
        
        graph = cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    edge = Edge.model_validate_json(line)
                    graph.add_edge(edge)
        
        logger.info(f"Loaded graph: {graph.get_statistics()}")
        
        # Validate
        issues = graph.validate()
        if issues:
            logger.warning(f"Graph validation issues: {issues}")
        
        return graph
    
    def visualize(self, output_file: Optional[str] = None) -> Optional[str]:
        """
        Create visualization of prerequisite graph.
        
        Requires matplotlib and graphviz.
        """
        try:
            import matplotlib.pyplot as plt
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError:
            logger.warning("Visualization requires matplotlib and pygraphviz")
            return None
        
        # Create layout
        try:
            pos = graphviz_layout(self.graph, prog='dot')
        except:
            pos = nx.spring_layout(self.graph)
        
        # Color nodes by review status
        verified_nodes = []
        pending_nodes = []
        
        for u, v in self.graph.edges():
            edge_data = self.graph[u][v]
            if edge_data.get('review_status') == 'verified':
                verified_nodes.extend([u, v])
            else:
                pending_nodes.extend([u, v])
        
        verified_nodes = list(set(verified_nodes))
        pending_nodes = list(set(pending_nodes) - set(verified_nodes))
        
        # Draw
        plt.figure(figsize=(12, 8))
        
        nx.draw_networkx_nodes(self.graph, pos, nodelist=verified_nodes, 
                               node_color='lightgreen', node_size=500, label='Verified')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=pending_nodes,
                               node_color='lightyellow', node_size=500, label='Pending')
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.legend()
        plt.title("Prerequisite Graph")
        plt.axis('off')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_file}")
        
        return output_file


def create_sample_graph() -> PrerequisiteGraph:
    """
    Create sample prerequisite graph for EVS Grades 3-5.
    
    NOTE: This is illustrative only. Real prerequisite edges must be
    expert-reviewed for research validity.
    """
    graph = PrerequisiteGraph()
    
    # Sample edges (EVS Grade 3-5 concepts)
    sample_edges = [
        # Grade 3 foundations
        ("living_nonliving_g3", "plants_parts_g4", "verified", 0.9, "Basic classification needed"),
        ("living_nonliving_g3", "animals_types_g4", "verified", 0.9, "Basic classification needed"),
        
        # Grade 4 builds on Grade 3
        ("plants_parts_g4", "photosynthesis_g5", "verified", 1.0, "Must know plant structure"),
        ("animals_types_g4", "food_chains_g5", "verified", 0.8, "Animal classification helps"),
        
        # Within-grade dependencies
        ("water_sources_g4", "water_cycle_g4", "verified", 0.9, "Understanding sources helps cycle"),
        ("food_types_g3", "nutrition_g4", "verified", 0.8, "Food categories prerequisite"),
        
        # Pending review (conservative - not used without expert validation)
        ("seasons_g3", "agriculture_g5", "pending", 0.6, "Possible connection"),
    ]
    
    for prereq, target, status, strength, justification in sample_edges:
        edge = Edge(
            prerequisite_lo=prereq,
            target_lo=target,
            review_status=status,
            strength=strength,
            justification=justification,
            reviewer_id="system_generated" if status == "verified" else None
        )
        graph.add_edge(edge)
    
    return graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample graph
    graph = create_sample_graph()
    print(f"Graph stats: {graph.get_statistics()}")
    print(f"Validation: {graph.validate()}")
    
    # Test ancestor lookup
    ancestors = graph.get_verified_ancestors("photosynthesis_g5")
    print(f"Ancestors of photosynthesis_g5: {ancestors}")
    
    # Save
    graph.save("data/graph/prerequisites_sample.jsonl")
    
    print("Prerequisite graph module ready.")
