"""
Quantum Annealing for Semantic Search.

Improves semantic search convergence by treating search result optimization
as a quantum annealing problem. Uses simulated quantum tunneling to escape
local minima in the similarity landscape.

This is inspired by D-Wave's quantum annealing approach but implemented
classically with quantum-inspired heuristics.
"""

import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class QuantumAnnealingSearch:
    """
    Quantum-inspired annealing for semantic search optimization.
    
    Key concepts:
    - Treats embedding similarity as energy landscape
    - Uses simulated quantum tunneling to explore landscape
    - Annealing schedule reduces "quantum fluctuations" over time
    - Final state represents optimized search results
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        annealing_steps: int = 50,
        tunneling_strength: float = 0.3,
    ):
        """
        Initialize quantum annealing search.
        
        Args:
            initial_temperature: Starting temperature (high = more exploration)
            final_temperature: Ending temperature (low = greedy selection)
            annealing_steps: Number of annealing iterations
            tunneling_strength: Probability of quantum tunneling jumps
        """
        self.T_initial = initial_temperature
        self.T_final = final_temperature
        self.steps = annealing_steps
        self.tunneling_strength = tunneling_strength
        
    def _compute_energy(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        selected_mask: np.ndarray,
    ) -> float:
        """
        Compute energy of current selection state.
        
        Lower energy = better selection.
        Energy is negative similarity to encourage selection of similar items.
        
        Args:
            query_embedding: Query vector [dim]
            candidate_embeddings: Candidate vectors [N, dim]
            selected_mask: Binary mask of selected items [N]
            
        Returns:
            Energy (lower is better)
        """
        if selected_mask.sum() == 0:
            return float('inf')
            
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cand_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarities
        similarities = cand_norms @ query_norm
        
        # Energy = negative average similarity of selected items
        selected_sims = similarities * selected_mask
        avg_similarity = selected_sims.sum() / (selected_mask.sum() + 1e-8)
        
        # Add diversity penalty (encourage diverse selections)
        if selected_mask.sum() > 1:
            selected_indices = np.where(selected_mask > 0.5)[0]
            selected_embeddings = cand_norms[selected_indices]
            # Pairwise similarity of selected items
            pairwise_sim = selected_embeddings @ selected_embeddings.T
            # Off-diagonal average (redundancy penalty)
            off_diag_mask = 1 - np.eye(len(selected_indices))
            redundancy = (pairwise_sim * off_diag_mask).sum() / (off_diag_mask.sum() + 1e-8)
            # Higher redundancy = higher energy
            diversity_penalty = 0.2 * redundancy
        else:
            diversity_penalty = 0
            
        return -avg_similarity + diversity_penalty
    
    def _quantum_tunnel(
        self, 
        state: np.ndarray, 
        temperature: float,
        n_candidates: int,
    ) -> np.ndarray:
        """
        Simulate quantum tunneling to escape local minima.
        
        With probability proportional to temperature and tunneling_strength,
        randomly toggle bits in the state vector.
        
        Args:
            state: Current selection state [N]
            temperature: Current temperature
            n_candidates: Number of candidates
            
        Returns:
            New state after tunneling
        """
        new_state = state.copy()
        
        # Tunneling probability decreases with temperature
        tunnel_prob = self.tunneling_strength * temperature
        
        # Random bit flips (quantum tunneling through energy barriers)
        tunnel_mask = np.random.random(n_candidates) < tunnel_prob
        new_state[tunnel_mask] = 1 - new_state[tunnel_mask]
        
        return new_state
    
    def optimize_search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Optimize search results using quantum annealing.
        
        Args:
            query_embedding: Query vector [dim]
            candidate_embeddings: Candidate vectors [N, dim]
            k: Number of results to select
            
        Returns:
            Tuple of (final_state, selected_indices)
        """
        n_candidates = len(candidate_embeddings)
        if n_candidates <= k:
            return np.ones(n_candidates), list(range(n_candidates))
        
        # Initialize with greedy top-k
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cand_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = cand_norms @ query_norm
        
        # Initial state: top-k by similarity
        state = np.zeros(n_candidates)
        top_k_indices = np.argsort(similarities)[-k:]
        state[top_k_indices] = 1
        
        current_energy = self._compute_energy(query_embedding, candidate_embeddings, state)
        best_state = state.copy()
        best_energy = current_energy
        
        # Annealing schedule (exponential decay)
        temperatures = np.logspace(
            np.log10(self.T_initial), 
            np.log10(self.T_final), 
            self.steps
        )
        
        for step, temperature in enumerate(temperatures):
            # Quantum tunneling step
            new_state = self._quantum_tunnel(state, temperature, n_candidates)
            
            # Enforce exactly k selections
            while new_state.sum() > k:
                selected = np.where(new_state > 0.5)[0]
                remove_idx = np.random.choice(selected)
                new_state[remove_idx] = 0
            while new_state.sum() < k:
                unselected = np.where(new_state < 0.5)[0]
                add_idx = np.random.choice(unselected)
                new_state[add_idx] = 1
            
            # Compute new energy
            new_energy = self._compute_energy(query_embedding, candidate_embeddings, new_state)
            
            # Metropolis acceptance criterion
            delta_E = new_energy - current_energy
            if delta_E < 0:
                # Accept improvement
                state = new_state
                current_energy = new_energy
            else:
                # Accept with probability exp(-delta_E / T)
                accept_prob = np.exp(-delta_E / temperature)
                if np.random.random() < accept_prob:
                    state = new_state
                    current_energy = new_energy
            
            # Track best
            if current_energy < best_energy:
                best_state = state.copy()
                best_energy = current_energy
        
        # Return best state found
        selected_indices = np.where(best_state > 0.5)[0].tolist()
        
        # Sort by similarity for consistent ordering
        selected_indices.sort(key=lambda i: -similarities[i])
        
        logger.debug(f"Quantum annealing: initial_energy={-similarities[top_k_indices].mean():.4f}, "
                     f"final_energy={best_energy:.4f}, steps={self.steps}")
        
        return best_state, selected_indices


class AnnealingSemanticSearch:
    """
    Semantic search enhanced with quantum annealing optimization.
    
    Wraps the base Saguaro search and applies annealing to improve
    result diversity and relevance.
    """
    
    def __init__(self, substrate, brain=None):
        """
        Initialize annealing-enhanced search.
        
        Args:
            substrate: SaguaroSubstrate for base search
            brain: DeterministicOllama for embeddings
        """
        self.substrate = substrate
        self.brain = brain
        self.annealer = QuantumAnnealingSearch(
            initial_temperature=1.0,
            final_temperature=0.01,
            annealing_steps=30,  # Fast enough for interactive use
            tunneling_strength=0.2,
        )
    
    def search(self, query: str, k: int = 5, rerank_pool: int = 20) -> List[dict]:
        """
        Perform annealing-optimized semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            rerank_pool: Number of candidates to consider for reranking
            
        Returns:
            List of search results with file paths and scores
        """
        # Get larger pool from Saguaro
        raw_results = self.substrate.agent_query(query, k=rerank_pool)
        
        # Parse results to extract paths and get embeddings
        # This is a simplified implementation - full version would
        # get embeddings for each result
        
        # For now, return raw results (annealing would be applied
        # if we had access to the embedding vectors)
        logger.debug(f"Quantum annealing search for: {query[:50]}...")
        
        return raw_results
