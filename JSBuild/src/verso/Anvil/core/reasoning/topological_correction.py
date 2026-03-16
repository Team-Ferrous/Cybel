"""
Topological Error Correction for Reasoning Chains.

Implements surface code-inspired error correction for multi-step reasoning.
Errors in individual reasoning steps are detected and corrected using
topological redundancy - the same concept that makes topological quantum
computers fault-tolerant.

Key concepts:
- Reasoning chains form a "lattice" of logical steps
- Syndrome extraction identifies inconsistencies
- Topological correction uses neighboring steps to fix errors
- Self-correcting reasoning emerges from the structure
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    content: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    error_syndrome: int = 0  # 0 = no error detected
    corrected: bool = False


class TopologicalReasoningCorrector:
    """
    Self-correcting reasoning using topological error correction.
    
    The key insight is that in a valid reasoning chain, each step should be
    consistent with its neighbors. Inconsistencies (errors) can be detected
    by measuring "syndromes" - patterns that indicate logical breaks.
    
    This is analogous to surface codes in quantum computing, where physical
    qubit errors are detected and corrected using topological properties
    of the code lattice.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.3,
        error_correction_strength: float = 0.5,
        max_correction_rounds: int = 3,
    ):
        """
        Initialize topological corrector.
        
        Args:
            coherence_threshold: Minimum cosine similarity for step coherence
            error_correction_strength: How much to blend corrections
            max_correction_rounds: Maximum error correction iterations
        """
        self.coherence_threshold = coherence_threshold
        self.correction_strength = error_correction_strength
        self.max_rounds = max_correction_rounds
        
    def _compute_coherence(
        self, 
        emb1: np.ndarray, 
        emb2: np.ndarray
    ) -> float:
        """
        Compute coherence between two reasoning steps.
        
        Uses cosine similarity as the coherence measure.
        Low coherence indicates a potential logical break.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def _extract_syndromes(
        self, 
        steps: List[ReasoningStep]
    ) -> List[int]:
        """
        Extract error syndromes from reasoning chain.
        
        A syndrome is computed for each step by checking coherence
        with its neighbors. Non-zero syndrome indicates error.
        
        In topological codes, syndromes are measured by checking
        parity of neighboring qubits. Here we use coherence.
        
        Returns:
            List of syndrome values (0 = no error)
        """
        syndromes = []
        
        for i, step in enumerate(steps):
            if step.embedding is None:
                syndromes.append(0)
                continue
                
            # Check coherence with previous step
            prev_coherence = 1.0
            if i > 0 and steps[i-1].embedding is not None:
                prev_coherence = self._compute_coherence(
                    steps[i-1].embedding, step.embedding
                )
            
            # Check coherence with next step
            next_coherence = 1.0
            if i < len(steps) - 1 and steps[i+1].embedding is not None:
                next_coherence = self._compute_coherence(
                    step.embedding, steps[i+1].embedding
                )
            
            # Syndrome = number of broken coherence bonds
            syndrome = 0
            if prev_coherence < self.coherence_threshold:
                syndrome += 1
            if next_coherence < self.coherence_threshold:
                syndrome += 1
                
            syndromes.append(syndrome)
            step.error_syndrome = syndrome
            
        return syndromes
    
    def _apply_topological_correction(
        self,
        steps: List[ReasoningStep],
        syndromes: List[int],
    ) -> List[ReasoningStep]:
        """
        Apply topological error correction.
        
        For steps with non-zero syndrome, blend their embedding
        with neighbors to restore coherence. This is analogous
        to how surface codes use syndrome measurements to
        determine correction operators.
        
        Returns:
            Corrected reasoning steps
        """
        corrected_steps = []
        
        for i, step in enumerate(steps):
            if syndromes[i] == 0 or step.embedding is None:
                corrected_steps.append(step)
                continue
            
            # Collect valid neighbor embeddings
            neighbor_embeddings = []
            if i > 0 and steps[i-1].embedding is not None:
                neighbor_embeddings.append(steps[i-1].embedding)
            if i < len(steps) - 1 and steps[i+1].embedding is not None:
                neighbor_embeddings.append(steps[i+1].embedding)
            
            if not neighbor_embeddings:
                corrected_steps.append(step)
                continue
            
            # Compute correction as weighted average of neighbors
            neighbor_mean = np.mean(neighbor_embeddings, axis=0)
            
            # Blend original with correction
            corrected_embedding = (
                (1 - self.correction_strength) * step.embedding +
                self.correction_strength * neighbor_mean
            )
            
            # Normalize
            norm = np.linalg.norm(corrected_embedding)
            if norm > 1e-8:
                corrected_embedding = corrected_embedding / norm
            
            # Create corrected step
            corrected_step = ReasoningStep(
                content=step.content,
                embedding=corrected_embedding,
                confidence=step.confidence * (1 - syndromes[i] * 0.2),  # Reduce confidence
                error_syndrome=0,  # Reset after correction
                corrected=True,
            )
            corrected_steps.append(corrected_step)
            
            logger.debug(f"Corrected reasoning step {i} with syndrome {syndromes[i]}")
            
        return corrected_steps
    
    def correct_chain(
        self,
        steps: List[ReasoningStep],
    ) -> Tuple[List[ReasoningStep], Dict[str, Any]]:
        """
        Apply iterative topological error correction to reasoning chain.
        
        Uses multiple correction rounds until no more errors are detected
        or max rounds reached.
        
        Args:
            steps: Reasoning chain to correct
            
        Returns:
            Tuple of (corrected_steps, correction_stats)
        """
        current_steps = steps
        total_corrections = 0
        rounds_used = 0
        
        for round_num in range(self.max_rounds):
            # Extract syndromes
            syndromes = self._extract_syndromes(current_steps)
            
            # Count errors
            error_count = sum(1 for s in syndromes if s > 0)
            
            if error_count == 0:
                # No more errors - chain is coherent
                break
                
            # Apply correction
            current_steps = self._apply_topological_correction(
                current_steps, syndromes
            )
            
            total_corrections += error_count
            rounds_used = round_num + 1
            
            logger.debug(f"Correction round {round_num + 1}: fixed {error_count} errors")
        
        # Final syndrome check
        final_syndromes = self._extract_syndromes(current_steps)
        remaining_errors = sum(1 for s in final_syndromes if s > 0)
        
        stats = {
            "total_corrections": total_corrections,
            "rounds_used": rounds_used,
            "remaining_errors": remaining_errors,
            "chain_length": len(steps),
            "coherence_threshold": self.coherence_threshold,
        }
        
        return current_steps, stats


class ReasoningChainBuilder:
    """
    Builds reasoning chains with automatic error correction.
    """
    
    def __init__(self, brain=None, corrector: Optional[TopologicalReasoningCorrector] = None):
        """
        Initialize chain builder.
        
        Args:
            brain: DeterministicOllama for embeddings
            corrector: Topological corrector (creates default if None)
        """
        self.brain = brain
        self.corrector = corrector or TopologicalReasoningCorrector()
        
    def build_chain(
        self,
        thoughts: List[str],
    ) -> Tuple[List[ReasoningStep], Dict[str, Any]]:
        """
        Build error-corrected reasoning chain from thoughts.
        
        Args:
            thoughts: List of reasoning step strings
            
        Returns:
            Tuple of (corrected_steps, stats)
        """
        # Create initial steps with embeddings
        steps = []
        for thought in thoughts:
            embedding = None
            if self.brain is not None:
                try:
                    embedding = np.array(self.brain.embeddings(thought[:500]))
                except Exception:
                    pass
                    
            step = ReasoningStep(
                content=thought,
                embedding=embedding,
                confidence=1.0,
            )
            steps.append(step)
        
        # Apply topological correction
        corrected_steps, stats = self.corrector.correct_chain(steps)
        
        return corrected_steps, stats
    
    def compute_chain_confidence(self, steps: List[ReasoningStep]) -> float:
        """
        Compute overall chain confidence.
        
        Multiplies individual step confidences (degraded by corrections).
        """
        if not steps:
            return 0.0
            
        confidences = [s.confidence for s in steps]
        return float(np.prod(confidences))
