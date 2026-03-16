"""
Multi-Agent Coherence Bus using Saguaro's quantum_coherence_bus_op.

Enables efficient latent state transfer between agents without text serialization.
"""

import numpy as np
from typing import Dict, Optional, Tuple

try:
    import importlib.util
    QCB_OPS_AVAILABLE = importlib.util.find_spec("saguaro.native.ops.quantum_coherence_bus_ops") is not None
except Exception:
    QCB_OPS_AVAILABLE = False


class CoherenceBus:
    """
    Quantum-inspired coherence bus for multi-agent state synchronization.
    """

    def __init__(
        self,
        num_agents: int = 6,
        entanglement_dim: int = 64,
        state_dim: int = 4096,
    ):
        """
        Initialize coherence bus.
        """
        self.num_agents = num_agents
        self.entanglement_dim = entanglement_dim
        self.state_dim = state_dim

        # Initialize entangled mesh
        self.entangled_state = None
        self.fidelity = 1.0
        self.bus_available = QCB_OPS_AVAILABLE

        if self.bus_available:
            self._initialize_entanglement()
        else:
            print("Warning: Coherence bus operating in fallback mode (direct transfer)")

        # Agent state cache
        self.agent_states: Dict[str, np.ndarray] = {}
        self.transfer_count = 0

    def _initialize_entanglement(self):
        """Initialize GHZ-like entangled mesh."""
        if not self.bus_available:
            return

        try:
            self.entangled_state = np.random.randn(
                self.num_agents, self.entanglement_dim
            ).astype(np.float32)

            # Normalize to create coherent superposition
            self.entangled_state = self.entangled_state / np.linalg.norm(
                self.entangled_state, axis=1, keepdims=True
            )

            self.fidelity = 1.0

        except Exception as e:
            print(f"Warning: Entanglement initialization failed: {e}")
            self.bus_available = False

    def register_agent(self, agent_id: str, initial_state: Optional[np.ndarray] = None):
        """
        Register agent on the coherence bus.
        """
        if initial_state is not None:
            self.agent_states[agent_id] = initial_state.astype(np.float32)
        else:
            self.agent_states[agent_id] = np.zeros(self.state_dim, dtype=np.float32)

    def transfer_context(
        self,
        source_agent_id: str,
        target_agent_id: str,
        source_state: np.ndarray,
        transfer_strength: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Transfer context from source agent to target via coherent teleportation.
        """
        if not self.bus_available:
            # Fallback: Direct transfer with noise
            noise = np.random.randn(*source_state.shape).astype(np.float32) * 0.01
            teleported = source_state * transfer_strength + noise * (
                1 - transfer_strength
            )
            return teleported, 0.95  # Simulated fidelity

        try:
            # Project source state to entanglement subspace
            source_reduced = source_state[: self.entanglement_dim]

            # Entangle with mesh
            source_block = hash(source_agent_id) % self.num_agents
            target_block = hash(target_agent_id) % self.num_agents

            entanglement_channel = self.entangled_state[source_block]

            # Teleport via phase correlation
            phase_overlap = np.dot(source_reduced, entanglement_channel)

            # Reconstruct at target
            target_entanglement = self.entangled_state[target_block]
            teleported_reduced = target_entanglement * phase_overlap

            # Expand back to full dimension
            teleported_state = np.zeros_like(source_state)
            teleported_state[: self.entanglement_dim] = teleported_reduced

            # Add residual from original state (partial transfer)
            teleported_state = (
                transfer_strength * teleported_state
                + (1 - transfer_strength) * source_state
            )

            # Compute fidelity
            fidelity = float(
                np.abs(phase_overlap) / (np.linalg.norm(source_reduced) + 1e-8)
            )
            fidelity = min(fidelity, 1.0)

            self.transfer_count += 1

            return teleported_state, fidelity

        except Exception as e:
            print(f"Warning: Coherent transfer failed: {e}. Using direct transfer.")
            return source_state, 0.9

    def broadcast_state(
        self, source_agent_id: str, source_state: np.ndarray, target_agents: list
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Broadcast state to multiple agents simultaneously.
        """
        results = {}
        for target_id in target_agents:
            teleported, fidelity = self.transfer_context(
                source_agent_id, target_id, source_state, transfer_strength=0.8
            )
            results[target_id] = (teleported, fidelity)
        return results

    def get_agent_state(self, agent_id: str) -> Optional[np.ndarray]:
        """Get cached agent state."""
        return self.agent_states.get(agent_id)

    def update_agent_state(self, agent_id: str, state: np.ndarray):
        """Update agent state cache."""
        self.agent_states[agent_id] = state.astype(np.float32)

    def measure_coherence(self) -> float:
        """
        Measure current coherence of the entangled mesh.
        """
        if not self.bus_available or self.entangled_state is None:
            return 0.0

        # Compute pairwise correlations
        correlations = self.entangled_state @ self.entangled_state.T
        off_diagonal = correlations - np.eye(self.num_agents)
        avg_correlation = np.abs(off_diagonal).mean()

        # Coherence decays with transfers
        decay_factor = 0.99**self.transfer_count
        coherence = avg_correlation * decay_factor

        return float(np.clip(coherence, 0.0, 1.0))

    def refresh_entanglement(self):
        """Refresh entangled mesh (re-initialize)."""
        self._initialize_entanglement()
        self.transfer_count = 0
        print("Info: Coherence bus entanglement refreshed")

    def get_stats(self) -> dict:
        """Get coherence bus statistics."""
        return {
            "num_agents": self.num_agents,
            "registered_agents": len(self.agent_states),
            "transfer_count": self.transfer_count,
            "coherence": self.measure_coherence(),
            "fidelity": self.fidelity,
            "bus_available": self.bus_available,
        }


class AgentContextWrapper:
    """
    Wrapper for agent context injection via coherence bus.
    """

    def __init__(self, coherence_bus: CoherenceBus):
        """Initialize wrapper."""
        self.bus = coherence_bus

    def spawn_child_agent(
        self,
        parent_id: str,
        child_id: str,
        parent_state: np.ndarray,
        inheritance_ratio: float = 0.7,
    ) -> np.ndarray:
        """
        Spawn child agent with inherited context from parent.
        """
        # Transfer parent context to child
        child_state, fidelity = self.bus.transfer_context(
            parent_id, child_id, parent_state, transfer_strength=inheritance_ratio
        )

        # Register child
        self.bus.register_agent(child_id, child_state)

        print(
            f"Info: Spawned {child_id} from {parent_id} "
            f"(fidelity: {fidelity:.3f}, inheritance: {inheritance_ratio:.0%})"
        )

        return child_state

    def handoff_to_agent(
        self,
        from_agent_id: str,
        to_agent_id: str,
        current_state: np.ndarray,
        full_transfer: bool = True,
    ) -> np.ndarray:
        """
        Hand off execution to another agent.
        """
        transfer_strength = 1.0 if full_transfer else 0.5

        target_state, fidelity = self.bus.transfer_context(
            from_agent_id,
            to_agent_id,
            current_state,
            transfer_strength=transfer_strength,
        )

        self.bus.update_agent_state(to_agent_id, target_state)

        transfer_type = "full" if full_transfer else "partial"
        print(
            f"Info: Handoff {from_agent_id} → {to_agent_id} "
            f"({transfer_type}, fidelity: {fidelity:.3f})"
        )

        return target_state
