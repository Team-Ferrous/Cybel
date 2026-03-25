"""
QAOA-Inspired Subagent Scheduling.

Implements a quantum-approximate optimization algorithm (QAOA) inspired
scheduler for optimal task distribution across subagents. This treats
task assignment as a combinatorial optimization problem and uses
quantum-inspired techniques to find near-optimal schedules.

Key concepts:
- Tasks and agents form a bipartite assignment problem
- QAOA uses alternating mixer and cost operators
- Variationally optimizes schedule quality
- Achieves better task distribution than greedy approaches
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Capabilities that agents may have."""
    CODE_ANALYSIS = "code_analysis"
    FILE_EDITING = "file_editing"
    RESEARCH = "research"
    TESTING = "testing"
    PLANNING = "planning"
    VERIFICATION = "verification"


@dataclass
class SchedulableTask:
    """A task that can be scheduled to an agent."""
    id: str
    description: str
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    estimated_tokens: int = 1000
    priority: float = 1.0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SchedulableAgent:
    """An agent that can execute tasks."""
    id: str
    name: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    context_capacity: int = 300000  # Tokens
    current_load: int = 0


@dataclass
class TaskAssignment:
    """Assignment of a task to an agent."""
    task: SchedulableTask
    agent: SchedulableAgent
    score: float  # Quality of this assignment


class QAOAScheduler:
    """
    QAOA-inspired task scheduler for optimal subagent assignment.
    
    The scheduler treats task assignment as a QUBO (Quadratic Unconstrained
    Binary Optimization) problem and uses QAOA-inspired variational
    optimization to find high-quality schedules.
    
    Key algorithm:
    1. Encode assignments as binary variables
    2. Define cost function (maximize quality, minimize conflicts)
    3. Apply alternating mixer and phase operators
    4. Measure and select best schedule
    """
    
    def __init__(
        self,
        qaoa_layers: int = 3,
        optimization_rounds: int = 10,
        beta_range: Tuple[float, float] = (0.1, 1.5),
        gamma_range: Tuple[float, float] = (0.5, 2.5),
    ):
        """
        Initialize QAOA scheduler.
        
        Args:
            qaoa_layers: Number of QAOA layers (p parameter)
            optimization_rounds: Variational optimization iterations
            beta_range: Range for mixer angle parameters
            gamma_range: Range for cost angle parameters
        """
        self.layers = qaoa_layers
        self.opt_rounds = optimization_rounds
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        
    def _compute_assignment_cost(
        self,
        task: SchedulableTask,
        agent: SchedulableAgent,
    ) -> float:
        """
        Compute cost of assigning a task to an agent.
        
        Lower cost = better assignment.
        
        Factors:
        - Capability match (required capabilities must be present)
        - Load balancing (prefer agents with lower load)
        - Context capacity (task must fit)
        """
        # Check hard constraints
        for cap in task.required_capabilities:
            if cap not in agent.capabilities:
                return float('inf')  # Invalid assignment
                
        if agent.current_load + task.estimated_tokens > agent.context_capacity:
            return float('inf')  # Capacity exceeded
            
        # Soft costs
        load_cost = agent.current_load / agent.context_capacity  # Prefer idle agents
        
        # Capability bonus (more matching capabilities = better)
        cap_overlap = len(set(task.required_capabilities) & set(agent.capabilities))
        cap_bonus = -0.2 * cap_overlap  # Negative cost = bonus
        
        # Priority scaling
        priority_factor = 1.0 / (task.priority + 0.1)
        
        return (load_cost + cap_bonus) * priority_factor
    
    def _build_cost_matrix(
        self,
        tasks: List[SchedulableTask],
        agents: List[SchedulableAgent],
    ) -> np.ndarray:
        """
        Build cost matrix for all task-agent pairs.
        
        Returns:
            Cost matrix [num_tasks, num_agents]
        """
        n_tasks = len(tasks)
        n_agents = len(agents)
        
        cost_matrix = np.zeros((n_tasks, n_agents))
        
        for i, task in enumerate(tasks):
            for j, agent in enumerate(agents):
                cost_matrix[i, j] = self._compute_assignment_cost(task, agent)
                
        return cost_matrix
    
    def _apply_mixer_operator(
        self,
        state: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        """
        Apply QAOA mixer operator (X rotations).
        
        The mixer creates superpositions, allowing exploration
        of different assignment combinations.
        
        For classical simulation, we use "soft" bit flips
        with probability related to beta.
        """
        n_vars = len(state)
        
        # Mixer creates transitions between computational basis states
        # Classically: probabilistic bit flips
        flip_prob = np.sin(beta) ** 2
        
        flip_mask = np.random.random(n_vars) < flip_prob
        new_state = state.copy()
        new_state[flip_mask] = 1 - new_state[flip_mask]
        
        return new_state
    
    def _apply_cost_operator(
        self,
        state: np.ndarray,
        cost_matrix: np.ndarray,
        gamma: float,
        n_tasks: int,
        n_agents: int,
    ) -> float:
        """
        Apply QAOA cost/phase operator.
        
        Computes the cost of current state and returns it.
        In quantum QAOA, this would apply a phase rotation.
        Here we return the cost for classical selection.
        """
        # Reshape state to assignment matrix
        assignment = state.reshape(n_tasks, n_agents)
        
        # Compute total cost
        total_cost = 0.0
        
        # Assignment costs
        total_cost += np.sum(assignment * cost_matrix)
        
        # Penalty for invalid assignments (task assigned to multiple agents)
        row_sums = assignment.sum(axis=1)
        constraint_violation = np.sum((row_sums - 1) ** 2)
        total_cost += 10.0 * constraint_violation  # Hard constraint penalty
        
        # Penalty for unassigned tasks
        unassigned = np.sum(row_sums == 0)
        total_cost += 20.0 * unassigned
        
        return total_cost * gamma
    
    def _decode_state(
        self,
        state: np.ndarray,
        tasks: List[SchedulableTask],
        agents: List[SchedulableAgent],
        cost_matrix: np.ndarray,
    ) -> List[TaskAssignment]:
        """
        Decode binary state into task assignments.
        """
        n_tasks = len(tasks)
        n_agents = len(agents)
        
        assignment_matrix = state.reshape(n_tasks, n_agents)
        assignments = []
        
        for i, task in enumerate(tasks):
            # Find assigned agent (argmax for this task)
            agent_scores = assignment_matrix[i]
            if agent_scores.max() > 0.5:
                agent_idx = np.argmax(agent_scores)
                agent = agents[agent_idx]
                score = 1.0 / (cost_matrix[i, agent_idx] + 1e-8)
                
                assignments.append(TaskAssignment(
                    task=task,
                    agent=agent,
                    score=score,
                ))
                
        return assignments
    
    def schedule(
        self,
        tasks: List[SchedulableTask],
        agents: List[SchedulableAgent],
    ) -> Tuple[List[TaskAssignment], Dict[str, Any]]:
        """
        Compute optimal task schedule using QAOA-inspired optimization.
        
        Args:
            tasks: Tasks to schedule
            agents: Available agents
            
        Returns:
            Tuple of (assignments, schedule_stats)
        """
        if not tasks or not agents:
            return [], {"error": "No tasks or agents"}
            
        n_tasks = len(tasks)
        n_agents = len(agents)
        n_vars = n_tasks * n_agents
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(tasks, agents)
        
        # Initialize with greedy assignment
        best_state = np.zeros(n_vars)
        for i in range(n_tasks):
            # Assign to lowest cost valid agent
            valid_costs = cost_matrix[i].copy()
            valid_costs[valid_costs == float('inf')] = 1e10
            best_agent = np.argmin(valid_costs)
            best_state[i * n_agents + best_agent] = 1
            
        best_cost = self._apply_cost_operator(
            best_state, cost_matrix, 1.0, n_tasks, n_agents
        )
        
        # QAOA-inspired variational optimization
        samples_evaluated = 0
        
        for opt_round in range(self.opt_rounds):
            # Sample variational parameters
            betas = np.random.uniform(*self.beta_range, self.layers)
            
            # Start from uniform superposition (random state)
            state = (np.random.random(n_vars) > 0.5).astype(float)
            
            # Apply QAOA layers
            for layer in range(self.layers):
                # Mixer
                state = self._apply_mixer_operator(state, betas[layer])
                
                # Enforce one-hot constraint per task
                assignment = state.reshape(n_tasks, n_agents)
                for i in range(n_tasks):
                    if assignment[i].sum() == 0:
                        # No assignment - pick random
                        valid = np.where(cost_matrix[i] < float('inf'))[0]
                        if len(valid) > 0:
                            assignment[i, np.random.choice(valid)] = 1
                    elif assignment[i].sum() > 1:
                        # Multiple - keep best
                        costs = assignment[i] * cost_matrix[i]
                        costs[assignment[i] == 0] = float('inf')
                        best = np.argmin(costs)
                        assignment[i] = 0
                        assignment[i, best] = 1
                state = assignment.flatten()
            
            # Evaluate cost
            cost = self._apply_cost_operator(
                state, cost_matrix, 1.0, n_tasks, n_agents
            )
            samples_evaluated += 1
            
            if cost < best_cost:
                best_cost = cost
                best_state = state.copy()
                
        # Decode best state
        assignments = self._decode_state(best_state, tasks, agents, cost_matrix)
        
        stats = {
            "qaoa_layers": self.layers,
            "optimization_rounds": self.opt_rounds,
            "samples_evaluated": samples_evaluated,
            "best_cost": best_cost,
            "assignments_made": len(assignments),
            "unassigned_tasks": n_tasks - len(assignments),
        }
        
        logger.info(f"QAOA scheduling: {len(assignments)}/{n_tasks} tasks assigned, "
                    f"cost={best_cost:.4f}")
        
        return assignments, stats


class SubagentScheduler:
    """
    High-level scheduler for distributing work across subagents.
    """
    
    def __init__(self, quantum_inspired: bool = True):
        """
        Initialize subagent scheduler.
        
        Args:
            quantum_inspired: Use QAOA-inspired optimization (vs greedy)
        """
        self.use_qaoa = quantum_inspired
        self.qaoa = QAOAScheduler() if quantum_inspired else None
        
    def schedule_tasks(
        self,
        task_descriptions: List[str],
        available_agents: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """
        Schedule tasks across available agents.
        
        Args:
            task_descriptions: List of task descriptions
            available_agents: Agent info dicts with name, capabilities, capacity
            
        Returns:
            Dict mapping agent names to assigned task descriptions
        """
        # Convert to internal format
        tasks = [
            SchedulableTask(
                id=f"task_{i}",
                description=desc,
                estimated_tokens=len(desc) * 2,  # Rough estimate
            )
            for i, desc in enumerate(task_descriptions)
        ]
        
        agents = [
            SchedulableAgent(
                id=f"agent_{i}",
                name=agent.get("name", f"Agent-{i}"),
                capabilities=[
                    AgentCapability(c) for c in agent.get("capabilities", [])
                    if c in [cap.value for cap in AgentCapability]
                ],
                context_capacity=agent.get("capacity", 300000),
            )
            for i, agent in enumerate(available_agents)
        ]
        
        if not agents:
            # No agents - return empty
            return {}
            
        if self.use_qaoa and self.qaoa is not None:
            assignments, _ = self.qaoa.schedule(tasks, agents)
        else:
            # Greedy fallback
            assignments = []
            for task in tasks:
                for agent in agents:
                    if agent.current_load + task.estimated_tokens <= agent.context_capacity:
                        assignments.append(TaskAssignment(task, agent, 1.0))
                        agent.current_load += task.estimated_tokens
                        break
        
        # Build result
        result: Dict[str, List[str]] = {}
        for assignment in assignments:
            agent_name = assignment.agent.name
            if agent_name not in result:
                result[agent_name] = []
            result[agent_name].append(assignment.task.description)
            
        return result
