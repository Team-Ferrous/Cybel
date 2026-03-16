"""Deterministic synthesis primitives for Saguaro."""

from .assembler import AssemblyPlan, ComponentAssembler
from .builder import DeterministicSynthesisBuilder
from .ast_builder import ASTBuilder, Emitter, SagASTNode, SagFunction, SagParameter
from .adapter_generator import AdapterGenerator, AdapterField, AdapterPlan
from .cache import SynthesisCache, SynthesisMemoryRecord
from .component_catalog import ComponentCatalog, ComponentDescriptor
from .contract_harvester import ContractHarvester, HarvestedContract
from .effects import BudgetConstraint, EffectEvaluation, ForbiddenFlow, SynthesisEffectEngine
from .eqsat_runner import BoundedEqsatRunner, EqsatResult, RewriteApplication
from .math_kernel_ir import MathKernelCompiler, MathKernelIR
from .patch_inductor import PatchInductor, SemanticPatchRule
from .policy import StrategyDecision, SynthesisPolicyEngine
from .portfolio_search import CandidateMeritScore, PortfolioSearch, SearchBudget
from .program_lattice import LatticeCandidate, ProgramLattice
from .proof_capsule import SynthesisProofCapsule
from .replay_tape import SynthesisReplayTape
from .solver import DeterministicSolver, ProofResult, SolverCounterexample
from .spec import SagSpec, SpecConstraint, SpecEvidenceRef, SpecLowerer
from .spec_lint import SpecLintIssue, SpecLintResult, lint_sagspec
from .translation_validator import TranslationValidator, TranslationWitness
from .variant_selector import HardwareAwareVariantSelector, VariantChoice

__all__ = [
    "AssemblyPlan",
    "ASTBuilder",
    "AdapterField",
    "AdapterGenerator",
    "AdapterPlan",
    "BoundedEqsatRunner",
    "BudgetConstraint",
    "CandidateMeritScore",
    "ComponentAssembler",
    "ComponentCatalog",
    "ComponentDescriptor",
    "ContractHarvester",
    "DeterministicSolver",
    "EffectEvaluation",
    "Emitter",
    "EqsatResult",
    "ForbiddenFlow",
    "HarvestedContract",
    "HardwareAwareVariantSelector",
    "LatticeCandidate",
    "MathKernelCompiler",
    "MathKernelIR",
    "DeterministicSynthesisBuilder",
    "PatchInductor",
    "PortfolioSearch",
    "ProofResult",
    "ProgramLattice",
    "RewriteApplication",
    "SagASTNode",
    "SagFunction",
    "SagParameter",
    "SagSpec",
    "SearchBudget",
    "SemanticPatchRule",
    "SolverCounterexample",
    "SpecConstraint",
    "SpecEvidenceRef",
    "SpecLintIssue",
    "SpecLintResult",
    "SpecLowerer",
    "StrategyDecision",
    "SynthesisCache",
    "SynthesisEffectEngine",
    "SynthesisMemoryRecord",
    "SynthesisPolicyEngine",
    "SynthesisProofCapsule",
    "SynthesisReplayTape",
    "TranslationValidator",
    "TranslationWitness",
    "VariantChoice",
    "lint_sagspec",
]
