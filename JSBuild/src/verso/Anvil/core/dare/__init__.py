"""Deep Analysis & Autonomous Research Engine (DARE)."""

from .campaign_sculptor import CampaignSculptor
from .deep_analyzer import DeepAnalyzer
from .knowledge_base import DareKnowledgeBase
from .pipeline import DarePipeline
from .refinement import RefinementProtocol
from .repo_ingestion import RepoIngestionEngine
from .synthesizer import NovelMethodSynthesizer
from .web_research import WebResearchEngine

__all__ = [
    "CampaignSculptor",
    "DareKnowledgeBase",
    "DarePipeline",
    "DeepAnalyzer",
    "NovelMethodSynthesizer",
    "RefinementProtocol",
    "RepoIngestionEngine",
    "WebResearchEngine",
]
