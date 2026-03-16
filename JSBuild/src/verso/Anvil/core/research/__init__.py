"""Research operating-system primitives for autonomy campaigns."""

from core.research.browser_runtime import BrowserResearchRuntime
from core.research.clusterer import TopicClusterer
from core.research.crawler import ResearchCrawler
from core.research.eid_master import EIDMasterLoop
from core.research.experiment_runner import ExperimentRunner
from core.research.hypothesis_lab import HypothesisLab
from core.research.normalizer import ResearchNormalizer
from core.research.repo_acquisition import RepoAcquisitionService
from core.research.research_evals import ResearchEvaluationHarness
from core.research.store import ResearchStore

EIDMaster = EIDMasterLoop

__all__ = [
    "BrowserResearchRuntime",
    "EIDMaster",
    "EIDMasterLoop",
    "ExperimentRunner",
    "HypothesisLab",
    "RepoAcquisitionService",
    "ResearchCrawler",
    "ResearchEvaluationHarness",
    "ResearchNormalizer",
    "ResearchStore",
    "TopicClusterer",
]
