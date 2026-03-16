"""Ownership subsystem for multi-agent coordination."""

from core.ownership.file_ownership import FileOwnershipRegistry
from core.ownership.ownership_crdt import LamportClock, LWWEntry, OwnershipCRDT
from core.ownership.ownership_models import (
    AccessDecision,
    ClaimResult,
    DeniedFile,
    OwnershipRecord,
    PhaseStartResult,
    PhaseStatus,
)
from core.ownership.phase_manager import PhaseManager
from core.ownership.ownership_sync import OwnershipSyncProtocol

__all__ = [
    "AccessDecision",
    "ClaimResult",
    "DeniedFile",
    "FileOwnershipRegistry",
    "LamportClock",
    "LWWEntry",
    "OwnershipCRDT",
    "OwnershipRecord",
    "OwnershipSyncProtocol",
    "PhaseManager",
    "PhaseStartResult",
    "PhaseStatus",
]
