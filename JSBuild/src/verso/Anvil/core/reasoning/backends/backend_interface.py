from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class CoconutBackend(ABC):
    """
    Abstract base class for COCONUT reasoning backends.
    Allows for multiple implementations (NumPy, TensorFlow, PyTorch).
    """

    @abstractmethod
    def __init__(self, **config):
        """Initialize the backend with configuration."""
        pass

    @abstractmethod
    def explore(self, context_embedding: np.ndarray) -> np.ndarray:
        """
        Explore latent reasoning paths.

        Args:
            context_embedding: Input embedding [Batch, Dim] or [Batch, Seq, Dim]

        Returns:
            refined_embedding: Enhanced embedding after reasoning [Batch, Dim]
        """
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return information about the device being used."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend and its dependencies are available."""
        pass

    def get_last_session_record(self) -> Dict[str, Any] | None:
        """Return an additive latent-session record when the backend provides one."""
        return None
