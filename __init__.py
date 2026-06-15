"""
Autonomous Data Science AI Package
An intelligent, zero-hardcoded-rules data science assistant.
"""

__version__ = "1.0.0"
__author__ = "Autonomous Data Science AI Team"
__license__ = "MIT"

from .base_agent import BaseAgent
from .smart_orchestrator_v2 import SmartOrchestratorV2

__all__ = [
    "BaseAgent",
    "SmartOrchestratorV2",
]
