"""
ASHES - Autonomous Scientific Hypothesis Evolution System

The world's first fully autonomous scientific research platform that combines
cutting-edge agentic AI with advanced laboratory automation to accelerate
scientific discovery beyond human-only capabilities.

Copyright (c) 2025 ASHES Development Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ASHES Development Team"
__email__ = "team@ashes-lab.com"
__description__ = "Autonomous Scientific Hypothesis Evolution System"

from .core.orchestrator import ASHESOrchestrator
from .agents.base import BaseAgent

# Use minimal lab controller if full one fails to import
try:
    from .laboratory.controller import LabController
except ImportError:
    from .laboratory.controller_minimal import LabController

from .api.main import create_app

__all__ = [
    "ASHESOrchestrator",
    "BaseAgent", 
    "LabController",
    "create_app"
]
