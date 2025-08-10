"""Laboratory Controller - Simplified version for testing."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LabController:
    """Simplified Laboratory Controller for testing."""
    
    def __init__(self):
        self.is_running = False
        self.experiments = []
        logger.info("Lab Controller initialized (minimal mode)")
    
    async def start(self):
        """Start the laboratory controller."""
        self.is_running = True
        logger.info("Lab Controller started")
    
    async def stop(self):
        """Stop the laboratory controller."""
        self.is_running = False
        logger.info("Lab Controller stopped")
    
    async def execute_experiment(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an experiment (simulated)."""
        logger.info(f"Executing experiment: {experiment_data.get('name', 'Unknown')}")
        
        # Simulate experiment execution
        await asyncio.sleep(0.1)
        
        result = {
            "experiment_id": experiment_data.get("id", "test_experiment"),
            "status": "completed",
            "result": {"success": True, "data": "simulated_result"},
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "running": self.is_running,
            "experiments_count": len(self.experiments),
            "timestamp": datetime.now().isoformat()
        }
