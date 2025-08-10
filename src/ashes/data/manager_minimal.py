"""Data Manager - Simplified version for testing."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataManager:
    """Simplified Data Manager for testing."""
    
    def __init__(self):
        self.experiments = []
        self.results = []
        logger.info("Data Manager initialized (minimal mode)")
    
    async def start(self):
        """Start the data manager."""
        logger.info("Data Manager started")
    
    async def stop(self):
        """Stop the data manager."""
        logger.info("Data Manager stopped")
    
    async def store_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Store experiment data (simulated)."""
        experiment_id = experiment_data.get("id", f"exp_{len(self.experiments)}")
        self.experiments.append(experiment_data)
        logger.info(f"Stored experiment: {experiment_id}")
        return experiment_id
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment data."""
        for exp in self.experiments:
            if exp.get("id") == experiment_id:
                return exp
        return None
    
    async def store_result(self, result_data: Dict[str, Any]) -> str:
        """Store experiment result."""
        result_id = result_data.get("id", f"result_{len(self.results)}")
        self.results.append(result_data)
        logger.info(f"Stored result: {result_id}")
        return result_id
    
    async def get_status(self) -> Dict[str, Any]:
        """Get data manager status."""
        return {
            "experiments_count": len(self.experiments),
            "results_count": len(self.results),
            "timestamp": datetime.now().isoformat()
        }
