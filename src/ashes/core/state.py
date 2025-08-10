"""
System and experiment state management for ASHES.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

# Import Task from agents.base for compatibility
from ..agents.base import Task, TaskStatus, TaskPriority, TaskResult


class SystemStatus(Enum):
    """System status enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    REJECTED_SAFETY = "rejected_safety"
    REJECTED_ETHICS = "rejected_ethics"


@dataclass
class SystemState:
    """Global system state tracking."""
    
    status: str = SystemStatus.INITIALIZING.value
    started_at: Optional[datetime] = None
    version: str = "1.0.0"
    
    # Performance metrics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # Component status
    agents_status: Dict[str, str] = field(default_factory=dict)
    laboratory_status: Dict[str, str] = field(default_factory=dict)
    database_status: Dict[str, str] = field(default_factory=dict)
    
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_status(self, status: SystemStatus):
        """Update system status."""
        self.status = status.value
        self.last_updated = datetime.utcnow()
    
    def increment_experiments(self, success: bool = True):
        """Increment experiment counters."""
        self.total_experiments += 1
        if success:
            self.successful_experiments += 1
        else:
            self.failed_experiments += 1
        self.last_updated = datetime.utcnow()


@dataclass
class ExperimentState:
    """Individual experiment state tracking."""
    
    # Basic information
    id: str
    domain: str
    priority: int = 1
    status: str = ExperimentStatus.CREATED.value
    
    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Scientific content
    initial_hypothesis: Optional[str] = None
    current_hypothesis: Optional[str] = None
    evolved_hypothesis: Optional[str] = None
    
    # Workflow stages
    literature_review: Optional[Dict[str, Any]] = None
    experiment_design: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    publication: Optional[Dict[str, Any]] = None
    
    # Configuration and parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking and metrics
    agent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    laboratory_events: List[Dict[str, Any]] = field(default_factory=list)
    safety_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_time: Optional[float] = None
    agent_response_times: Dict[str, float] = field(default_factory=dict)
    laboratory_execution_time: Optional[float] = None
    
    def update_status(self, status: ExperimentStatus):
        """Update experiment status."""
        self.status = status.value
        
        if status == ExperimentStatus.RUNNING and not self.started_at:
            self.started_at = datetime.utcnow()
        elif status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.STOPPED]:
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
                if self.started_at:
                    self.total_time = (self.completed_at - self.started_at).total_seconds()
    
    def add_agent_interaction(self, agent_type: str, action: str, duration: float, **kwargs):
        """Record an agent interaction."""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_type": agent_type,
            "action": action,
            "duration": duration,
            **kwargs
        }
        self.agent_interactions.append(interaction)
        self.agent_response_times[f"{agent_type}_{action}"] = duration
    
    def add_laboratory_event(self, device_name: str, event_type: str, **kwargs):
        """Record a laboratory event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "device_name": device_name,
            "event_type": event_type,
            **kwargs
        }
        self.laboratory_events.append(event)
    
    def add_safety_event(self, safety_level: str, description: str, **kwargs):
        """Record a safety event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "safety_level": safety_level,
            "description": description,
            **kwargs
        }
        self.safety_events.append(event)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(f"{datetime.utcnow().isoformat()}: {warning}")
    
    def get_progress_percentage(self) -> float:
        """Calculate experiment progress percentage."""
        stages = [
            "hypothesis_generated",
            "literature_reviewed", 
            "experiment_designed",
            "safety_approved",
            "laboratory_executed",
            "results_analyzed",
            "hypothesis_evolved",
            "publication_generated"
        ]
        
        completed_stages = 0
        
        if self.current_hypothesis:
            completed_stages += 1
        if self.literature_review:
            completed_stages += 1
        if self.experiment_design:
            completed_stages += 1
        if self.status not in [ExperimentStatus.REJECTED_SAFETY.value, ExperimentStatus.REJECTED_ETHICS.value]:
            completed_stages += 1
        if self.results:
            completed_stages += 1
        if self.analysis:
            completed_stages += 1
        if self.evolved_hypothesis:
            completed_stages += 1
        if self.publication:
            completed_stages += 1
        
        return (completed_stages / len(stages)) * 100.0


@dataclass
class AgentState:
    """Individual agent state tracking."""
    
    agent_id: str
    agent_type: str
    status: str = "idle"
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    # Current task
    current_experiment_id: Optional[str] = None
    current_task: Optional[str] = None
    task_started_at: Optional[datetime] = None
    
    # Resource usage
    memory_usage: float = 0.0
    
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_metrics(self, response_time: float, success: bool = True):
        """Update agent performance metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Update average response time
        self.average_response_time = (
            (self.average_response_time * (self.total_requests - 1) + response_time) 
            / self.total_requests
        )
        
        self.last_updated = datetime.utcnow()


@dataclass
class LaboratoryDeviceState:
    """Laboratory device state tracking."""
    
    device_id: str
    device_type: str
    device_name: str
    status: str = "idle"
    
    # Connection information
    ip_address: Optional[str] = None
    port: Optional[int] = None
    connected: bool = False
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Current operation
    current_experiment_id: Optional[str] = None
    current_operation: Optional[str] = None
    operation_started_at: Optional[datetime] = None
    
    # Safety status
    safety_status: str = "safe"
    last_safety_check: Optional[datetime] = None
    
    # Maintenance
    last_maintenance: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_operation_metrics(self, success: bool = True):
        """Update device operation metrics."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        self.last_updated = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Calculate device success rate."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100.0
