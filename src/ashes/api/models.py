"""
Pydantic models for ASHES API.

This module defines all the request and response models used by the API endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ExperimentStatus(str, Enum):
    """Experiment status enumeration."""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    REJECTED_SAFETY = "rejected_safety"
    REJECTED_ETHICS = "rejected_ethics"


class AgentType(str, Enum):
    """Agent type enumeration."""
    THEORIST = "theorist"
    EXPERIMENTALIST = "experimentalist"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    ETHICS = "ethics"


class DeviceStatus(str, Enum):
    """Device status enumeration."""
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    OFFLINE = "offline"


# Authentication Models
class TokenResponse(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class UserResponse(BaseModel):
    """User information response model."""
    username: str
    email: str
    role: str
    permissions: List[str] = []
    last_login: Optional[datetime] = None


class UserCreate(BaseModel):
    """User creation request model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    role: str = "user"


# Experiment Models
class ExperimentCreate(BaseModel):
    """Experiment creation request model."""
    research_domain: str = Field(..., description="Scientific research domain")
    initial_hypothesis: Optional[str] = Field(None, description="Initial hypothesis (optional)")
    priority: int = Field(default=1, ge=1, le=10, description="Experiment priority (1-10)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class ExperimentResponse(BaseModel):
    """Basic experiment response model."""
    id: str
    status: ExperimentStatus
    research_domain: str
    initial_hypothesis: Optional[str] = None
    current_hypothesis: Optional[str] = None
    priority: int
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    progress_percentage: float = 0.0


class ExperimentDetailResponse(BaseModel):
    """Detailed experiment response model."""
    id: str
    status: ExperimentStatus
    research_domain: str
    initial_hypothesis: Optional[str] = None
    current_hypothesis: Optional[str] = None
    evolved_hypothesis: Optional[str] = None
    priority: int
    
    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Workflow stages
    literature_review: Optional[Dict[str, Any]] = None
    experiment_design: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    publication: Optional[Dict[str, Any]] = None
    
    # Tracking
    agent_interactions: List[Dict[str, Any]] = []
    laboratory_events: List[Dict[str, Any]] = []
    safety_events: List[Dict[str, Any]] = []
    
    # Performance
    total_time: Optional[float] = None
    agent_response_times: Dict[str, float] = {}
    laboratory_execution_time: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = []
    
    progress_percentage: float = 0.0


class ExperimentUpdate(BaseModel):
    """Experiment update request model."""
    priority: Optional[int] = Field(None, ge=1, le=10)
    parameters: Optional[Dict[str, Any]] = None


# Agent Models
class AgentCapabilityInfo(BaseModel):
    """Agent capability information model."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    estimated_duration: float
    confidence_level: float


class AgentPerformance(BaseModel):
    """Agent performance metrics model."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_response_time: float


class AgentDetailResponse(BaseModel):
    """Detailed agent information response model."""
    agent_id: str
    agent_type: AgentType
    status: str
    current_task: Optional[str] = None
    queue_size: int
    capabilities: List[str]
    performance: AgentPerformance


class AgentStatusResponse(BaseModel):
    """Agent manager status response model."""
    total_agents: int
    ready_agents: int
    busy_agents: int
    agent_types: Dict[str, int]
    agents: Dict[str, AgentDetailResponse]
    total_requests: int


class AgentScaleRequest(BaseModel):
    """Agent scaling request model."""
    target_count: int = Field(..., ge=1, le=20, description="Target number of agents")


class AgentTaskRequest(BaseModel):
    """Agent task execution request model."""
    capability: str
    parameters: Dict[str, Any]
    timeout: float = 300.0


# Laboratory Models
class DeviceResponse(BaseModel):
    """Laboratory device response model."""
    device_id: str
    device_type: str
    device_name: str
    status: DeviceStatus
    ip_address: Optional[str] = None
    port: Optional[int] = None
    connected: bool = False
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    success_rate: float = 100.0
    
    # Current operation
    current_experiment_id: Optional[str] = None
    current_operation: Optional[str] = None
    operation_started_at: Optional[datetime] = None
    
    # Safety and maintenance
    safety_status: str = "safe"
    last_safety_check: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    
    last_updated: datetime


class LabStatusResponse(BaseModel):
    """Laboratory status response model."""
    total_devices: int
    online_devices: int
    busy_devices: int
    error_devices: int
    devices: List[DeviceResponse]
    safety_status: str = "safe"
    emergency_stop_active: bool = False


class DeviceCommand(BaseModel):
    """Device command request model."""
    command: str
    parameters: Dict[str, Any] = {}
    safety_override: bool = False


class ExperimentProtocol(BaseModel):
    """Experiment protocol model."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    materials: List[str]
    equipment: List[str]
    safety_protocols: List[str]
    estimated_duration: float


# System Models
class SystemHealth(BaseModel):
    """System health metrics model."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    disk_usage: float
    network_status: str = "healthy"


class ComponentStatus(BaseModel):
    """System component status model."""
    name: str
    status: str
    health: str
    last_updated: datetime
    details: Dict[str, Any] = {}


class SystemStatusResponse(BaseModel):
    """System status response model."""
    status: str
    version: str = "1.0.0"
    uptime: Optional[float] = None
    
    # Performance metrics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    success_rate: float = 0.0
    
    # System health
    health: SystemHealth
    components: List[ComponentStatus] = []
    
    last_updated: datetime


# Data Models
class ExperimentResults(BaseModel):
    """Experiment results model."""
    experiment_id: str
    results: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None
    visualizations: List[Dict[str, Any]] = []
    confidence_score: float = 0.0
    timestamp: datetime


class DashboardMetrics(BaseModel):
    """Dashboard metrics model."""
    experiments: Dict[str, Union[int, float]]
    agents: Dict[str, Any]
    laboratory: Dict[str, Any]
    system_health: str
    recent_activities: List[Dict[str, Any]] = []


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Dict[str, Any] = {}
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    took_ms: float


# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class NotificationPreferences(BaseModel):
    """User notification preferences model."""
    experiment_updates: bool = True
    system_alerts: bool = True
    agent_notifications: bool = False
    laboratory_events: bool = True
    email_notifications: bool = False


# Configuration Models
class SystemConfig(BaseModel):
    """System configuration model."""
    max_concurrent_experiments: int = 10
    default_experiment_timeout: int = 3600
    auto_cleanup_completed: bool = True
    enable_safety_monitoring: bool = True
    log_level: str = "INFO"


class ExportRequest(BaseModel):
    """Data export request model."""
    format: str = Field(..., pattern=r'^(json|csv|xlsx|pdf)$')
    data_type: str = Field(..., pattern=r'^(experiments|agents|laboratory|system)$')
    date_range: Optional[Dict[str, datetime]] = None
    filters: Dict[str, Any] = {}


# Error Models
class ErrorResponse(BaseModel):
    """API error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ValidationError(BaseModel):
    """Validation error model."""
    field: str
    message: str
    value: Any


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    error: str = "validation_error"
    message: str = "Input validation failed"
    errors: List[ValidationError]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
