"""
Database models for ASHES system.

SQLAlchemy models for persistent data storage.
"""

from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    permissions = Column(JSON, default=list)
    
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")


class Experiment(Base):
    """Experiment model for tracking scientific experiments."""
    
    __tablename__ = "experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="created", index=True)
    priority = Column(Integer, default=1)
    
    # Scientific content
    initial_hypothesis = Column(Text, nullable=True)
    current_hypothesis = Column(Text, nullable=True)
    evolved_hypothesis = Column(Text, nullable=True)
    
    # Workflow data
    literature_review = Column(JSON, nullable=True)
    experiment_design = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    analysis = Column(JSON, nullable=True)
    publication = Column(JSON, nullable=True)
    
    # Configuration
    parameters = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    total_time = Column(Float, nullable=True)
    agent_response_times = Column(JSON, default=dict)
    laboratory_execution_time = Column(Float, nullable=True)
    
    # Error handling
    error = Column(Text, nullable=True)
    warnings = Column(JSON, default=list)
    
    # Foreign keys
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Relationships
    created_by_user = relationship("User", back_populates="experiments")
    interactions = relationship("AgentInteraction", back_populates="experiment")
    lab_events = relationship("LaboratoryEvent", back_populates="experiment")
    safety_events = relationship("SafetyEvent", back_populates="experiment")


class AgentInteraction(Base):
    """Agent interaction tracking."""
    
    __tablename__ = "agent_interactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    agent_type = Column(String(50), nullable=False)
    agent_id = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    
    # Performance
    duration = Column(Float, nullable=False)
    success = Column(Boolean, default=True)
    
    # Data
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="interactions")


class LaboratoryEvent(Base):
    """Laboratory equipment event tracking."""
    
    __tablename__ = "laboratory_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=True)
    device_name = Column(String(100), nullable=False, index=True)
    device_type = Column(String(50), nullable=False)
    event_type = Column(String(50), nullable=False)
    
    # Event data
    event_data = Column(JSON, nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="lab_events")


class SafetyEvent(Base):
    """Safety event tracking."""
    
    __tablename__ = "safety_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=True)
    safety_level = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    
    # Event details
    event_data = Column(JSON, nullable=True)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="safety_events")


class DeviceState(Base):
    """Laboratory device state tracking."""
    
    __tablename__ = "device_states"
    
    device_id = Column(String(100), primary_key=True)
    device_type = Column(String(50), nullable=False)
    device_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="idle")
    
    # Connection info
    ip_address = Column(String(15), nullable=True)
    port = Column(Integer, nullable=True)
    connected = Column(Boolean, default=False)
    
    # Performance metrics
    total_operations = Column(Integer, default=0)
    successful_operations = Column(Integer, default=0)
    failed_operations = Column(Integer, default=0)
    
    # Current operation
    current_experiment_id = Column(UUID(as_uuid=True), nullable=True)
    current_operation = Column(String(100), nullable=True)
    operation_started_at = Column(DateTime, nullable=True)
    
    # Safety and maintenance
    safety_status = Column(String(20), default="safe")
    last_safety_check = Column(DateTime, nullable=True)
    last_maintenance = Column(DateTime, nullable=True)
    next_maintenance = Column(DateTime, nullable=True)
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemMetrics(Base):
    """System performance metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    
    # Context
    tags = Column(JSON, default=dict)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class AuditLog(Base):
    """Audit log for security and compliance."""
    
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    
    # Details
    details = Column(JSON, nullable=True)
    ip_address = Column(String(15), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Result
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


class KnowledgeBase(Base):
    """Scientific knowledge base entries."""
    
    __tablename__ = "knowledge_base"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False)  # hypothesis, result, publication, etc.
    domain = Column(String(100), nullable=False, index=True)
    
    # Additional metadata
    additional_metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)
    confidence_score = Column(Float, default=0.0)
    
    # Vector embedding for similarity search
    embedding = Column(JSON, nullable=True)
    
    # Source tracking
    source_experiment_id = Column(UUID(as_uuid=True), nullable=True)
    source_type = Column(String(50), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Publication(Base):
    """Generated scientific publications."""
    
    __tablename__ = "publications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(300), nullable=False)
    abstract = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    
    # Metadata
    authors = Column(JSON, default=list)
    keywords = Column(JSON, default=list)
    research_domain = Column(String(100), nullable=False)
    
    # Publication details
    status = Column(String(20), default="draft")  # draft, submitted, published
    target_journal = Column(String(200), nullable=True)
    doi = Column(String(100), nullable=True)
    
    # Source experiments
    source_experiments = Column(JSON, default=list)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
