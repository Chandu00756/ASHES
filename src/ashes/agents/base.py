"""
Enterprise-grade base agent class for ASHES AI Agent System.

This module provides the foundational architecture for all AI agents
in the ASHES system, implementing distributed agent communication,
task orchestration, and enterprise-grade capabilities.
"""

import asyncio
import uuid
import logging
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# AI/ML imports
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
except ImportError:
    # Fallback for when LangChain is not available
    ChatOpenAI = None
    BaseMessage = None
    HumanMessage = None
    SystemMessage = None
    AIMessage = None
    BaseCallbackHandler = None
    ChatPromptTemplate = None

# Configuration
from ..core.logging import get_logger
from ..core.config import get_config



class AgentStatus(str, Enum):
    """Agent operational status."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class AgentCapability(str, Enum):
    """Standard agent capabilities."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    ETHICS_REVIEW = "ethics_review"
    EXPERIMENT_DESIGN = "experiment_design"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class Task:
    """Agent task definition."""
    id: str
    type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """Agent task execution result."""
    task_id: str
    agent_id: str
    status: str  # 'completed', 'failed', 'partial'
    result: Any
    execution_time: float
    timestamp: datetime
    error: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Standardized message format for agent communication."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }


class AgentMetrics:
    """Agent performance and health metrics."""
    
    def __init__(self):
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.average_response_time = 0.0
        self.last_activity = datetime.utcnow()
        self.error_count = 0
        self.uptime_start = datetime.utcnow()
    
    def record_task_completion(self, execution_time: float, success: bool = True):
        """Record task completion metrics."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
            self.error_count += 1
        
        self.total_execution_time += execution_time
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.average_response_time = self.total_execution_time / total_tasks
        
        self.last_activity = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 100.0
        return (self.tasks_completed / total_tasks) * 100.0
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return (datetime.utcnow() - self.uptime_start).total_seconds()


class AgentCallbackHandler:
    """Custom callback handler for LangChain operations."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = get_logger(f"agent.{agent_id}")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts."""
        self.logger.debug(f"LLM started for agent {self.agent_id}")
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Called when LLM ends."""
        self.logger.debug(f"LLM completed for agent {self.agent_id}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM errors."""
        self.logger.error(f"LLM error for agent {self.agent_id}: {error}")


class BaseAgent(ABC):
    """
    Enterprise base class for all ASHES AI agents.
    
    Provides comprehensive infrastructure for distributed agent operations,
    including task management, inter-agent communication, monitoring,
    and enterprise-grade reliability features.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.config = config or {}
        
        # State management
        self.status = AgentStatus.INITIALIZING
        self.current_task: Optional[Task] = None
        self.task_queue: List[Task] = []
        self.metrics = AgentMetrics()
        
        # Logging
        self.logger = get_logger(f"agent.{agent_type}.{agent_id}")
        
        # LLM setup
        self.llm_config = get_config()
        self.llm = self._initialize_llm()
        self.callback_handler = AgentCallbackHandler(agent_id)
        
        # Task execution
        self._running = False
        self._task_executor_task: Optional[asyncio.Task] = None
        
        # Inter-agent communication
        self.message_handlers: Dict[str, Callable] = {}
        self.subscriptions: List[str] = []
        
        # Health monitoring
        self.health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Memory and context
        self.short_term_memory = {}
        self.conversation_history = []
        self.context_window_size = 10
        
        # Safety and validation
        self.safety_enabled = True
        self.max_retries = 3
        
        self.logger.info(f"Initialized {agent_type} agent {agent_id}")
    
    def _initialize_llm(self):
        """Initialize the language model for this agent."""
        try:
            if ChatOpenAI:
                return ChatOpenAI(
                    model_name=self.config.get("model_name", "gpt-4"),
                    temperature=self.config.get("temperature", 0.7),
                    max_tokens=self.config.get("max_tokens", 2000),
                    callbacks=[self.callback_handler] if self.callback_handler else []
                )
            else:
                # Fallback placeholder when LangChain not available
                return "placeholder_llm"
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM: {e}")
            return "placeholder_llm"
    
    async def start(self):
        """Start the agent and begin processing tasks."""
        if self._running:
            return
        
        self._running = True
        self.status = AgentStatus.IDLE
        
        # Start task executor
        self._task_executor_task = asyncio.create_task(self._task_executor_loop())
        
        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent gracefully."""
        self._running = False
        self.status = AgentStatus.OFFLINE
        
        # Cancel tasks
        if self._task_executor_task:
            self._task_executor_task.cancel()
            try:
                await self._task_executor_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def _task_executor_loop(self):
        """Main task execution loop."""
        while self._running:
            try:
                if self.task_queue and self.status == AgentStatus.IDLE:
                    # Get next task
                    task = self.task_queue.pop(0)
                    await self._execute_task(task)
                else:
                    # No tasks, wait briefly
                    await asyncio.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Task executor error: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        self.status = AgentStatus.BUSY
        self.current_task = task
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing task {task.id} of type {task.type}")
            
            # Check dependencies
            if not await self._check_task_dependencies(task):
                raise Exception("Task dependencies not met")
            
            # Execute the task
            result = await self._process_task(task)
            
            execution_time = time.time() - start_time
            
            # Create task result
            task_result = TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status="completed",
                result=result,
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
            
            # Update metrics
            self.metrics.record_task_completion(execution_time, success=True)
            
            self.logger.info(f"Completed task {task.id} in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle task failure
            task_result = TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                status="failed",
                result=None,
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                error=str(e)
            )
            
            # Update metrics
            self.metrics.record_task_completion(execution_time, success=False)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.task_queue.insert(0, task)  # Retry at front of queue
                self.logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries})")
            else:
                self.logger.error(f"Task {task.id} failed permanently: {e}")
        
        finally:
            self.current_task = None
            self.status = AgentStatus.IDLE
        
        return task_result
    
    @abstractmethod
    async def _process_task(self, task: Task) -> Any:
        """
        Process a specific task. Must be implemented by subclasses.
        
        Args:
            task: The task to process
            
        Returns:
            The task result
        """
        pass
    
    @abstractmethod
    async def _execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute a specific capability.
        
        This method must be implemented by each specialized agent.
        """
        pass
    
    async def _check_task_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        # Default implementation - always return True
        # Subclasses can implement specific dependency checking
        return True
    
    async def _health_check_loop(self):
        """Periodic health check loop."""
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_check(self):
        """Perform health check operations."""
        # Check if agent is responsive
        if self.status == AgentStatus.ERROR:
            # Attempt to recover
            self.logger.info("Attempting to recover from error state")
            self.status = AgentStatus.IDLE
        
        # Log health metrics
        self.logger.debug(f"Health check: status={self.status}, queue_size={len(self.task_queue)}")
    
    async def add_task(self, task: Task):
        """Add a task to the agent's queue."""
        # Insert task in priority order
        inserted = False
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                self.task_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task)
        
        self.logger.info(f"Added task {task.id} to queue (priority: {task.priority.name})")
    
    async def queue_task(self, capability: str, parameters: Dict[str, Any]) -> str:
        """Queue a task for execution."""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=capability,
            priority=TaskPriority.NORMAL,
            payload=parameters,
            created_at=datetime.utcnow()
        )
        
        await self.add_task(task)
        
        self.logger.debug(f"Queued task {task_id} for capability {capability}")
        
        return task_id
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "current_task": self.current_task.id if self.current_task else None,
            "queue_size": len(self.task_queue),
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "success_rate": self.metrics.get_success_rate(),
                "average_response_time": self.metrics.average_response_time,
                "uptime": self.metrics.get_uptime(),
                "last_activity": self.metrics.last_activity.isoformat()
            }
        }
    
    async def send_message(self, recipient_id: str, message: Dict[str, Any]):
        """Send a message to another agent."""
        # This would integrate with the message bus in a full implementation
        self.logger.info(f"Sending message to {recipient_id}: {message}")
    
    async def receive_message(self, message: AgentMessage) -> bool:
        """Receive and process a message from another agent."""
        try:
            self.logger.debug(
                f"Received message from {message.sender_id}",
                extra={"message": message.to_dict()}
            )
            
            # Add to conversation history
            self.conversation_history.append(message)
            
            # Keep conversation history within window size
            if len(self.conversation_history) > self.context_window_size:
                self.conversation_history = self.conversation_history[-self.context_window_size:]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all subscribed agents."""
        # This would integrate with the message bus in a full implementation
        self.logger.info(f"Broadcasting message: {message}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for incoming messages."""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")
    
    async def handle_message(self, message: Dict[str, Any]):
        """Handle an incoming message."""
        message_type = message.get("type")
        if message_type in self.message_handlers:
            await self.message_handlers[message_type](message)
        else:
            self.logger.warning(f"No handler for message type: {message_type}")
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a response using the LLM."""
        try:
            if not ChatOpenAI or not HumanMessage:
                # Fallback when LangChain not available
                return f"Generated response for: {prompt[:100]}..."
            
            messages = []
            
            # Add system prompt if provided
            if system_prompt and SystemMessage:
                messages.append(SystemMessage(content=system_prompt))
            
            # Add context if provided
            if context:
                context_str = f"Context: {json.dumps(context, indent=2)}"
                messages.append(HumanMessage(content=context_str))
            
            # Add main prompt
            messages.append(HumanMessage(content=prompt))
            
            # Generate response
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
            
        except Exception as e:
            self.logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {e}"
    
    def add_to_memory(self, key: str, value: Any):
        """Add information to short-term memory."""
        self.short_term_memory[key] = {
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_from_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from short-term memory."""
        memory_item = self.short_term_memory.get(key)
        return memory_item["value"] if memory_item else None
    
    def clear_memory(self):
        """Clear short-term memory."""
        self.short_term_memory.clear()
        self.conversation_history.clear()
        self.logger.debug(f"Cleared memory for agent {self.agent_id}")
    
    @abstractmethod
    async def get_capabilities_info(self) -> Dict[str, Any]:
        """
        Return detailed information about agent capabilities.
        Must be implemented by subclasses.
        """
        pass
    
    def get_capability_info(self, capability_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific capability."""
        if capability_name in [cap.value for cap in self.capabilities]:
            return {
                "name": capability_name,
                "description": f"Capability: {capability_name}",
                "available": True
            }
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.status.value})"


# Exception classes
class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, agent_id: str = None, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    pass


class AgentValidationError(AgentError):
    """Raised when agent input validation fails."""
    pass


class AgentSafetyError(AgentError):
    """Raised when a safety violation is detected."""
    pass
