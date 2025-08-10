"""
Enterprise ASHES Agent Orchestrator System.

This module implements the central orchestration system for managing
multiple AI agents, coordinating tasks, handling inter-agent communication,
and ensuring enterprise-grade reliability and monitoring.
"""

import asyncio
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict

# Agent system imports
from ..agents.base import BaseAgent, Task, TaskResult, TaskPriority, AgentStatus, AgentCapability
from .logging import get_logger
from .config import get_config


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class MessageType(str, Enum):
    """Inter-agent message types."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COLLABORATION = "collaboration"
    PEER_REVIEW = "peer_review"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    ALERT = "alert"
    COORDINATION = "coordination"


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    id: str
    name: str
    agent_type: str
    capability: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Multi-agent workflow definition."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    created_at: datetime
    status: WorkflowStatus = WorkflowStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    priority: TaskPriority = TaskPriority.NORMAL


@dataclass
class AgentPool:
    """Pool of available agents."""
    agents_by_type: Dict[str, List[BaseAgent]] = field(default_factory=lambda: defaultdict(list))
    agents_by_id: Dict[str, BaseAgent] = field(default_factory=dict)
    agent_capabilities: Dict[str, Set[AgentCapability]] = field(default_factory=lambda: defaultdict(set))
    load_balancer: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class MessageBus:
    """Enterprise message bus for inter-agent communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queue: List[Dict[str, Any]] = []
        self.message_history: List[Dict[str, Any]] = []
        self.logger = get_logger("message_bus")
    
    async def subscribe(self, message_type: str, handler: Callable):
        """Subscribe to a message type."""
        self.subscribers[message_type].append(handler)
        self.logger.debug(f"Subscribed handler to message type: {message_type}")
    
    async def publish(self, message_type: str, message: Dict[str, Any]):
        """Publish a message to all subscribers."""
        message_with_metadata = {
            "id": str(uuid.uuid4()),
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": message
        }
        
        # Add to history
        self.message_history.append(message_with_metadata)
        
        # Keep history bounded
        if len(self.message_history) > 10000:
            self.message_history = self.message_history[-5000:]
        
        # Notify subscribers
        handlers = self.subscribers.get(message_type, [])
        for handler in handlers:
            try:
                await handler(message_with_metadata)
            except Exception as e:
                self.logger.error(f"Message handler error: {e}")
        
        self.logger.debug(f"Published message type {message_type} to {len(handlers)} handlers")
    
    async def get_message_history(self, message_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history with optional filtering."""
        messages = self.message_history
        
        if message_type:
            messages = [msg for msg in messages if msg["type"] == message_type]
        
        return messages[-limit:]


class TaskScheduler:
    """Enterprise task scheduler with priority queues and load balancing."""
    
    def __init__(self):
        self.task_queue: List[tuple] = []  # Priority queue
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.logger = get_logger("task_scheduler")
    
    async def schedule_task(self, task: Task, dependencies: Optional[List[str]] = None):
        """Schedule a task with optional dependencies."""
        if dependencies:
            self.task_dependencies[task.id] = set(dependencies)
        
        # Add to priority queue (negative priority for max-heap behavior)
        priority = (-task.priority.value, task.created_at.timestamp())
        heapq.heappush(self.task_queue, (priority, task))
        
        self.logger.info(f"Scheduled task {task.id} with priority {task.priority.name}")
    
    async def get_next_ready_task(self) -> Optional[Task]:
        """Get the next ready task (all dependencies satisfied)."""
        ready_tasks = []
        
        # Check all tasks in queue for readiness
        temp_queue = []
        while self.task_queue:
            priority, task = heapq.heappop(self.task_queue)
            
            if self._are_dependencies_satisfied(task.id):
                ready_tasks.append((priority, task))
            else:
                temp_queue.append((priority, task))
        
        # Put non-ready tasks back
        for item in temp_queue:
            heapq.heappush(self.task_queue, item)
        
        # Return highest priority ready task
        if ready_tasks:
            ready_tasks.sort(key=lambda x: x[0])  # Sort by priority
            _, task = ready_tasks[0]
            
            # Put other ready tasks back
            for item in ready_tasks[1:]:
                heapq.heappush(self.task_queue, item)
            
            return task
        
        return None
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all task dependencies are satisfied."""
        dependencies = self.task_dependencies.get(task_id, set())
        
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            # Check if dependency succeeded
            result = self.completed_tasks[dep_id]
            if result.status != "completed":
                return False
        
        return True
    
    async def mark_task_running(self, task: Task):
        """Mark task as running."""
        self.running_tasks[task.id] = task
        self.logger.debug(f"Task {task.id} marked as running")
    
    async def mark_task_completed(self, result: TaskResult):
        """Mark task as completed."""
        if result.task_id in self.running_tasks:
            del self.running_tasks[result.task_id]
        
        self.completed_tasks[result.task_id] = result
        self.logger.info(f"Task {result.task_id} completed with status {result.status}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get scheduler status information."""
        return {
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_dependencies": sum(len(deps) for deps in self.task_dependencies.values())
        }


class EnterpriseAgentOrchestrator:
    """
    Enterprise-grade AI Agent Orchestrator for ASHES system.
    
    Manages multiple AI agents, coordinates complex workflows,
    handles inter-agent communication, and provides enterprise
    monitoring and reliability features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.logger = get_logger("orchestrator")
        
        # Agent management
        self.agent_pool = AgentPool()
        self.agent_factories = self._initialize_agent_factories()
        
        # Communication and coordination
        self.message_bus = MessageBus()
        self.task_scheduler = TaskScheduler()
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.workflow_executions: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring and metrics
        self.system_metrics = {
            "total_tasks_executed": 0,
            "total_workflows_executed": 0,
            "average_task_time": 0.0,
            "system_uptime": datetime.utcnow(),
            "active_agents": 0,
            "errors_count": 0
        }
        
        # Control
        self._running = False
        self._orchestrator_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("Enterprise Agent Orchestrator initialized")
    
    def _initialize_agent_factories(self) -> Dict[str, Callable]:
        """Initialize agent factory functions."""
        return {
            "theorist": self._create_theorist_agent,
            "experimentalist": self._create_experimentalist_agent,
            "critic": self._create_critic_agent,
            "synthesizer": self._create_synthesizer_agent,
            "ethics": self._create_ethics_agent,
            "manager": self._create_manager_agent
        }
    
    async def start(self):
        """Start the orchestrator system."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize default agents
        await self._initialize_default_agents()
        
        # Start orchestration loop
        self._orchestrator_task = asyncio.create_task(self._orchestration_loop())
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Enterprise Agent Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator system gracefully."""
        self._running = False
        
        # Stop all agents
        for agent in self.agent_pool.agents_by_id.values():
            await agent.stop()
        
        # Cancel tasks
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Enterprise Agent Orchestrator stopped")
    
    async def _initialize_default_agents(self):
        """Initialize default set of agents."""
        default_agents = [
            ("theorist", "theorist_01"),
            ("experimentalist", "experimentalist_01"),
            ("critic", "critic_01"),
            ("synthesizer", "synthesizer_01"),
            ("ethics", "ethics_01"),
            ("manager", "manager_01")
        ]
        
        for agent_type, agent_id in default_agents:
            try:
                await self.create_agent(agent_type, agent_id)
            except Exception as e:
                self.logger.warning(f"Failed to create {agent_type} agent: {e}")
    
    async def create_agent(self, agent_type: str, agent_id: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
        """Create and register a new agent."""
        if agent_type not in self.agent_factories:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if agent_id in self.agent_pool.agents_by_id:
            raise ValueError(f"Agent {agent_id} already exists")
        
        # Create agent
        factory = self.agent_factories[agent_type]
        agent = await factory(agent_id, config or {})
        
        # Register agent
        self.agent_pool.agents_by_type[agent_type].append(agent)
        self.agent_pool.agents_by_id[agent_id] = agent
        self.agent_pool.agent_capabilities[agent_id] = set(agent.capabilities)
        
        # Start agent
        await agent.start()
        
        # Subscribe to agent messages
        await self._setup_agent_communication(agent)
        
        self.logger.info(f"Created and started {agent_type} agent: {agent_id}")
        return agent
    
    async def _setup_agent_communication(self, agent: BaseAgent):
        """Setup communication channels for an agent."""
        # Subscribe agent to relevant message types
        await self.message_bus.subscribe(
            MessageType.TASK_REQUEST,
            lambda msg: self._handle_agent_task_request(agent, msg)
        )
        
        await self.message_bus.subscribe(
            MessageType.COLLABORATION,
            lambda msg: self._handle_collaboration_request(agent, msg)
        )
    
    async def _handle_agent_task_request(self, agent: BaseAgent, message: Dict[str, Any]):
        """Handle task request from an agent."""
        payload = message["payload"]
        
        # Create task from request
        task = Task(
            id=str(uuid.uuid4()),
            type=payload.get("task_type", "general"),
            priority=TaskPriority(payload.get("priority", TaskPriority.NORMAL.value)),
            payload=payload,
            created_at=datetime.utcnow()
        )
        
        # Schedule task
        await self.task_scheduler.schedule_task(task)
        
        self.logger.debug(f"Scheduled task from agent {agent.agent_id}: {task.id}")
    
    async def _handle_collaboration_request(self, agent: BaseAgent, message: Dict[str, Any]):
        """Handle collaboration request between agents."""
        payload = message["payload"]
        target_agent_id = payload.get("target_agent")
        
        if target_agent_id in self.agent_pool.agents_by_id:
            target_agent = self.agent_pool.agents_by_id[target_agent_id]
            # Forward collaboration message
            await target_agent.handle_message(payload)
            
            self.logger.debug(f"Facilitated collaboration between {agent.agent_id} and {target_agent_id}")
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self._running:
            try:
                # Process ready tasks
                await self._process_ready_tasks()
                
                # Execute workflow steps
                await self._process_workflow_executions()
                
                # Health checks
                await self._perform_system_health_checks()
                
                # Brief pause
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                self.system_metrics["errors_count"] += 1
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _process_ready_tasks(self):
        """Process tasks that are ready for execution."""
        task = await self.task_scheduler.get_next_ready_task()
        
        if task:
            # Find suitable agent
            agent = await self._find_suitable_agent(task)
            
            if agent:
                # Execute task
                await self.task_scheduler.mark_task_running(task)
                await agent.add_task(task)
                
                self.logger.debug(f"Assigned task {task.id} to agent {agent.agent_id}")
            else:
                # No suitable agent available, reschedule
                await self.task_scheduler.schedule_task(task)
                self.logger.warning(f"No suitable agent for task {task.id}, rescheduling")
    
    async def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        """Find the most suitable agent for a task."""
        # Simple load balancing - find least busy agent with required capability
        suitable_agents = []
        
        for agent in self.agent_pool.agents_by_id.values():
            if agent.status == AgentStatus.IDLE:
                # Check if agent has required capabilities
                required_cap = task.type
                if any(cap.value == required_cap for cap in agent.capabilities):
                    suitable_agents.append(agent)
        
        if suitable_agents:
            # Return agent with smallest queue
            return min(suitable_agents, key=lambda a: len(a.task_queue))
        
        return None
    
    async def _process_workflow_executions(self):
        """Process active workflow executions."""
        for workflow_id, execution in list(self.workflow_executions.items()):
            if execution["status"] == WorkflowStatus.RUNNING:
                await self._process_workflow_step(workflow_id, execution)
    
    async def _process_workflow_step(self, workflow_id: str, execution: Dict[str, Any]):
        """Process a single workflow step."""
        workflow = self.workflows[workflow_id]
        current_step_index = execution.get("current_step", 0)
        
        if current_step_index >= len(workflow.steps):
            # Workflow completed
            execution["status"] = WorkflowStatus.COMPLETED
            execution["completed_at"] = datetime.utcnow()
            
            await self.message_bus.publish(
                MessageType.STATUS_UPDATE,
                {"workflow_id": workflow_id, "status": "completed"}
            )
            
            self.logger.info(f"Workflow {workflow_id} completed")
            return
        
        step = workflow.steps[current_step_index]
        
        # Check if step dependencies are satisfied
        if await self._are_workflow_step_dependencies_satisfied(step, execution):
            # Execute step
            await self._execute_workflow_step(workflow_id, step, execution)
    
    async def _are_workflow_step_dependencies_satisfied(self, step: WorkflowStep, execution: Dict[str, Any]) -> bool:
        """Check if workflow step dependencies are satisfied."""
        completed_steps = execution.get("completed_steps", set())
        
        for dep in step.dependencies:
            if dep not in completed_steps:
                return False
        
        return True
    
    async def _execute_workflow_step(self, workflow_id: str, step: WorkflowStep, execution: Dict[str, Any]):
        """Execute a workflow step."""
        # Create task for the step
        task = Task(
            id=f"{workflow_id}_{step.id}",
            type=step.capability,
            priority=TaskPriority.HIGH,  # Workflow tasks get high priority
            payload=step.parameters,
            created_at=datetime.utcnow(),
            metadata={"workflow_id": workflow_id, "step_id": step.id}
        )
        
        # Find agent of required type
        agents = self.agent_pool.agents_by_type.get(step.agent_type, [])
        if not agents:
            self.logger.error(f"No agents of type {step.agent_type} available for workflow step")
            execution["status"] = WorkflowStatus.FAILED
            return
        
        # Select least busy agent
        agent = min(agents, key=lambda a: len(a.task_queue))
        
        # Execute task
        await agent.add_task(task)
        
        # Update execution state
        execution["running_steps"] = execution.get("running_steps", set())
        execution["running_steps"].add(step.id)
        
        self.logger.info(f"Executing workflow step {step.id} on agent {agent.agent_id}")
    
    async def _perform_system_health_checks(self):
        """Perform system-wide health checks."""
        # Check agent health
        unhealthy_agents = []
        for agent in self.agent_pool.agents_by_id.values():
            if agent.status == AgentStatus.ERROR:
                unhealthy_agents.append(agent.agent_id)
        
        if unhealthy_agents:
            self.logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
            
            # Attempt recovery
            for agent_id in unhealthy_agents:
                await self._attempt_agent_recovery(agent_id)
    
    async def _attempt_agent_recovery(self, agent_id: str):
        """Attempt to recover a failed agent."""
        agent = self.agent_pool.agents_by_id.get(agent_id)
        if agent:
            try:
                await agent.stop()
                await agent.start()
                self.logger.info(f"Successfully recovered agent {agent_id}")
            except Exception as e:
                self.logger.error(f"Failed to recover agent {agent_id}: {e}")
    
    async def _monitoring_loop(self):
        """System monitoring and metrics collection loop."""
        while self._running:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check for system alerts
                await self._check_system_alerts()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _update_system_metrics(self):
        """Update system-wide metrics."""
        self.system_metrics["active_agents"] = len([
            agent for agent in self.agent_pool.agents_by_id.values()
            if agent.status in [AgentStatus.IDLE, AgentStatus.BUSY]
        ])
        
        # Calculate average task time
        completed_tasks = list(self.task_scheduler.completed_tasks.values())
        if completed_tasks:
            total_time = sum(result.execution_time for result in completed_tasks)
            self.system_metrics["average_task_time"] = total_time / len(completed_tasks)
    
    async def _check_system_alerts(self):
        """Check for system-wide alerts."""
        # Check if too many agents are failing
        error_agents = [
            agent for agent in self.agent_pool.agents_by_id.values()
            if agent.status == AgentStatus.ERROR
        ]
        
        if len(error_agents) > len(self.agent_pool.agents_by_id) * 0.3:  # 30% failure rate
            await self.message_bus.publish(
                MessageType.ALERT,
                {
                    "severity": "critical",
                    "message": f"High agent failure rate: {len(error_agents)} agents in error state",
                    "affected_agents": [agent.agent_id for agent in error_agents]
                }
            )
    
    async def _cleanup_old_data(self):
        """Cleanup old data to prevent memory leaks."""
        # Remove old completed tasks (keep last 1000)
        if len(self.task_scheduler.completed_tasks) > 1000:
            tasks_to_remove = list(self.task_scheduler.completed_tasks.keys())[:-1000]
            for task_id in tasks_to_remove:
                del self.task_scheduler.completed_tasks[task_id]
    
    # Agent factory methods
    async def _create_theorist_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create a theorist agent."""
        try:
            from ..agents.theorist import TheoristAgent
            return TheoristAgent(agent_id, config)
        except ImportError:
            # Fallback to base agent with theorist capabilities
            return self._create_fallback_agent(agent_id, "theorist", config)
    
    async def _create_experimentalist_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create an experimentalist agent."""
        try:
            from ..agents.experimentalist import ExperimentalistAgent
            return ExperimentalistAgent(agent_id, config)
        except ImportError:
            return self._create_fallback_agent(agent_id, "experimentalist", config)
    
    async def _create_critic_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create a critic agent."""
        try:
            from ..agents.critic import CriticAgent
            return CriticAgent(agent_id, config)
        except ImportError:
            return self._create_fallback_agent(agent_id, "critic", config)
    
    async def _create_synthesizer_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create a synthesizer agent."""
        try:
            from ..agents.synthesizer import SynthesizerAgent
            return SynthesizerAgent(agent_id, config)
        except ImportError:
            return self._create_fallback_agent(agent_id, "synthesizer", config)
    
    async def _create_ethics_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create an ethics agent."""
        try:
            from ..agents.ethics import EthicsAgent
            return EthicsAgent(agent_id, config)
        except ImportError:
            return self._create_fallback_agent(agent_id, "ethics", config)
    
    async def _create_manager_agent(self, agent_id: str, config: Dict[str, Any]) -> BaseAgent:
        """Create a manager agent."""
        try:
            from ..agents.manager import ManagerAgent
            return ManagerAgent(agent_id, config)
        except ImportError:
            return self._create_fallback_agent(agent_id, "manager", config)
    
    def _create_fallback_agent(self, agent_id: str, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
        """Create a fallback agent when specific implementation not available."""
        from ..agents.base import BaseAgent, AgentCapability
        
        class FallbackAgent(BaseAgent):
            async def _process_task(self, task: Task) -> Any:
                return f"Fallback response for {task.type} task"
            
            async def _execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
                return f"Executed {capability_name} with fallback implementation"
            
            async def get_capabilities_info(self) -> Dict[str, Any]:
                return {
                    "agent_type": self.agent_type,
                    "capabilities": [cap.value for cap in self.capabilities],
                    "implementation": "fallback"
                }
        
        # Map agent types to capabilities
        capability_mapping = {
            "theorist": [AgentCapability.HYPOTHESIS_GENERATION, AgentCapability.REASONING],
            "experimentalist": [AgentCapability.EXPERIMENT_DESIGN, AgentCapability.DATA_ANALYSIS],
            "critic": [AgentCapability.CRITIQUE, AgentCapability.LITERATURE_REVIEW],
            "synthesizer": [AgentCapability.SYNTHESIS, AgentCapability.ANALYSIS],
            "ethics": [AgentCapability.ETHICS_REVIEW],
            "manager": [AgentCapability.ANALYSIS, AgentCapability.REASONING]
        }
        
        capabilities = capability_mapping.get(agent_type, [AgentCapability.ANALYSIS])
        
        return FallbackAgent(agent_id, agent_type, capabilities, config)
    
    # Public API methods
    async def submit_task(self, task_type: str, payload: Dict[str, Any], priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit a task for execution."""
        task = Task(
            id=str(uuid.uuid4()),
            type=task_type,
            priority=priority,
            payload=payload,
            created_at=datetime.utcnow()
        )
        
        await self.task_scheduler.schedule_task(task)
        
        self.logger.info(f"Submitted task {task.id} of type {task_type}")
        return task.id
    
    async def submit_workflow(self, workflow: Workflow) -> str:
        """Submit a workflow for execution."""
        workflow.id = workflow.id or str(uuid.uuid4())
        
        self.workflows[workflow.id] = workflow
        self.workflow_executions[workflow.id] = {
            "status": WorkflowStatus.RUNNING,
            "started_at": datetime.utcnow(),
            "current_step": 0,
            "completed_steps": set(),
            "running_steps": set()
        }
        
        self.logger.info(f"Submitted workflow {workflow.id}: {workflow.name}")
        return workflow.id
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "orchestrator": {
                "running": self._running,
                "agents": {
                    "total": len(self.agent_pool.agents_by_id),
                    "by_type": {
                        agent_type: len(agents)
                        for agent_type, agents in self.agent_pool.agents_by_type.items()
                    },
                    "by_status": {
                        status.value: len([
                            agent for agent in self.agent_pool.agents_by_id.values()
                            if agent.status == status
                        ])
                        for status in AgentStatus
                    }
                },
                "tasks": self.task_scheduler.get_queue_status(),
                "workflows": {
                    "total": len(self.workflows),
                    "running": len([
                        wf for wf in self.workflow_executions.values()
                        if wf["status"] == WorkflowStatus.RUNNING
                    ])
                },
                "metrics": self.system_metrics
            }
        }
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        agent = self.agent_pool.agents_by_id.get(agent_id)
        if agent:
            return await agent.get_status()
        return None
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            execution = self.workflow_executions.get(workflow_id, {})
            
            return {
                "workflow": {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": execution.get("status", WorkflowStatus.PENDING),
                    "created_at": workflow.created_at.isoformat(),
                    "started_at": execution.get("started_at", {}).isoformat() if execution.get("started_at") else None,
                    "completed_at": execution.get("completed_at", {}).isoformat() if execution.get("completed_at") else None,
                    "current_step": execution.get("current_step", 0),
                    "total_steps": len(workflow.steps),
                    "completed_steps": len(execution.get("completed_steps", set())),
                    "running_steps": len(execution.get("running_steps", set()))
                }
            }
        
        return None
