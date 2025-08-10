"""
Agent Manager for ASHES system.

Manages the lifecycle and coordination of all autonomous agents
in the scientific research system.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseAgent, AgentCapability
from .theorist import TheoristAgent
from .experimentalist import ExperimentalistAgent
from .critic import CriticAgent
from .synthesizer import SynthesizerAgent
from .ethics import EthicsAgent
from ..core.logging import get_logger
from ..core.config import get_config


class AgentManager:
    """
    Manages all agents in the ASHES system.
    
    Responsibilities:
    - Agent lifecycle management (create, start, stop)
    - Agent discovery and routing
    - Load balancing across agent instances
    - Performance monitoring and optimization
    - Inter-agent communication coordination
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, List[str]] = {
            "theorist": [],
            "experimentalist": [],
            "critic": [],
            "synthesizer": [],
            "ethics": []
        }
        
        # Communication and coordination
        self.message_broker = None
        self.orchestration_graph = None
        
        # Performance tracking
        self.total_requests = 0
        self.agent_utilization = {}
        
    async def start(self):
        """Start the agent manager and initialize all agents."""
        self.logger.info("Starting ASHES Agent Manager")
        
        try:
            # Initialize message broker
            await self._setup_message_broker()
            
            # Create and start default agent pool
            await self._create_default_agents()
            
            # Start all agents
            await self._start_all_agents()
            
            # Setup orchestration
            await self._setup_orchestration()
            
            self.logger.info("Agent Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Agent Manager: {e}")
            raise
    
    async def stop(self):
        """Stop all agents gracefully."""
        self.logger.info("Stopping Agent Manager")
        
        try:
            # Stop all agents
            await self._stop_all_agents()
            
            # Cleanup resources
            await self._cleanup_resources()
            
            self.logger.info("Agent Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during Agent Manager shutdown: {e}")
            raise
    
    async def _setup_message_broker(self):
        """Setup message broker for inter-agent communication."""
        # This would typically setup Kafka or Redis for message passing
        self.logger.debug("Setting up message broker")
        # Placeholder implementation
        self.message_broker = "placeholder_broker"
    
    async def _create_default_agents(self):
        """Create the default set of agents for the system."""
        self.logger.info("Creating default agent pool")
        
        # Create Theorist agents
        for i in range(2):  # Create 2 theorist agents for load balancing
            agent_id = f"theorist_{i:03d}"
            agent = TheoristAgent(agent_id)
            await self._register_agent(agent)
        
        # Create Experimentalist agents
        for i in range(3):  # Create 3 experimentalist agents
            agent_id = f"experimentalist_{i:03d}"
            agent = ExperimentalistAgent(agent_id)
            await self._register_agent(agent)
        
        # Create Critic agents
        for i in range(2):  # Create 2 critic agents
            agent_id = f"critic_{i:03d}"
            agent = CriticAgent(agent_id)
            await self._register_agent(agent)
        
        # Create Synthesizer agents
        for i in range(2):  # Create 2 synthesizer agents
            agent_id = f"synthesizer_{i:03d}"
            agent = SynthesizerAgent(agent_id)
            await self._register_agent(agent)
        
        # Create Ethics agent
        ethics_agent = EthicsAgent("ethics_001")
        await self._register_agent(ethics_agent)
        
        self.logger.info(f"Created {len(self.agents)} agents")
    
    async def _register_agent(self, agent: BaseAgent):
        """Register an agent with the manager."""
        self.agents[agent.agent_id] = agent
        self.agent_types[agent.agent_type].append(agent.agent_id)
        
        self.logger.debug(f"Registered agent {agent.agent_id} of type {agent.agent_type}")
    
    async def _start_all_agents(self):
        """Start all registered agents."""
        self.logger.info("Starting all agents")
        
        start_tasks = []
        for agent in self.agents.values():
            start_tasks.append(agent.start())
        
        # Start all agents concurrently
        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        
        # Check for failures
        failed_agents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_id = list(self.agents.keys())[i]
                failed_agents.append((agent_id, result))
        
        if failed_agents:
            self.logger.error(f"Failed to start {len(failed_agents)} agents: {failed_agents}")
            # Could implement retry logic here
        
        running_agents = len(self.agents) - len(failed_agents)
        self.logger.info(f"Started {running_agents}/{len(self.agents)} agents successfully")
    
    async def _stop_all_agents(self):
        """Stop all agents gracefully."""
        self.logger.info("Stopping all agents")
        
        stop_tasks = []
        for agent in self.agents.values():
            if agent.status != "stopped":
                stop_tasks.append(agent.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.logger.info("All agents stopped")
    
    async def _setup_orchestration(self):
        """Setup the orchestration graph for agent workflows."""
        # This would typically use LangGraph for complex workflows
        self.logger.debug("Setting up agent orchestration")
        # Placeholder implementation
        self.orchestration_graph = "placeholder_graph"
    
    async def _cleanup_resources(self):
        """Cleanup any resources used by the agent manager."""
        self.agents.clear()
        for agent_list in self.agent_types.values():
            agent_list.clear()
    
    async def get_agent(self, agent_type: str, preferred_id: Optional[str] = None) -> Optional[BaseAgent]:
        """
        Get an available agent of the specified type.
        
        Args:
            agent_type: Type of agent needed
            preferred_id: Optional specific agent ID to request
            
        Returns:
            Available agent instance or None if none available
        """
        if preferred_id and preferred_id in self.agents:
            agent = self.agents[preferred_id]
            if agent.agent_type == agent_type and agent.status == "ready":
                return agent
        
        # Find any available agent of the requested type
        agent_ids = self.agent_types.get(agent_type, [])
        
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent and agent.status == "ready":
                return agent
        
        self.logger.warning(f"No available agents of type {agent_type}")
        return None
    
    async def execute_capability(
        self, 
        agent_type: str, 
        capability: str, 
        parameters: Dict[str, Any],
        timeout: float = 300.0
    ) -> Any:
        """
        Execute a capability on an available agent.
        
        Args:
            agent_type: Type of agent to use
            capability: Name of the capability to execute
            parameters: Parameters for the capability
            timeout: Maximum time to wait for completion
            
        Returns:
            Result from the agent execution
        """
        agent = await self.get_agent(agent_type)
        if not agent:
            raise RuntimeError(f"No available {agent_type} agents")
        
        # Queue the task
        task_id = await agent.queue_task(capability, parameters)
        
        # Wait for completion (simplified - in production would use proper task tracking)
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Agent task {task_id} timed out")
            
            # Check if agent is ready (task completed)
            if agent.status == "ready" and agent.current_task is None:
                break
            
            await asyncio.sleep(0.1)
        
        # In a real implementation, we'd return the actual result
        return {"task_id": task_id, "status": "completed"}
    
    async def broadcast_message(self, message_type: str, content: Dict[str, Any]):
        """Broadcast a message to all agents."""
        self.logger.debug(f"Broadcasting message type {message_type} to all agents")
        
        # In production, this would use the message broker
        for agent in self.agents.values():
            # Create message and send to agent
            pass
    
    async def coordinate_workflow(
        self, 
        workflow_type: str, 
        experiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate a multi-agent workflow.
        
        Args:
            workflow_type: Type of workflow to execute
            experiment_data: Data about the experiment
            
        Returns:
            Workflow results
        """
        if workflow_type == "autonomous_research":
            return await self._execute_autonomous_research_workflow(experiment_data)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
    
    async def _execute_autonomous_research_workflow(
        self, 
        experiment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the complete autonomous research workflow."""
        self.logger.info("Executing autonomous research workflow")
        
        results = {}
        
        try:
            # Step 1: Hypothesis Generation
            theorist = await self.get_agent("theorist")
            if theorist:
                hypothesis_task = await theorist.queue_task(
                    "generate_hypothesis", 
                    experiment_data
                )
                results["hypothesis"] = f"Generated by {theorist.agent_id}"
            
            # Step 2: Experiment Design
            experimentalist = await self.get_agent("experimentalist")
            if experimentalist:
                design_task = await experimentalist.queue_task(
                    "design_experiment",
                    {"hypothesis": results.get("hypothesis"), **experiment_data}
                )
                results["experiment_design"] = f"Designed by {experimentalist.agent_id}"
            
            # Step 3: Critical Review
            critic = await self.get_agent("critic")
            if critic:
                review_task = await critic.queue_task(
                    "review_design",
                    {"design": results.get("experiment_design")}
                )
                results["review"] = f"Reviewed by {critic.agent_id}"
            
            # Step 4: Ethics Review
            ethics = await self.get_agent("ethics")
            if ethics:
                ethics_task = await ethics.queue_task(
                    "ethics_review",
                    {"experiment": results.get("experiment_design")}
                )
                results["ethics_approval"] = f"Approved by {ethics.agent_id}"
            
            self.logger.info("Autonomous research workflow completed successfully")
            
        except Exception as e:
            self.logger.error(f"Autonomous research workflow failed: {e}")
            results["error"] = str(e)
        
        return results
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents."""
        agent_statuses = {}
        
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        # Calculate overall statistics
        total_agents = len(self.agents)
        ready_agents = sum(1 for agent in self.agents.values() if agent.status == "ready")
        busy_agents = sum(1 for agent in self.agents.values() if agent.status == "busy")
        
        return {
            "total_agents": total_agents,
            "ready_agents": ready_agents,
            "busy_agents": busy_agents,
            "agent_types": {
                agent_type: len(agent_ids) 
                for agent_type, agent_ids in self.agent_types.items()
            },
            "agents": agent_statuses,
            "total_requests": self.total_requests
        }
    
    async def scale_agents(self, agent_type: str, target_count: int):
        """Scale the number of agents of a specific type."""
        current_count = len(self.agent_types.get(agent_type, []))
        
        if target_count > current_count:
            # Create additional agents
            for i in range(current_count, target_count):
                agent_id = f"{agent_type}_{i:03d}"
                
                if agent_type == "theorist":
                    agent = TheoristAgent(agent_id)
                elif agent_type == "experimentalist":
                    agent = ExperimentalistAgent(agent_id)
                elif agent_type == "critic":
                    agent = CriticAgent(agent_id)
                elif agent_type == "synthesizer":
                    agent = SynthesizerAgent(agent_id)
                elif agent_type == "ethics":
                    agent = EthicsAgent(agent_id)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")
                
                await self._register_agent(agent)
                await agent.start()
                
                self.logger.info(f"Scaled up {agent_type} agents to {target_count}")
        
        elif target_count < current_count:
            # Remove excess agents
            agents_to_remove = self.agent_types[agent_type][target_count:]
            
            for agent_id in agents_to_remove:
                agent = self.agents[agent_id]
                await agent.stop()
                del self.agents[agent_id]
                self.agent_types[agent_type].remove(agent_id)
            
            self.logger.info(f"Scaled down {agent_type} agents to {target_count}")
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        metrics = {
            "total_requests": self.total_requests,
            "agent_performance": {},
            "utilization": {}
        }
        
        for agent_id, agent in self.agents.items():
            status = agent.get_status()
            metrics["agent_performance"][agent_id] = status["performance"]
            
            # Calculate utilization
            if status["performance"]["total_requests"] > 0:
                utilization = (
                    status["performance"]["successful_requests"] / 
                    status["performance"]["total_requests"]
                )
                metrics["utilization"][agent_id] = utilization
        
        return metrics
