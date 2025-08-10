"""
Core orchestrator for the ASHES system.

This module implements the main orchestration engine that coordinates
all system components including agents, laboratory equipment, and data management.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .config import get_config
from .logging import get_logger
from .state import SystemState, ExperimentState


@dataclass
class ASHESOrchestrator:
    """
    Main orchestrator for the ASHES autonomous scientific research system.
    
    Coordinates multi-agent interactions, laboratory automation,
    and experimental workflows to enable fully autonomous scientific discovery.
    """
    
    config: Any = field(default_factory=get_config)
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))
    system_state: SystemState = field(default_factory=SystemState)
    
    # Component managers
    agent_manager: Optional[Any] = None
    lab_controller: Optional[Any] = None
    data_manager: Optional[Any] = None
    safety_monitor: Optional[Any] = None
    
    # Active experiments tracking
    active_experiments: Dict[str, ExperimentState] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize orchestrator components."""
        self._initialize_components()
        self._setup_monitoring()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize agent manager
            from ..agents.manager import AgentManager
            self.agent_manager = AgentManager()
            
            # Initialize laboratory controller (use minimal version if full one fails)
            try:
                from ..laboratory.controller import LabController
                self.lab_controller = LabController()
            except ImportError:
                from ..laboratory.controller_minimal import LabController
                self.lab_controller = LabController()
                self.logger.info("Using minimal laboratory controller due to missing dependencies")
            
            # Initialize data manager (use minimal version if full one fails)
            try:
                from ..data.manager import DataManager
                self.data_manager = DataManager()
            except (ImportError, AttributeError) as e:
                from ..data.manager_minimal import DataManager
                self.data_manager = DataManager()
                self.logger.info("Using minimal data manager due to missing dependencies")
            
            # Initialize safety monitor
            from ..safety.monitor import SafetyMonitor
            self.safety_monitor = SafetyMonitor()
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_monitoring(self):
        """Setup system monitoring and metrics collection."""
        self.logger.info("Setting up system monitoring")
        # Monitoring setup will be implemented in monitoring module
    
    async def start(self):
        """Start the ASHES orchestrator and all subsystems."""
        self.logger.info("Starting ASHES orchestrator")
        
        try:
            # Start all components
            if self.agent_manager:
                await self.agent_manager.start()
            
            if self.lab_controller:
                await self.lab_controller.start()
            
            if self.data_manager:
                await self.data_manager.start()
            
            if self.safety_monitor:
                await self.safety_monitor.start()
            
            self.system_state.status = "running"
            self.system_state.started_at = datetime.utcnow()
            
            self.logger.info("ASHES orchestrator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {e}")
            self.system_state.status = "error"
            raise
    
    async def stop(self):
        """Stop the ASHES orchestrator and all subsystems."""
        self.logger.info("Stopping ASHES orchestrator")
        
        try:
            # Stop all active experiments
            await self._stop_all_experiments()
            
            # Stop all components
            if self.safety_monitor:
                await self.safety_monitor.stop()
            
            if self.lab_controller:
                await self.lab_controller.stop()
            
            if self.agent_manager:
                await self.agent_manager.stop()
            
            if self.data_manager:
                await self.data_manager.stop()
            
            self.system_state.status = "stopped"
            self.logger.info("ASHES orchestrator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    async def create_experiment(
        self, 
        research_domain: str,
        initial_hypothesis: Optional[str] = None,
        priority: int = 1,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new autonomous experiment.
        
        Args:
            research_domain: The scientific domain for the experiment
            initial_hypothesis: Optional initial hypothesis to test
            priority: Experiment priority (1-10, higher is more priority)
            parameters: Additional experiment parameters
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        experiment_state = ExperimentState(
            id=experiment_id,
            domain=research_domain,
            initial_hypothesis=initial_hypothesis,
            priority=priority,
            parameters=parameters or {},
            status="created",
            created_at=datetime.utcnow()
        )
        
        self.active_experiments[experiment_id] = experiment_state
        
        self.logger.info(
            f"Created experiment {experiment_id} in domain {research_domain}",
            extra={
                "experiment_id": experiment_id,
                "domain": research_domain,
                "priority": priority
            }
        )
        
        # Queue experiment for execution
        await self._queue_experiment(experiment_id)
        
        return experiment_id
    
    async def _queue_experiment(self, experiment_id: str):
        """Queue an experiment for autonomous execution."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = "queued"
        
        # Start the autonomous research workflow
        await self._execute_autonomous_workflow(experiment_id)
    
    async def _execute_autonomous_workflow(self, experiment_id: str):
        """
        Execute the complete autonomous research workflow.
        
        This is the core of ASHES - fully autonomous scientific research.
        """
        experiment = self.active_experiments[experiment_id]
        
        try:
            experiment.status = "running"
            experiment.started_at = datetime.utcnow()
            
            self.logger.info(f"Starting autonomous workflow for experiment {experiment_id}")
            
            # Phase 1: Hypothesis Generation
            if not experiment.initial_hypothesis:
                hypothesis = await self._generate_hypothesis(experiment)
                experiment.current_hypothesis = hypothesis
            else:
                experiment.current_hypothesis = experiment.initial_hypothesis
            
            # Phase 2: Literature Review and Validation
            literature_review = await self._conduct_literature_review(experiment)
            experiment.literature_review = literature_review
            
            # Phase 3: Experiment Design
            experiment_design = await self._design_experiment(experiment)
            experiment.experiment_design = experiment_design
            
            # Phase 4: Safety and Ethics Review
            safety_approval = await self._conduct_safety_review(experiment)
            if not safety_approval:
                experiment.status = "rejected_safety"
                return
            
            # Phase 5: Laboratory Execution
            experimental_results = await self._execute_laboratory_experiment(experiment)
            experiment.results = experimental_results
            
            # Phase 6: Data Analysis and Interpretation
            analysis_results = await self._analyze_results(experiment)
            experiment.analysis = analysis_results
            
            # Phase 7: Hypothesis Validation and Evolution
            hypothesis_evolution = await self._evolve_hypothesis(experiment)
            experiment.evolved_hypothesis = hypothesis_evolution
            
            # Phase 8: Scientific Publication Generation
            publication = await self._generate_publication(experiment)
            experiment.publication = publication
            
            experiment.status = "completed"
            experiment.completed_at = datetime.utcnow()
            
            self.logger.info(f"Completed autonomous workflow for experiment {experiment_id}")
            
        except Exception as e:
            experiment.status = "failed"
            experiment.error = str(e)
            self.logger.error(f"Autonomous workflow failed for experiment {experiment_id}: {e}")
            raise
    
    async def _generate_hypothesis(self, experiment: ExperimentState) -> str:
        """Generate a novel scientific hypothesis using AI agents."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        theorist_agent = await self.agent_manager.get_agent("theorist")
        
        hypothesis_request = {
            "domain": experiment.domain,
            "parameters": experiment.parameters,
            "context": "Generate a novel, testable scientific hypothesis"
        }
        
        hypothesis = await theorist_agent.generate_hypothesis(hypothesis_request)
        
        self.logger.info(
            f"Generated hypothesis for experiment {experiment.id}: {hypothesis}",
            extra={"experiment_id": experiment.id, "hypothesis": hypothesis}
        )
        
        return hypothesis
    
    async def _conduct_literature_review(self, experiment: ExperimentState) -> Dict[str, Any]:
        """Conduct autonomous literature review and prior art analysis."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        critic_agent = await self.agent_manager.get_agent("critic")
        
        review_request = {
            "hypothesis": experiment.current_hypothesis,
            "domain": experiment.domain,
            "depth": "comprehensive"
        }
        
        literature_review = await critic_agent.conduct_literature_review(review_request)
        
        self.logger.info(f"Completed literature review for experiment {experiment.id}")
        
        return literature_review
    
    async def _design_experiment(self, experiment: ExperimentState) -> Dict[str, Any]:
        """Design experimental protocol using experimentalist agent."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        experimentalist_agent = await self.agent_manager.get_agent("experimentalist")
        
        design_request = {
            "hypothesis": experiment.current_hypothesis,
            "domain": experiment.domain,
            "available_equipment": await self.lab_controller.get_available_equipment(),
            "safety_constraints": await self.safety_monitor.get_constraints()
        }
        
        experiment_design = await experimentalist_agent.design_experiment(design_request)
        
        self.logger.info(f"Designed experiment protocol for experiment {experiment.id}")
        
        return experiment_design
    
    async def _conduct_safety_review(self, experiment: ExperimentState) -> bool:
        """Conduct comprehensive safety and ethics review."""
        if not self.safety_monitor:
            raise RuntimeError("Safety monitor not initialized")
        
        safety_request = {
            "experiment_design": experiment.experiment_design,
            "materials": experiment.experiment_design.get("materials", []),
            "procedures": experiment.experiment_design.get("procedure", []),
            "domain": experiment.domain
        }
        
        safety_approval = await self.safety_monitor.review_experiment(safety_request)
        
        self.logger.info(
            f"Safety review for experiment {experiment.id}: {'approved' if safety_approval else 'rejected'}"
        )
        
        return safety_approval
    
    async def _execute_laboratory_experiment(self, experiment: ExperimentState) -> Dict[str, Any]:
        """Execute the physical experiment in the laboratory."""
        if not self.lab_controller:
            raise RuntimeError("Laboratory controller not initialized")
        
        execution_request = {
            "experiment_id": experiment.id,
            "design": experiment.experiment_design,
            "safety_protocols": experiment.experiment_design.get("safety_protocols", [])
        }
        
        results = await self.lab_controller.execute_experiment(execution_request)
        
        self.logger.info(f"Completed laboratory execution for experiment {experiment.id}")
        
        return results
    
    async def _analyze_results(self, experiment: ExperimentState) -> Dict[str, Any]:
        """Analyze experimental results using data analysis agents."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        synthesizer_agent = await self.agent_manager.get_agent("synthesizer")
        
        analysis_request = {
            "experimental_data": experiment.results,
            "hypothesis": experiment.current_hypothesis,
            "experiment_design": experiment.experiment_design
        }
        
        analysis = await synthesizer_agent.analyze_results(analysis_request)
        
        self.logger.info(f"Completed results analysis for experiment {experiment.id}")
        
        return analysis
    
    async def _evolve_hypothesis(self, experiment: ExperimentState) -> str:
        """Evolve hypothesis based on experimental results."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        theorist_agent = await self.agent_manager.get_agent("theorist")
        
        evolution_request = {
            "original_hypothesis": experiment.current_hypothesis,
            "experimental_results": experiment.results,
            "analysis": experiment.analysis,
            "literature_review": experiment.literature_review
        }
        
        evolved_hypothesis = await theorist_agent.evolve_hypothesis(evolution_request)
        
        self.logger.info(f"Evolved hypothesis for experiment {experiment.id}")
        
        return evolved_hypothesis
    
    async def _generate_publication(self, experiment: ExperimentState) -> Dict[str, Any]:
        """Generate scientific publication from experimental results."""
        if not self.agent_manager:
            raise RuntimeError("Agent manager not initialized")
        
        synthesizer_agent = await self.agent_manager.get_agent("synthesizer")
        
        publication_request = {
            "experiment": experiment.__dict__,
            "format": "scientific_paper",
            "target_journal": "autonomous_research"
        }
        
        publication = await synthesizer_agent.generate_publication(publication_request)
        
        self.logger.info(f"Generated publication for experiment {experiment.id}")
        
        return publication
    
    async def _stop_all_experiments(self):
        """Stop all active experiments safely."""
        for experiment_id, experiment in self.active_experiments.items():
            if experiment.status in ["running", "queued"]:
                experiment.status = "stopped"
                self.logger.info(f"Stopped experiment {experiment_id}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "system_state": self.system_state.__dict__,
            "active_experiments": len(self.active_experiments),
            "components": {}
        }
        
        if self.agent_manager:
            status["components"]["agents"] = await self.agent_manager.get_status()
        
        if self.lab_controller:
            status["components"]["laboratory"] = await self.lab_controller.get_status()
        
        if self.data_manager:
            status["components"]["data"] = await self.data_manager.get_status()
        
        if self.safety_monitor:
            status["components"]["safety"] = await self.safety_monitor.get_status()
        
        return status
    
    async def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific experiment."""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return None
        
        return experiment.__dict__
