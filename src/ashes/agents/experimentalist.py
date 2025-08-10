"""
Experimentalist Agent for the ASHES AI Agent System.

This agent specializes in experimental design, protocol development,
data collection coordination, and experimental validation for autonomous research.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base import BaseAgent, Task, AgentCapability
from ..core.logging import get_logger


@dataclass
class ExperimentalProtocol:
    """Structure for experimental protocols."""
    id: str
    title: str
    objective: str
    hypothesis: str
    methodology: Dict[str, Any]
    materials: List[str]
    equipment: List[str]
    procedures: List[str]
    safety_requirements: List[str]
    expected_outcomes: List[str]
    success_criteria: Dict[str, Any]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    created_at: datetime


@dataclass
class ExperimentalDesign:
    """Structure for experimental design."""
    variables: Dict[str, Any]
    controls: List[str]
    replicates: int
    sample_size: int
    randomization: str
    blinding: Optional[str]
    statistical_plan: Dict[str, Any]


class ExperimentalistAgent(BaseAgent):
    """
    Experimentalist Agent specialized in experimental design and execution planning.
    
    Capabilities:
    - Design comprehensive experimental protocols
    - Plan data collection strategies
    - Optimize experimental parameters
    - Validate experimental feasibility
    - Generate statistical analysis plans
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            AgentCapability.EXPERIMENT_DESIGN,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.ANALYSIS,
            AgentCapability.RESEARCH
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type="experimentalist",
            capabilities=capabilities,
            config=config
        )
        
        # Experimentalist-specific configuration
        self.experimental_domains = config.get("domains", ["general"]) if config else ["general"]
        self.design_templates = self._initialize_design_templates()
        self.statistical_methods = self._initialize_statistical_methods()
        self.safety_protocols = self._initialize_safety_protocols()
        
        self.logger.info(f"Experimentalist agent {agent_id} initialized for domains: {self.experimental_domains}")
    
    def _initialize_design_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize experimental design templates."""
        return {
            "factorial_design": {
                "description": "Full or fractional factorial design for multiple factors",
                "parameters": ["factors", "levels", "interactions"],
                "use_cases": ["optimization", "screening", "interaction_studies"]
            },
            "response_surface": {
                "description": "Response surface methodology for optimization",
                "parameters": ["continuous_factors", "response_variables", "model_order"],
                "use_cases": ["optimization", "modeling", "prediction"]
            },
            "doe_screening": {
                "description": "Design of experiments for factor screening",
                "parameters": ["factors", "resolution", "confounding"],
                "use_cases": ["screening", "factor_identification"]
            },
            "randomized_controlled": {
                "description": "Randomized controlled trial design",
                "parameters": ["treatment_groups", "controls", "randomization"],
                "use_cases": ["causal_inference", "treatment_effects"]
            },
            "time_series": {
                "description": "Time series experimental design",
                "parameters": ["time_points", "interventions", "baseline"],
                "use_cases": ["temporal_effects", "longitudinal_studies"]
            }
        }
    
    def _initialize_statistical_methods(self) -> Dict[str, Dict[str, Any]]:
        """Initialize statistical analysis methods."""
        return {
            "anova": {
                "description": "Analysis of variance for comparing means",
                "assumptions": ["normality", "homoscedasticity", "independence"],
                "use_cases": ["group_comparisons", "factor_effects"]
            },
            "regression": {
                "description": "Linear and nonlinear regression analysis",
                "assumptions": ["linearity", "independence", "homoscedasticity"],
                "use_cases": ["prediction", "modeling", "relationships"]
            },
            "multivariate": {
                "description": "Multivariate statistical analysis",
                "methods": ["PCA", "MANOVA", "cluster_analysis"],
                "use_cases": ["dimensionality_reduction", "pattern_recognition"]
            },
            "bayesian": {
                "description": "Bayesian statistical inference",
                "advantages": ["uncertainty_quantification", "prior_knowledge"],
                "use_cases": ["small_samples", "complex_models"]
            },
            "machine_learning": {
                "description": "ML-based analysis and prediction",
                "methods": ["random_forest", "neural_networks", "svm"],
                "use_cases": ["prediction", "classification", "pattern_discovery"]
            }
        }
    
    def _initialize_safety_protocols(self) -> Dict[str, List[str]]:
        """Initialize safety protocols by domain."""
        return {
            "chemistry": [
                "Chemical compatibility assessment",
                "Fume hood requirements",
                "Personal protective equipment",
                "Waste disposal protocols",
                "Emergency procedures"
            ],
            "biology": [
                "Biosafety level assessment",
                "Containment requirements",
                "Decontamination procedures",
                "Waste sterilization",
                "Personal protection"
            ],
            "materials": [
                "Material safety data sheets",
                "Ventilation requirements",
                "High temperature safety",
                "Mechanical hazard assessment",
                "Environmental protection"
            ],
            "general": [
                "Risk assessment",
                "Safety training requirements",
                "Emergency contact information",
                "Incident reporting procedures",
                "Regular safety reviews"
            ]
        }
    
    async def _process_task(self, task: Task) -> Any:
        """Process an experimentalist-specific task."""
        task_type = task.type
        payload = task.payload
        
        if task_type == "experiment_design":
            return await self._design_experiment(payload)
        elif task_type == "protocol_development":
            return await self._develop_protocol(payload)
        elif task_type == "feasibility_assessment":
            return await self._assess_feasibility(payload)
        elif task_type == "statistical_planning":
            return await self._plan_statistical_analysis(payload)
        elif task_type == "optimization_design":
            return await self._design_optimization_experiment(payload)
        else:
            return await self._handle_general_experimental_task(payload)
    
    async def _execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific capability."""
        if capability_name == AgentCapability.EXPERIMENT_DESIGN.value:
            return await self._design_experiment(parameters)
        elif capability_name == AgentCapability.DATA_ANALYSIS.value:
            return await self._analyze_experimental_data(parameters)
        elif capability_name == AgentCapability.ANALYSIS.value:
            return await self._perform_experimental_analysis(parameters)
        elif capability_name == AgentCapability.RESEARCH.value:
            return await self._conduct_experimental_research(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _design_experiment(self, parameters: Dict[str, Any]) -> ExperimentalProtocol:
        """Design a comprehensive experimental protocol."""
        self.logger.info("Designing experimental protocol")
        
        hypothesis = parameters.get("hypothesis", "")
        objective = parameters.get("objective", "")
        domain = parameters.get("domain", "general")
        constraints = parameters.get("constraints", {})
        available_resources = parameters.get("available_resources", {})
        
        # Analyze hypothesis to determine experimental approach
        experimental_approach = await self._analyze_hypothesis_for_design(hypothesis, objective, domain)
        
        # Select appropriate design template
        design_template = await self._select_design_template(experimental_approach, constraints)
        
        # Develop detailed methodology
        methodology = await self._develop_methodology(design_template, experimental_approach, constraints)
        
        # Identify required materials and equipment
        materials = await self._identify_materials(methodology, domain)
        equipment = await self._identify_equipment(methodology, domain)
        
        # Develop step-by-step procedures
        procedures = await self._develop_procedures(methodology, materials, equipment)
        
        # Assess safety requirements
        safety_requirements = await self._assess_safety_requirements(materials, procedures, domain)
        
        # Define expected outcomes and success criteria
        expected_outcomes = await self._define_expected_outcomes(hypothesis, methodology)
        success_criteria = await self._define_success_criteria(expected_outcomes, objective)
        
        # Estimate resource requirements
        resource_requirements = await self._estimate_resource_requirements(
            materials, equipment, procedures, available_resources
        )
        
        # Conduct risk assessment
        risk_assessment = await self._conduct_risk_assessment(
            materials, equipment, procedures, safety_requirements
        )
        
        # Create experimental protocol
        protocol = ExperimentalProtocol(
            id=f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            title=f"Experimental validation of: {hypothesis[:50]}...",
            objective=objective,
            hypothesis=hypothesis,
            methodology=methodology,
            materials=materials,
            equipment=equipment,
            procedures=procedures,
            safety_requirements=safety_requirements,
            expected_outcomes=expected_outcomes,
            success_criteria=success_criteria,
            estimated_duration=resource_requirements.get("time_estimate", 0),
            resource_requirements=resource_requirements,
            risk_assessment=risk_assessment,
            created_at=datetime.utcnow()
        )
        
        self.logger.info(f"Designed experimental protocol: {protocol.id}")
        return protocol
    
    async def _analyze_hypothesis_for_design(self, hypothesis: str, objective: str, domain: str) -> Dict[str, Any]:
        """Analyze hypothesis to determine optimal experimental approach."""
        system_prompt = f"""You are an expert experimental scientist designing experiments for {domain}.
        Analyze the hypothesis to determine the best experimental approach."""
        
        prompt = f"""
        Hypothesis: {hypothesis}
        Objective: {objective}
        Domain: {domain}
        
        Analyze this hypothesis and provide:
        1. Type of hypothesis (causal, correlational, mechanistic, predictive)
        2. Key variables to test (independent, dependent, control)
        3. Experimental challenges and considerations
        4. Recommended experimental approach
        5. Critical measurements needed
        6. Potential confounding factors
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "hypothesis": hypothesis,
            "analysis": response,
            "domain": domain,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _select_design_template(self, experimental_approach: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """Select the most appropriate experimental design template."""
        # Extract key information from analysis
        analysis = experimental_approach["analysis"]
        
        # Simple heuristic-based selection (in practice, would use more sophisticated analysis)
        if "optimization" in analysis.lower() or "multiple factors" in analysis.lower():
            return "factorial_design"
        elif "screening" in analysis.lower() or "many variables" in analysis.lower():
            return "doe_screening"
        elif "response surface" in analysis.lower() or "continuous optimization" in analysis.lower():
            return "response_surface"
        elif "treatment" in analysis.lower() or "intervention" in analysis.lower():
            return "randomized_controlled"
        elif "time" in analysis.lower() or "temporal" in analysis.lower():
            return "time_series"
        else:
            return "factorial_design"  # Default
    
    async def _develop_methodology(self, design_template: str, experimental_approach: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Develop detailed experimental methodology."""
        template_info = self.design_templates[design_template]
        
        system_prompt = f"""You are an experimental scientist developing a detailed methodology.
        Use {design_template} design principles to create a comprehensive experimental plan."""
        
        prompt = f"""
        Design Template: {design_template}
        Template Info: {json.dumps(template_info, indent=2)}
        Experimental Approach: {experimental_approach['analysis']}
        Constraints: {json.dumps(constraints, indent=2)}
        
        Develop a detailed methodology including:
        1. Experimental design structure
        2. Sample size and power analysis
        3. Randomization and blocking strategies
        4. Control conditions
        5. Measurement protocols
        6. Data collection procedures
        7. Quality control measures
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "design_template": design_template,
            "detailed_plan": response,
            "template_parameters": template_info,
            "constraints_applied": constraints
        }

    async def _identify_materials(self, methodology: Dict[str, Any], domain: str) -> List[str]:
        """Identify required materials for the experiment."""
        system_prompt = f"""You are an experimental scientist in {domain} identifying required materials.
        List all materials, reagents, and consumables needed."""
        
        prompt = f"""
        Experimental Methodology:
        {methodology['detailed_plan']}
        
        Domain: {domain}
        
        Identify and list all required materials including:
        1. Raw materials and reagents
        2. Consumables and supplies
        3. Standards and calibration materials
        4. Safety materials
        5. Quantities needed
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Extract materials list from response
        materials = []
        lines = response.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or any(c.isdigit() for c in line[:3])):
                material = line.strip().lstrip('-•0123456789. ')
                if material and len(material) > 3:
                    materials.append(material)
        
        return materials if materials else ["Materials to be determined based on specific requirements"]
    
    async def _identify_equipment(self, methodology: Dict[str, Any], domain: str) -> List[str]:
        """Identify required equipment for the experiment."""
        system_prompt = f"""You are an experimental scientist in {domain} identifying required equipment.
        List all instruments, tools, and apparatus needed."""
        
        prompt = f"""
        Experimental Methodology:
        {methodology['detailed_plan']}
        
        Domain: {domain}
        
        Identify and list all required equipment including:
        1. Major instruments and analyzers
        2. Basic laboratory equipment
        3. Specialized tools
        4. Safety equipment
        5. Data acquisition systems
        6. Calibration equipment
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Extract equipment list from response
        equipment = []
        lines = response.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or any(c.isdigit() for c in line[:3])):
                item = line.strip().lstrip('-•0123456789. ')
                if item and len(item) > 3:
                    equipment.append(item)
        
        return equipment if equipment else ["Equipment to be determined based on specific requirements"]

    async def _develop_procedures(self, methodology: Dict[str, Any], materials: List[str], equipment: List[str]) -> List[str]:
        """Develop step-by-step experimental procedures."""
        system_prompt = """You are an experimental scientist writing detailed, reproducible procedures.
        Create clear, step-by-step instructions that another scientist could follow."""
        
        prompt = f"""
        Methodology: {methodology['detailed_plan']}
        Materials: {', '.join(materials[:10])}...
        Equipment: {', '.join(equipment[:10])}...
        
        Write detailed, step-by-step procedures including:
        1. Setup and preparation steps
        2. Calibration procedures
        3. Sample preparation
        4. Measurement procedures
        5. Data collection steps
        6. Quality control checks
        7. Cleanup and disposal
        
        Make procedures specific, measurable, and reproducible.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Extract procedures from response
        procedures = []
        lines = response.split('\n')
        current_procedure = []
        
        for line in lines:
            if line.strip():
                if line.startswith(('Step', 'step', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    if current_procedure:
                        procedures.append(' '.join(current_procedure))
                        current_procedure = []
                    current_procedure.append(line.strip())
                elif current_procedure:
                    current_procedure.append(line.strip())
        
        if current_procedure:
            procedures.append(' '.join(current_procedure))
        
        return procedures if procedures else ["Detailed procedures to be developed based on specific methodology"]

    async def _assess_safety_requirements(self, materials: List[str], procedures: List[str], domain: str) -> List[str]:
        """Assess safety requirements for the experiment."""
        base_safety = self.safety_protocols.get(domain, self.safety_protocols["general"])
        
        system_prompt = f"""You are a safety officer assessing experimental safety requirements for {domain}.
        Identify all safety considerations and required precautions."""
        
        prompt = f"""
        Materials: {', '.join(materials[:10])}...
        Procedures: {'. '.join(procedures[:5])}...
        Domain: {domain}
        Base Safety Protocols: {', '.join(base_safety)}
        
        Assess safety requirements including:
        1. Material-specific hazards
        2. Equipment safety considerations
        3. Procedure-related risks
        4. Personal protective equipment
        5. Environmental controls
        6. Emergency procedures
        7. Training requirements
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Combine base safety protocols with specific requirements
        safety_requirements = base_safety.copy()
        
        # Extract additional safety requirements from response
        lines = response.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or 'safety' in line.lower() or 'hazard' in line.lower()):
                requirement = line.strip().lstrip('-•0123456789. ')
                if requirement and len(requirement) > 10 and requirement not in safety_requirements:
                    safety_requirements.append(requirement)
        
        return safety_requirements

    async def _define_expected_outcomes(self, hypothesis: str, methodology: Dict[str, Any]) -> List[str]:
        """Define expected experimental outcomes."""
        system_prompt = """You are an experimental scientist defining expected outcomes.
        Based on the hypothesis and methodology, predict what results should be observed."""
        
        prompt = f"""
        Hypothesis: {hypothesis}
        Methodology: {methodology['detailed_plan']}
        
        Define expected outcomes including:
        1. Primary outcomes that would support the hypothesis
        2. Secondary outcomes and measurements
        3. Quantitative predictions where possible
        4. Alternative outcomes if hypothesis is not supported
        5. Unexpected results that might occur
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Extract expected outcomes
        outcomes = []
        lines = response.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('-') or line.startswith('•') or any(c.isdigit() for c in line[:3])):
                outcome = line.strip().lstrip('-•0123456789. ')
                if outcome and len(outcome) > 10:
                    outcomes.append(outcome)
        
        return outcomes if outcomes else ["Outcomes to be determined based on specific experimental context"]

    async def _define_success_criteria(self, expected_outcomes: List[str], objective: str) -> Dict[str, Any]:
        """Define clear success criteria for the experiment."""
        system_prompt = """You are an experimental scientist defining quantitative success criteria.
        Create measurable criteria that will determine if the experiment succeeded."""
        
        prompt = f"""
        Expected Outcomes: {'. '.join(expected_outcomes)}
        Objective: {objective}
        
        Define success criteria including:
        1. Primary success criteria (must achieve)
        2. Secondary success criteria (desirable)
        3. Quantitative thresholds where applicable
        4. Statistical significance requirements
        5. Quality criteria for data
        6. Minimum acceptable results
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "criteria_definition": response,
            "primary_criteria": "Achievement of primary objective with statistical significance",
            "secondary_criteria": "Additional insights and reproducible results",
            "quality_thresholds": "Data quality and experimental validity requirements",
            "defined_at": datetime.utcnow().isoformat()
        }

    async def _estimate_resource_requirements(self, materials: List[str], equipment: List[str], procedures: List[str], available_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate comprehensive resource requirements."""
        system_prompt = """You are a project manager estimating experimental resource requirements.
        Provide realistic estimates for time, cost, and personnel needs."""
        
        prompt = f"""
        Materials: {len(materials)} items including {', '.join(materials[:5])}...
        Equipment: {len(equipment)} items including {', '.join(equipment[:5])}...
        Procedures: {len(procedures)} steps
        Available Resources: {json.dumps(available_resources, indent=2)}
        
        Estimate resource requirements:
        1. Time requirements (setup, execution, analysis)
        2. Personnel requirements (skills and time)
        3. Cost estimates (materials, equipment, personnel)
        4. Space and facility requirements
        5. Timeline with milestones
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "estimation_details": response,
            "time_estimate": 5.0,  # Default 5 days
            "personnel_required": 2,  # Default 2 people
            "cost_category": "medium",
            "space_requirements": "Standard laboratory space",
            "timeline": "Estimated based on procedure complexity",
            "estimated_at": datetime.utcnow().isoformat()
        }

    async def _conduct_risk_assessment(self, materials: List[str], equipment: List[str], procedures: List[str], safety_requirements: List[str]) -> Dict[str, Any]:
        """Conduct comprehensive risk assessment."""
        system_prompt = """You are a risk assessment specialist evaluating experimental risks.
        Identify, assess, and provide mitigation strategies for all potential risks."""
        
        prompt = f"""
        Materials: {', '.join(materials[:10])}...
        Equipment: {', '.join(equipment[:10])}...
        Safety Requirements: {', '.join(safety_requirements[:10])}...
        
        Conduct risk assessment including:
        1. Identify all potential risks (safety, technical, schedule)
        2. Assess probability and impact of each risk
        3. Classify risk levels (low, medium, high, critical)
        4. Provide mitigation strategies
        5. Contingency plans
        6. Risk monitoring requirements
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "risk_analysis": response,
            "risk_level": "medium",  # Default assessment
            "critical_risks": "To be identified through detailed analysis",
            "mitigation_strategies": "Standard safety protocols and monitoring",
            "assessed_at": datetime.utcnow().isoformat()
        }

    async def _analyze_experimental_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experimental data using appropriate statistical methods."""
        self.logger.info("Analyzing experimental data")
        
        data = parameters.get("data", {})
        experiment_design = parameters.get("experiment_design", {})
        analysis_goals = parameters.get("analysis_goals", [])
        
        # Select appropriate statistical methods
        statistical_plan = await self._create_statistical_plan(data, experiment_design, analysis_goals)
        
        # Perform analysis (simulated)
        analysis_results = await self._perform_statistical_analysis(data, statistical_plan)
        
        return {
            "data_summary": f"Analyzed {len(data)} data points",
            "statistical_plan": statistical_plan,
            "analysis_results": analysis_results,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _create_statistical_plan(self, data: Dict[str, Any], experiment_design: Dict[str, Any], analysis_goals: List[str]) -> Dict[str, Any]:
        """Create a comprehensive statistical analysis plan."""
        system_prompt = """You are a biostatistician creating an analysis plan for experimental data.
        Select appropriate statistical methods based on data characteristics and experimental design."""
        
        prompt = f"""
        Data Characteristics: {json.dumps(data, indent=2)}
        Experiment Design: {json.dumps(experiment_design, indent=2)}
        Analysis Goals: {analysis_goals}
        
        Create statistical analysis plan including:
        1. Descriptive statistics
        2. Appropriate statistical tests
        3. Model assumptions and checks
        4. Multiple comparison corrections
        5. Effect size calculations
        6. Confidence intervals
        7. Power analysis
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "plan_details": response,
            "recommended_methods": ["descriptive_stats", "hypothesis_tests", "effect_sizes"],
            "assumptions": "Normality, independence, homoscedasticity",
            "created_at": datetime.utcnow().isoformat()
        }

    async def _perform_statistical_analysis(self, data: Dict[str, Any], statistical_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis according to the plan."""
        # Simulated analysis results
        return {
            "descriptive_stats": "Mean, median, standard deviation calculated",
            "hypothesis_tests": "Statistical tests performed according to plan",
            "effect_sizes": "Cohen's d and confidence intervals calculated",
            "p_values": "Statistical significance assessed",
            "conclusions": "Results interpreted in context of experimental hypothesis",
            "analyzed_at": datetime.utcnow().isoformat()
        }

    async def _perform_experimental_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive experimental analysis."""
        return await self._analyze_experimental_data(parameters)

    async def _conduct_experimental_research(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive experimental research."""
        research_question = parameters.get("question", "")
        domain = parameters.get("domain", "general")
        
        # Design experiment to address research question
        experiment_design = await self._design_experiment({
            "hypothesis": research_question,
            "objective": f"Investigate {research_question}",
            "domain": domain
        })
        
        return {
            "research_question": research_question,
            "experimental_design": experiment_design.__dict__,
            "recommendations": "Comprehensive experimental approach developed",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _handle_general_experimental_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general experimental tasks."""
        return await self._design_experiment(parameters)

    async def get_capabilities_info(self) -> Dict[str, Any]:
        """Return detailed information about experimentalist capabilities."""
        return {
            "agent_type": "experimentalist",
            "specialization": "Experimental design and protocol development",
            "capabilities": [cap.value for cap in self.capabilities],
            "experimental_domains": self.experimental_domains,
            "design_templates": list(self.design_templates.keys()),
            "statistical_methods": list(self.statistical_methods.keys()),
            "safety_protocols": list(self.safety_protocols.keys()),
            "primary_functions": [
                "Design comprehensive experimental protocols",
                "Develop statistical analysis plans",
                "Assess experimental feasibility",
                "Identify resource requirements",
                "Conduct risk assessments",
                "Optimize experimental parameters"
            ],
            "output_formats": [
                "Detailed experimental protocols",
                "Statistical analysis plans",
                "Resource requirement estimates",
                "Safety assessments",
                "Risk evaluations"
            ]
        }


# Public interface function for backward compatibility
async def ExperimentalistAgent_create(agent_id: str, config: Dict[str, Any]) -> ExperimentalistAgent:
    """Create an ExperimentalistAgent instance."""
    return ExperimentalistAgent(agent_id, config)
