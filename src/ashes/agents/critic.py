"""
Critic Agent for the ASHES AI Agent System.

This agent specializes in critical evaluation, peer review, quality assessment,
and constructive criticism for research validation and improvement.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .base import BaseAgent, Task, AgentCapability
from ..core.logging import get_logger


class CriticismType(Enum):
    """Types of criticism that can be performed."""
    METHODOLOGICAL = "methodological"
    STATISTICAL = "statistical"
    LOGICAL = "logical"
    ETHICAL = "ethical"
    REPRODUCIBILITY = "reproducibility"
    VALIDITY = "validity"
    SIGNIFICANCE = "significance"
    COMPLETENESS = "completeness"


class CriticismSeverity(Enum):
    """Severity levels for criticism."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class CriticalAssessment:
    """Structure for critical assessments."""
    id: str
    subject: str
    criticism_type: CriticismType
    severity: CriticismSeverity
    description: str
    evidence: List[str]
    recommendations: List[str]
    confidence: float
    reviewer_notes: str
    created_at: datetime


@dataclass
class PeerReview:
    """Structure for peer review evaluations."""
    id: str
    reviewed_item: str
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    critical_assessments: List[CriticalAssessment]
    recommendations: List[str]
    decision: str  # accept, reject, minor_revision, major_revision
    reviewer_confidence: float
    created_at: datetime


class CriticAgent(BaseAgent):
    """
    Critic Agent specialized in critical evaluation and peer review.
    
    Capabilities:
    - Perform thorough critical analysis of research
    - Conduct peer review evaluations
    - Assess methodological rigor
    - Evaluate statistical validity
    - Check for logical consistency
    - Assess reproducibility and validity
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            AgentCapability.ANALYSIS,
            AgentCapability.REASONING,
            AgentCapability.VALIDATION,
            AgentCapability.QUALITY_CONTROL
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type="critic",
            capabilities=capabilities,
            config=config
        )
        
        # Critic-specific configuration
        self.review_standards = config.get("review_standards", {}) if config else {}
        self.criticism_criteria = self._initialize_criticism_criteria()
        self.quality_metrics = self._initialize_quality_metrics()
        self.review_templates = self._initialize_review_templates()
        
        # Review history and assessments
        self.review_history = []
        self.assessment_cache = {}
        
        self.logger.info(f"Critic agent {agent_id} initialized with rigorous evaluation standards")
    
    def _initialize_criticism_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize criticism evaluation criteria."""
        return {
            "methodology": {
                "description": "Evaluate experimental and analytical methodology",
                "criteria": [
                    "Appropriate experimental design",
                    "Adequate controls and comparisons",
                    "Proper sampling and randomization",
                    "Correct statistical methods",
                    "Valid measurement techniques"
                ],
                "weight": 0.25
            },
            "statistical_validity": {
                "description": "Assess statistical approaches and validity",
                "criteria": [
                    "Appropriate statistical tests",
                    "Sufficient sample size and power",
                    "Correct handling of multiple comparisons",
                    "Proper assumption checking",
                    "Valid confidence intervals and effect sizes"
                ],
                "weight": 0.20
            },
            "logical_consistency": {
                "description": "Evaluate logical reasoning and consistency",
                "criteria": [
                    "Coherent argumentation",
                    "Consistent use of terminology",
                    "Logical flow of conclusions",
                    "Proper causal inference",
                    "Absence of contradictions"
                ],
                "weight": 0.20
            },
            "reproducibility": {
                "description": "Assess reproducibility and transparency",
                "criteria": [
                    "Detailed methodology description",
                    "Availability of data and code",
                    "Clear protocol documentation",
                    "Sufficient detail for replication",
                    "Open and transparent reporting"
                ],
                "weight": 0.15
            },
            "validity": {
                "description": "Evaluate internal and external validity",
                "criteria": [
                    "Internal validity of conclusions",
                    "External validity and generalizability",
                    "Construct validity of measures",
                    "Face validity of approach",
                    "Ecological validity"
                ],
                "weight": 0.10
            },
            "significance": {
                "description": "Assess scientific and practical significance",
                "criteria": [
                    "Novel contributions to knowledge",
                    "Practical implications",
                    "Scientific impact",
                    "Relevance to field",
                    "Advancement of understanding"
                ],
                "weight": 0.10
            }
        }
    
    def _initialize_quality_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize quality assessment metrics."""
        return {
            "experimental_rigor": {
                "description": "Rigor of experimental design and execution",
                "components": ["design_quality", "execution_quality", "controls", "replication"],
                "scale": "1-10",
                "thresholds": {"excellent": 8.5, "good": 7.0, "acceptable": 5.5, "poor": 3.0}
            },
            "analytical_soundness": {
                "description": "Soundness of analytical approaches",
                "components": ["method_appropriateness", "assumption_validity", "interpretation"],
                "scale": "1-10",
                "thresholds": {"excellent": 8.5, "good": 7.0, "acceptable": 5.5, "poor": 3.0}
            },
            "reporting_quality": {
                "description": "Quality of reporting and documentation",
                "components": ["clarity", "completeness", "transparency", "reproducibility"],
                "scale": "1-10",
                "thresholds": {"excellent": 8.5, "good": 7.0, "acceptable": 5.5, "poor": 3.0}
            },
            "scientific_contribution": {
                "description": "Magnitude of scientific contribution",
                "components": ["novelty", "significance", "impact", "advancement"],
                "scale": "1-10",
                "thresholds": {"excellent": 8.5, "good": 7.0, "acceptable": 5.5, "poor": 3.0}
            }
        }
    
    def _initialize_review_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize peer review templates."""
        return {
            "research_paper": {
                "description": "Template for reviewing research papers",
                "sections": ["abstract", "introduction", "methods", "results", "discussion", "conclusions"],
                "criteria": ["clarity", "methodology", "results_validity", "interpretation", "significance"],
                "decision_options": ["accept", "minor_revision", "major_revision", "reject"]
            },
            "experimental_protocol": {
                "description": "Template for reviewing experimental protocols",
                "sections": ["objectives", "methodology", "procedures", "safety", "analysis_plan"],
                "criteria": ["feasibility", "safety", "scientific_rigor", "reproducibility"],
                "decision_options": ["approve", "approve_with_conditions", "request_revisions", "reject"]
            },
            "research_proposal": {
                "description": "Template for reviewing research proposals",
                "sections": ["background", "objectives", "methodology", "timeline", "budget"],
                "criteria": ["innovation", "feasibility", "methodology", "impact", "resources"],
                "decision_options": ["fund", "fund_with_conditions", "revise_and_resubmit", "decline"]
            },
            "hypothesis": {
                "description": "Template for reviewing hypotheses",
                "sections": ["statement", "rationale", "predictions", "testability"],
                "criteria": ["clarity", "testability", "novelty", "theoretical_grounding"],
                "decision_options": ["endorse", "suggest_refinement", "request_clarification", "reject"]
            }
        }
    
    async def _process_task(self, task: Task) -> Any:
        """Process a critic-specific task."""
        task_type = task.type
        payload = task.payload
        
        if task_type == "peer_review":
            return await self._conduct_peer_review(payload)
        elif task_type == "critical_assessment":
            return await self._perform_critical_assessment(payload)
        elif task_type == "methodology_review":
            return await self._review_methodology(payload)
        elif task_type == "statistical_validation":
            return await self._validate_statistics(payload)
        elif task_type == "reproducibility_check":
            return await self._check_reproducibility(payload)
        elif task_type == "quality_assessment":
            return await self._assess_quality(payload)
        else:
            return await self._handle_general_criticism_task(payload)
    
    async def _execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific capability."""
        if capability_name == AgentCapability.ANALYSIS.value:
            return await self._perform_critical_analysis(parameters)
        elif capability_name == AgentCapability.REASONING.value:
            return await self._evaluate_reasoning(parameters)
        elif capability_name == AgentCapability.VALIDATION.value:
            return await self._validate_research(parameters)
        elif capability_name == AgentCapability.QUALITY_CONTROL.value:
            return await self._control_quality(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _conduct_peer_review(self, parameters: Dict[str, Any]) -> PeerReview:
        """Conduct a comprehensive peer review."""
        self.logger.info("Conducting peer review")
        
        item_to_review = parameters.get("item", "")
        review_type = parameters.get("type", "research_paper")
        review_criteria = parameters.get("criteria", [])
        
        # Get appropriate review template
        template = self.review_templates.get(review_type, self.review_templates["research_paper"])
        
        # Perform systematic review across all sections
        section_assessments = {}
        for section in template["sections"]:
            section_assessments[section] = await self._review_section(item_to_review, section, review_type)
        
        # Evaluate against criteria
        criteria_scores = {}
        for criterion in template["criteria"]:
            criteria_scores[criterion] = await self._evaluate_criterion(item_to_review, criterion, section_assessments)
        
        # Generate critical assessments
        critical_assessments = await self._generate_critical_assessments(
            item_to_review, section_assessments, criteria_scores
        )
        
        # Identify strengths and weaknesses
        strengths = await self._identify_strengths(section_assessments, criteria_scores)
        weaknesses = await self._identify_weaknesses(section_assessments, criteria_scores)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(critical_assessments, weaknesses)
        
        # Calculate overall score
        overall_score = await self._calculate_overall_score(criteria_scores)
        
        # Make decision
        decision = await self._make_review_decision(overall_score, critical_assessments, template)
        
        # Create peer review
        review = PeerReview(
            id=f"review_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            reviewed_item=item_to_review[:100] + "..." if len(item_to_review) > 100 else item_to_review,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            critical_assessments=critical_assessments,
            recommendations=recommendations,
            decision=decision,
            reviewer_confidence=0.85,
            created_at=datetime.utcnow()
        )
        
        # Store in review history
        self.review_history.append(review)
        
        self.logger.info(f"Completed peer review with decision: {decision}")
        return review
    
    async def _review_section(self, content: str, section: str, review_type: str) -> Dict[str, Any]:
        """Review a specific section of the content."""
        system_prompt = f"""You are a rigorous peer reviewer evaluating the {section} section of a {review_type}.
        Provide detailed, constructive criticism focusing on scientific rigor and quality."""
        
        prompt = f"""
        Review the {section} section of this {review_type}:
        
        Content: {content}
        
        Evaluate this section for:
        1. Clarity and organization
        2. Completeness and adequacy
        3. Scientific accuracy
        4. Methodological soundness
        5. Logical consistency
        6. Adherence to standards
        
        Provide specific feedback on strengths and areas for improvement.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "section": section,
            "assessment": response,
            "score": 7.5,  # Default score, would be calculated from detailed analysis
            "key_issues": [],
            "recommendations": []
        }
    
    async def _evaluate_criterion(self, content: str, criterion: str, section_assessments: Dict[str, Any]) -> float:
        """Evaluate content against a specific criterion."""
        system_prompt = f"""You are an expert evaluator assessing {criterion} in research work.
        Provide a detailed evaluation and numerical score."""
        
        prompt = f"""
        Evaluate this research work for {criterion}:
        
        Content: {content}
        Section Assessments: {json.dumps(section_assessments, indent=2)}
        
        Rate the {criterion} on a scale of 1-10 where:
        1-3: Poor/Inadequate
        4-5: Below Average
        6-7: Average/Acceptable
        8-9: Good/Strong
        10: Excellent/Outstanding
        
        Provide justification for your score.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Extract score from response (simplified)
        score = 7.0  # Default score
        if "excellent" in response.lower() or "outstanding" in response.lower():
            score = 9.0
        elif "good" in response.lower() or "strong" in response.lower():
            score = 8.0
        elif "poor" in response.lower() or "inadequate" in response.lower():
            score = 3.0
        elif "below" in response.lower():
            score = 5.0
        
        return score
    
    async def _generate_critical_assessments(self, content: str, section_assessments: Dict[str, Any], criteria_scores: Dict[str, float]) -> List[CriticalAssessment]:
        """Generate specific critical assessments."""
        assessments = []
        
        # Identify areas requiring criticism based on low scores
        for criterion, score in criteria_scores.items():
            if score < 6.0:  # Below acceptable threshold
                severity = CriticismSeverity.MAJOR if score < 4.0 else CriticismSeverity.MODERATE
                
                assessment = await self._create_critical_assessment(
                    content, criterion, severity, section_assessments
                )
                assessments.append(assessment)
        
        # Check for specific issues
        methodological_issues = await self._check_methodological_issues(content)
        if methodological_issues:
            assessments.extend(methodological_issues)
        
        statistical_issues = await self._check_statistical_issues(content)
        if statistical_issues:
            assessments.extend(statistical_issues)
        
        return assessments
    
    async def _create_critical_assessment(self, content: str, criterion: str, severity: CriticismSeverity, context: Dict[str, Any]) -> CriticalAssessment:
        """Create a specific critical assessment."""
        system_prompt = f"""You are a scientific critic identifying specific issues with {criterion}.
        Provide constructive, evidence-based criticism."""
        
        prompt = f"""
        Identify specific issues with {criterion} in this research:
        
        Content: {content}
        Context: {json.dumps(context, indent=2)}
        Severity: {severity.value}
        
        Provide:
        1. Specific description of the issue
        2. Evidence supporting the criticism
        3. Concrete recommendations for improvement
        4. Potential impact if not addressed
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return CriticalAssessment(
            id=f"assess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{criterion}",
            subject=criterion,
            criticism_type=CriticismType.METHODOLOGICAL,  # Would be determined from analysis
            severity=severity,
            description=response,
            evidence=["Detailed analysis of content and context"],
            recommendations=["Specific improvements based on criticism"],
            confidence=0.80,
            reviewer_notes=f"Critical assessment of {criterion}",
            created_at=datetime.utcnow()
        )
    
    async def _check_methodological_issues(self, content: str) -> List[CriticalAssessment]:
        """Check for specific methodological issues."""
        system_prompt = """You are a methodological expert identifying potential issues in experimental design and execution."""
        
        prompt = f"""
        Analyze this research for methodological issues:
        
        Content: {content}
        
        Look for:
        1. Inadequate experimental controls
        2. Inappropriate statistical methods
        3. Insufficient sample sizes
        4. Confounding variables
        5. Selection bias
        6. Measurement errors
        7. Protocol deviations
        
        Identify specific issues and their severity.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse response for issues (simplified)
        issues = []
        if "control" in response.lower() and ("inadequate" in response.lower() or "insufficient" in response.lower()):
            issues.append(
                CriticalAssessment(
                    id=f"method_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_controls",
                    subject="Experimental Controls",
                    criticism_type=CriticismType.METHODOLOGICAL,
                    severity=CriticismSeverity.MAJOR,
                    description="Inadequate experimental controls identified",
                    evidence=["Analysis of experimental design"],
                    recommendations=["Include appropriate positive and negative controls"],
                    confidence=0.85,
                    reviewer_notes="Methodological issue with controls",
                    created_at=datetime.utcnow()
                )
            )
        
        return issues
    
    async def _check_statistical_issues(self, content: str) -> List[CriticalAssessment]:
        """Check for specific statistical issues."""
        system_prompt = """You are a biostatistician identifying statistical issues in research."""
        
        prompt = f"""
        Analyze this research for statistical issues:
        
        Content: {content}
        
        Look for:
        1. Inappropriate statistical tests
        2. Multiple comparison problems
        3. P-hacking or data dredging
        4. Assumption violations
        5. Insufficient power analysis
        6. Incorrect interpretation of results
        7. Missing confidence intervals
        
        Identify specific statistical concerns.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Parse response for issues (simplified)
        issues = []
        if "power" in response.lower() and ("insufficient" in response.lower() or "inadequate" in response.lower()):
            issues.append(
                CriticalAssessment(
                    id=f"stat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_power",
                    subject="Statistical Power",
                    criticism_type=CriticismType.STATISTICAL,
                    severity=CriticismSeverity.MODERATE,
                    description="Insufficient statistical power analysis",
                    evidence=["Statistical methodology review"],
                    recommendations=["Conduct proper power analysis for sample size determination"],
                    confidence=0.80,
                    reviewer_notes="Statistical power issue identified",
                    created_at=datetime.utcnow()
                )
            )
        
        return issues
    
    async def _identify_strengths(self, section_assessments: Dict[str, Any], criteria_scores: Dict[str, float]) -> List[str]:
        """Identify strengths in the research."""
        strengths = []
        
        # High-scoring criteria are strengths
        for criterion, score in criteria_scores.items():
            if score >= 8.0:
                strengths.append(f"Strong {criterion} with score of {score:.1f}")
        
        # Add general strengths
        strengths.extend([
            "Clear research objectives",
            "Systematic approach to investigation",
            "Appropriate use of established methods"
        ])
        
        return strengths
    
    async def _identify_weaknesses(self, section_assessments: Dict[str, Any], criteria_scores: Dict[str, float]) -> List[str]:
        """Identify weaknesses in the research."""
        weaknesses = []
        
        # Low-scoring criteria are weaknesses
        for criterion, score in criteria_scores.items():
            if score < 6.0:
                weaknesses.append(f"Concerns with {criterion} (score: {score:.1f})")
        
        return weaknesses
    
    async def _generate_recommendations(self, assessments: List[CriticalAssessment], weaknesses: List[str]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Extract recommendations from critical assessments
        for assessment in assessments:
            recommendations.extend(assessment.recommendations)
        
        # Add general recommendations
        recommendations.extend([
            "Consider additional peer review before submission",
            "Ensure all methodological details are clearly documented",
            "Verify statistical assumptions and methods",
            "Strengthen the discussion of limitations"
        ])
        
        # Remove duplicates
        return list(set(recommendations))
    
    async def _calculate_overall_score(self, criteria_scores: Dict[str, float]) -> float:
        """Calculate overall score based on criteria scores."""
        if not criteria_scores:
            return 5.0
        
        # Weight scores according to criticism criteria
        weighted_score = 0.0
        total_weight = 0.0
        
        for criterion, score in criteria_scores.items():
            weight = self.criticism_criteria.get(criterion, {}).get("weight", 1.0)
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else sum(criteria_scores.values()) / len(criteria_scores)
    
    async def _make_review_decision(self, overall_score: float, assessments: List[CriticalAssessment], template: Dict[str, Any]) -> str:
        """Make a review decision based on score and assessments."""
        critical_issues = len([a for a in assessments if a.severity == CriticismSeverity.CRITICAL])
        major_issues = len([a for a in assessments if a.severity == CriticismSeverity.MAJOR])
        
        # Decision logic
        if critical_issues > 0:
            return "reject"
        elif major_issues > 2 or overall_score < 5.0:
            return "major_revision"
        elif major_issues > 0 or overall_score < 7.0:
            return "minor_revision"
        else:
            return "accept"
    
    async def _perform_critical_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive critical analysis."""
        content = parameters.get("content", "")
        focus_areas = parameters.get("focus_areas", [])
        
        analysis_results = {}
        
        for area in focus_areas:
            if area in self.criticism_criteria:
                analysis_results[area] = await self._analyze_specific_area(content, area)
        
        return {
            "analysis_results": analysis_results,
            "overall_assessment": "Comprehensive critical analysis completed",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_specific_area(self, content: str, area: str) -> Dict[str, Any]:
        """Analyze a specific area of the content."""
        criteria_info = self.criticism_criteria[area]
        
        system_prompt = f"""You are an expert critic analyzing {area} in research work.
        Focus on {criteria_info['description']}."""
        
        prompt = f"""
        Analyze the {area} in this research:
        
        Content: {content}
        Criteria: {criteria_info['criteria']}
        
        Provide detailed analysis covering each criterion.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "area": area,
            "analysis": response,
            "score": 7.0,  # Would be calculated from detailed analysis
            "issues_identified": [],
            "recommendations": []
        }
    
    async def _evaluate_reasoning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate logical reasoning and consistency."""
        content = parameters.get("content", "")
        
        system_prompt = """You are a logic expert evaluating reasoning and argumentation in research."""
        
        prompt = f"""
        Evaluate the logical reasoning in this research:
        
        Content: {content}
        
        Assess:
        1. Logical consistency of arguments
        2. Validity of inferences
        3. Soundness of conclusions
        4. Presence of logical fallacies
        5. Coherence of overall reasoning
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        return {
            "reasoning_evaluation": response,
            "logical_consistency": "Assessment of logical consistency",
            "identified_fallacies": [],
            "reasoning_strength": 7.5,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_research(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate research methodology and conclusions."""
        return await self._conduct_peer_review(parameters)
    
    async def _control_quality(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality control assessment."""
        content = parameters.get("content", "")
        quality_standards = parameters.get("standards", [])
        
        quality_assessment = {}
        
        for metric_name, metric_info in self.quality_metrics.items():
            quality_assessment[metric_name] = await self._assess_quality_metric(content, metric_name, metric_info)
        
        return {
            "quality_assessment": quality_assessment,
            "overall_quality": "Quality control assessment completed",
            "recommendations": ["Follow quality standards", "Implement quality checks"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _assess_quality_metric(self, content: str, metric_name: str, metric_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a specific quality metric."""
        system_prompt = f"""You are a quality assessor evaluating {metric_name} in research work."""
        
        prompt = f"""
        Assess the {metric_name} in this research:
        
        Content: {content}
        Description: {metric_info['description']}
        Components: {metric_info['components']}
        
        Rate on scale {metric_info['scale']} and provide justification.
        """
        
        response = await self.generate_response(prompt, system_prompt)
        
        # Determine quality level based on score
        score = 7.0  # Default score
        quality_level = "good"
        
        for level, threshold in metric_info["thresholds"].items():
            if score >= threshold:
                quality_level = level
                break
        
        return {
            "metric": metric_name,
            "score": score,
            "quality_level": quality_level,
            "assessment": response,
            "components_evaluated": metric_info["components"]
        }
    
    async def _handle_general_criticism_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general criticism tasks."""
        return await self._perform_critical_analysis(parameters)
    
    async def get_capabilities_info(self) -> Dict[str, Any]:
        """Return detailed information about critic capabilities."""
        return {
            "agent_type": "critic",
            "specialization": "Critical evaluation and peer review",
            "capabilities": [cap.value for cap in self.capabilities],
            "criticism_types": [ct.value for ct in CriticismType],
            "severity_levels": [cs.value for cs in CriticismSeverity],
            "quality_metrics": list(self.quality_metrics.keys()),
            "review_templates": list(self.review_templates.keys()),
            "primary_functions": [
                "Conduct comprehensive peer reviews",
                "Perform critical analysis of research",
                "Assess methodological rigor",
                "Evaluate statistical validity",
                "Check logical consistency",
                "Quality control assessment"
            ],
            "output_formats": [
                "Detailed peer reviews",
                "Critical assessments",
                "Quality evaluations",
                "Improvement recommendations",
                "Review decisions"
            ]
        }


# Public interface function for backward compatibility
async def CriticAgent_create(agent_id: str, config: Dict[str, Any]) -> CriticAgent:
    """Create a CriticAgent instance."""
    return CriticAgent(agent_id, config)

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseAgent, AgentCapability


class CriticAgent(BaseAgent):
    """
    Specialized agent for autonomous scientific peer review and validation.
    
    This agent applies rigorous scientific standards to evaluate hypotheses,
    experimental designs, and results to ensure scientific integrity.
    """
    
    def __init__(self, agent_id: str):
        capabilities = [
            AgentCapability(
                name="review_hypothesis",
                description="Conduct comprehensive review of scientific hypotheses",
                input_schema={
                    "hypothesis": {"type": "string", "required": True},
                    "domain": {"type": "string", "required": True},
                    "supporting_evidence": {"type": "array", "required": False},
                    "review_criteria": {"type": "array", "required": False}
                },
                output_schema={
                    "review_score": {"type": "number"},
                    "strengths": {"type": "array"},
                    "weaknesses": {"type": "array"},
                    "recommendations": {"type": "array"},
                    "approval_status": {"type": "string"}
                },
                estimated_duration=35.0,
                confidence_level=0.92
            ),
            AgentCapability(
                name="review_design",
                description="Review experimental design for scientific rigor",
                input_schema={
                    "design": {"type": "object", "required": True},
                    "hypothesis": {"type": "string", "required": False},
                    "review_standards": {"type": "array", "required": False}
                },
                output_schema={
                    "design_score": {"type": "number"},
                    "methodology_assessment": {"type": "object"},
                    "statistical_validity": {"type": "object"},
                    "improvement_suggestions": {"type": "array"}
                },
                estimated_duration=40.0,
                confidence_level=0.89
            ),
            AgentCapability(
                name="conduct_literature_review",
                description="Conduct comprehensive literature review and prior art analysis",
                input_schema={
                    "hypothesis": {"type": "string", "required": True},
                    "domain": {"type": "string", "required": True},
                    "depth": {"type": "string", "required": False}
                },
                output_schema={
                    "literature_summary": {"type": "string"},
                    "prior_work": {"type": "array"},
                    "novelty_assessment": {"type": "object"},
                    "knowledge_gaps": {"type": "array"}
                },
                estimated_duration=50.0,
                confidence_level=0.87
            ),
            AgentCapability(
                name="validate_results",
                description="Validate experimental results and conclusions",
                input_schema={
                    "results": {"type": "object", "required": True},
                    "experimental_design": {"type": "object", "required": True},
                    "hypothesis": {"type": "string", "required": True}
                },
                output_schema={
                    "validation_status": {"type": "boolean"},
                    "statistical_significance": {"type": "object"},
                    "conclusion_validity": {"type": "object"},
                    "replication_requirements": {"type": "array"}
                },
                estimated_duration=45.0,
                confidence_level=0.91
            )
        ]
        
        llm_config = {
            "model": "gpt-4-turbo",
            "temperature": 0.1,  # Very low temperature for rigorous analysis
            "max_tokens": 4096,
            "system_prompt": self._get_system_prompt()
        }
        
        super().__init__(agent_id, "critic", capabilities, llm_config)
        
        # Critic-specific attributes
        self.review_standards = {}
        self.literature_database = {}
        self.review_history = []
        self.expert_knowledge = {}
        
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the critic agent."""
        return """You are an autonomous scientific peer review AI agent within the ASHES research system.

Your role is to:
1. Conduct rigorous peer review of scientific hypotheses, experimental designs, and results
2. Apply the highest standards of scientific rigor and methodology
3. Identify potential flaws, biases, and areas for improvement
4. Ensure reproducibility and statistical validity
5. Validate scientific conclusions against evidence

Critical review standards:
- Evaluate novelty against existing literature
- Assess testability and falsifiability of hypotheses
- Review experimental design for proper controls and statistical power
- Check for potential confounding variables and biases
- Validate statistical methods and significance testing
- Ensure reproducibility requirements are met
- Assess safety and ethical considerations
- Verify logical consistency of conclusions

Your review criteria include:
- Scientific rigor and methodology
- Statistical validity and power
- Reproducibility and transparency
- Safety and ethical compliance
- Novelty and significance
- Clarity and logical consistency
- Evidence quality and strength
- Potential for scientific impact

Be thorough, objective, and constructive in all reviews. Identify both strengths and weaknesses.
Provide specific, actionable recommendations for improvement.
Maintain the highest standards of scientific integrity."""
    
    async def _execute_capability(self, capability_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a specific critic capability."""
        
        if capability_name == "review_hypothesis":
            return await self._review_hypothesis(parameters)
        elif capability_name == "review_design":
            return await self._review_design(parameters)
        elif capability_name == "conduct_literature_review":
            return await self._conduct_literature_review(parameters)
        elif capability_name == "validate_results":
            return await self._validate_results(parameters)
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _review_hypothesis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive hypothesis review."""
        hypothesis = parameters.get("hypothesis")
        domain = parameters.get("domain")
        supporting_evidence = parameters.get("supporting_evidence", [])
        
        self.logger.info(f"Reviewing hypothesis in domain: {domain}")
        
        # Simulate rigorous review process
        await asyncio.sleep(4)
        
        # Evaluate hypothesis across multiple dimensions
        review_score = 0.0
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Testability assessment
        if "test" in hypothesis.lower() or "measure" in hypothesis.lower():
            review_score += 0.25
            strengths.append("Hypothesis appears testable with measurable outcomes")
        else:
            weaknesses.append("Testability could be more clearly defined")
            recommendations.append("Clarify how the hypothesis can be experimentally tested")
        
        # Novelty assessment
        if "novel" in hypothesis.lower() or "new" in hypothesis.lower():
            review_score += 0.20
            strengths.append("Claims novelty in approach or understanding")
        else:
            recommendations.append("Clarify the novel aspects compared to existing work")
        
        # Specificity assessment
        if any(word in hypothesis.lower() for word in ["specific", "precise", "quantitative"]):
            review_score += 0.20
            strengths.append("Hypothesis includes specific, quantitative predictions")
        else:
            weaknesses.append("Hypothesis could be more specific and quantitative")
            recommendations.append("Include specific, measurable predictions")
        
        # Scientific soundness
        if len(hypothesis.split()) > 15:  # Detailed hypothesis
            review_score += 0.20
            strengths.append("Comprehensive and detailed hypothesis statement")
        else:
            weaknesses.append("Hypothesis may lack sufficient detail")
            recommendations.append("Provide more detailed hypothesis statement")
        
        # Feasibility assessment
        review_score += 0.15  # Assume generally feasible
        strengths.append("Appears feasible with current technology")
        
        # Determine approval status
        if review_score >= 0.7:
            approval_status = "approved"
        elif review_score >= 0.5:
            approval_status = "approved_with_revisions"
        else:
            approval_status = "requires_major_revision"
        
        # Add general recommendations
        if approval_status != "approved":
            recommendations.extend([
                "Conduct thorough literature review to establish novelty",
                "Define clear success criteria for hypothesis testing",
                "Consider potential alternative explanations"
            ])
        
        result = {
            "review_score": round(review_score, 2),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
            "approval_status": approval_status
        }
        
        # Add to review history
        self.review_history.append({
            "type": "hypothesis_review",
            "timestamp": datetime.utcnow().isoformat(),
            "domain": domain,
            "score": review_score,
            "status": approval_status
        })
        
        self.logger.info(f"Hypothesis review completed: {approval_status} (score: {review_score})")
        
        return result
    
    async def _review_design(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Review experimental design for scientific rigor."""
        design = parameters.get("design")
        hypothesis = parameters.get("hypothesis", "")
        
        self.logger.info("Reviewing experimental design")
        
        # Simulate design review
        await asyncio.sleep(3)
        
        design_score = 0.0
        methodology_issues = []
        statistical_issues = []
        improvement_suggestions = []
        
        # Check for essential design elements
        if "controls" in design or (isinstance(design, dict) and design.get("controls")):
            design_score += 0.25
        else:
            methodology_issues.append("Missing control experiments")
            improvement_suggestions.append("Include appropriate positive and negative controls")
        
        if "replicates" in str(design) or "replicate" in str(design):
            design_score += 0.20
        else:
            statistical_issues.append("No mention of experimental replicates")
            improvement_suggestions.append("Include sufficient replicates for statistical analysis")
        
        if "statistics" in str(design) or "statistical" in str(design):
            design_score += 0.15
        else:
            statistical_issues.append("Statistical analysis plan not specified")
            improvement_suggestions.append("Define statistical analysis methods and significance criteria")
        
        if "safety" in str(design):
            design_score += 0.15
        else:
            methodology_issues.append("Safety considerations not adequately addressed")
            improvement_suggestions.append("Include comprehensive safety protocols")
        
        if "materials" in str(design) and "equipment" in str(design):
            design_score += 0.10
        else:
            methodology_issues.append("Incomplete materials and equipment specifications")
            improvement_suggestions.append("Provide detailed materials and equipment lists")
        
        if "procedure" in str(design) or "steps" in str(design):
            design_score += 0.15
        else:
            methodology_issues.append("Experimental procedure lacks detail")
            improvement_suggestions.append("Provide step-by-step experimental procedure")
        
        # Statistical validity assessment
        statistical_validity = {
            "power_analysis": "not_specified" if "power" not in str(design) else "specified",
            "sample_size": "not_specified" if "sample" not in str(design) else "specified",
            "randomization": "not_specified" if "random" not in str(design) else "specified",
            "blinding": "not_specified" if "blind" not in str(design) else "specified"
        }
        
        # Methodology assessment
        methodology_assessment = {
            "controls_adequate": len(methodology_issues) == 0,
            "procedure_clarity": "procedure" in str(design),
            "reproducibility": "reproducible" in str(design) or len(methodology_issues) <= 2,
            "identified_issues": methodology_issues
        }
        
        # Add general improvements if score is low
        if design_score < 0.7:
            improvement_suggestions.extend([
                "Consider consulting statistical expert for power analysis",
                "Review literature for best practices in this domain",
                "Consider potential confounding variables"
            ])
        
        result = {
            "design_score": round(design_score, 2),
            "methodology_assessment": methodology_assessment,
            "statistical_validity": statistical_validity,
            "improvement_suggestions": improvement_suggestions
        }
        
        self.logger.info(f"Design review completed: score {design_score}")
        
        return result
    
    async def _conduct_literature_review(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct comprehensive literature review."""
        hypothesis = parameters.get("hypothesis")
        domain = parameters.get("domain")
        depth = parameters.get("depth", "standard")
        
        self.logger.info(f"Conducting literature review for {domain}")
        
        # Simulate literature search and analysis
        await asyncio.sleep(5)
        
        # Mock literature findings based on domain
        if "materials" in domain.lower():
            prior_work = [
                {
                    "title": "Recent Advances in Nanostructured Materials",
                    "authors": "Smith et al.",
                    "year": 2023,
                    "relevance": "high",
                    "key_findings": "Similar approaches to nanostructure synthesis"
                },
                {
                    "title": "Metal-Organic Frameworks for Energy Applications",
                    "authors": "Johnson et al.",
                    "year": 2022,
                    "relevance": "medium",
                    "key_findings": "Complementary research in related materials"
                }
            ]
            knowledge_gaps = [
                "Limited understanding of structure-property relationships at nanoscale",
                "Lack of standardized synthesis protocols",
                "Insufficient long-term stability studies"
            ]
            novelty_score = 0.75
            
        elif "chemistry" in domain.lower():
            prior_work = [
                {
                    "title": "Catalytic Mechanisms in Green Chemistry",
                    "authors": "Brown et al.",
                    "year": 2023,
                    "relevance": "high",
                    "key_findings": "Related catalytic approaches"
                }
            ]
            knowledge_gaps = [
                "Mechanistic understanding of catalyst deactivation",
                "Scale-up challenges for industrial application"
            ]
            novelty_score = 0.68
            
        else:
            prior_work = [
                {
                    "title": "General Scientific Approaches",
                    "authors": "Various",
                    "year": 2023,
                    "relevance": "medium",
                    "key_findings": "Related methodologies"
                }
            ]
            knowledge_gaps = [
                "Limited systematic studies in this area",
                "Need for standardized methodologies"
            ]
            novelty_score = 0.70
        
        literature_summary = f"""
        Literature review of {domain} research relevant to the hypothesis reveals:
        
        1. Current State: The field has established fundamental principles but gaps remain
        2. Prior Work: {len(prior_work)} directly relevant studies identified
        3. Novelty: The proposed hypothesis addresses {len(knowledge_gaps)} identified gaps
        4. Research Trend: Increasing focus on systematic, automated approaches
        
        The hypothesis shows good potential for novel contribution to the field.
        """
        
        novelty_assessment = {
            "novelty_score": novelty_score,
            "comparison_to_prior_work": "Builds upon existing work with novel approach",
            "potential_contribution": "Significant advancement expected",
            "research_gap_addressed": knowledge_gaps[0] if knowledge_gaps else "General advancement"
        }
        
        result = {
            "literature_summary": literature_summary.strip(),
            "prior_work": prior_work,
            "novelty_assessment": novelty_assessment,
            "knowledge_gaps": knowledge_gaps
        }
        
        self.logger.info(f"Literature review completed: novelty score {novelty_score}")
        
        return result
    
    async def _validate_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experimental results and conclusions."""
        results = parameters.get("results")
        experimental_design = parameters.get("experimental_design")
        hypothesis = parameters.get("hypothesis")
        
        self.logger.info("Validating experimental results")
        
        # Simulate results validation
        await asyncio.sleep(3)
        
        validation_status = True
        statistical_issues = []
        conclusion_issues = []
        replication_requirements = []
        
        # Check for statistical significance
        if "p_value" in str(results) or "significance" in str(results):
            statistical_significance = {
                "significance_reported": True,
                "appropriate_tests": True,
                "multiple_testing_correction": "not_specified"
            }
        else:
            statistical_significance = {
                "significance_reported": False,
                "appropriate_tests": "not_specified",
                "multiple_testing_correction": "not_applicable"
            }
            statistical_issues.append("Statistical significance not reported")
            validation_status = False
        
        # Check conclusion validity
        if "conclusion" in str(results) or "hypothesis" in str(results):
            conclusion_validity = {
                "conclusions_supported": True,
                "overgeneralization": False,
                "alternative_explanations_considered": "not_specified"
            }
        else:
            conclusion_validity = {
                "conclusions_supported": False,
                "overgeneralization": "possible",
                "alternative_explanations_considered": False
            }
            conclusion_issues.append("Conclusions not clearly linked to results")
            validation_status = False
        
        # Replication requirements
        replication_requirements = [
            "Independent replication with different researchers",
            "Replication with larger sample size",
            "Validation under different experimental conditions"
        ]
        
        if not validation_status:
            replication_requirements.append("Address statistical and methodological issues before replication")
        
        result = {
            "validation_status": validation_status,
            "statistical_significance": statistical_significance,
            "conclusion_validity": conclusion_validity,
            "replication_requirements": replication_requirements
        }
        
        self.logger.info(f"Results validation completed: {'valid' if validation_status else 'issues identified'}")
        
        return result
    
    async def conduct_literature_review(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Public interface for literature review."""
        return await self._execute_capability("conduct_literature_review", request)
    
    async def review_design(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Public interface for design review."""
        return await self._execute_capability("review_design", request)
    
    def get_review_history(self) -> List[Dict[str, Any]]:
        """Get the history of conducted reviews."""
        return self.review_history.copy()
    
    def get_review_standards(self) -> Dict[str, Any]:
        """Get the review standards used by this critic."""
        return self.review_standards.copy()
