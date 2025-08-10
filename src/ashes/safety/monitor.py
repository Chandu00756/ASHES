"""
Safety Monitor - Comprehensive safety and emergency management system for ASHES.

This module implements real-time safety monitoring, emergency protocols,
and risk assessment for autonomous laboratory operations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.config import get_config
from ..core.logging import get_logger


class SafetyLevel(Enum):
    """Safety levels for operations and conditions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyType(Enum):
    """Types of emergency situations."""
    FIRE = "fire"
    CHEMICAL_SPILL = "chemical_spill"
    EQUIPMENT_FAILURE = "equipment_failure"
    POWER_FAILURE = "power_failure"
    SAFETY_BREACH = "safety_breach"
    HUMAN_EMERGENCY = "human_emergency"


@dataclass
class SafetySensor:
    """Safety sensor configuration and state."""
    sensor_id: str
    sensor_type: str
    location: str
    status: str = "active"
    
    # Sensor thresholds
    min_safe_value: Optional[float] = None
    max_safe_value: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Current readings
    current_value: Optional[float] = None
    last_reading_time: Optional[datetime] = None
    
    # Calibration
    calibration_date: Optional[datetime] = None
    calibration_due: Optional[datetime] = None


@dataclass
class SafetyProtocol:
    """Safety protocol definition."""
    protocol_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    required_actions: List[str]
    safety_level: SafetyLevel
    auto_execute: bool = True


@dataclass
class EmergencyProcedure:
    """Emergency response procedure."""
    procedure_id: str
    emergency_type: EmergencyType
    severity: SafetyLevel
    immediate_actions: List[str]
    notification_contacts: List[str]
    equipment_shutdowns: List[str]


class SafetyMonitor:
    """Main safety monitoring and emergency response system."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Safety sensors
        self.sensors: Dict[str, SafetySensor] = {}
        
        # Safety protocols and procedures
        self.safety_protocols: Dict[str, SafetyProtocol] = {}
        self.emergency_procedures: Dict[str, EmergencyProcedure] = {}
        
        # Emergency state
        self.emergency_active = False
        self.emergency_level = SafetyLevel.LOW
        self.emergency_log: List[Dict[str, Any]] = []
        
        # Safety checks
        self.safety_checks_enabled = True
        self.last_safety_check = None
        self.safety_violations: List[Dict[str, Any]] = []
        
        # Environmental monitoring
        self.environment_status = {
            "temperature": 22.0,
            "humidity": 45.0,
            "air_quality": "good",
            "radiation_level": 0.0,
            "gas_levels": {}
        }
        
        self._initialize_safety_systems()
    
    def _initialize_safety_systems(self):
        """Initialize safety sensors, protocols, and procedures."""
        self._setup_sensors()
        self._setup_safety_protocols()
        self._setup_emergency_procedures()
    
    def _setup_sensors(self):
        """Setup safety sensors throughout the laboratory."""
        # Temperature sensors
        self.sensors["temp_001"] = SafetySensor(
            sensor_id="temp_001",
            sensor_type="temperature",
            location="main_lab",
            min_safe_value=15.0,
            max_safe_value=35.0,
            critical_threshold=40.0
        )
        
        self.sensors["temp_002"] = SafetySensor(
            sensor_id="temp_002",
            sensor_type="temperature",
            location="furnace_area",
            min_safe_value=15.0,
            max_safe_value=45.0,
            critical_threshold=60.0
        )
        
        # Pressure sensors
        self.sensors["press_001"] = SafetySensor(
            sensor_id="press_001",
            sensor_type="pressure",
            location="main_lab",
            min_safe_value=0.9,  # atm
            max_safe_value=1.1,
            critical_threshold=1.5
        )
        
        # Gas detection sensors
        for gas in ["CO", "CO2", "H2S", "NH3", "O2"]:
            self.sensors[f"gas_{gas.lower()}"] = SafetySensor(
                sensor_id=f"gas_{gas.lower()}",
                sensor_type="gas_detector",
                location="main_lab",
                max_safe_value=self._get_gas_safe_limit(gas),
                critical_threshold=self._get_gas_critical_limit(gas)
            )
        
        # Radiation sensors
        self.sensors["rad_001"] = SafetySensor(
            sensor_id="rad_001",
            sensor_type="radiation",
            location="xrd_area",
            max_safe_value=0.02,  # mSv/h
            critical_threshold=0.1
        )
        
        # Fire detection sensors
        self.sensors["fire_001"] = SafetySensor(
            sensor_id="fire_001",
            sensor_type="smoke_detector",
            location="main_lab",
            critical_threshold=1.0  # Smoke density
        )
        
        self.sensors["fire_002"] = SafetySensor(
            sensor_id="fire_002",
            sensor_type="heat_detector",
            location="furnace_area",
            critical_threshold=70.0  # °C
        )
        
        self.logger.info(f"Initialized {len(self.sensors)} safety sensors")
    
    def _get_gas_safe_limit(self, gas: str) -> float:
        """Get safe exposure limits for gases (ppm)."""
        limits = {
            "CO": 35.0,     # Carbon monoxide
            "CO2": 5000.0,  # Carbon dioxide
            "H2S": 10.0,    # Hydrogen sulfide
            "NH3": 25.0,    # Ammonia
            "O2": 23.5      # Oxygen (max for safe range)
        }
        return limits.get(gas, 10.0)
    
    def _get_gas_critical_limit(self, gas: str) -> float:
        """Get critical exposure limits for gases (ppm)."""
        limits = {
            "CO": 200.0,
            "CO2": 40000.0,
            "H2S": 100.0,
            "NH3": 300.0,
            "O2": 25.0
        }
        return limits.get(gas, 50.0)
    
    def _setup_safety_protocols(self):
        """Setup automated safety protocols."""
        # High temperature protocol
        self.safety_protocols["high_temp"] = SafetyProtocol(
            protocol_id="high_temp",
            name="High Temperature Safety",
            description="Automated response to elevated temperatures",
            trigger_conditions=["temperature > max_safe_value"],
            required_actions=[
                "alert_operators",
                "increase_ventilation",
                "reduce_heat_sources"
            ],
            safety_level=SafetyLevel.MEDIUM,
            auto_execute=True
        )
        
        # Gas leak protocol
        self.safety_protocols["gas_leak"] = SafetyProtocol(
            protocol_id="gas_leak",
            name="Gas Leak Response",
            description="Response to hazardous gas detection",
            trigger_conditions=["gas_level > max_safe_value"],
            required_actions=[
                "activate_ventilation",
                "isolate_gas_sources",
                "evacuate_if_critical",
                "alert_emergency_services"
            ],
            safety_level=SafetyLevel.HIGH,
            auto_execute=True
        )
        
        # Radiation safety protocol
        self.safety_protocols["radiation"] = SafetyProtocol(
            protocol_id="radiation",
            name="Radiation Safety",
            description="Response to elevated radiation levels",
            trigger_conditions=["radiation > max_safe_value"],
            required_actions=[
                "shutdown_xray_sources",
                "restrict_access",
                "monitor_exposure"
            ],
            safety_level=SafetyLevel.HIGH,
            auto_execute=True
        )
        
        # Equipment failure protocol
        self.safety_protocols["equipment_failure"] = SafetyProtocol(
            protocol_id="equipment_failure",
            name="Equipment Failure Response",
            description="Response to equipment failures",
            trigger_conditions=["device_error", "communication_loss"],
            required_actions=[
                "emergency_stop_device",
                "isolate_failed_equipment",
                "assess_safety_impact"
            ],
            safety_level=SafetyLevel.MEDIUM,
            auto_execute=True
        )
        
        self.logger.info(f"Setup {len(self.safety_protocols)} safety protocols")
    
    def _setup_emergency_procedures(self):
        """Setup emergency response procedures."""
        # Fire emergency
        self.emergency_procedures["fire"] = EmergencyProcedure(
            procedure_id="fire",
            emergency_type=EmergencyType.FIRE,
            severity=SafetyLevel.CRITICAL,
            immediate_actions=[
                "activate_fire_suppression",
                "evacuate_laboratory",
                "call_fire_department",
                "emergency_stop_all_equipment"
            ],
            notification_contacts=["emergency_services", "lab_manager", "safety_officer"],
            equipment_shutdowns=["all_devices", "electrical_systems", "gas_lines"]
        )
        
        # Chemical spill emergency
        self.emergency_procedures["chemical_spill"] = EmergencyProcedure(
            procedure_id="chemical_spill",
            emergency_type=EmergencyType.CHEMICAL_SPILL,
            severity=SafetyLevel.HIGH,
            immediate_actions=[
                "contain_spill",
                "activate_ventilation",
                "evacuate_affected_area",
                "assess_chemical_hazards"
            ],
            notification_contacts=["lab_manager", "safety_officer", "hazmat_team"],
            equipment_shutdowns=["affected_area_equipment"]
        )
        
        # Equipment failure emergency
        self.emergency_procedures["equipment_failure"] = EmergencyProcedure(
            procedure_id="equipment_failure",
            emergency_type=EmergencyType.EQUIPMENT_FAILURE,
            severity=SafetyLevel.MEDIUM,
            immediate_actions=[
                "emergency_stop_device",
                "isolate_power_source",
                "assess_damage",
                "secure_area"
            ],
            notification_contacts=["lab_manager", "maintenance_team"],
            equipment_shutdowns=["failed_equipment", "dependent_systems"]
        )
        
        self.logger.info(f"Setup {len(self.emergency_procedures)} emergency procedures")
    
    async def start(self):
        """Start the safety monitoring system."""
        self.logger.info("Starting safety monitoring system")
        
        # Start sensor monitoring
        asyncio.create_task(self._monitor_sensors())
        
        # Start safety checks
        asyncio.create_task(self._periodic_safety_checks())
        
        # Start emergency response system
        asyncio.create_task(self._emergency_response_loop())
        
        self.logger.info("Safety monitoring system started")
    
    async def stop(self):
        """Stop the safety monitoring system."""
        self.logger.info("Stopping safety monitoring system")
        self.safety_checks_enabled = False
    
    async def check_safety(self) -> bool:
        """Perform comprehensive safety check."""
        try:
            # Check all sensors
            sensor_check = await self._check_all_sensors()
            
            # Check environmental conditions
            environment_check = await self._check_environment()
            
            # Check equipment status
            equipment_check = await self._check_equipment_safety()
            
            # Update last check time
            self.last_safety_check = datetime.utcnow()
            
            overall_safe = sensor_check and environment_check and equipment_check
            
            if not overall_safe:
                await self._trigger_safety_protocol("general_safety_violation")
            
            return overall_safe
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    async def check_experiment_safety(self, experiment_request: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an experiment is safe to execute."""
        experiment_id = experiment_request.get("experiment_id", "unknown")
        design = experiment_request.get("design", {})
        
        self.logger.info(f"Checking safety for experiment {experiment_id}")
        
        safety_assessment = {
            "approved": True,
            "safety_level": SafetyLevel.LOW,
            "risks": [],
            "mitigations": [],
            "requirements": [],
            "reasons": []
        }
        
        # Check materials safety
        materials = design.get("materials", [])
        for material in materials:
            material_risk = await self._assess_material_risk(material)
            if material_risk["risk_level"] != SafetyLevel.LOW:
                safety_assessment["risks"].append(material_risk)
                safety_assessment["safety_level"] = max(
                    safety_assessment["safety_level"], 
                    material_risk["risk_level"], 
                    key=lambda x: x.value
                )
        
        # Check procedure safety
        procedure = design.get("procedure", [])
        for step in procedure:
            step_risk = await self._assess_procedure_step_risk(step)
            if step_risk["risk_level"] != SafetyLevel.LOW:
                safety_assessment["risks"].append(step_risk)
        
        # Check equipment requirements
        equipment_requirements = design.get("equipment_requirements", [])
        equipment_check = await self._check_equipment_availability(equipment_requirements)
        if not equipment_check["available"]:
            safety_assessment["approved"] = False
            safety_assessment["reasons"].append("Required equipment not available")
        
        # Check environmental conditions
        if not await self._check_environment():
            safety_assessment["approved"] = False
            safety_assessment["reasons"].append("Unsafe environmental conditions")
        
        # Determine if experiment should be approved
        if safety_assessment["safety_level"] == SafetyLevel.CRITICAL:
            safety_assessment["approved"] = False
            safety_assessment["reasons"].append("Critical safety risks identified")
        elif safety_assessment["safety_level"] == SafetyLevel.HIGH:
            # High-risk experiments require additional safety measures
            safety_assessment["requirements"].extend([
                "continuous_monitoring",
                "safety_officer_presence",
                "emergency_response_ready"
            ])
        
        self.logger.info(
            f"Experiment {experiment_id} safety check: "
            f"{'APPROVED' if safety_assessment['approved'] else 'REJECTED'} "
            f"(Level: {safety_assessment['safety_level'].value})"
        )
        
        return safety_assessment
    
    async def _assess_material_risk(self, material: str) -> Dict[str, Any]:
        """Assess safety risk of a material."""
        # Simulate material safety database lookup
        material_hazards = {
            "sodium": {"toxicity": "high", "flammability": "high", "reactivity": "high"},
            "acids": {"toxicity": "medium", "corrosivity": "high", "reactivity": "medium"},
            "titanium_dioxide": {"toxicity": "low", "flammability": "low", "reactivity": "low"},
            "oxygen": {"toxicity": "low", "flammability": "high", "reactivity": "high"}
        }
        
        # Determine risk level based on material
        if any(hazard in material.lower() for hazard in ["sodium", "potassium", "lithium"]):
            risk_level = SafetyLevel.HIGH
        elif any(hazard in material.lower() for hazard in ["acid", "base", "oxidizer"]):
            risk_level = SafetyLevel.MEDIUM
        else:
            risk_level = SafetyLevel.LOW
        
        return {
            "material": material,
            "risk_level": risk_level,
            "hazards": material_hazards.get(material.lower(), {}),
            "safety_measures": self._get_material_safety_measures(material)
        }
    
    def _get_material_safety_measures(self, material: str) -> List[str]:
        """Get required safety measures for a material."""
        if any(hazard in material.lower() for hazard in ["sodium", "potassium"]):
            return ["inert_atmosphere", "fire_suppression", "protective_equipment"]
        elif "acid" in material.lower():
            return ["fume_hood", "acid_resistant_equipment", "neutralization_agent"]
        else:
            return ["standard_ventilation", "protective_equipment"]
    
    async def _assess_procedure_step_risk(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Assess safety risk of a procedure step."""
        step_type = step.get("type", "unknown")
        parameters = step.get("parameters", {})
        
        risk_level = SafetyLevel.LOW
        risks = []
        
        if step_type == "synthesis":
            temperature = parameters.get("temperature", 25)
            if temperature > 1000:
                risk_level = SafetyLevel.HIGH
                risks.append("High temperature operation")
            elif temperature > 500:
                risk_level = SafetyLevel.MEDIUM
                risks.append("Elevated temperature")
        
        elif step_type == "characterization":
            if "xrd" in step.get("equipment", "").lower():
                risk_level = SafetyLevel.MEDIUM
                risks.append("X-ray radiation exposure")
        
        return {
            "step_type": step_type,
            "risk_level": risk_level,
            "risks": risks,
            "safety_measures": self._get_step_safety_measures(step_type)
        }
    
    def _get_step_safety_measures(self, step_type: str) -> List[str]:
        """Get required safety measures for a procedure step."""
        measures = {
            "synthesis": ["temperature_monitoring", "emergency_cooling", "ventilation"],
            "characterization": ["radiation_monitoring", "access_control", "protective_equipment"],
            "sample_handling": ["contamination_control", "protective_equipment"]
        }
        return measures.get(step_type, ["standard_precautions"])
    
    async def _check_equipment_availability(self, equipment_list: List[str]) -> Dict[str, Any]:
        """Check if required equipment is available and safe."""
        # This would integrate with the laboratory controller
        return {
            "available": True,
            "equipment_status": {equipment: "available" for equipment in equipment_list},
            "safety_checks_passed": True
        }
    
    async def _check_all_sensors(self) -> bool:
        """Check all safety sensors."""
        all_safe = True
        
        for sensor in self.sensors.values():
            if not await self._check_sensor(sensor):
                all_safe = False
        
        return all_safe
    
    async def _check_sensor(self, sensor: SafetySensor) -> bool:
        """Check individual sensor status and readings."""
        # Simulate sensor reading
        sensor.current_value = await self._read_sensor(sensor)
        sensor.last_reading_time = datetime.utcnow()
        
        # Check if reading is within safe limits
        if sensor.max_safe_value and sensor.current_value > sensor.max_safe_value:
            await self._handle_sensor_violation(sensor, "above_safe_limit")
            return False
        
        if sensor.min_safe_value and sensor.current_value < sensor.min_safe_value:
            await self._handle_sensor_violation(sensor, "below_safe_limit")
            return False
        
        if sensor.critical_threshold and sensor.current_value > sensor.critical_threshold:
            await self._handle_sensor_violation(sensor, "critical_threshold")
            return False
        
        return True
    
    async def _read_sensor(self, sensor: SafetySensor) -> float:
        """Read sensor value (simulated)."""
        # Simulate different sensor types
        import random
        
        if sensor.sensor_type == "temperature":
            if "furnace" in sensor.location:
                return 20 + random.uniform(0, 10)  # Slightly elevated
            else:
                return 22 + random.uniform(-2, 3)  # Room temperature
        
        elif sensor.sensor_type == "pressure":
            return 1.0 + random.uniform(-0.05, 0.05)  # Near atmospheric
        
        elif sensor.sensor_type == "gas_detector":
            return random.uniform(0, sensor.max_safe_value * 0.1)  # Low levels
        
        elif sensor.sensor_type == "radiation":
            return random.uniform(0, 0.005)  # Very low background
        
        elif sensor.sensor_type in ["smoke_detector", "heat_detector"]:
            return random.uniform(0, 0.1)  # No detection
        
        else:
            return 0.0
    
    async def _handle_sensor_violation(self, sensor: SafetySensor, violation_type: str):
        """Handle sensor safety violations."""
        violation = {
            "sensor_id": sensor.sensor_id,
            "sensor_type": sensor.sensor_type,
            "location": sensor.location,
            "violation_type": violation_type,
            "current_value": sensor.current_value,
            "safe_limit": sensor.max_safe_value or sensor.min_safe_value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.safety_violations.append(violation)
        
        self.logger.warning(
            f"Safety violation detected: {sensor.sensor_id} "
            f"({violation_type}) - Value: {sensor.current_value}"
        )
        
        # Trigger appropriate safety protocol
        if sensor.sensor_type == "temperature":
            await self._trigger_safety_protocol("high_temp")
        elif sensor.sensor_type == "gas_detector":
            await self._trigger_safety_protocol("gas_leak")
        elif sensor.sensor_type == "radiation":
            await self._trigger_safety_protocol("radiation")
    
    async def _check_environment(self) -> bool:
        """Check environmental safety conditions."""
        # Update environment status
        self.environment_status.update({
            "temperature": 22.5,
            "humidity": 45.0,
            "air_quality": "good",
            "radiation_level": 0.001
        })
        
        # Check if conditions are safe
        if self.environment_status["temperature"] > 35:
            return False
        if self.environment_status["humidity"] > 70:
            return False
        if self.environment_status["radiation_level"] > 0.02:
            return False
        
        return True
    
    async def _check_equipment_safety(self) -> bool:
        """Check safety status of all equipment."""
        # This would integrate with laboratory controller
        # For now, assume equipment is safe
        return True
    
    async def _trigger_safety_protocol(self, protocol_id: str):
        """Trigger a safety protocol."""
        if protocol_id not in self.safety_protocols:
            self.logger.error(f"Unknown safety protocol: {protocol_id}")
            return
        
        protocol = self.safety_protocols[protocol_id]
        
        self.logger.warning(f"Triggering safety protocol: {protocol.name}")
        
        if protocol.auto_execute:
            for action in protocol.required_actions:
                await self._execute_safety_action(action)
    
    async def _execute_safety_action(self, action: str):
        """Execute a safety action."""
        self.logger.info(f"Executing safety action: {action}")
        
        if action == "alert_operators":
            await self._send_alert("Safety alert: Immediate attention required")
        elif action == "increase_ventilation":
            await self._control_ventilation("increase")
        elif action == "emergency_stop_device":
            await self._emergency_stop_equipment()
        elif action == "evacuate_if_critical":
            await self._initiate_evacuation()
        # Add more action implementations as needed
    
    async def _send_alert(self, message: str):
        """Send safety alert to operators."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "safety_alert",
            "message": message,
            "severity": "high"
        }
        
        # In a real implementation, this would send alerts via email, SMS, etc.
        self.logger.warning(f"SAFETY ALERT: {message}")
    
    async def _control_ventilation(self, action: str):
        """Control laboratory ventilation system."""
        self.logger.info(f"Ventilation control: {action}")
        # Implementation would control actual ventilation system
    
    async def _emergency_stop_equipment(self):
        """Emergency stop all laboratory equipment."""
        self.logger.warning("Emergency stop triggered for all equipment")
        # This would interface with the laboratory controller
    
    async def _initiate_evacuation(self):
        """Initiate laboratory evacuation procedures."""
        self.logger.critical("EVACUATION INITIATED")
        await self._send_alert("EVACUATION ORDER: Clear the laboratory immediately")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety system status."""
        sensor_statuses = {}
        for sensor_id, sensor in self.sensors.items():
            sensor_statuses[sensor_id] = {
                "type": sensor.sensor_type,
                "location": sensor.location,
                "status": sensor.status,
                "current_value": sensor.current_value,
                "safe_range": [sensor.min_safe_value, sensor.max_safe_value],
                "last_reading": sensor.last_reading_time.isoformat() if sensor.last_reading_time else None
            }
        
        return {
            "safety_system_status": "operational",
            "emergency_active": self.emergency_active,
            "emergency_level": self.emergency_level.value,
            "last_safety_check": self.last_safety_check.isoformat() if self.last_safety_check else None,
            "sensors": sensor_statuses,
            "recent_violations": self.safety_violations[-10:],  # Last 10 violations
            "environment": self.environment_status,
            "protocols_active": len([p for p in self.safety_protocols.values() if p.auto_execute]),
            "emergency_procedures_ready": len(self.emergency_procedures)
        }
    
    async def get_constraints(self) -> Dict[str, Any]:
        """Get current safety constraints for experiments."""
        return {
            "max_temperature": 1200,  # °C
            "max_pressure": 10,       # atm
            "restricted_materials": ["plutonium", "enriched_uranium"],
            "required_safety_equipment": ["emergency_stop", "fire_suppression", "ventilation"],
            "operating_hours": {"start": "08:00", "end": "18:00"},
            "max_concurrent_experiments": 3,
            "radiation_limits": {"xrd": 0.02, "general": 0.001}  # mSv/h
        }
    
    async def _monitor_sensors(self):
        """Background task to continuously monitor sensors."""
        while self.safety_checks_enabled:
            try:
                for sensor in self.sensors.values():
                    await self._check_sensor(sensor)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Sensor monitoring error: {e}")
    
    async def _periodic_safety_checks(self):
        """Background task for periodic comprehensive safety checks."""
        while self.safety_checks_enabled:
            try:
                await self.check_safety()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Periodic safety check error: {e}")
    
    async def _emergency_response_loop(self):
        """Background task for emergency response monitoring."""
        while self.safety_checks_enabled:
            try:
                # Check for emergency conditions
                if len(self.safety_violations) > 0:
                    recent_violations = [
                        v for v in self.safety_violations
                        if datetime.fromisoformat(v["timestamp"]) > datetime.utcnow() - timedelta(minutes=5)
                    ]
                    
                    if len(recent_violations) >= 3:
                        await self._declare_emergency(EmergencyType.SAFETY_BREACH)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Emergency response loop error: {e}")
    
    async def _declare_emergency(self, emergency_type: EmergencyType):
        """Declare a laboratory emergency."""
        if not self.emergency_active:
            self.emergency_active = True
            self.emergency_level = SafetyLevel.HIGH
            
            emergency_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": emergency_type.value,
                "level": self.emergency_level.value,
                "triggered_by": "automatic_system"
            }
            
            self.emergency_log.append(emergency_log)
            
            self.logger.critical(f"EMERGENCY DECLARED: {emergency_type.value}")
            
            # Execute emergency procedures
            if emergency_type.value in self.emergency_procedures:
                procedure = self.emergency_procedures[emergency_type.value]
                for action in procedure.immediate_actions:
                    await self._execute_safety_action(action)
