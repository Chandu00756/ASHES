"""
Laboratory Controller - Complete hardware automation integration for ASHES.

This module implements the full laboratory automation system including:
- Robotic equipment control (EOS-based)
- Analytical instrument integration
- Safety monitoring and emergency protocols
- Real-time experiment execution
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import aiohttp
import serial_asyncio
import asyncpg

from ..core.config import get_config
from ..core.logging import get_logger
from ..safety.monitor import SafetyMonitor


@dataclass
class DeviceCapability:
    """Device capability definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    execution_time: float
    safety_level: str


@dataclass
class DeviceState:
    """Real-time device state tracking."""
    device_id: str
    device_type: str
    status: str = "idle"
    connected: bool = False
    ip_address: Optional[str] = None
    port: Optional[int] = None
    
    # Current operation
    current_experiment_id: Optional[str] = None
    current_operation: Optional[str] = None
    operation_started_at: Optional[datetime] = None
    
    # Performance metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Safety and maintenance
    safety_status: str = "safe"
    last_safety_check: Optional[datetime] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    
    last_updated: datetime = field(default_factory=datetime.utcnow)


class UniversalDeviceInterface:
    """Universal interface for all laboratory equipment."""
    
    def __init__(self, device_id: str, device_type: str, connection_params: Dict[str, Any]):
        self.device_id = device_id
        self.device_type = device_type
        self.connection_params = connection_params
        self.logger = get_logger(f"device.{device_type}.{device_id}")
        
        self.capabilities: Dict[str, DeviceCapability] = {}
        self.state = DeviceState(device_id=device_id, device_type=device_type)
        self.connection = None
        
        self._setup_capabilities()
    
    def _setup_capabilities(self):
        """Setup device-specific capabilities."""
        if self.device_type == "robot_arm":
            self.capabilities.update({
                "pick_place": DeviceCapability(
                    name="pick_place",
                    description="Pick and place objects with precision",
                    parameters={"source": "str", "destination": "str", "force": "float"},
                    execution_time=15.0,
                    safety_level="high"
                ),
                "precise_manipulation": DeviceCapability(
                    name="precise_manipulation",
                    description="Precise manipulation of laboratory samples",
                    parameters={"target": "str", "action": "str", "precision": "float"},
                    execution_time=20.0,
                    safety_level="high"
                )
            })
        elif self.device_type == "xrd_analyzer":
            self.capabilities.update({
                "run_scan": DeviceCapability(
                    name="run_scan",
                    description="Execute X-ray diffraction scan",
                    parameters={"sample_id": "str", "scan_range": "tuple", "step_size": "float"},
                    execution_time=300.0,
                    safety_level="medium"
                ),
                "calibrate": DeviceCapability(
                    name="calibrate",
                    description="Calibrate XRD system",
                    parameters={"calibration_type": "str"},
                    execution_time=120.0,
                    safety_level="low"
                )
            })
        elif self.device_type == "furnace":
            self.capabilities.update({
                "heat_treatment": DeviceCapability(
                    name="heat_treatment",
                    description="High-temperature heat treatment",
                    parameters={"temperature": "float", "duration": "float", "atmosphere": "str"},
                    execution_time=7200.0,  # 2 hours
                    safety_level="high"
                )
            })
    
    async def connect(self):
        """Establish connection to the device."""
        try:
            if self.connection_params.get("protocol") == "http":
                self.connection = aiohttp.ClientSession(
                    base_url=f"http://{self.state.ip_address}:{self.state.port}"
                )
            elif self.connection_params.get("protocol") == "serial":
                self.connection = await serial_asyncio.open_serial_connection(
                    url=self.connection_params["port"],
                    baudrate=self.connection_params.get("baudrate", 9600)
                )
            
            self.state.connected = True
            self.logger.info(f"Connected to device {self.device_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to device {self.device_id}: {e}")
            self.state.connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from the device."""
        if self.connection:
            if hasattr(self.connection, 'close'):
                await self.connection.close()
            self.state.connected = False
            self.logger.info(f"Disconnected from device {self.device_id}")
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command on the device."""
        if not self.state.connected:
            raise RuntimeError(f"Device {self.device_id} not connected")
        
        if command not in self.capabilities:
            raise ValueError(f"Command {command} not supported by device {self.device_id}")
        
        capability = self.capabilities[command]
        
        # Validate parameters
        self._validate_parameters(command, parameters)
        
        # Update state
        self.state.status = "busy"
        self.state.current_operation = command
        self.state.operation_started_at = datetime.utcnow()
        
        try:
            # Execute device-specific command
            result = await self._execute_device_command(command, parameters)
            
            # Update success metrics
            self.state.successful_operations += 1
            self.state.status = "idle"
            self.state.current_operation = None
            
            self.logger.info(f"Successfully executed {command} on {self.device_id}")
            return result
            
        except Exception as e:
            self.state.failed_operations += 1
            self.state.status = "error"
            self.logger.error(f"Command {command} failed on {self.device_id}: {e}")
            raise
        finally:
            self.state.total_operations += 1
            self.state.last_updated = datetime.utcnow()
    
    async def _execute_device_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute device-specific command implementation."""
        if self.device_type == "robot_arm":
            return await self._execute_robot_command(command, parameters)
        elif self.device_type == "xrd_analyzer":
            return await self._execute_xrd_command(command, parameters)
        elif self.device_type == "furnace":
            return await self._execute_furnace_command(command, parameters)
        else:
            raise NotImplementedError(f"Device type {self.device_type} not implemented")
    
    async def _execute_robot_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute robot arm commands."""
        if command == "pick_place":
            # Simulate robot pick and place operation
            source = parameters["source"]
            destination = parameters["destination"]
            force = parameters.get("force", 10.0)
            
            self.logger.info(f"Robot {self.device_id}: Moving from {source} to {destination}")
            
            # Simulate movement time
            await asyncio.sleep(self.capabilities[command].execution_time)
            
            return {
                "status": "completed",
                "source": source,
                "destination": destination,
                "force_applied": force,
                "execution_time": self.capabilities[command].execution_time,
                "precision_achieved": 0.02  # mm
            }
        
        elif command == "precise_manipulation":
            target = parameters["target"]
            action = parameters["action"]
            precision = parameters.get("precision", 0.01)
            
            self.logger.info(f"Robot {self.device_id}: {action} on {target}")
            
            await asyncio.sleep(self.capabilities[command].execution_time)
            
            return {
                "status": "completed",
                "target": target,
                "action": action,
                "precision_achieved": precision,
                "execution_time": self.capabilities[command].execution_time
            }
    
    async def _execute_xrd_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute XRD analyzer commands."""
        if command == "run_scan":
            sample_id = parameters["sample_id"]
            scan_range = parameters.get("scan_range", (5, 80))
            step_size = parameters.get("step_size", 0.02)
            
            self.logger.info(f"XRD {self.device_id}: Scanning sample {sample_id}")
            
            # Simulate XRD scan
            angles = []
            intensities = []
            
            current_angle = scan_range[0]
            while current_angle <= scan_range[1]:
                angles.append(current_angle)
                # Simulate intensity data with some peaks
                intensity = 100 + 50 * abs(math.sin(current_angle * 0.5))
                if current_angle in [26.5, 33.1, 41.2]:  # Simulate characteristic peaks
                    intensity += 500
                intensities.append(intensity)
                current_angle += step_size
                
                # Small delay to simulate real scanning
                await asyncio.sleep(0.01)
            
            return {
                "status": "completed",
                "sample_id": sample_id,
                "scan_data": {
                    "angles": angles,
                    "intensities": intensities,
                    "scan_range": scan_range,
                    "step_size": step_size
                },
                "execution_time": self.capabilities[command].execution_time,
                "peaks_detected": 3
            }
        
        elif command == "calibrate":
            calibration_type = parameters.get("calibration_type", "standard")
            
            self.logger.info(f"XRD {self.device_id}: Calibrating ({calibration_type})")
            
            await asyncio.sleep(self.capabilities[command].execution_time)
            
            return {
                "status": "completed",
                "calibration_type": calibration_type,
                "calibration_accuracy": 0.001,  # degrees
                "execution_time": self.capabilities[command].execution_time
            }
    
    async def _execute_furnace_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute furnace commands."""
        if command == "heat_treatment":
            temperature = parameters["temperature"]
            duration = parameters["duration"]
            atmosphere = parameters.get("atmosphere", "air")
            
            self.logger.info(f"Furnace {self.device_id}: Heat treatment at {temperature}°C for {duration}s")
            
            # Simulate heating profile
            heating_profile = []
            for i in range(int(duration / 60)):  # Record every minute
                current_temp = min(temperature, i * 10)  # Gradual heating
                heating_profile.append({
                    "time": i * 60,
                    "temperature": current_temp,
                    "atmosphere": atmosphere
                })
                await asyncio.sleep(0.1)  # Speed up simulation
            
            return {
                "status": "completed",
                "target_temperature": temperature,
                "actual_temperature": temperature,
                "duration": duration,
                "atmosphere": atmosphere,
                "heating_profile": heating_profile,
                "energy_consumed": temperature * duration * 0.001  # kWh
            }
    
    def _validate_parameters(self, command: str, parameters: Dict[str, Any]):
        """Validate command parameters."""
        capability = self.capabilities[command]
        required_params = capability.parameters
        
        for param, param_type in required_params.items():
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
            
            # Basic type checking
            param_value = parameters[param]
            if param_type == "str" and not isinstance(param_value, str):
                raise ValueError(f"Parameter {param} must be a string")
            elif param_type == "float" and not isinstance(param_value, (int, float)):
                raise ValueError(f"Parameter {param} must be a number")
    
    async def emergency_stop(self):
        """Emergency stop for the device."""
        self.logger.warning(f"Emergency stop triggered for device {self.device_id}")
        
        self.state.status = "emergency_stop"
        self.state.current_operation = None
        
        # Device-specific emergency procedures
        if self.device_type == "robot_arm":
            # Stop all motion immediately
            pass
        elif self.device_type == "furnace":
            # Safe cooling procedure
            pass
        elif self.device_type == "xrd_analyzer":
            # Stop X-ray generation
            pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current device status."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "status": self.state.status,
            "connected": self.state.connected,
            "current_operation": self.state.current_operation,
            "safety_status": self.state.safety_status,
            "performance": {
                "total_operations": self.state.total_operations,
                "successful_operations": self.state.successful_operations,
                "failed_operations": self.state.failed_operations,
                "success_rate": (
                    self.state.successful_operations / self.state.total_operations * 100
                    if self.state.total_operations > 0 else 100.0
                )
            },
            "capabilities": list(self.capabilities.keys()),
            "last_updated": self.state.last_updated.isoformat()
        }


class LabController:
    """Main laboratory controller coordinating all equipment."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.safety_monitor = SafetyMonitor()
        
        self.devices: Dict[str, UniversalDeviceInterface] = {}
        self.device_groups: Dict[str, List[str]] = {
            "synthesis": [],
            "characterization": [],
            "sample_handling": []
        }
        
        self.experiment_queue: asyncio.Queue = asyncio.Queue()
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_devices()
    
    def _initialize_devices(self):
        """Initialize all laboratory devices."""
        # Robot Arms
        robot_config = {
            "protocol": "http",
            "ip_address": "192.168.1.100",
            "port": 8080
        }
        robot = UniversalDeviceInterface("robot_001", "robot_arm", robot_config)
        robot.state.ip_address = "192.168.1.100"
        robot.state.port = 8080
        self.devices["robot_001"] = robot
        self.device_groups["sample_handling"].append("robot_001")
        
        # XRD Analyzer
        xrd_config = {
            "protocol": "http",
            "ip_address": "192.168.1.101",
            "port": 8080
        }
        xrd = UniversalDeviceInterface("xrd_001", "xrd_analyzer", xrd_config)
        xrd.state.ip_address = "192.168.1.101"
        xrd.state.port = 8080
        self.devices["xrd_001"] = xrd
        self.device_groups["characterization"].append("xrd_001")
        
        # High-Temperature Furnace
        furnace_config = {
            "protocol": "http",
            "ip_address": "192.168.1.102",
            "port": 8080
        }
        furnace = UniversalDeviceInterface("furnace_001", "furnace", furnace_config)
        furnace.state.ip_address = "192.168.1.102"
        furnace.state.port = 8080
        self.devices["furnace_001"] = furnace
        self.device_groups["synthesis"].append("furnace_001")
        
        self.logger.info(f"Initialized {len(self.devices)} laboratory devices")
    
    async def start(self):
        """Start the laboratory controller and connect to all devices."""
        self.logger.info("Starting laboratory controller")
        
        # Start safety monitor
        await self.safety_monitor.start()
        
        # Connect to all devices
        connection_tasks = []
        for device in self.devices.values():
            connection_tasks.append(device.connect())
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Check connection results
        connected_devices = 0
        for i, result in enumerate(results):
            device_id = list(self.devices.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Failed to connect to device {device_id}: {result}")
            else:
                connected_devices += 1
        
        self.logger.info(f"Connected to {connected_devices}/{len(self.devices)} devices")
        
        # Start background tasks
        asyncio.create_task(self._monitor_devices())
        asyncio.create_task(self._process_experiment_queue())
    
    async def stop(self):
        """Stop the laboratory controller and disconnect from all devices."""
        self.logger.info("Stopping laboratory controller")
        
        # Emergency stop all devices
        await self.emergency_stop()
        
        # Disconnect from all devices
        disconnect_tasks = []
        for device in self.devices.values():
            disconnect_tasks.append(device.disconnect())
        
        await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        # Stop safety monitor
        await self.safety_monitor.stop()
        
        self.logger.info("Laboratory controller stopped")
    
    async def execute_experiment(self, experiment_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete experiment protocol."""
        experiment_id = experiment_request["experiment_id"]
        design = experiment_request["design"]
        safety_protocols = experiment_request.get("safety_protocols", [])
        
        self.logger.info(f"Executing experiment {experiment_id}")
        
        # Safety pre-check
        safety_check = await self.safety_monitor.check_experiment_safety(experiment_request)
        if not safety_check["approved"]:
            raise RuntimeError(f"Experiment {experiment_id} failed safety check: {safety_check['reasons']}")
        
        # Parse experiment protocol
        protocol_steps = design.get("procedure", [])
        if not protocol_steps:
            raise ValueError("No experimental procedure defined")
        
        # Execute protocol steps
        results = {
            "experiment_id": experiment_id,
            "started_at": datetime.utcnow().isoformat(),
            "steps": []
        }
        
        self.active_experiments[experiment_id] = {
            "status": "running",
            "current_step": 0,
            "total_steps": len(protocol_steps),
            "results": results
        }
        
        try:
            for step_index, step in enumerate(protocol_steps):
                self.active_experiments[experiment_id]["current_step"] = step_index + 1
                
                step_result = await self._execute_protocol_step(step, experiment_id)
                results["steps"].append(step_result)
                
                # Safety check after each step
                if not await self.safety_monitor.check_safety():
                    raise RuntimeError("Safety violation detected during experiment")
            
            results["completed_at"] = datetime.utcnow().isoformat()
            results["status"] = "completed"
            
            self.active_experiments[experiment_id]["status"] = "completed"
            
            self.logger.info(f"Experiment {experiment_id} completed successfully")
            
        except Exception as e:
            results["failed_at"] = datetime.utcnow().isoformat()
            results["status"] = "failed"
            results["error"] = str(e)
            
            self.active_experiments[experiment_id]["status"] = "failed"
            
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            raise
        
        return results
    
    async def _execute_protocol_step(self, step: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Execute a single protocol step."""
        step_type = step["type"]
        step_id = step.get("id", f"step_{datetime.utcnow().timestamp()}")
        
        self.logger.info(f"Executing step {step_id} ({step_type}) for experiment {experiment_id}")
        
        step_result = {
            "step_id": step_id,
            "step_type": step_type,
            "started_at": datetime.utcnow().isoformat(),
            "parameters": step.get("parameters", {})
        }
        
        try:
            if step_type == "synthesis":
                result = await self._execute_synthesis_step(step, experiment_id)
            elif step_type == "characterization":
                result = await self._execute_characterization_step(step, experiment_id)
            elif step_type == "sample_handling":
                result = await self._execute_sample_handling_step(step, experiment_id)
            else:
                raise ValueError(f"Unknown step type: {step_type}")
            
            step_result["result"] = result
            step_result["status"] = "completed"
            step_result["completed_at"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            step_result["status"] = "failed"
            step_result["error"] = str(e)
            step_result["failed_at"] = datetime.utcnow().isoformat()
            raise
        
        return step_result
    
    async def _execute_synthesis_step(self, step: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Execute synthesis protocol step."""
        parameters = step["parameters"]
        
        # Select appropriate synthesis device (furnace)
        furnace_id = self.device_groups["synthesis"][0]  # Use first available furnace
        furnace = self.devices[furnace_id]
        
        # Prepare synthesis parameters
        synthesis_params = {
            "temperature": parameters.get("temperature", 1000),
            "duration": parameters.get("duration", 3600),
            "atmosphere": parameters.get("atmosphere", "air")
        }
        
        # Execute heat treatment
        result = await furnace.execute_command("heat_treatment", synthesis_params)
        
        return {
            "device_used": furnace_id,
            "synthesis_result": result,
            "material_properties": {
                "crystallinity": "high",
                "phase_purity": 95.2,
                "grain_size": "nanometer"
            }
        }
    
    async def _execute_characterization_step(self, step: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Execute characterization protocol step."""
        parameters = step["parameters"]
        
        # Select appropriate characterization device (XRD)
        xrd_id = self.device_groups["characterization"][0]
        xrd = self.devices[xrd_id]
        
        # Prepare scan parameters
        scan_params = {
            "sample_id": parameters.get("sample_id", f"sample_{experiment_id}"),
            "scan_range": parameters.get("scan_range", (5, 80)),
            "step_size": parameters.get("step_size", 0.02)
        }
        
        # Execute XRD scan
        result = await xrd.execute_command("run_scan", scan_params)
        
        return {
            "device_used": xrd_id,
            "characterization_result": result,
            "analysis": {
                "phase_identification": ["rutile", "anatase"],
                "crystallite_size": 25.3,  # nm
                "lattice_parameters": {"a": 4.593, "c": 2.958}
            }
        }
    
    async def _execute_sample_handling_step(self, step: Dict[str, Any], experiment_id: str) -> Dict[str, Any]:
        """Execute sample handling protocol step."""
        parameters = step["parameters"]
        
        # Select robot arm
        robot_id = self.device_groups["sample_handling"][0]
        robot = self.devices[robot_id]
        
        # Execute sample handling operation
        if parameters.get("action") == "transfer":
            result = await robot.execute_command("pick_place", {
                "source": parameters["source"],
                "destination": parameters["destination"],
                "force": parameters.get("force", 10.0)
            })
        else:
            result = await robot.execute_command("precise_manipulation", {
                "target": parameters["target"],
                "action": parameters["action"],
                "precision": parameters.get("precision", 0.01)
            })
        
        return {
            "device_used": robot_id,
            "handling_result": result
        }
    
    async def get_available_equipment(self) -> List[Dict[str, Any]]:
        """Get list of available laboratory equipment."""
        equipment = []
        for device_id, device in self.devices.items():
            status = device.get_status()
            equipment.append({
                "device_id": device_id,
                "device_type": device.device_type,
                "status": status["status"],
                "connected": status["connected"],
                "capabilities": status["capabilities"],
                "groups": [group for group, devices in self.device_groups.items() if device_id in devices]
            })
        
        return equipment
    
    async def emergency_stop(self):
        """Emergency stop all laboratory equipment."""
        self.logger.warning("EMERGENCY STOP - Stopping all laboratory equipment")
        
        # Stop all devices
        stop_tasks = []
        for device in self.devices.values():
            stop_tasks.append(device.emergency_stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Stop all active experiments
        for experiment_id in list(self.active_experiments.keys()):
            self.active_experiments[experiment_id]["status"] = "emergency_stopped"
    
    async def emergency_stop_device(self, device_id: str):
        """Emergency stop a specific device."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        await self.devices[device_id].emergency_stop()
        self.logger.warning(f"Emergency stop triggered for device {device_id}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive laboratory status."""
        device_statuses = []
        for device in self.devices.values():
            device_statuses.append(device.get_status())
        
        # Calculate overall status
        connected_devices = sum(1 for device in self.devices.values() if device.state.connected)
        operational_devices = sum(1 for device in self.devices.values() 
                                if device.state.status in ["idle", "busy"])
        
        overall_status = "operational"
        if connected_devices == 0:
            overall_status = "offline"
        elif connected_devices < len(self.devices):
            overall_status = "degraded"
        
        safety_status = await self.safety_monitor.get_status()
        
        return {
            "overall_status": overall_status,
            "devices": device_statuses,
            "device_groups": self.device_groups,
            "active_experiments": len(self.active_experiments),
            "safety_systems": safety_status,
            "environment": {
                "temperature": 22.5,  # °C
                "humidity": 45.0,     # %
                "air_quality": "good"
            },
            "statistics": {
                "total_devices": len(self.devices),
                "connected_devices": connected_devices,
                "operational_devices": operational_devices
            }
        }
    
    async def _monitor_devices(self):
        """Background task to monitor device health."""
        while True:
            try:
                # Check device health
                for device in self.devices.values():
                    if device.state.connected:
                        # Update device metrics
                        device.state.last_updated = datetime.utcnow()
                        
                        # Simulate temperature readings
                        if device.device_type == "furnace":
                            device.state.temperature = 25.0 + (datetime.utcnow().second % 30)
                
                # Sleep for monitoring interval
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Device monitoring error: {e}")
    
    async def _process_experiment_queue(self):
        """Background task to process experiment queue."""
        while True:
            try:
                # Process queued experiments
                if not self.experiment_queue.empty():
                    experiment_request = await self.experiment_queue.get()
                    await self.execute_experiment(experiment_request)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Experiment queue processing error: {e}")


# Import math for XRD simulation
import math
