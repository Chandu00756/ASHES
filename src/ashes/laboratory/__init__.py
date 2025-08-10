"""
Laboratory automation and device control system.

Handles communication with robotic systems, analytical instruments,
and laboratory safety monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Device status enumeration."""
    OFFLINE = "offline"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class SafetyLevel(Enum):
    """Safety level enumeration."""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    EMERGENCY = "emergency"


@dataclass
class DeviceCapability:
    """Device capability definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    safety_requirements: List[str]


@dataclass
class SafetyProtocol:
    """Safety protocol definition."""
    name: str
    description: str
    conditions: List[str]
    actions: List[str]
    emergency_stop: bool = False


class BaseDevice(ABC):
    """Base class for laboratory devices."""
    
    def __init__(self, device_id: str, name: str, device_type: str):
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.status = DeviceStatus.OFFLINE
        self.current_operation = None
        self.error_message = None
        self.capabilities: List[DeviceCapability] = []
        self.safety_protocols: List[SafetyProtocol] = []
        self._connection_params = {}
        self._event_callbacks: List[Callable] = []
    
    @abstractmethod
    async def connect(self, **params) -> bool:
        """Connect to the device."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the device."""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current device status."""
        pass
    
    @abstractmethod
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute device operation."""
        pass
    
    async def emergency_stop(self) -> bool:
        """Emergency stop operation."""
        logger.critical(f"Emergency stop triggered for device {self.device_id}")
        self.status = DeviceStatus.ERROR
        self.current_operation = None
        return True
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add event callback."""
        self._event_callbacks.append(callback)
    
    async def _notify_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify event callbacks."""
        for callback in self._event_callbacks:
            try:
                await callback(self.device_id, event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


class RoboticArm(BaseDevice):
    """Robotic arm controller (Mitsubishi RV-7F)."""
    
    def __init__(self, device_id: str):
        super().__init__(device_id, "Mitsubishi RV-7F", "robotic_arm")
        
        # Define capabilities
        self.capabilities = [
            DeviceCapability(
                "move_to_position",
                "Move arm to specified position",
                {"x": float, "y": float, "z": float, "rx": float, "ry": float, "rz": float},
                ["collision_detection", "workspace_limits"]
            ),
            DeviceCapability(
                "pick_object",
                "Pick up object with gripper",
                {"grip_force": float, "approach_speed": float},
                ["force_monitoring", "object_detection"]
            ),
            DeviceCapability(
                "place_object",
                "Place object at target location",
                {"release_height": float, "placement_force": float},
                ["precision_placement", "force_monitoring"]
            )
        ]
        
        # Safety protocols
        self.safety_protocols = [
            SafetyProtocol(
                "collision_avoidance",
                "Prevent collisions with obstacles",
                ["proximity_sensors", "force_feedback"],
                ["reduce_speed", "stop_movement"],
                emergency_stop=True
            )
        ]
    
    async def connect(self, ip_address: str = "192.168.1.100", port: int = 8080) -> bool:
        """Connect to robotic arm."""
        try:
            # Simulate connection to robotic arm
            await asyncio.sleep(0.1)  # Simulate connection delay
            
            self._connection_params = {"ip": ip_address, "port": port}
            self.status = DeviceStatus.IDLE
            
            await self._notify_event("connected", {"ip": ip_address, "port": port})
            logger.info(f"Connected to robotic arm {self.device_id} at {ip_address}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to robotic arm {self.device_id}: {e}")
            self.status = DeviceStatus.ERROR
            self.error_message = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from robotic arm."""
        try:
            self.status = DeviceStatus.OFFLINE
            self.current_operation = None
            
            await self._notify_event("disconnected", {})
            logger.info(f"Disconnected from robotic arm {self.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from robotic arm {self.device_id}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """Get robotic arm status."""
        return {
            "device_id": self.device_id,
            "status": self.status.value,
            "current_operation": self.current_operation,
            "error_message": self.error_message,
            "position": {"x": 100.0, "y": 50.0, "z": 200.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            "gripper_state": "open",
            "force_readings": {"fx": 0.1, "fy": 0.2, "fz": 0.5}
        }
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute robotic arm operation."""
        if self.status != DeviceStatus.IDLE:
            raise ValueError(f"Device {self.device_id} is not available (status: {self.status})")
        
        self.status = DeviceStatus.BUSY
        self.current_operation = operation
        
        try:
            if operation == "move_to_position":
                return await self._move_to_position(parameters)
            elif operation == "pick_object":
                return await self._pick_object(parameters)
            elif operation == "place_object":
                return await self._place_object(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            self.status = DeviceStatus.ERROR
            self.error_message = str(e)
            await self._notify_event("operation_failed", {"operation": operation, "error": str(e)})
            raise
        finally:
            if self.status != DeviceStatus.ERROR:
                self.status = DeviceStatus.IDLE
            self.current_operation = None
    
    async def _move_to_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Move arm to specified position."""
        # Simulate movement time
        await asyncio.sleep(2.0)
        
        position = {
            "x": params.get("x", 0.0),
            "y": params.get("y", 0.0),
            "z": params.get("z", 0.0),
            "rx": params.get("rx", 0.0),
            "ry": params.get("ry", 0.0),
            "rz": params.get("rz", 0.0)
        }
        
        await self._notify_event("position_reached", {"position": position})
        return {"success": True, "final_position": position}
    
    async def _pick_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pick up object."""
        # Simulate pick operation
        await asyncio.sleep(1.5)
        
        result = {
            "success": True,
            "grip_force": params.get("grip_force", 10.0),
            "object_detected": True
        }
        
        await self._notify_event("object_picked", result)
        return result
    
    async def _place_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Place object."""
        # Simulate place operation
        await asyncio.sleep(1.0)
        
        result = {
            "success": True,
            "placement_accuracy": 0.1,  # mm
            "placement_force": params.get("placement_force", 5.0)
        }
        
        await self._notify_event("object_placed", result)
        return result


class AnalyticalInstrument(BaseDevice):
    """Base class for analytical instruments."""
    
    def __init__(self, device_id: str, name: str, instrument_type: str):
        super().__init__(device_id, name, instrument_type)
    
    async def calibrate(self) -> Dict[str, Any]:
        """Calibrate instrument."""
        # Simulate calibration
        await asyncio.sleep(3.0)
        
        result = {
            "success": True,
            "calibration_curve": [1.0, 0.95, 0.02],  # Example coefficients
            "r_squared": 0.998
        }
        
        await self._notify_event("calibrated", result)
        return result


class XRDDiffractometer(AnalyticalInstrument):
    """X-Ray Diffractometer controller."""
    
    def __init__(self, device_id: str):
        super().__init__(device_id, "X-Ray Diffractometer", "xrd")
        
        self.capabilities = [
            DeviceCapability(
                "powder_diffraction",
                "Collect powder diffraction pattern",
                {"scan_range": tuple, "step_size": float, "count_time": float},
                ["sample_alignment", "radiation_safety"]
            ),
            DeviceCapability(
                "single_crystal",
                "Single crystal diffraction analysis",
                {"exposure_time": float, "detector_distance": float},
                ["crystal_mounting", "radiation_safety"]
            )
        ]
    
    async def connect(self, **params) -> bool:
        """Connect to XRD system."""
        try:
            await asyncio.sleep(0.2)
            self.status = DeviceStatus.IDLE
            await self._notify_event("connected", params)
            return True
        except Exception as e:
            self.status = DeviceStatus.ERROR
            self.error_message = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from XRD system."""
        self.status = DeviceStatus.OFFLINE
        await self._notify_event("disconnected", {})
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get XRD status."""
        return {
            "device_id": self.device_id,
            "status": self.status.value,
            "current_operation": self.current_operation,
            "x_ray_source": "Cu K-alpha",
            "detector_temperature": -40.0,
            "vacuum_level": 1e-6
        }
    
    async def execute_operation(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute XRD operation."""
        if self.status != DeviceStatus.IDLE:
            raise ValueError(f"XRD {self.device_id} is not available")
        
        self.status = DeviceStatus.BUSY
        self.current_operation = operation
        
        try:
            if operation == "powder_diffraction":
                return await self._powder_diffraction(parameters)
            elif operation == "single_crystal":
                return await self._single_crystal_analysis(parameters)
            else:
                raise ValueError(f"Unknown XRD operation: {operation}")
        finally:
            self.status = DeviceStatus.IDLE
            self.current_operation = None
    
    async def _powder_diffraction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform powder diffraction measurement."""
        # Simulate measurement time
        scan_time = params.get("count_time", 1.0) * 100  # Simplified calculation
        await asyncio.sleep(min(scan_time, 10.0))  # Cap simulation time
        
        # Generate mock diffraction data
        import random
        angles = list(range(10, 80, 1))
        intensities = [random.randint(50, 1000) + 1000 * (1 / (1 + abs(angle - 26))) for angle in angles]
        
        result = {
            "success": True,
            "scan_range": params.get("scan_range", (10, 80)),
            "data": {"angles": angles, "intensities": intensities},
            "peak_positions": [26.2, 33.1, 47.5],  # Example peaks
            "measurement_time": scan_time
        }
        
        await self._notify_event("measurement_complete", result)
        return result
    
    async def _single_crystal_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform single crystal analysis."""
        exposure_time = params.get("exposure_time", 10.0)
        await asyncio.sleep(min(exposure_time, 5.0))
        
        result = {
            "success": True,
            "unit_cell": {"a": 5.64, "b": 5.64, "c": 5.64, "alpha": 90, "beta": 90, "gamma": 90},
            "space_group": "Fm-3m",
            "resolution": 0.8,
            "completeness": 99.2
        }
        
        await self._notify_event("analysis_complete", result)
        return result


class LaboratoryController:
    """Main laboratory automation controller."""
    
    def __init__(self):
        self.devices: Dict[str, BaseDevice] = {}
        self.safety_manager = SafetyManager()
        self.experiment_queue: List[Dict[str, Any]] = []
        self.active_operations: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize laboratory controller."""
        logger.info("Initializing laboratory controller")
        
        # Initialize devices
        await self._initialize_devices()
        
        # Start safety monitoring
        await self.safety_manager.start_monitoring()
        
        logger.info("Laboratory controller initialized")
    
    async def _initialize_devices(self) -> None:
        """Initialize all laboratory devices."""
        # Initialize robotic arm
        robotic_arm = RoboticArm("robot_01")
        robotic_arm.add_event_callback(self._device_event_handler)
        self.devices["robot_01"] = robotic_arm
        
        # Initialize XRD
        xrd = XRDDiffractometer("xrd_01")
        xrd.add_event_callback(self._device_event_handler)
        self.devices["xrd_01"] = xrd
        
        # Connect to devices
        for device in self.devices.values():
            try:
                await device.connect()
            except Exception as e:
                logger.error(f"Failed to connect to device {device.device_id}: {e}")
    
    async def _device_event_handler(self, device_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Handle device events."""
        logger.info(f"Device event: {device_id} - {event_type} - {data}")
        
        # Log to database (would be implemented with actual database)
        # await self._log_device_event(device_id, event_type, data)
    
    async def get_device_status(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of devices."""
        if device_id:
            if device_id not in self.devices:
                raise ValueError(f"Device {device_id} not found")
            return await self.devices[device_id].get_status()
        
        # Get status of all devices
        status = {}
        for dev_id, device in self.devices.items():
            status[dev_id] = await device.get_status()
        
        return status
    
    async def execute_device_operation(self, device_id: str, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operation on specific device."""
        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not found")
        
        device = self.devices[device_id]
        
        # Check safety protocols
        await self.safety_manager.validate_operation(device_id, operation, parameters)
        
        # Execute operation
        result = await device.execute_operation(operation, parameters)
        
        return result
    
    async def emergency_stop_all(self) -> Dict[str, bool]:
        """Emergency stop all devices."""
        logger.critical("Emergency stop triggered for all devices")
        
        results = {}
        for device_id, device in self.devices.items():
            results[device_id] = await device.emergency_stop()
        
        return results
    
    async def shutdown(self) -> None:
        """Shutdown laboratory controller."""
        logger.info("Shutting down laboratory controller")
        
        # Set shutdown event
        self._shutdown_event.set()
        
        # Cancel active operations
        for task in self.active_operations.values():
            task.cancel()
        
        # Disconnect devices
        for device in self.devices.values():
            await device.disconnect()
        
        # Stop safety monitoring
        await self.safety_manager.stop_monitoring()
        
        logger.info("Laboratory controller shutdown complete")


class SafetyManager:
    """Laboratory safety monitoring and management."""
    
    def __init__(self):
        self.safety_level = SafetyLevel.SAFE
        self.active_protocols: List[SafetyProtocol] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.safety_events: List[Dict[str, Any]] = []
    
    async def start_monitoring(self) -> None:
        """Start safety monitoring."""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitor_safety())
        logger.info("Safety monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop safety monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        logger.info("Safety monitoring stopped")
    
    async def _monitor_safety(self) -> None:
        """Monitor safety conditions."""
        while True:
            try:
                # Simulate safety checks
                await asyncio.sleep(1.0)
                
                # Check environmental conditions
                await self._check_environmental_safety()
                
                # Check device safety
                await self._check_device_safety()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in safety monitoring: {e}")
    
    async def _check_environmental_safety(self) -> None:
        """Check environmental safety conditions."""
        # Simulate environmental monitoring
        # In real implementation, this would check sensors
        pass
    
    async def _check_device_safety(self) -> None:
        """Check device safety conditions."""
        # Simulate device safety checks
        # In real implementation, this would check device status
        pass
    
    async def validate_operation(self, device_id: str, operation: str, parameters: Dict[str, Any]) -> None:
        """Validate operation safety."""
        # Check current safety level
        if self.safety_level == SafetyLevel.EMERGENCY:
            raise ValueError("Emergency safety level - all operations suspended")
        
        if self.safety_level == SafetyLevel.DANGER:
            raise ValueError("Danger safety level - only emergency operations allowed")
        
        # Operation-specific safety checks would go here
        logger.debug(f"Safety validation passed for {device_id}:{operation}")
    
    def report_safety_event(self, level: SafetyLevel, description: str, device_id: Optional[str] = None) -> None:
        """Report safety event."""
        event = {
            "timestamp": datetime.utcnow(),
            "level": level,
            "description": description,
            "device_id": device_id
        }
        
        self.safety_events.append(event)
        
        if level in [SafetyLevel.DANGER, SafetyLevel.EMERGENCY]:
            logger.critical(f"Safety event: {level.value} - {description}")
        else:
            logger.warning(f"Safety event: {level.value} - {description}")
