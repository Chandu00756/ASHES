"""
Monitoring and metrics collection system.

Provides comprehensive system monitoring, performance metrics,
and health checks for the ASHES platform.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemMetric:
    """System metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]
    labels: Dict[str, str]


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # healthy, warning, critical
    message: str
    timestamp: datetime
    response_time: float
    details: Dict[str, Any]


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    level: str  # info, warning, critical
    message: str
    source: str
    timestamp: datetime
    resolved: bool
    metadata: Dict[str, Any]


class MetricsCollector(ABC):
    """Abstract metrics collector interface."""
    
    @abstractmethod
    async def collect_metrics(self) -> List[SystemMetric]:
        """Collect metrics from source."""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthCheck:
        """Perform health check."""
        pass


class SystemMetricsCollector(MetricsCollector):
    """System-level metrics collector."""
    
    def __init__(self):
        self.name = "system"
    
    async def collect_metrics(self) -> List[SystemMetric]:
        """Collect system metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(SystemMetric(
            name="cpu_usage_percent",
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "cpu"}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(SystemMetric(
            name="memory_usage_percent",
            value=memory.percent,
            unit="percent",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "memory"}
        ))
        
        metrics.append(SystemMetric(
            name="memory_used_bytes",
            value=memory.used,
            unit="bytes",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "memory"}
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(SystemMetric(
            name="disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            unit="percent",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "disk", "mount": "/"}
        ))
        
        # Network metrics
        network = psutil.net_io_counters()
        metrics.append(SystemMetric(
            name="network_bytes_sent",
            value=network.bytes_sent,
            unit="bytes",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "network", "direction": "sent"}
        ))
        
        metrics.append(SystemMetric(
            name="network_bytes_recv",
            value=network.bytes_recv,
            unit="bytes",
            timestamp=timestamp,
            tags={"collector": "system"},
            labels={"type": "network", "direction": "recv"}
        ))
        
        return metrics
    
    async def health_check(self) -> HealthCheck:
        """Perform system health check."""
        start_time = time.time()
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            # Determine status
            status = "healthy"
            message = "System operating normally"
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            }
            
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = "critical"
                message = "High resource usage detected"
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 80:
                status = "warning"
                message = "Elevated resource usage"
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                name="system",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheck(
                name="system",
                status="critical",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )


class DatabaseMetricsCollector(MetricsCollector):
    """Database metrics collector."""
    
    def __init__(self, db_manager):
        self.name = "database"
        self.db_manager = db_manager
    
    async def collect_metrics(self) -> List[SystemMetric]:
        """Collect database metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Connection pool metrics
            if hasattr(self.db_manager, 'engine') and self.db_manager.engine:
                pool = self.db_manager.engine.pool
                
                metrics.append(SystemMetric(
                    name="db_connections_active",
                    value=pool.checkedout(),
                    unit="count",
                    timestamp=timestamp,
                    tags={"collector": "database"},
                    labels={"type": "connections"}
                ))
                
                metrics.append(SystemMetric(
                    name="db_connections_pool_size",
                    value=pool.size(),
                    unit="count",
                    timestamp=timestamp,
                    tags={"collector": "database"},
                    labels={"type": "connections"}
                ))
            
            # Query performance metrics (mock)
            metrics.append(SystemMetric(
                name="db_query_duration_avg",
                value=0.025,  # Mock average query time
                unit="seconds",
                timestamp=timestamp,
                tags={"collector": "database"},
                labels={"type": "performance"}
            ))
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
        
        return metrics
    
    async def health_check(self) -> HealthCheck:
        """Perform database health check."""
        start_time = time.time()
        
        try:
            # Test database connection
            if self.db_manager and hasattr(self.db_manager, 'get_session'):
                with self.db_manager.session_scope() as session:
                    # Simple query to test connection
                    session.execute("SELECT 1")
                
                response_time = time.time() - start_time
                
                return HealthCheck(
                    name="database",
                    status="healthy",
                    message="Database connection successful",
                    timestamp=datetime.utcnow(),
                    response_time=response_time,
                    details={"connection_test": "passed"}
                )
            else:
                return HealthCheck(
                    name="database",
                    status="critical",
                    message="Database manager not initialized",
                    timestamp=datetime.utcnow(),
                    response_time=time.time() - start_time,
                    details={"error": "db_manager_not_initialized"}
                )
                
        except Exception as e:
            return HealthCheck(
                name="database",
                status="critical",
                message=f"Database health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )


class AgentMetricsCollector(MetricsCollector):
    """Agent performance metrics collector."""
    
    def __init__(self, agent_manager):
        self.name = "agents"
        self.agent_manager = agent_manager
    
    async def collect_metrics(self) -> List[SystemMetric]:
        """Collect agent metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Get agent status
            agent_status = await self.agent_manager.get_all_agent_status()
            
            # Count agents by status
            status_counts = {}
            total_agents = 0
            
            for agent_id, status in agent_status.items():
                agent_state = status.get('status', 'unknown')
                status_counts[agent_state] = status_counts.get(agent_state, 0) + 1
                total_agents += 1
            
            # Create metrics for each status
            for status, count in status_counts.items():
                metrics.append(SystemMetric(
                    name="agents_by_status",
                    value=count,
                    unit="count",
                    timestamp=timestamp,
                    tags={"collector": "agents", "status": status},
                    labels={"type": "count"}
                ))
            
            # Total agents metric
            metrics.append(SystemMetric(
                name="agents_total",
                value=total_agents,
                unit="count",
                timestamp=timestamp,
                tags={"collector": "agents"},
                labels={"type": "total"}
            ))
            
            # Performance metrics (mock)
            metrics.append(SystemMetric(
                name="agent_response_time_avg",
                value=1.25,  # Mock average response time
                unit="seconds",
                timestamp=timestamp,
                tags={"collector": "agents"},
                labels={"type": "performance"}
            ))
            
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
        
        return metrics
    
    async def health_check(self) -> HealthCheck:
        """Perform agent health check."""
        start_time = time.time()
        
        try:
            agent_status = await self.agent_manager.get_all_agent_status()
            
            healthy_agents = 0
            total_agents = len(agent_status)
            
            for agent_id, status in agent_status.items():
                if status.get('status') in ['idle', 'running']:
                    healthy_agents += 1
            
            response_time = time.time() - start_time
            
            if healthy_agents == total_agents:
                status = "healthy"
                message = f"All {total_agents} agents are healthy"
            elif healthy_agents >= total_agents * 0.8:
                status = "warning"
                message = f"{healthy_agents}/{total_agents} agents are healthy"
            else:
                status = "critical"
                message = f"Only {healthy_agents}/{total_agents} agents are healthy"
            
            return HealthCheck(
                name="agents",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={
                    "total_agents": total_agents,
                    "healthy_agents": healthy_agents,
                    "health_ratio": healthy_agents / total_agents if total_agents > 0 else 0
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="agents",
                status="critical",
                message=f"Agent health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )


class LaboratoryMetricsCollector(MetricsCollector):
    """Laboratory equipment metrics collector."""
    
    def __init__(self, lab_controller):
        self.name = "laboratory"
        self.lab_controller = lab_controller
    
    async def collect_metrics(self) -> List[SystemMetric]:
        """Collect laboratory metrics."""
        metrics = []
        timestamp = datetime.utcnow()
        
        try:
            # Get device status
            device_status = await self.lab_controller.get_device_status()
            
            # Count devices by status
            status_counts = {}
            total_devices = 0
            
            for device_id, status in device_status.items():
                device_state = status.get('status', 'unknown')
                status_counts[device_state] = status_counts.get(device_state, 0) + 1
                total_devices += 1
            
            # Create metrics for each status
            for status, count in status_counts.items():
                metrics.append(SystemMetric(
                    name="devices_by_status",
                    value=count,
                    unit="count",
                    timestamp=timestamp,
                    tags={"collector": "laboratory", "status": status},
                    labels={"type": "count"}
                ))
            
            # Total devices metric
            metrics.append(SystemMetric(
                name="devices_total",
                value=total_devices,
                unit="count",
                timestamp=timestamp,
                tags={"collector": "laboratory"},
                labels={"type": "total"}
            ))
            
            # Equipment utilization (mock)
            metrics.append(SystemMetric(
                name="equipment_utilization",
                value=0.65,  # Mock utilization rate
                unit="percent",
                timestamp=timestamp,
                tags={"collector": "laboratory"},
                labels={"type": "utilization"}
            ))
            
        except Exception as e:
            logger.error(f"Error collecting laboratory metrics: {e}")
        
        return metrics
    
    async def health_check(self) -> HealthCheck:
        """Perform laboratory health check."""
        start_time = time.time()
        
        try:
            device_status = await self.lab_controller.get_device_status()
            
            connected_devices = 0
            total_devices = len(device_status)
            
            for device_id, status in device_status.items():
                if status.get('status') in ['idle', 'busy']:
                    connected_devices += 1
            
            response_time = time.time() - start_time
            
            if connected_devices == total_devices:
                status = "healthy"
                message = f"All {total_devices} devices are connected"
            elif connected_devices >= total_devices * 0.8:
                status = "warning"
                message = f"{connected_devices}/{total_devices} devices are connected"
            else:
                status = "critical"
                message = f"Only {connected_devices}/{total_devices} devices are connected"
            
            return HealthCheck(
                name="laboratory",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time=response_time,
                details={
                    "total_devices": total_devices,
                    "connected_devices": connected_devices,
                    "connection_ratio": connected_devices / total_devices if total_devices > 0 else 0
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="laboratory",
                status="critical",
                message=f"Laboratory health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time=time.time() - start_time,
                details={"error": str(e)}
            )


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict[str, Any]] = []
        self.subscribers: List[Callable] = []
    
    def add_alert_rule(self, metric_name: str, condition: str, threshold: float, 
                      level: str = "warning", message: str = None) -> None:
        """Add alert rule."""
        rule = {
            "metric_name": metric_name,
            "condition": condition,  # "gt", "lt", "eq"
            "threshold": threshold,
            "level": level,
            "message": message or f"{metric_name} {condition} {threshold}"
        }
        self.alert_rules.append(rule)
    
    def add_subscriber(self, callback: Callable[[Alert], None]) -> None:
        """Add alert subscriber."""
        self.subscribers.append(callback)
    
    async def check_metrics(self, metrics: List[SystemMetric]) -> List[Alert]:
        """Check metrics against alert rules."""
        new_alerts = []
        
        for metric in metrics:
            for rule in self.alert_rules:
                if metric.name == rule["metric_name"]:
                    if self._evaluate_condition(metric.value, rule["condition"], rule["threshold"]):
                        alert = Alert(
                            id=f"alert_{datetime.utcnow().timestamp()}",
                            level=rule["level"],
                            message=rule["message"],
                            source=f"metric:{metric.name}",
                            timestamp=datetime.utcnow(),
                            resolved=False,
                            metadata={
                                "metric_value": metric.value,
                                "threshold": rule["threshold"],
                                "condition": rule["condition"],
                                "metric_tags": metric.tags
                            }
                        )
                        
                        new_alerts.append(alert)
                        self.alerts.append(alert)
                        
                        # Notify subscribers
                        for subscriber in self.subscribers:
                            try:
                                await subscriber(alert)
                            except Exception as e:
                                logger.error(f"Error notifying alert subscriber: {e}")
        
        return new_alerts
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        else:
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False


class MonitoringManager:
    """Main monitoring and metrics management system."""
    
    def __init__(self):
        self.collectors: List[MetricsCollector] = []
        self.alert_manager = AlertManager()
        self.data_manager = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.collection_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self._shutdown_event = asyncio.Event()
    
    def add_collector(self, collector: MetricsCollector) -> None:
        """Add metrics collector."""
        self.collectors.append(collector)
        logger.info(f"Added metrics collector: {collector.name}")
    
    def set_data_manager(self, data_manager) -> None:
        """Set data manager for metrics storage."""
        self.data_manager = data_manager
    
    async def start_monitoring(self) -> None:
        """Start monitoring tasks."""
        if self.monitoring_task:
            return
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring tasks."""
        self._shutdown_event.set()
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring stopped")
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        # CPU usage alerts
        self.alert_manager.add_alert_rule(
            "cpu_usage_percent", "gt", 80.0, "warning", "High CPU usage detected"
        )
        self.alert_manager.add_alert_rule(
            "cpu_usage_percent", "gt", 95.0, "critical", "Critical CPU usage detected"
        )
        
        # Memory usage alerts
        self.alert_manager.add_alert_rule(
            "memory_usage_percent", "gt", 80.0, "warning", "High memory usage detected"
        )
        self.alert_manager.add_alert_rule(
            "memory_usage_percent", "gt", 95.0, "critical", "Critical memory usage detected"
        )
        
        # Disk usage alerts
        self.alert_manager.add_alert_rule(
            "disk_usage_percent", "gt", 85.0, "warning", "High disk usage detected"
        )
        self.alert_manager.add_alert_rule(
            "disk_usage_percent", "gt", 95.0, "critical", "Critical disk usage detected"
        )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics from all collectors
                all_metrics = []
                for collector in self.collectors:
                    try:
                        metrics = await collector.collect_metrics()
                        all_metrics.extend(metrics)
                    except Exception as e:
                        logger.error(f"Error collecting metrics from {collector.name}: {e}")
                
                # Store metrics in time series database
                if self.data_manager:
                    for metric in all_metrics:
                        await self.data_manager.record_metric(
                            metric=metric.name,
                            value=metric.value,
                            tags={**metric.tags, **metric.labels},
                            fields={"unit": metric.unit}
                        )
                
                # Check for alerts
                alerts = await self.alert_manager.check_metrics(all_metrics)
                if alerts:
                    logger.info(f"Generated {len(alerts)} new alerts")
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self._shutdown_event.is_set():
            try:
                # Perform health checks
                health_results = []
                for collector in self.collectors:
                    try:
                        health = await collector.health_check()
                        health_results.append(health)
                    except Exception as e:
                        logger.error(f"Error performing health check for {collector.name}: {e}")
                
                # Log health check results
                for health in health_results:
                    if health.status == "critical":
                        logger.error(f"Health check CRITICAL: {health.name} - {health.message}")
                    elif health.status == "warning":
                        logger.warning(f"Health check WARNING: {health.name} - {health.message}")
                    else:
                        logger.debug(f"Health check OK: {health.name} - {health.message}")
                
                # Wait for next health check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        # Perform immediate health checks
        health_results = []
        for collector in self.collectors:
            try:
                health = await collector.health_check()
                health_results.append(asdict(health))
            except Exception as e:
                logger.error(f"Error getting health status from {collector.name}: {e}")
        
        # Get active alerts
        active_alerts = [asdict(alert) for alert in self.alert_manager.get_active_alerts()]
        
        # Determine overall status
        overall_status = "healthy"
        for health in health_results:
            if health["status"] == "critical":
                overall_status = "critical"
                break
            elif health["status"] == "warning" and overall_status == "healthy":
                overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": health_results,
            "active_alerts": active_alerts,
            "metrics_collectors": len(self.collectors)
        }
