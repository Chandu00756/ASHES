"""
Structured logging configuration for ASHES system.
"""

import logging
import structlog
import sys
from typing import Dict, Any
from .config import get_config


def configure_logging():
    """Configure structured logging for the entire ASHES system."""
    config = get_config()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if config.monitoring.structured_logging 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.monitoring.log_level.upper())
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class ASHESLogger:
    """Enhanced logger for ASHES system with experiment tracking."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        
    def log_experiment_event(
        self, 
        event_type: str, 
        experiment_id: str, 
        **kwargs
    ):
        """Log an experiment-related event."""
        self.logger.info(
            event_type,
            experiment_id=experiment_id,
            **kwargs
        )
    
    def log_agent_action(
        self,
        agent_type: str,
        action: str,
        experiment_id: str = None,
        **kwargs
    ):
        """Log an agent action."""
        self.logger.info(
            "agent_action",
            agent_type=agent_type,
            action=action,
            experiment_id=experiment_id,
            **kwargs
        )
    
    def log_laboratory_event(
        self,
        device_name: str,
        event_type: str,
        experiment_id: str = None,
        **kwargs
    ):
        """Log a laboratory equipment event."""
        self.logger.info(
            "laboratory_event",
            device_name=device_name,
            event_type=event_type,
            experiment_id=experiment_id,
            **kwargs
        )
    
    def log_safety_event(
        self,
        safety_level: str,
        event_description: str,
        experiment_id: str = None,
        **kwargs
    ):
        """Log a safety-related event."""
        self.logger.warning(
            "safety_event",
            safety_level=safety_level,
            description=event_description,
            experiment_id=experiment_id,
            **kwargs
        )
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)


# Initialize logging on module import
configure_logging()
