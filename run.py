"""
Production-ready FastAPI application runner.

Configures and runs the ASHES backend API with proper production settings,
middleware, monitoring, and error handling.
"""

import asyncio
import logging
import sys
import signal
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ashes.core.config import get_settings
from src.ashes.core.logging import setup_logging
from src.ashes.core.orchestrator import SystemOrchestrator
from src.ashes.api.main import app as api_app
from src.ashes.monitoring import MonitoringManager, SystemMetricsCollector


logger = logging.getLogger(__name__)


class ASHESApplication:
    """Main ASHES application manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.orchestrator: SystemOrchestrator = None
        self.monitoring_manager: MonitoringManager = None
        self.app: FastAPI = None
        
    async def initialize(self) -> None:
        """Initialize all ASHES components."""
        logger.info("Initializing ASHES application...")
        
        # Initialize system orchestrator
        self.orchestrator = SystemOrchestrator()
        await self.orchestrator.initialize()
        
        # Initialize monitoring
        self.monitoring_manager = MonitoringManager()
        self.monitoring_manager.add_collector(SystemMetricsCollector())
        
        # Add more collectors if components are available
        if hasattr(self.orchestrator, 'database_manager'):
            from src.ashes.monitoring import DatabaseMetricsCollector
            self.monitoring_manager.add_collector(
                DatabaseMetricsCollector(self.orchestrator.database_manager)
            )
        
        if hasattr(self.orchestrator, 'agent_manager'):
            from src.ashes.monitoring import AgentMetricsCollector
            self.monitoring_manager.add_collector(
                AgentMetricsCollector(self.orchestrator.agent_manager)
            )
        
        if hasattr(self.orchestrator, 'laboratory_controller'):
            from src.ashes.monitoring import LaboratoryMetricsCollector
            self.monitoring_manager.add_collector(
                LaboratoryMetricsCollector(self.orchestrator.laboratory_controller)
            )
        
        # Set data manager for metrics storage
        if hasattr(self.orchestrator, 'data_manager'):
            self.monitoring_manager.set_data_manager(self.orchestrator.data_manager)
        
        # Start monitoring
        await self.monitoring_manager.start_monitoring()
        
        logger.info("ASHES application initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown all ASHES components."""
        logger.info("Shutting down ASHES application...")
        
        if self.monitoring_manager:
            await self.monitoring_manager.stop_monitoring()
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        logger.info("ASHES application shutdown complete")


# Global application instance
ashes_app = ASHESApplication()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup
    await ashes_app.initialize()
    
    # Make orchestrator available to API routes
    app.state.orchestrator = ashes_app.orchestrator
    app.state.monitoring = ashes_app.monitoring_manager
    
    yield
    
    # Shutdown
    await ashes_app.shutdown()


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Create main application with lifespan
    app = FastAPI(
        title="ASHES - Autonomous Scientific Hypothesis Evolution System",
        description="A fully autonomous scientific research platform combining multi-agent AI orchestration with advanced laboratory automation",
        version="1.0.0",
        docs_url="/docs" if not settings.production else None,
        redoc_url="/redoc" if not settings.production else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Mount API routes
    app.mount("/api/v1", api_app)
    
    # Mount Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Global exception handler caught: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error_id": str(id(exc))
            }
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            if hasattr(app.state, 'monitoring'):
                status = await app.state.monitoring.get_system_status()
                return status
            else:
                return {"status": "starting", "message": "System is starting up"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": str(e)}
            )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "ASHES - Autonomous Scientific Hypothesis Evolution System",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }
    
    return app


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # The actual shutdown will be handled by the lifespan context manager
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for running ASHES application."""
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Get settings
    settings = get_settings()
    
    # Create application
    app = create_application()
    
    # Configure uvicorn
    uvicorn_config = {
        "app": app,
        "host": settings.host,
        "port": settings.port,
        "log_config": None,  # We handle logging ourselves
        "access_log": False,  # Disable uvicorn access logs
    }
    
    # Production-specific settings
    if settings.production:
        uvicorn_config.update({
            "workers": settings.workers,
            "loop": "uvloop",
            "http": "httptools",
        })
    else:
        uvicorn_config.update({
            "reload": True,
            "reload_dirs": ["src"],
        })
    
    logger.info(f"Starting ASHES application on {settings.host}:{settings.port}")
    logger.info(f"Production mode: {settings.production}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Run the application
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
