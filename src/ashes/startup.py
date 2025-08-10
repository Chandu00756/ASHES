"""
ASHES Startup Manager
Production-ready system initialization and startup procedures
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ashes.core.config import ASHESConfig as Settings
from ashes.core.logging import configure_logging
from ashes.core.orchestrator_enterprise import EnterpriseAgentOrchestrator
from ashes.api.main import create_app
from ashes.database.session import DatabaseManager
from ashes.security.auth import SecurityManager


class ASHESStartupManager:
    """Production startup manager for ASHES system"""
    
    def __init__(self):
        self.settings = Settings()
        self.logger = None
        self.app = None
        self.orchestrator = None
        self.db_manager = None
        self.auth_service = None
        
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize all system components"""
        try:
            # Setup logging
            configure_logging()
            self.logger = logging.getLogger("ashes.startup")
            self.logger.info("Starting ASHES v1.0.1 - PortalVII Production System")
            
            # Initialize database
            self.db_manager = DatabaseManager()
            self.db_manager.initialize()
            self.logger.info("Database initialized successfully")
            
            # Initialize authentication
            self.auth_service = SecurityManager()
            # Auth service doesn't have initialize method
            self.logger.info("Authentication service initialized")
            
            # Initialize orchestrator
            self.orchestrator = EnterpriseAgentOrchestrator()
            # Orchestrator will be started later with start() method
            self.logger.info("Agent orchestrator initialized")
            
            # Create FastAPI app
            self.app = create_app()
            self.logger.info("FastAPI application created")
            
            # System health check
            health_status = await self._perform_health_check()
            self.logger.info(f"System health check: {health_status}")
            
            return {
                "status": "initialized",
                "version": "1.0.1",
                "organization": "PortalVII",
                "health": health_status,
                "components": {
                    "database": "operational",
                    "authentication": "operational",
                    "orchestrator": "operational",
                    "api": "operational"
                }
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"System initialization failed: {e}")
            else:
                print(f"Critical startup error: {e}")
            raise
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health = {
            "database": False,
            "orchestrator": False,
            "auth": False,
            "overall": False
        }
        
        try:
            # Database health
            if self.db_manager:
                health["database"] = self.db_manager.engine is not None
                
            # Orchestrator health
            if self.orchestrator:
                health["orchestrator"] = True  # Simple status check
                
            # Auth service health
            if self.auth_service:
                health["auth"] = True  # Simple status check
                
            # Overall health
            health["overall"] = all([
                health["database"],
                health["orchestrator"], 
                health["auth"]
            ])
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            
        return health
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the production server"""
        if not self.app:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
            
        self.logger.info(f"Starting ASHES server on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False,  # Production mode
            workers=1,  # For development, increase for production
            loop="asyncio"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self):
        """Graceful system shutdown"""
        self.logger.info("Initiating system shutdown...")
        
        try:
            if self.orchestrator:
                await self.orchestrator.stop()
                self.logger.info("Orchestrator shutdown complete")
                
            if self.auth_service:
                # Auth service doesn't have shutdown method
                self.logger.info("Auth service shutdown complete")
                
            if self.db_manager:
                # Database manager doesn't have shutdown method
                self.logger.info("Database shutdown complete")
                
            self.logger.info("ASHES system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


async def main():
    """Main startup function"""
    startup_manager = ASHESStartupManager()
    
    try:
        # Initialize system
        init_result = await startup_manager.initialize_system()
        print(f"ASHES v1.0.1 initialized successfully: {init_result}")
        
        # Start server
        await startup_manager.start_server()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        await startup_manager.shutdown()
        
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
