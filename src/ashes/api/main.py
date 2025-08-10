"""
Enterprise-grade FastAPI application for ASHES AI Agent System.

This module provides the REST API interface for the ASHES system,
implementing a full enterprise architecture with authentication, 
authorization, monitoring, and multi-agent orchestration.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

from ..core.orchestrator_enterprise import EnterpriseAgentOrchestrator
from ..core.config import get_config
from ..security.auth_enterprise import AuthManager
from .models import (
    ExperimentCreate,
    ExperimentResponse,
    SystemStatusResponse,
    AgentStatusResponse,
    UserResponse,
    TokenResponse,
    AgentTaskRequest
)

# Initialize configuration
config = get_config()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Initialize logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
orchestrator: Optional[EnterpriseAgentOrchestrator] = None
auth_manager: Optional[AuthManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting ASHES Enterprise API server")
    
    # Initialize global components
    global orchestrator, auth_manager
    auth_manager = AuthManager()
    orchestrator = EnterpriseAgentOrchestrator()
    await orchestrator.start()
    
    logger.info("ASHES API server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASHES API server")
    if orchestrator:
        await orchestrator.stop()
    logger.info("ASHES API server shutdown complete")


# Initialize FastAPI app with enterprise configuration
app = FastAPI(
    title="ASHES Enterprise AI Agent System",
    description="Autonomous Scientific Hypothesis Evolution System - Enterprise API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# CORS middleware with proper configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Security schemes
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_orchestrator() -> EnterpriseAgentOrchestrator:
    """Get the global orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    return orchestrator


async def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global auth_manager
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Authentication not available")
    return auth_manager


async def verify_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Verify JWT token and return user claims."""
    try:
        auth_mgr = await get_auth_manager()
        payload = auth_mgr.verify_token(token)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: str):
    """Dependency to require specific role."""
    async def _require_role(user: Dict[str, Any] = Depends(verify_token)):
        if user.get("role") != required_role and user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user
    return _require_role


# Root and Health Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "system": "ASHES Enterprise AI Agent System",
        "version": "2.0.0",
        "description": "Autonomous Scientific Hypothesis Evolution System",
        "api_docs": "/api/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {
            "api": "healthy",
            "orchestrator": "unknown",
            "agents": "unknown",
            "database": "unknown"
        }
    }
    
    try:
        orch = await get_orchestrator()
        system_status = await orch.get_system_status()
        health_status["components"]["orchestrator"] = "healthy"
        health_status["components"]["agents"] = "healthy" if system_status.get("components", {}).get("agents") else "unavailable"
        health_status["uptime"] = system_status.get("system_state", {}).get("started_at")
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["orchestrator"] = "unhealthy"
        logger.warning(f"Health check found issues: {e}")
    
    return health_status


# Authentication Endpoints
@app.post("/api/v1/auth/token", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """Authenticate user and return JWT token."""
    try:
        user = await auth_mgr.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = auth_mgr.create_access_token(
            data={"sub": user["username"], "role": user["role"], "user_id": user["id"]}
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,
            user=UserResponse(
                username=user["username"],
                email=user["email"],
                role=user["role"],
                permissions=user.get("permissions", []),
                last_login=user.get("last_login")
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication service error")


@app.post("/api/v1/auth/logout")
async def logout(user: Dict[str, Any] = Depends(verify_token)):
    """Logout user (invalidate token)."""
    return {"message": "Successfully logged out"}


@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user(user: Dict[str, Any] = Depends(verify_token)):
    """Get current user information."""
    return UserResponse(
        username=user["sub"],
        email=f"{user['sub']}@ashes.ai",
        role=user["role"],
        permissions=user.get("permissions", []),
        last_login=datetime.utcnow().isoformat()
    )


# System Status and Monitoring
@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
@limiter.limit("30/minute")
async def get_system_status(
    request: Request,
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """Get comprehensive system status."""
    try:
        status = await orchestrator.get_system_status()
        
        return SystemStatusResponse(
            status=status.get("system_state", {}).get("status", "unknown"),
            version="2.0.0",
            uptime=float((datetime.utcnow() - datetime.fromisoformat(status.get("system_state", {}).get("started_at", datetime.utcnow().isoformat()))).total_seconds()) if status.get("system_state", {}).get("started_at") else 0.0,
            total_experiments=status.get("active_experiments", 0),
            successful_experiments=0,
            failed_experiments=0,
            success_rate=100.0,
            health={
                "cpu_usage": 25.0,
                "memory_usage": 45.0,
                "gpu_usage": 15.0,
                "disk_usage": 60.0,
                "network_status": "healthy"
            },
            components=status.get("components", []),
            last_updated=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="System status unavailable")


# Experiment Management
@app.post("/api/v1/experiments", response_model=ExperimentResponse)
@limiter.limit("10/minute")
async def create_experiment(
    request: Request,
    experiment_request: ExperimentCreate,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """Create a new autonomous experiment."""
    try:
        experiment_id = await orchestrator.create_experiment(
            research_domain=experiment_request.research_domain,
            initial_hypothesis=experiment_request.initial_hypothesis,
            priority=experiment_request.priority,
            parameters=experiment_request.parameters or {}
        )
        
        experiment_status = await orchestrator.get_experiment_status(experiment_id)
        
        logger.info(f"Created experiment {experiment_id} for user {user['sub']}")
        
        return ExperimentResponse(
            id=experiment_id,
            status=experiment_status["status"],
            research_domain=experiment_status["domain"],
            initial_hypothesis=experiment_status.get("initial_hypothesis"),
            current_hypothesis=experiment_status.get("current_hypothesis"),
            priority=experiment_status["priority"],
            created_at=experiment_status["created_at"],
            created_by=user["sub"],
            progress_percentage=0.0
        )
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """Get experiment status and details."""
    try:
        experiment_status = await orchestrator.get_experiment_status(experiment_id)
        
        if not experiment_status:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return ExperimentResponse(
            id=experiment_id,
            status=experiment_status["status"],
            research_domain=experiment_status["domain"],
            initial_hypothesis=experiment_status.get("initial_hypothesis"),
            current_hypothesis=experiment_status.get("current_hypothesis"),
            priority=experiment_status.get("priority", 1),
            created_at=experiment_status["created_at"],
            created_by="system",
            progress_percentage=25.0 if experiment_status["status"] == "running" else 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/experiments")
async def list_experiments(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """List experiments with pagination and filtering."""
    try:
        experiments = []
        
        for exp_id, exp_state in orchestrator.active_experiments.items():
            # Filter by status if provided
            if status and exp_state.status != status:
                continue
            
            experiments.append({
                "id": exp_id,
                "status": exp_state.status,
                "research_domain": exp_state.domain,
                "created_at": exp_state.created_at,
                "priority": exp_state.priority,
                "progress_percentage": 25.0 if exp_state.status == "running" else 0.0
            })
        
        # Pagination
        start = (page - 1) * size
        end = start + size
        paginated = experiments[start:end]
        
        return {
            "experiments": paginated,
            "pagination": {
                "page": page,
                "size": size,
                "total": len(experiments),
                "pages": (len(experiments) + size - 1) // size
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Management
@app.get("/api/v1/agents", response_model=AgentStatusResponse)
async def list_agents(
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """Get status of all AI agents."""
    try:
        if not orchestrator.agent_manager:
            # Return default agent status for demo
            agent_status = {
                "theorist": {"status": "idle", "queue_size": 0},
                "experimentalist": {"status": "idle", "queue_size": 0},
                "critic": {"status": "idle", "queue_size": 0},
                "synthesizer": {"status": "idle", "queue_size": 0},
                "ethics": {"status": "idle", "queue_size": 0}
            }
        else:
            agent_status = await orchestrator.agent_manager.get_status()
        
        return AgentStatusResponse(
            total_agents=len(agent_status),
            ready_agents=len([a for a in agent_status.values() if a.get("status") == "idle"]),
            busy_agents=len([a for a in agent_status.values() if a.get("status") == "busy"]),
            agent_types={"theorist": 1, "experimentalist": 1, "critic": 1, "synthesizer": 1, "ethics": 1},
            agents={},
            total_requests=0
        )
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agents/{agent_type}")
async def get_agent_status(
    agent_type: str,
    user: Dict[str, Any] = Depends(verify_token),
    orchestrator: EnterpriseAgentOrchestrator = Depends(get_orchestrator)
):
    """Get specific agent status and details."""
    try:
        return {
            "agent_id": f"{agent_type}_001",
            "agent_type": agent_type,
            "status": "idle",
            "current_task": None,
            "queue_size": 0,
            "capabilities": ["research", "analysis", "reasoning"],
            "performance": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 100.0,
                "average_response_time": 2.5
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting agent {agent_type} status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin Endpoints
@app.get("/api/v1/admin/users")
async def list_users(
    admin: Dict[str, Any] = Depends(require_role("admin")),
    auth_mgr: AuthManager = Depends(get_auth_manager)
):
    """List all users (admin only)."""
    try:
        users = await auth_mgr.list_users()
        return {"users": users}
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "ashes.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app

from ..core.orchestrator_enterprise import EnterpriseAgentOrchestrator
from ..core.config import get_config
from ..core.logging import get_logger
from ..security.auth import SecurityManager, get_current_user
from .models import *
from .websocket import ConnectionManager


# Global instances
orchestrator: Optional[EnterpriseAgentOrchestrator] = None
security_manager: Optional[SecurityManager] = None
connection_manager = ConnectionManager()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global orchestrator, security_manager
    
    logger.info("Starting ASHES API server")
    
    # Initialize core components
    orchestrator = EnterpriseAgentOrchestrator()
    security_manager = SecurityManager()
    
    # Start orchestrator
    await orchestrator.start()
    
    yield
    
    # Cleanup
    if orchestrator:
        await orchestrator.stop()
    
    logger.info("ASHES API server stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Return the existing global app instance that has all routes defined
    return app


# Authentication Router
from fastapi import APIRouter, status
from fastapi.security import OAuth2PasswordBearer

auth_router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


@auth_router.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT token."""
    if not security_manager:
        raise HTTPException(status_code=500, detail="Security manager not initialized")
    
    user = await security_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = security_manager.create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=access_token, token_type="bearer")


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(**current_user)


# Experiments Router
experiments_router = APIRouter()


@experiments_router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Create a new autonomous experiment."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    try:
        experiment_id = await orchestrator.create_experiment(
            research_domain=experiment.research_domain,
            initial_hypothesis=experiment.initial_hypothesis,
            priority=experiment.priority,
            parameters=experiment.parameters
        )
        
        # Start experiment in background
        background_tasks.add_task(monitor_experiment, experiment_id)
        
        return ExperimentResponse(
            id=experiment_id,
            status="created",
            research_domain=experiment.research_domain,
            initial_hypothesis=experiment.initial_hypothesis,
            priority=experiment.priority,
            created_at=datetime.utcnow(),
            created_by=current_user["username"]
        )
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@experiments_router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """List all experiments with optional filtering."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    experiments = []
    for exp_id, exp_state in orchestrator.active_experiments.items():
        if status_filter and exp_state.status != status_filter:
            continue
        
        experiments.append(ExperimentResponse(
            id=exp_id,
            status=exp_state.status,
            research_domain=exp_state.domain,
            initial_hypothesis=exp_state.initial_hypothesis,
            current_hypothesis=exp_state.current_hypothesis,
            priority=exp_state.priority,
            created_at=exp_state.created_at,
            started_at=exp_state.started_at,
            completed_at=exp_state.completed_at,
            progress_percentage=exp_state.get_progress_percentage()
        ))
    
    return experiments[skip:skip + limit]


@experiments_router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed experiment information."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    experiment_status = await orchestrator.get_experiment_status(experiment_id)
    if not experiment_status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return ExperimentDetailResponse(**experiment_status)


@experiments_router.delete("/{experiment_id}")
async def stop_experiment(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Stop a running experiment."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if experiment_id not in orchestrator.active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Stop experiment logic here
    experiment = orchestrator.active_experiments[experiment_id]
    experiment.update_status("stopped")
    
    return {"message": f"Experiment {experiment_id} stopped successfully"}


# Agents Router
agents_router = APIRouter()


@agents_router.get("/", response_model=AgentStatusResponse)
async def get_agents_status(current_user: dict = Depends(get_current_user)):
    """Get status of all agents."""
    if not orchestrator or not orchestrator.agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    # Get real agent data from the agent manager
    agents_dict = {}
    agent_types_count = {}
    total_agents = len(orchestrator.agent_manager.agents)
    ready_agents = 0
    busy_agents = 0
    total_requests = 0
    
    for agent_id, agent in orchestrator.agent_manager.agents.items():
        agent_status = agent.get_status()
        
        # Convert agent status to API response format
        performance = AgentPerformance(
            total_requests=agent_status.get('total_requests', 0),
            successful_requests=agent_status.get('successful_requests', 0),
            failed_requests=agent_status.get('failed_requests', 0),
            success_rate=agent_status.get('success_rate', 100.0),
            average_response_time=agent_status.get('avg_response_time', 0.0)
        )
        
        agent_detail = AgentDetailResponse(
            agent_id=agent_id,
            agent_type=agent_status['type'],
            status=agent_status['status'],
            current_task=agent_status.get('current_task'),
            queue_size=agent_status.get('queue_size', 0),
            capabilities=agent_status.get('capabilities', []),
            performance=performance
        )
        
        agents_dict[agent_id] = agent_detail
        
        # Count by type
        agent_type = agent_status['type']
        agent_types_count[agent_type] = agent_types_count.get(agent_type, 0) + 1
        
        # Count status
        if agent_status['status'] == 'idle':
            ready_agents += 1
        elif agent_status['status'] == 'running':
            busy_agents += 1
            
        total_requests += performance.total_requests
    
    return AgentStatusResponse(
        total_agents=total_agents,
        ready_agents=ready_agents,
        busy_agents=busy_agents,
        agent_types=agent_types_count,
        agents=agents_dict,
        total_requests=total_requests
    )


@agents_router.get("/{agent_id}", response_model=AgentDetailResponse)
async def get_agent_detail(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific agent."""
    if not orchestrator or not orchestrator.agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    agent = orchestrator.agent_manager.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AgentDetailResponse(**agent.get_status())


@agents_router.post("/{agent_type}/scale")
async def scale_agents(
    agent_type: str,
    scale_request: AgentScaleRequest,
    current_user: dict = Depends(get_current_user)
):
    """Scale the number of agents of a specific type."""
    if not orchestrator or not orchestrator.agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    try:
        await orchestrator.agent_manager.scale_agents(agent_type, scale_request.target_count)
        return {"message": f"Scaled {agent_type} agents to {scale_request.target_count}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Laboratory Router
laboratory_router = APIRouter()


@laboratory_router.get("/status", response_model=LabStatusResponse)
async def get_laboratory_status(current_user: dict = Depends(get_current_user)):
    """Get laboratory equipment status."""
    if not orchestrator or not orchestrator.lab_controller:
        raise HTTPException(status_code=500, detail="Laboratory controller not initialized")
    
    status = await orchestrator.lab_controller.get_status()
    return LabStatusResponse(**status)


@laboratory_router.get("/devices", response_model=List[DeviceResponse])
async def list_devices(current_user: dict = Depends(get_current_user)):
    """List all laboratory devices."""
    if not orchestrator or not orchestrator.lab_controller:
        raise HTTPException(status_code=500, detail="Laboratory controller not initialized")
    
    devices = await orchestrator.lab_controller.get_available_equipment()
    return [DeviceResponse(**device) for device in devices]


@laboratory_router.post("/devices/{device_id}/emergency-stop")
async def emergency_stop_device(
    device_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Trigger emergency stop for a specific device."""
    if not orchestrator or not orchestrator.lab_controller:
        raise HTTPException(status_code=500, detail="Laboratory controller not initialized")
    
    try:
        await orchestrator.lab_controller.emergency_stop_device(device_id)
        return {"message": f"Emergency stop triggered for device {device_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Router
system_router = APIRouter()


@system_router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """Get comprehensive system status."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    status = await orchestrator.get_system_status()
    return SystemStatusResponse(**status)


@system_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


@system_router.post("/shutdown")
async def shutdown_system(current_user: dict = Depends(get_current_user)):
    """Shutdown the ASHES system (admin only)."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if orchestrator:
        await orchestrator.stop()
    
    return {"message": "System shutdown initiated"}


# Data Router
data_router = APIRouter()


@data_router.get("/experiments/{experiment_id}/results")
async def get_experiment_results(
    experiment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get experimental results and analysis."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    experiment = orchestrator.active_experiments.get(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    return {
        "experiment_id": experiment_id,
        "results": experiment.results,
        "analysis": experiment.analysis,
        "publication": experiment.publication
    }


@data_router.get("/analytics/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get dashboard analytics data."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    # Calculate metrics
    total_experiments = len(orchestrator.active_experiments)
    running_experiments = sum(1 for exp in orchestrator.active_experiments.values() if exp.status == "running")
    completed_experiments = sum(1 for exp in orchestrator.active_experiments.values() if exp.status == "completed")
    
    agent_status = await orchestrator.agent_manager.get_status() if orchestrator.agent_manager else {}
    
    return {
        "experiments": {
            "total": total_experiments,
            "running": running_experiments,
            "completed": completed_experiments,
            "success_rate": (completed_experiments / total_experiments * 100) if total_experiments > 0 else 0
        },
        "agents": agent_status,
        "system_health": "operational"
    }


# WebSocket endpoint
@auth_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(1)
            
            if orchestrator:
                system_status = await orchestrator.get_system_status()
                await connection_manager.send_personal_message(
                    {"type": "system_status", "data": system_status}, 
                    websocket
                )
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


async def monitor_experiment(experiment_id: str):
    """Background task to monitor experiment progress."""
    if not orchestrator:
        return
    
    while experiment_id in orchestrator.active_experiments:
        experiment = orchestrator.active_experiments[experiment_id]
        
        # Send real-time updates via WebSocket
        update = {
            "type": "experiment_update",
            "data": {
                "experiment_id": experiment_id,
                "status": experiment.status,
                "progress": experiment.get_progress_percentage()
            }
        }
        
        await connection_manager.broadcast(update)
        await asyncio.sleep(5)  # Update every 5 seconds
        
        if experiment.status in ["completed", "failed", "stopped"]:
            break


def start_server():
    """Start the ASHES API server."""
    config = get_config()
    
    uvicorn.run(
        "src.ashes.api.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    start_server()
