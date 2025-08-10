# ASHES v1.0.1 - PortalVII Research Platform Prototype

> ⚠️ **PROTOTYPE VERSION** ⚠️  
> **This is a development prototype, NOT production-ready software**  
> **For research, development, and evaluation purposes only**  
> **Not recommended for enterprise production deployment**

## Overview

ASHES (Autonomous Scientific Hypothesis Evaluation System) is a prototype research platform developed by PortalVII. This system demonstrates a multi-agent AI framework for autonomous scientific research, hypothesis generation, and experimental validation.

**Current Status: v1.0.1 Prototype**
- ✅ Core functionality implemented
- ✅ Basic agent orchestration working  
- ✅ API endpoints functional
- ⚠️ Limited testing and validation
- ⚠️ Development dependencies and configurations
- ⚠️ Prototype-level security implementations

## Organization

- **Organization**: PortalVII
- **Contact**: <chandu@portalvii.com>
- **Version**: 1.0.1 (Prototype)
- **Status**: Intial Prototype

## Features

### Enterprise AI Agent System

- **Theorist Agent**: Advanced hypothesis generation and theoretical framework development
- **Experimentalist Agent**: Experiment design, execution, and data collection
- **Critic Agent**: Rigorous peer review and validation processes
- **Synthesizer Agent**: Knowledge integration and pattern recognition
- **Ethics Agent**: Ethical compliance and safety validation
- **Manager Agent**: Project coordination and resource management

### Production Backend

- **FastAPI**: High-performance API with enterprise-grade security
- **PostgreSQL + pgvector**: Vector database for semantic search and knowledge storage
- **Redis**: High-speed caching and session management
- **OAuth2.0 + JWT**: Enterprise authentication and authorization
- **Rate Limiting**: API protection with slowapi
- **Monitoring**: Prometheus metrics and health checks

### Modern Frontend

- **React 19**: Latest React with concurrent features
- **Material-UI**: Professional enterprise UI components
- **Real-time Dashboard**: Live system monitoring and analytics
- **Responsive Design**: Cross-platform compatibility

### Advanced Capabilities

- **Vector Search**: Semantic similarity and knowledge retrieval
- **Knowledge Graphs**: Dynamic relationship mapping
- **Real-time Analytics**: Live experiment tracking and metrics
- **WebSocket Communication**: Real-time agent coordination
- **Enterprise Security**: RBAC, audit logging, and compliance

## Production Deployment

### Quick Start

```bash
# Clone and setup
git clone <repository>
cd ASHES

# Install dependencies
pip install -e .

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost/ashes"
export SECRET_KEY="your-secret-key"
export OPENAI_API_KEY="your-openai-key"

# Run production server
python run_production.py
or

./start_ashes.sh
```

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql://username:password@host:port/database
SECRET_KEY=your-jwt-secret-key
OPENAI_API_KEY=your-openai-api-key

# Optional
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=your-anthropic-key
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Production Services

#### Backend API

- **URL**: <http://localhost:8000>
- **Documentation**: <http://localhost:8000/docs>
- **Health Check**: <http://localhost:8000/health>
- **Metrics**: <http://localhost:8000/metrics>

#### Frontend Dashboard

- **URL**: <http://localhost:3000>
- **Features**: Real-time monitoring, agent management, experiment tracking
- **Authentication**: Enterprise SSO integration

## System Architecture

### Agent Communication

```
EnterpriseAgentOrchestrator
├── Theorist Agent (Hypothesis Generation)
├── Experimentalist Agent (Experiment Design)
├── Critic Agent (Validation & Review)
├── Synthesizer Agent (Knowledge Integration)
├── Ethics Agent (Safety & Compliance)
└── Manager Agent (Coordination)
```

### Data Flow

```
Frontend Dashboard → FastAPI Backend → Agent Orchestrator → Specialized Agents
                                   ↓
                              PostgreSQL + Vector DB
                                   ↓
                              Knowledge Graph Engine
```

### Security Architecture

```
Client → Rate Limiter → Authentication → Authorization → API Endpoints
                                      ↓
                               Enterprise Audit Logging
```

## API Endpoints

### Authentication

- `POST /auth/login` - User authentication
- `POST /auth/refresh` - Token refresh
- `GET /auth/profile` - User profile

### System Management

- `GET /health` - System health check
- `GET /metrics` - System metrics
- `GET /system/status` - Detailed system status

### Agent Operations

- `GET /agents` - List all agents
- `POST /agents/{agent_id}/tasks` - Assign tasks
- `GET /agents/{agent_id}/status` - Agent status

### Research Operations

- `POST /research/experiments` - Create experiments
- `GET /research/experiments` - List experiments
- `POST /research/hypotheses` - Submit hypotheses
- `GET /research/results` - Retrieve results

## Development

### Project Structure

```
ASHES/
├── src/ashes/
│   ├── agents/          # AI Agent implementations
│   ├── api/             # FastAPI application
│   ├── core/            # Core system components
│   ├── db/              # Database models and operations
│   └── services/        # Business logic services
├── frontend/            # React dashboard
├── tests/               # Test suites
├── docs/                # Documentation
└── scripts/             # Deployment scripts
```

### Key Components

#### Agent System (`src/ashes/agents/`)

- `base.py` - Abstract agent foundation
- `theorist.py` - Hypothesis generation agent
- `experimentalist.py` - Experiment design agent
- `critic.py` - Validation and review agent
- `synthesizer.py` - Knowledge integration agent
- `ethics.py` - Ethics and safety agent
- `manager.py` - Project management agent

#### API Layer (`src/ashes/api/`)

- `main.py` - FastAPI application setup
- `auth.py` - Authentication endpoints
- `agents.py` - Agent management endpoints
- `research.py` - Research operation endpoints

#### Core System (`src/ashes/core/`)

- `orchestrator.py` - Enterprise agent orchestration
- `config.py` - System configuration
- `logging.py` - Enterprise logging setup
- `state.py` - System state management

## Production Features

### Enterprise Security

- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- API rate limiting and DDoS protection
- Comprehensive audit logging
- Data encryption at rest and in transit

### Monitoring & Analytics

- Real-time system metrics via Prometheus
- Performance monitoring and alerting
- Agent activity tracking
- Experiment success rate analytics
- Resource utilization monitoring

### Scalability

- Horizontal scaling support
- Load balancing ready
- Database connection pooling
- Redis caching layer
- Asynchronous processing

### Reliability

- Graceful error handling
- Automatic retry mechanisms
- Circuit breaker patterns
- Health checks and auto-recovery
- Comprehensive logging

## Support

### PortalVII Contact

- **Email**: <chandu@portalvii.com>
- **Organization**: PortalVII
- **System Version**: 1.0.1

### Documentation

- API Documentation: Available at `/docs` endpoint
- Agent Documentation: See `docs/agents/` directory
- Deployment Guide: See `docs/deployment/` directory

### Troubleshooting

1. Check system health: `GET /health`

---

## ⚠️ IMPORTANT PROTOTYPE DISCLAIMERS ⚠️

### **THIS IS A PROTOTYPE VERSION - NOT PRODUCTION READY**

**Development Status:**
- This is ASHES v1.0.1 **PROTOTYPE** 
- Built for research, development, and demonstration purposes
- **NOT intended for production enterprise deployment**
- Contains development configurations and dependencies

**Known Limitations:**
- Limited security hardening
- Development-grade error handling
- Prototype-level testing coverage
- Some agent implementations incomplete (Theorist, Critic, Ethics agents)
- Default credentials in use (change for any real deployment)

**Use Cases:**
- ✅ Research and development
- ✅ Proof of concept demonstrations
- ✅ Academic exploration
- ✅ Learning and experimentation
- ❌ Production enterprise systems
- ❌ Mission-critical applications
- ❌ Unsupervised autonomous operation

**Before Production Use:**
- Comprehensive security audit required
- Production-grade authentication system
- Extensive testing and validation
- Security hardening and configuration
- Complete agent implementations
- Performance optimization
- Error handling improvements

**Contact PortalVII for enterprise-ready versions**

---

## License & Legal

**Prototype Software - Development Use Only**  
© 2025 PortalVII - Chandu Chitikam  
This prototype is provided for research and development purposes.
2. Review logs: Check application logs for detailed error information
3. Verify environment: Ensure all required environment variables are set
4. Database connectivity: Verify PostgreSQL and Redis connections

## License

Enterprise software developed by PortalVII. All rights reserved.

---

**ASHES v1.0.1** - Production-ready autonomous research platform by PortalVII
Contact: <chandu@portalvii.com>
