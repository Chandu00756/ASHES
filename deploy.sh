#!/bin/bash

# ASHES v1.0.1 Production Deployment Script
# PortalVII Enterprise Research Platform

set -e

echo "=========================================="
echo "ASHES v1.0.1 Production Deployment"
echo "Organization: PortalVII"
echo "Contact: chandu@portalvii.com"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Warning: Node.js not found. Frontend will not be available."
    FRONTEND_AVAILABLE=false
else
    FRONTEND_AVAILABLE=true
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

# Install frontend dependencies if Node.js is available
if [ "$FRONTEND_AVAILABLE" = true ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Set default environment variables if not set
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./ashes.db"}
export SECRET_KEY=${SECRET_KEY:-"development-secret-key-change-in-production"}
export ENVIRONMENT=${ENVIRONMENT:-"production"}

echo ""
echo "Environment Configuration:"
echo "- DATABASE_URL: $DATABASE_URL"
echo "- SECRET_KEY: [HIDDEN]"
echo "- ENVIRONMENT: $ENVIRONMENT"
echo ""

# Create startup script
cat > start_ashes.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./ashes.db"}
export SECRET_KEY=${SECRET_KEY:-"development-secret-key-change-in-production"}
export ENVIRONMENT=${ENVIRONMENT:-"production"}

echo "Starting ASHES v1.0.1 Production Server..."
python run_production.py
EOF

chmod +x start_ashes.sh

# Create frontend startup script if available
if [ "$FRONTEND_AVAILABLE" = true ]; then
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm start
EOF
    chmod +x start_frontend.sh
fi

echo "=========================================="
echo "ASHES v1.0.1 Deployment Complete!"
echo "=========================================="
echo ""
echo "To start the system:"
echo "1. Backend API: ./start_ashes.sh"

if [ "$FRONTEND_AVAILABLE" = true ]; then
    echo "2. Frontend Dashboard: ./start_frontend.sh"
fi

echo ""
echo "System URLs:"
echo "- Backend API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Health Check: http://localhost:8000/health"

if [ "$FRONTEND_AVAILABLE" = true ]; then
    echo "- Frontend Dashboard: http://localhost:3000"
fi

echo ""
echo "Production deployment ready!"
echo "Contact: chandu@portalvii.com"
