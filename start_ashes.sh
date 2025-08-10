#!/bin/bash
source venv/bin/activate
export DATABASE_URL=${DATABASE_URL:-"sqlite:///./ashes.db"}
export SECRET_KEY=${SECRET_KEY:-"development-secret-key-change-in-production"}
export ENVIRONMENT=${ENVIRONMENT:-"production"}

echo "Starting ASHES v1.0.1 Production Server..."
python run_production.py
