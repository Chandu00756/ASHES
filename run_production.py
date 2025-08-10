#!/usr/bin/env python3
"""
Production deployment script for ASHES v1.0.1
PortalVII Enterprise Research Platform
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ashes.startup import ASHESStartupManager


async def deploy_production():
    """Deploy ASHES in production mode"""
    print("=" * 60)
    print("ASHES v1.0.1 - PortalVII Production Deployment")
    print("Organization: PortalVII")
    print("Contact: chandu@portalvii.com")
    print("=" * 60)
    
    # Environment checks
    required_env_vars = [
        'DATABASE_URL',
        'SECRET_KEY',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"WARNING: Missing environment variables: {', '.join(missing_vars)}")
        print("The system will use default values where possible.")
        print()
    
    # Initialize and start system
    startup_manager = ASHESStartupManager()
    
    try:
        print("Initializing ASHES production system...")
        init_result = await startup_manager.initialize_system()
        
        print("✓ System initialization complete")
        print(f"✓ Version: {init_result['version']}")
        print(f"✓ Organization: {init_result['organization']}")
        print(f"✓ Health Status: {init_result['health']['overall']}")
        print()
        
        # Component status
        for component, status in init_result['components'].items():
            print(f"  - {component.title()}: {status}")
        print()
        
        print("Starting production server...")
        print("Server will be available at: http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        print("Health Check: http://localhost:8000/health")
        print()
        print("Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Start the server
        await startup_manager.start_server(host="0.0.0.0", port=8000)
        
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Shutdown initiated by user...")
        await startup_manager.shutdown()
        print("ASHES system stopped successfully")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("System startup failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(deploy_production())
