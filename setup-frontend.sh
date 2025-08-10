#!/bin/bash

# ASHES Frontend Build and Development Script
# Creates a complete React TypeScript frontend with black/white aesthetic

set -e

echo "🚀 Setting up ASHES Frontend with Black/White Aesthetic..."

# Create frontend directory
mkdir -p frontend
cd frontend

# Check if we already have a React app
if [ ! -f "package.json" ]; then
    echo "📦 Creating new React TypeScript application..."
    npx create-react-app . --template typescript
fi

# Install additional dependencies
echo "📦 Installing additional dependencies..."
npm install --save \
    @types/react @types/react-dom \
    @mui/material @emotion/react @emotion/styled \
    @mui/icons-material \
    react-router-dom @types/react-router-dom \
    axios \
    recharts \
    @reduxjs/toolkit react-redux \
    socket.io-client \
    date-fns \
    react-hook-form \
    @hookform/resolvers \
    yup

npm install --save-dev \
    @types/socket.io-client \
    eslint-config-prettier \
    prettier

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src/components/{common,layout,dashboard,experiments,agents,laboratory,data}
mkdir -p src/pages
mkdir -p src/hooks
mkdir -p src/services
mkdir -p src/store
mkdir -p src/types
mkdir -p src/utils
mkdir -p src/styles
mkdir -p public/icons

echo "✅ ASHES Frontend setup complete!"
echo "📍 Next steps:"
echo "   1. cd frontend"
echo "   2. npm start (to run development server)"
echo "   3. npm run build (to build for production)"
