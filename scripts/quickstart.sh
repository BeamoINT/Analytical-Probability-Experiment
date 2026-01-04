#!/bin/bash
# Quick start script for PolyB0T

set -e

echo "=================================="
echo "PolyB0T Quick Start"
echo "=================================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check for Poetry
if ! command -v poetry &> /dev/null; then
    echo "⚠️  Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ Poetry available"
echo ""

# Install dependencies
echo "Installing dependencies..."
poetry install
echo "✓ Dependencies installed"
echo ""

# Setup environment
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ Created .env (please review and customize)"
else
    echo "✓ .env already exists"
fi
echo ""

# Initialize database
echo "Initializing database..."
poetry run polyb0t db init
echo "✓ Database initialized"
echo ""

# Run tests
echo "Running tests to verify installation..."
poetry run pytest -q
echo "✓ Tests passed"
echo ""

echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review and customize .env configuration:"
echo "   nano .env"
echo ""
echo "2. Start paper trading:"
echo "   poetry run polyb0t run --paper"
echo ""
echo "3. Or use the Makefile:"
echo "   make run         # Start bot"
echo "   make api         # Start API server"
echo "   make report      # Generate report"
echo "   make universe    # View tradable markets"
echo ""
echo "4. View help:"
echo "   poetry run polyb0t --help"
echo ""
echo "Happy (paper) trading! 📈"

