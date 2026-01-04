#!/bin/bash
# PolyB0t - Ubuntu Auto-Setup Script
# Automatically installs all dependencies and sets up the bot

set -e  # Exit on error

echo "================================================"
echo "🤖 PolyB0t - Ubuntu Auto-Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    echo -e "${RED}❌ Error: This script requires Ubuntu/Debian (apt package manager)${NC}"
    echo "For other Linux distributions, install manually:"
    echo "  - Python 3.11+"
    echo "  - pip"
    echo "  - poetry"
    exit 1
fi

echo -e "${GREEN}✅ Detected apt-based system (Ubuntu/Debian)${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        version=$(python3 --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            return 0
        fi
    fi
    return 1
}

echo "================================================"
echo "📦 STEP 1: System Dependencies"
echo "================================================"
echo ""

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies
SYSTEM_DEPS=(
    "build-essential"
    "curl"
    "git"
    "wget"
    "software-properties-common"
    "libssl-dev"
    "libffi-dev"
    "sqlite3"
    "libsqlite3-dev"
)

for dep in "${SYSTEM_DEPS[@]}"; do
    if dpkg -l | grep -q "^ii  $dep "; then
        echo -e "${GREEN}✅ $dep already installed${NC}"
    else
        echo -e "${YELLOW}📦 Installing $dep...${NC}"
        sudo apt install -y "$dep"
    fi
done

echo ""
echo "================================================"
echo "🐍 STEP 2: Python 3.11+"
echo "================================================"
echo ""

if check_python_version; then
    version=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✅ Python $version already installed${NC}"
else
    echo -e "${YELLOW}📦 Installing Python 3.11...${NC}"
    
    # Add deadsnakes PPA for latest Python versions
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    
    # Install Python 3.11 and related packages
    sudo apt install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip
    
    # Make Python 3.11 the default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    
    echo -e "${GREEN}✅ Python 3.11 installed${NC}"
fi

# Verify Python version
python3 --version

echo ""
echo "================================================"
echo "📦 STEP 3: pip (Python Package Manager)"
echo "================================================"
echo ""

if command_exists pip3; then
    echo -e "${GREEN}✅ pip already installed${NC}"
else
    echo -e "${YELLOW}📦 Installing pip...${NC}"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    echo -e "${GREEN}✅ pip installed${NC}"
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "================================================"
echo "📦 STEP 4: Poetry (Dependency Manager)"
echo "================================================"
echo ""

if command_exists poetry; then
    echo -e "${GREEN}✅ Poetry already installed${NC}"
    poetry --version
else
    echo -e "${YELLOW}📦 Installing Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add Poetry to PATH permanently
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    if [ -f ~/.zshrc ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.zshrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    fi
    
    echo -e "${GREEN}✅ Poetry installed${NC}"
    poetry --version
fi

echo ""
echo "================================================"
echo "🔧 STEP 5: PolyB0t Dependencies"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: pyproject.toml not found${NC}"
    echo "Please run this script from the PolyB0t directory"
    exit 1
fi

echo "Installing PolyB0t dependencies with Poetry..."
poetry install

echo ""
echo "================================================"
echo "🗄️  STEP 6: Database Setup"
echo "================================================"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Please edit .env and configure your settings${NC}"
    else
        echo -e "${YELLOW}⚠️  No .env.example found, skipping .env creation${NC}"
    fi
else
    echo -e "${GREEN}✅ .env already exists${NC}"
fi

# Initialize database
echo "Initializing database..."
poetry run polyb0t db init

echo ""
echo "================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "================================================"
echo ""
echo "🎉 PolyB0t is ready to use!"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your environment:"
echo "   ${YELLOW}nano .env${NC}"
echo ""
echo "2. Add required settings:"
echo "   - POLYBOT_MODE=live"
echo "   - POLYBOT_USER_ADDRESS=your_wallet_address"
echo "   - POLYBOT_POLYGON_RPC_URL=https://polygon-rpc.com"
echo ""
echo "3. (Optional) Add L2 credentials for live trading:"
echo "   - See README_L2_SETUP.md for instructions"
echo ""
echo "4. Test the installation:"
echo "   ${GREEN}poetry run polyb0t doctor${NC}"
echo ""
echo "5. Check status:"
echo "   ${GREEN}poetry run polyb0t status${NC}"
echo ""
echo "6. Run the bot (dry-run mode):"
echo "   ${GREEN}poetry run polyb0t run --live${NC}"
echo ""
echo "7. View all commands:"
echo "   ${GREEN}poetry run polyb0t --help${NC}"
echo ""
echo "================================================"
echo "📚 Documentation:"
echo "================================================"
echo ""
echo "  • README.md - Main documentation"
echo "  • README_L2_SETUP.md - L2 credentials guide"
echo "  • ML_SYSTEM_GUIDE.md - Machine learning guide"
echo "  • DATA_RETENTION_UPGRADE.md - Data retention guide"
echo "  • BROAD_MARKET_LEARNING.md - Market learning guide"
echo ""
echo "================================================"
echo "🆘 Troubleshooting:"
echo "================================================"
echo ""
echo "If you see 'poetry: command not found':"
echo "  ${YELLOW}source ~/.bashrc${NC}  (or restart terminal)"
echo ""
echo "If Poetry commands fail:"
echo "  ${YELLOW}poetry env use python3.11${NC}"
echo ""
echo "If dependencies fail to install:"
echo "  ${YELLOW}poetry lock --no-update && poetry install${NC}"
echo ""
echo "For help:"
echo "  ${YELLOW}poetry run polyb0t --help${NC}"
echo ""
echo "================================================"
echo "Happy trading! 🚀"
echo "================================================"

