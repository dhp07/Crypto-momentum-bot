#!/bin/bash
# ===========================================================================
# Crypto Momentum Bot — Server Setup Script
# ===========================================================================
#
# Run this on a FRESH Ubuntu 24.04 server (Oracle Cloud Always Free works well).
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# What this script does:
#   1. Updates system packages
#   2. Installs Python 3.12 and pip
#   3. Installs Redis (for runtime state)
#   4. Installs Git
#   5. Creates a Python virtual environment
#   6. Installs all bot dependencies from requirements.txt
#   7. Creates necessary data directories
#
# What this script does NOT do:
#   - Clone the bot repo (you do this manually first)
#   - Set up API credentials (you do this in .env)
#   - Configure firewall rules (Oracle handles this in their console)
#
# ===========================================================================

set -e  # Exit immediately if any command fails
set -u  # Exit if any variable is undefined

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ===========================================================================
# Pre-flight checks
# ===========================================================================

if [ "$EUID" -eq 0 ]; then
    log_error "Don't run this script as root. Use a normal user with sudo access."
    exit 1
fi

if ! command -v sudo &> /dev/null; then
    log_error "sudo is required. Install it first."
    exit 1
fi

log_info "Starting bot environment setup..."
log_info "User: $(whoami)"
log_info "Working directory: $(pwd)"

# ===========================================================================
# System packages
# ===========================================================================

log_info "Updating package lists..."
sudo apt-get update -qq

log_info "Installing system dependencies..."
sudo apt-get install -y -qq \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# ===========================================================================
# Python 3.12
# ===========================================================================

# Ubuntu 24.04 ships with Python 3.12 by default
log_info "Installing Python 3.12 and venv..."
sudo apt-get install -y -qq \
    python3.12 \
    python3.12-venv \
    python3-pip

PYTHON_VERSION=$(python3.12 --version)
log_info "Python installed: $PYTHON_VERSION"

# ===========================================================================
# Redis
# ===========================================================================

log_info "Installing Redis..."
sudo apt-get install -y -qq redis-server

# Configure Redis to start on boot
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verify Redis is running
if redis-cli ping > /dev/null 2>&1; then
    log_info "Redis is running"
else
    log_error "Redis failed to start"
    exit 1
fi

# ===========================================================================
# Python virtual environment
# ===========================================================================

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found. Run this script from the bot's root directory."
    exit 1
fi

log_info "Creating Python virtual environment..."
python3.12 -m venv venv

log_info "Activating venv and upgrading pip..."
# shellcheck source=/dev/null
source venv/bin/activate
pip install --upgrade pip --quiet

log_info "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet

log_info "Python dependencies installed"

# ===========================================================================
# Data directories
# ===========================================================================

log_info "Creating data directories..."
mkdir -p data/ticks data/bars data/trades data/signals
mkdir -p logs

# ===========================================================================
# Final checks
# ===========================================================================

log_info "Verifying installation..."

# Check that key Python packages import successfully
python3.12 -c "
import sys
errors = []
for pkg in ['polars', 'pydantic', 'yaml', 'redis', 'websockets', 'httpx']:
    try:
        __import__(pkg)
    except ImportError as e:
        errors.append(f'{pkg}: {e}')

if errors:
    print('Failed imports:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('All required packages import successfully')
"

# ===========================================================================
# Done
# ===========================================================================

log_info "============================================"
log_info "Setup complete!"
log_info "============================================"
echo
log_info "Next steps:"
echo "  1. Copy .env.example to .env: cp .env.example .env"
echo "  2. Edit .env and fill in your Coinbase API credentials"
echo "  3. Test the configuration loads: python3 config/settings.py"
echo "  4. Activate the venv when working: source venv/bin/activate"
echo
log_warn "Reminder: Coinbase API keys should have VIEW permission only at first."
log_warn "Do NOT enable TRADE permission until you're ready to actually place orders."
log_warn "NEVER enable TRANSFER permission."
