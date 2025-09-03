#!/bin/bash

# Ultimate Tic Tac Toe - Poetry Installation Script
# This script installs Poetry and sets up the project

set -e  # Exit on any error

echo "🎮 Ultimate Tic Tac Toe - Poetry Setup"
echo "======================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ first"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION found"

# Check if Poetry is already installed
if command -v poetry &> /dev/null; then
    echo "✅ Poetry is already installed"
    POETRY_VERSION=$(poetry --version)
    echo "   $POETRY_VERSION"
else
    echo "📦 Installing Poetry..."
    
    # Install Poetry using the official installer
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry installed successfully"
        POETRY_VERSION=$(poetry --version)
        echo "   $POETRY_VERSION"
    else
        echo "❌ Poetry installation failed"
        echo "Please add Poetry to your PATH manually:"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
fi

echo ""
echo "🔧 Setting up Ultimate Tic Tac Toe..."

# Install project dependencies
echo "📥 Installing project dependencies..."
poetry install

echo ""
echo "🧪 Testing installation..."

# Run basic test
if poetry run python basic_test.py; then
    echo "✅ Basic test passed!"
else
    echo "❌ Basic test failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 To play the game:"
echo "   poetry run python main.py"
echo ""
echo "🧪 To run tests:"
echo "   poetry run pytest"
echo ""
echo "📖 For more information, see:"
echo "   POETRY_SETUP.md"
echo "   QUICKSTART.md"
echo ""

# Optional: Activate virtual environment
read -p "Would you like to activate the virtual environment now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔧 Activating virtual environment..."
    echo "Type 'exit' to deactivate when done"
    poetry shell
fi
