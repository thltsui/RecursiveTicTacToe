# Poetry Setup Guide for Ultimate Tic Tac Toe

## 🚀 **Why Poetry?**

Poetry is a modern dependency management and packaging tool for Python that offers:
- **Dependency Resolution**: Automatically resolves package conflicts
- **Virtual Environments**: Creates isolated environments automatically
- **Lock Files**: Ensures reproducible builds across different machines
- **Modern Standards**: Uses `pyproject.toml` (PEP 518/621 compliant)
- **Better Dependency Management**: More reliable than pip + requirements.txt

## 📦 **Installing Poetry**

### **On macOS (Recommended)**
```bash
# Using Homebrew
brew install poetry

# Or using the official installer
curl -sSL https://install.python-poetry.org | python3 -
```

### **On Linux/Windows**
```bash
# Using the official installer
curl -sSL https://install.python-poetry.org | python3 -

# Or using pip (if you have pip)
pip install poetry
```

### **Verify Installation**
```bash
poetry --version
# Should show: Poetry (version X.X.X)
```

## 🎮 **Setting Up Ultimate Tic Tac Toe with Poetry**

### **1. Navigate to Project Directory**
```bash
cd ultimate_ttt
```

### **2. Install Dependencies**
```bash
# Install all dependencies (production + development)
poetry install

# Install only production dependencies
poetry install --only main

# Install with development dependencies
poetry install --with dev
```

### **3. Activate Virtual Environment**
```bash
# Poetry creates a virtual environment automatically
poetry shell

# Or run commands directly without activating
poetry run python main.py
```

### **4. Test Installation**
```bash
# Run basic test
poetry run python basic_test.py

# Run demo
poetry run python demo.py

# Run main program
poetry run python main.py
```

## 🔧 **Poetry Commands Reference**

### **Dependency Management**
```bash
# Add a new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Remove dependency
poetry remove package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree
```

### **Virtual Environment Management**
```bash
# Activate virtual environment
poetry shell

# Deactivate (exit shell)
exit

# Run command in virtual environment
poetry run command

# Show virtual environment info
poetry env info
```

### **Project Management**
```bash
# Build package
poetry build

# Publish to PyPI
poetry publish

# Check for security vulnerabilities
poetry check

# Export requirements.txt (if needed)
poetry export -f requirements.txt --output requirements.txt
```

## 🎯 **Quick Start with Poetry**

### **Complete Setup in One Go**
```bash
# 1. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Navigate to project
cd ultimate_ttt

# 3. Install dependencies
poetry install

# 4. Activate environment and play
poetry shell
python main.py
```

### **Alternative: Run Without Activating**
```bash
cd ultimate_ttt
poetry install
poetry run python main.py
```

## 🧪 **Testing with Poetry**

### **Run All Tests**
```bash
# Using Poetry
poetry run pytest

# Or activate environment first
poetry shell
pytest
```

### **Run Specific Tests**
```bash
# Run basic test
poetry run python basic_test.py

# Run main test suite
poetry run python main.py test

# Run with coverage
poetry run pytest --cov=game
```

## 🔍 **Troubleshooting Poetry Issues**

### **Common Issues and Solutions**

#### **Poetry Command Not Found**
```bash
# Add Poetry to PATH (macOS/Linux)
export PATH="$HOME/.local/bin:$PATH"

# Or reinstall Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

#### **Permission Errors**
```bash
# Use user installation
poetry config virtualenvs.in-project true
poetry install
```

#### **Python Version Issues**
```bash
# Check Python version
poetry env info

# Specify Python version
poetry env use python3.9
```

#### **Virtual Environment Issues**
```bash
# Remove existing environment
poetry env remove python

# Recreate environment
poetry install
```

## 📚 **Poetry vs Traditional pip**

| Feature | Poetry | pip + requirements.txt |
|---------|--------|----------------------|
| Dependency Resolution | ✅ Automatic | ❌ Manual |
| Virtual Environments | ✅ Auto-created | ❌ Manual (venv) |
| Lock Files | ✅ Reproducible | ❌ No guarantees |
| Modern Standards | ✅ PEP 518/621 | ❌ Legacy |
| Security | ✅ Built-in checks | ❌ Manual tools |
| Development Dependencies | ✅ Groups | ❌ Separate files |

## 🎉 **Benefits of Using Poetry**

1. **Reproducible Builds**: Lock file ensures same versions everywhere
2. **Automatic Virtual Environments**: No need to manage venv manually
3. **Better Dependency Resolution**: Handles conflicts automatically
4. **Modern Python Standards**: Uses latest Python packaging standards
5. **Development Tools**: Built-in support for testing, linting, etc.
6. **Easier Collaboration**: Team members get identical environments

## 🚀 **Next Steps**

After setting up with Poetry:

1. **Play the Game**: `poetry run python main.py`
2. **Run Tests**: `poetry run pytest`
3. **Watch Demo**: `poetry run python demo.py`
4. **Generate Data**: `poetry run python main.py generate --num-games 1000`
5. **Develop**: `poetry shell` then use your favorite editor

## 📖 **Additional Resources**

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Poetry GitHub](https://github.com/python-poetry/poetry)
- [Python Packaging User Guide](https://packaging.python.org/)

Poetry makes Python development much more enjoyable and reliable. Once you're set up, you'll wonder how you ever managed without it! 🎯
