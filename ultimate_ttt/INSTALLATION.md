# Installation Guide

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.7 or higher
- **Memory**: At least 100MB free RAM
- **Storage**: At least 10MB free disk space

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd ultimate_ttt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python basic_test.py
   ```

### Method 2: Development Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ultimate_ttt
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

### Method 3: Manual Installation

1. **Install Python dependencies manually**
   ```bash
   pip install numpy torch
   ```

2. **Verify Python can import the modules**
   ```bash
   python -c "import numpy; import torch; print('Dependencies installed successfully')"
   ```

## Platform-Specific Instructions

### Windows

1. **Install Python from python.org**
   - Download Python 3.7+ from https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Install using pip**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Run the game**
   ```cmd
   python main.py
   ```
   Or double-click `run_game.bat`

### macOS

1. **Install Python using Homebrew (recommended)**
   ```bash
   brew install python3
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the game**
   ```bash
   python3 main.py
   ```

### Linux (Ubuntu/Debian)

1. **Install Python and pip**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```

2. **Install dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the game**
   ```bash
   python3 main.py
   ```

## Verification

After installation, verify everything works:

1. **Run basic test**
   ```bash
   python basic_test.py
   ```
   Should show: "🎉 All basic tests passed!"

2. **Run demo**
   ```bash
   python demo.py
   ```
   Should show a game between two random players

3. **Run main program**
   ```bash
   python main.py
   ```
   Should show the interactive menu

## Troubleshooting

### Common Issues

#### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'numpy'`
**Solution**: Install dependencies
```bash
pip install numpy torch
```

#### Permission Errors
**Problem**: Permission denied when installing packages
**Solution**: Use user installation
```bash
pip install --user -r requirements.txt
```

#### Python Version Issues
**Problem**: `python` command not found
**Solution**: Use `python3` instead
```bash
python3 main.py
```

#### Path Issues
**Problem**: Can't find the game modules
**Solution**: Make sure you're in the correct directory
```bash
cd ultimate_ttt
python main.py
```

### Advanced Troubleshooting

#### Virtual Environment
Create a virtual environment to avoid conflicts:
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the game
python main.py
```

#### Conda Environment
If using Anaconda:
```bash
# Create conda environment
conda create -n ultimate-ttt python=3.9

# Activate environment
conda activate ultimate-ttt

# Install dependencies
conda install numpy
pip install torch

# Run the game
python main.py
```

## Uninstallation

To remove the game and dependencies:

```bash
# Remove the package
pip uninstall ultimate-ttt

# Remove dependencies (be careful - may affect other projects)
pip uninstall numpy torch

# Remove the directory
rm -rf ultimate_ttt/
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python basic_test.py` to identify specific problems
3. Check Python and pip versions: `python --version` and `pip --version`
4. Ensure you're in the correct directory
5. Try creating a fresh virtual environment

## Next Steps

After successful installation:

1. **Play a game**: `python main.py`
2. **Run tests**: `python main.py test`
3. **Watch demo**: `python demo.py`
4. **Generate data**: `python main.py generate --num-games 1000`
5. **Read documentation**: Check `README.md` and `QUICKSTART.md`
