@echo off
setlocal enabledelayedexpansion

echo 🎮 Ultimate Tic Tac Toe - Poetry Setup
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.7+ first
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if Poetry is already installed
poetry --version >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing Poetry...
    
    REM Install Poetry using the official installer
    powershell -Command "Invoke-WebRequest -Uri https://install.python-poetry.org -OutFile install-poetry.py"
    python install-poetry.py
    
    REM Clean up installer
    del install-poetry.py
    
    REM Add Poetry to PATH for current session
    set PATH=%USERPROFILE%\AppData\Roaming\Python\Scripts;%PATH%
    
    REM Verify installation
    poetry --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Poetry installation failed
        echo Please add Poetry to your PATH manually
        pause
        exit /b 1
    )
) else (
    echo ✅ Poetry is already installed
    poetry --version
)

echo.
echo 🔧 Setting up Ultimate Tic Tac Toe...

REM Install project dependencies
echo 📥 Installing project dependencies...
poetry install

echo.
echo 🧪 Testing installation...

REM Run basic test
poetry run python basic_test.py
if errorlevel 1 (
    echo ❌ Basic test failed
    pause
    exit /b 1
)

echo ✅ Basic test passed!
echo.
echo 🎉 Setup completed successfully!
echo.
echo 🚀 To play the game:
echo    poetry run python main.py
echo.
echo 🧪 To run tests:
echo    poetry run pytest
echo.
echo 📖 For more information, see:
echo    POETRY_SETUP.md
echo    QUICKSTART.md
echo.

REM Optional: Activate virtual environment
set /p activate="Would you like to activate the virtual environment now? (y/n): "
if /i "!activate!"=="y" (
    echo 🔧 Activating virtual environment...
    echo Type 'exit' to deactivate when done
    poetry shell
)

pause
