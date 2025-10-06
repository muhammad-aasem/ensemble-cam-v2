@echo off
REM Script to import the ensemble-cam project from GitHub
REM Make sure you have git configured

echo Importing ensemble-cam project from GitHub...

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Git is not installed. Please install Git first.
    exit /b 1
)

REM Check if we're in a git repository
if exist ".git" (
    echo Warning: Already in a git repository.
    echo This script is designed to clone a fresh copy of the repository.
    echo Do you want to continue? (y/N)
    set /p response=
    if /i not "%response%"=="y" (
        echo Import cancelled.
        exit /b 0
    )
)

REM Clone the repository
echo Cloning repository from GitHub...
git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git

if %errorlevel% equ 0 (
    echo Repository cloned successfully!
    echo Next steps:
    echo 1. cd ensemble-cam-v2
    echo 2. Copy .env.example to .env and configure your tokens
    echo 3. Run: uv sync
    echo 4. Follow the README.md instructions
) else (
    echo Failed to clone repository. Please check your internet connection and try again.
    exit /b 1
)
