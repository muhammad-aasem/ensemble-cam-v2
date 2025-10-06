#!/bin/bash

# Script to import the ensemble-cam project from GitHub
# Make sure you have git configured

echo "Importing ensemble-cam project from GitHub..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if [ -d ".git" ]; then
    echo "Warning: Already in a git repository."
    echo "This script is designed to clone a fresh copy of the repository."
    echo "Do you want to continue? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Import cancelled."
        exit 0
    fi
fi

# Clone the repository
echo "Cloning repository from GitHub..."
git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git

if [ $? -eq 0 ]; then
    echo "Repository cloned successfully!"
    echo "Next steps:"
    echo "1. cd ensemble-cam-v2"
    echo "2. Copy .env.example to .env and configure your tokens"
    echo "3. Run: uv sync"
    echo "4. Follow the README.md instructions"
else
    echo "Failed to clone repository. Please check your internet connection and try again."
    exit 1
fi
