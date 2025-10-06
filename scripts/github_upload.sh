#!/bin/bash

# Script to prepare and upload the ensemble-cam project to GitHub
# Make sure you have git configured and the remote repository set up

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

echo "Preparing ensemble-cam project for GitHub upload..."

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files (respecting .gitignore)
echo "Adding files to git..."
git add .

# Check if there are any changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit. Repository is up to date."
else
    # Commit changes
    echo "Committing changes..."
    git commit -m "Initial commit: Ensemble CAM - NIH Chest X-ray Dataset

- Complete dataset preparation pipeline
- Multiple CNN architectures (ResNet50, DenseNet121, DenseNet169, InceptionResNetV2, Xception)
- Grad-CAM++ compatible models
- Jupyter notebooks for easy workflow execution
- Comprehensive evaluation and visualization
- UV package management setup"
fi

# Add remote origin if not already added
if ! git remote get-url origin >/dev/null 2>&1; then
    echo "Adding remote origin..."
    git remote add origin https://github.com/muhammad-aasem/ensemble-cam-v2.git
fi

# Push to GitHub
echo "Pushing to GitHub..."

# Use GitHub token from environment if available
if [ -n "$GITHUB_TOKEN" ]; then
    echo "Using GitHub token from environment..."
    git push https://muhammad-aasem:$GITHUB_TOKEN@github.com/muhammad-aasem/ensemble-cam-v2.git main
else
    echo "No GitHub token found in environment. You'll be prompted for credentials."
    git push -u origin main
fi

echo "Upload complete!"
echo "Repository available at: https://github.com/muhammad-aasem/ensemble-cam-v2"
