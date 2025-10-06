#!/bin/bash

# Script to prepare and upload the ensemble-cam project to GitHub
# Make sure you have git configured and the remote repository set up

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
git push -u origin main

echo "Upload complete!"
echo "Repository available at: https://github.com/muhammad-aasem/ensemble-cam-v2"
