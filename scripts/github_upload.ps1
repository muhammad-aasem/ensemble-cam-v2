# Script to prepare and upload the ensemble-cam project to GitHub
# Make sure you have git configured and the remote repository set up

# Load environment variables from .env file if it exists
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env file..."
    Get-Content ".env" | ForEach-Object {
        if ($_ -and -not $_.StartsWith("#") -and $_.Contains("=")) {
            $parts = $_.Split("=", 2)
            if ($parts.Length -eq 2) {
                [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
            }
        }
    }
}

Write-Host "Preparing ensemble-cam project for GitHub upload..."

# Function to clean up sensitive files
function Cleanup-SensitiveFiles {
    Write-Host "Cleaning up sensitive files..."
    
    # Remove any files that might contain tokens
    Remove-Item -Path "env.txt" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "temp_token.txt" -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "token_backup.txt" -Force -ErrorAction SilentlyContinue
    
    # Clean up SETUP.md if it contains tokens
    if (Test-Path "SETUP.md") {
        $content = Get-Content "SETUP.md" -Raw
        if ($content -match "ghp_|hf_") {
            Write-Host "Removing token references from SETUP.md..."
            $content = $content -replace ".*ghp_.*`r?`n", "" -replace ".*hf_.*`r?`n", ""
            Set-Content "SETUP.md" $content
        }
    }
    
    # Clean up README.md if it contains tokens
    if (Test-Path "README.md") {
        $content = Get-Content "README.md" -Raw
        if ($content -match "ghp_|hf_") {
            Write-Host "Removing token references from README.md..."
            $content = $content -replace ".*ghp_.*`r?`n", "" -replace ".*hf_.*`r?`n", ""
            Set-Content "README.md" $content
        }
    }
    
    # Clean up any .txt files that might contain tokens
    Get-ChildItem "*.txt" | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        if ($content -match "ghp_|hf_") {
            Write-Host "Removing token references from $($_.Name)..."
            $content = $content -replace ".*ghp_.*`r?`n", "" -replace ".*hf_.*`r?`n", ""
            Set-Content $_.FullName $content
        }
    }
}

# Function to push to GitHub with retry logic
function Push-ToGitHub {
    $attempt = 1
    $maxAttempts = 3
    
    while ($attempt -le $maxAttempts) {
        Write-Host "Push attempt $attempt of $maxAttempts..."
        
        # Use GitHub token from environment if available
        if ($env:GITHUB_TOKEN) {
            Write-Host "Using GitHub token from environment..."
            $pushUrl = "https://muhammad-aasem:$env:GITHUB_TOKEN@github.com/muhammad-aasem/ensemble-cam-v2.git"
            $result = git push $pushUrl main 2>&1
        } else {
            Write-Host "No GitHub token found in environment. You'll be prompted for credentials."
            $result = git push -u origin main 2>&1
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Push successful!"
            return $true
        } else {
            Write-Host "Push failed with exit code $LASTEXITCODE"
            
            # Check if it's a secrets violation
            if ($result -match "GH013|secrets|Push cannot contain") {
                Write-Host "Secrets violation detected. Cleaning up and retrying..."
                
                # Clean up sensitive files again
                Cleanup-SensitiveFiles
                
                # Remove files from git cache
                git rm --cached env.txt 2>$null
                git rm --cached temp_token.txt 2>$null
                git rm --cached token_backup.txt 2>$null
                
                # Add cleaned files
                git add .
                
                # Commit cleanup
                git commit -m "Security: Remove sensitive information

- Clean up token references from documentation
- Remove temporary token files
- Ensure no sensitive data in repository"
                
                Write-Host "Cleanup committed. Retrying push..."
            } else {
                Write-Host "Non-secrets error. Attempt $attempt failed."
            }
            
            $attempt++
            
            if ($attempt -le $maxAttempts) {
                Write-Host "Waiting 2 seconds before retry..."
                Start-Sleep -Seconds 2
            }
        }
    }
    
    Write-Host "All push attempts failed."
    return $false
}

# Clean up sensitive files before git operations
Cleanup-SensitiveFiles

# Initialize git repository if not already done
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..."
    git init
}

# Add all files (respecting .gitignore)
Write-Host "Adding files to git..."
git add .

# Check if there are any changes to commit
$stagedChanges = git diff --staged --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No changes to commit. Repository is up to date."
} else {
    # Commit changes
    Write-Host "Committing changes..."
    git commit -m "Update: Ensemble CAM - NIH Chest X-ray Dataset

- Complete dataset preparation pipeline
- Multiple CNN architectures (ResNet50, DenseNet121, DenseNet169, InceptionResNetV2, Xception)
- Grad-CAM++ compatible models
- Jupyter notebooks for easy workflow execution
- Comprehensive evaluation and visualization
- UV package management setup
- Environment variable configuration"
}

# Add remote origin if not already added
$remoteCheck = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Adding remote origin..."
    git remote add origin https://github.com/muhammad-aasem/ensemble-cam-v2.git
}

# Push to GitHub
Write-Host "Pushing to GitHub..."

# Attempt to push
if (Push-ToGitHub) {
    Write-Host "Upload complete!"
    Write-Host "Repository available at: https://github.com/muhammad-aasem/ensemble-cam-v2"
} else {
    Write-Host "Upload failed after multiple attempts."
    Write-Host "Please check the error messages above and try again."
    exit 1
}
