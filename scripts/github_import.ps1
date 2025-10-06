# Script to import the ensemble-cam project from GitHub
# Make sure you have git configured

Write-Host "Importing ensemble-cam project from GitHub..."

# Check if git is installed
try {
    git --version | Out-Null
} catch {
    Write-Host "Error: Git is not installed. Please install Git first." -ForegroundColor Red
    exit 1
}

# Check if we're in a git repository
if (Test-Path ".git") {
    Write-Host "Warning: Already in a git repository." -ForegroundColor Yellow
    Write-Host "This script is designed to clone a fresh copy of the repository."
    $response = Read-Host "Do you want to continue? (y/N)"
    if ($response -notmatch "^[Yy]$") {
        Write-Host "Import cancelled."
        exit 0
    }
}

# Clone the repository
Write-Host "Cloning repository from GitHub..."
$result = git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git

if ($LASTEXITCODE -eq 0) {
    Write-Host "Repository cloned successfully!" -ForegroundColor Green
    Write-Host "Next steps:"
    Write-Host "1. cd ensemble-cam-v2"
    Write-Host "2. Copy .env.example to .env and configure your tokens"
    Write-Host "3. Run: uv sync"
    Write-Host "4. Follow the README.md instructions"
} else {
    Write-Host "Failed to clone repository. Please check your internet connection and try again." -ForegroundColor Red
    exit 1
}
