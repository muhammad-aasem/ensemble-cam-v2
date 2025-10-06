# GitHub Scripts

This directory contains cross-platform scripts for GitHub operations.

## Upload Scripts

Upload your local changes to the GitHub repository:

- `github_upload.sh` - macOS/Linux shell script
- `github_upload.bat` - Windows Command Prompt batch file
- `github_upload.ps1` - Windows PowerShell script

### Features:
- Automatic environment variable loading from `.env` file
- Sensitive file cleanup (removes token references)
- Retry logic for GitHub push protection violations
- Cross-platform compatibility
- Secure token handling

## Import Scripts

Clone the repository from GitHub:

- `github_import.sh` - macOS/Linux shell script
- `github_import.bat` - Windows Command Prompt batch file
- `github_import.ps1` - Windows PowerShell script

### Features:
- Automatic repository cloning
- Git installation check
- Safety warnings for existing repositories
- Clear next-step instructions

## Usage

### Upload to GitHub

#### macOS/Linux
```bash
./github_upload.sh
```

#### Windows (Command Prompt)
```cmd
github_upload.bat
```

#### Windows (PowerShell)
```powershell
.\github_upload.ps1
```

### Import from GitHub

#### macOS/Linux
```bash
./github_import.sh
```

#### Windows (Command Prompt)
```cmd
github_import.bat
```

#### Windows (PowerShell)
```powershell
.\github_import.ps1
```

## Requirements

- Git installed and configured
- GitHub Personal Access Token (for upload scripts)
- Hugging Face Access Token (for dataset operations)
- Environment variables configured in `.env` file

## Environment Variables

Create a `.env` file in the project root with:

```env
# Hugging Face Hub Token
HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here

# GitHub Personal Access Token
GITHUB_TOKEN=your_github_token_here
```
