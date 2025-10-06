@echo off
REM Script to prepare and upload the ensemble-cam project to GitHub
REM Make sure you have git configured and the remote repository set up

REM Load environment variables from .env file if it exists
if exist .env (
    echo Loading environment variables from .env file...
    for /f "tokens=1,2 delims==" %%a in (.env) do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
)

echo Preparing ensemble-cam project for GitHub upload...

REM Function to clean up sensitive files
call :cleanup_sensitive_files

REM Initialize git repository if not already done
if not exist ".git" (
    echo Initializing git repository...
    git init
)

REM Add all files (respecting .gitignore)
echo Adding files to git...
git add .

REM Check if there are any changes to commit
git diff --staged --quiet
if %errorlevel% equ 0 (
    echo No changes to commit. Repository is up to date.
) else (
    REM Commit changes
    echo Committing changes...
    git commit -m "Update: Ensemble CAM - NIH Chest X-ray Dataset

- Complete dataset preparation pipeline
- Multiple CNN architectures (ResNet50, DenseNet121, DenseNet169, InceptionResNetV2, Xception)
- Grad-CAM++ compatible models
- Jupyter notebooks for easy workflow execution
- Comprehensive evaluation and visualization
- UV package management setup
- Environment variable configuration"
)

REM Add remote origin if not already added
git remote get-url origin >nul 2>&1
if %errorlevel% neq 0 (
    echo Adding remote origin...
    git remote add origin https://github.com/muhammad-aasem/ensemble-cam-v2.git
)

REM Push to GitHub with retry logic
echo Pushing to GitHub...
call :push_to_github

if %errorlevel% equ 0 (
    echo Upload complete!
    echo Repository available at: https://github.com/muhammad-aasem/ensemble-cam-v2
) else (
    echo Upload failed after multiple attempts.
    echo Please check the error messages above and try again.
    exit /b 1
)

goto :eof

REM Function to clean up sensitive files
:cleanup_sensitive_files
echo Cleaning up sensitive files...

REM Remove any files that might contain tokens
if exist env.txt del /f env.txt
if exist temp_token.txt del /f temp_token.txt
if exist token_backup.txt del /f token_backup.txt

REM Clean up SETUP.md if it contains tokens
if exist SETUP.md (
    findstr /c:"ghp_" /c:"hf_" SETUP.md >nul 2>&1
    if %errorlevel% equ 0 (
        echo Removing token references from SETUP.md...
        findstr /v /c:"ghp_" /c:"hf_" SETUP.md > SETUP.md.tmp
        move SETUP.md.tmp SETUP.md
    )
)

REM Clean up README.md if it contains tokens
if exist README.md (
    findstr /c:"ghp_" /c:"hf_" README.md >nul 2>&1
    if %errorlevel% equ 0 (
        echo Removing token references from README.md...
        findstr /v /c:"ghp_" /c:"hf_" README.md > README.md.tmp
        move README.md.tmp README.md
    )
)

REM Clean up any .txt files that might contain tokens
for %%f in (*.txt) do (
    if exist "%%f" (
        findstr /c:"ghp_" /c:"hf_" "%%f" >nul 2>&1
        if %errorlevel% equ 0 (
            echo Removing token references from %%f...
            findstr /v /c:"ghp_" /c:"hf_" "%%f" > "%%f.tmp"
            move "%%f.tmp" "%%f"
        )
    )
)

goto :eof

REM Function to push to GitHub with retry logic
:push_to_github
set attempt=1
set max_attempts=3

:push_loop
echo Push attempt %attempt% of %max_attempts%...

REM Use GitHub token from environment if available
if defined GITHUB_TOKEN (
    echo Using GitHub token from environment...
    git push https://muhammad-aasem:%GITHUB_TOKEN%@github.com/muhammad-aasem/ensemble-cam-v2.git main
) else (
    echo No GitHub token found in environment. You'll be prompted for credentials.
    git push -u origin main
)

if %errorlevel% equ 0 (
    echo Push successful!
    exit /b 0
) else (
    echo Push failed with exit code %errorlevel%
    
    REM Check if it's a secrets violation
    git push https://muhammad-aasem:%GITHUB_TOKEN%@github.com/muhammad-aasem/ensemble-cam-v2.git main 2>&1 | findstr /c:"GH013" /c:"secrets" /c:"Push cannot contain" >nul
    if %errorlevel% equ 0 (
        echo Secrets violation detected. Cleaning up and retrying...
        
        REM Clean up sensitive files again
        call :cleanup_sensitive_files
        
        REM Remove files from git cache
        git rm --cached env.txt >nul 2>&1
        git rm --cached temp_token.txt >nul 2>&1
        git rm --cached token_backup.txt >nul 2>&1
        
        REM Add cleaned files
        git add .
        
        REM Commit cleanup
        git commit -m "Security: Remove sensitive information

- Clean up token references from documentation
- Remove temporary token files
- Ensure no sensitive data in repository"
        
        echo Cleanup committed. Retrying push...
    ) else (
        echo Non-secrets error. Attempt %attempt% failed.
    )
    
    set /a attempt+=1
    
    if %attempt% leq %max_attempts% (
        echo Waiting 2 seconds before retry...
        timeout /t 2 /nobreak >nul
        goto push_loop
    )
)

echo All push attempts failed.
exit /b 1
