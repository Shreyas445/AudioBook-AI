@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "REPO_URL=https://github.com/Shreyas445/AudioBook-AI.git"

echo ==============================================================
echo   AudioBook-AI - Git Sync ^& Secret Safety Verification
echo ==============================================================

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed or not in PATH.
    pause
    exit /b 1
)

:: 1. Initialize .git if missing (fixes 'fatal: not a git repository' on git status)
if not exist ".git" (
    echo [1/5] Initializing local Git repository...
    git init -b main
) else (
    echo [1/5] Local .git directory detected.
)

:: 2. Configure remote origin -> Shreyas445/AudioBook-AI
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [2/5] Linking remote origin -> %REPO_URL%
    git remote add origin "%REPO_URL%"
) else (
    echo [2/5] Updating remote origin -> %REPO_URL%
    git remote set-url origin "%REPO_URL%"
)

:: 3. Fetch existing commits from GitHub & attach history without overwriting local files!
echo [3/5] Fetching existing commit history from GitHub...
git fetch origin

git show-ref --verify --quiet refs/remotes/origin/main
if not errorlevel 1 (
    git rev-parse --verify HEAD >nul 2>&1
    if errorlevel 1 (
        echo       Linking local 'main' to origin/main history ^(preserving all local files^)...
        git symbolic-ref HEAD refs/heads/main
        git update-ref refs/heads/main refs/remotes/origin/main
        git reset --mixed HEAD >nul
        git branch --set-upstream-to=origin/main main >nul 2>&1
    )
) else (
    git show-ref --verify --quiet refs/remotes/origin/master
    if not errorlevel 1 (
        git rev-parse --verify HEAD >nul 2>&1
        if errorlevel 1 (
            echo       Linking local 'master' to origin/master history ^(preserving all local files^)...
            git symbolic-ref HEAD refs/heads/master
            git update-ref refs/heads/master refs/remotes/origin/master
            git reset --mixed HEAD >nul
            git branch --set-upstream-to=origin/master master >nul 2>&1
        )
    )
)

:: 4. Untrack any sensitive or heavy generated folders if previously tracked
echo [4/5] Ensuring .env.local, API keys, and output/ audio files are untracked...
git rm -r --cached .env .env.local output .venv venv __pycache__ >nul 2>&1

:: Stage clean project files respecting .gitignore
git add .

:: 5. Automated Secret Scan across staged changes
echo [5/5] Scanning staged diff for accidental API keys ^(nvapi-, AIza, sk-^)...
git diff --cached | findstr /R /C:"nvapi-[A-Za-z0-9_-][A-Za-z0-9_-][A-Za-z0-9_-]" /C:"AIza[0-9A-Za-z_-][0-9A-Za-z_-][0-9A-Za-z_-]" /C:"sk-[A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9][A-Za-z0-9]" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [CRITICAL BLOCK] Potential API key pattern detected in staged changes!
    echo Unstaging changes for your safety. Please inspect with: git diff
    git reset
    pause
    exit /b 1
)

echo       [OK] Zero API keys or secrets detected in staged files!
echo.
echo ==============================================================
echo   Current Git Status ^(Ready to Commit ^& Push^):
echo ==============================================================
git status -s
echo.
echo Next commands to publish your upgrade to GitHub:
echo   git commit -m "feat: Noir glass web studio, multi-API support, latent voice blending & 24kHz soundscape"
echo   git push -u origin HEAD
echo ==============================================================
endlocal
