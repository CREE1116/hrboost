#!/bin/bash
set -e

# Load .env file
if [ -f .env ]; then
    # Filter out comments and load env variables
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$PYPI_API_TOKEN" ]; then
    echo "Error: PYPI_API_TOKEN is not set in .env"
    exit 1
fi

echo "======================================================"
echo "          Starting HRBoost Automated Release"
echo "======================================================"

# 1. Rebuild C++ libraries
echo "[1/4] Re-building C++ shared libraries..."
sh build.sh

# 2. Extract current version from pyproject.toml
VERSION=$(grep -E '^version\s*=\s*' pyproject.toml | head -n1 | cut -d'"' -f2)
echo "Detected package version: $VERSION"

# 3. Clean and build distribution wheels
echo "[2/4] Building Python distribution packages..."
rm -rf dist/ build/ python/*.egg-info/
.venv/bin/python -m build

# 4. Git commit and push to GitHub
echo "[3/4] Staging and pushing changes to GitHub..."
git add .
COMMIT_MSG=${1:-"Release version $VERSION"}
git commit -m "$COMMIT_MSG" || echo "No local changes to commit."
git push origin master

# 5. Upload to PyPI via twine
echo "[4/4] Uploading version $VERSION to PyPI using twine..."
.venv/bin/twine upload --username __token__ --password "$PYPI_API_TOKEN" dist/*"$VERSION"*

echo "======================================================"
echo "🚀 Release process successfully finished!"
echo "======================================================"
