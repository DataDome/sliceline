#!/bin/bash
# Check Python 3.13 readiness status
# Usage: ./scripts/check_python_313_readiness.sh

set -e

echo "================================================"
echo "Python 3.13 Readiness Check"
echo "================================================"
echo ""

# Check optbinning PR status
echo "📋 Checking optbinning PR #366 status..."
if command -v gh &> /dev/null; then
    gh pr view 366 --repo guillermo-navas-palencia/optbinning --json state,mergedAt,url,title | \
        jq -r '"Status: \(.state)\nMerged At: \(.mergedAt // "Not merged yet")\nURL: \(.url)\nTitle: \(.title)"'
    echo ""
else
    echo "⚠️  GitHub CLI (gh) not installed. Install with: brew install gh"
    echo "   Manual check: https://github.com/guillermo-navas-palencia/optbinning/pull/366"
    echo ""
fi

# Check latest optbinning version on PyPI
echo "📦 Checking latest optbinning version on PyPI..."
LATEST_VERSION=$(curl -s https://pypi.org/pypi/optbinning/json | jq -r '.info.version')
echo "Latest version: $LATEST_VERSION"
echo ""

# Check if latest version supports Python 3.13
echo "🐍 Checking Python 3.13 support..."
PYTHON_VERSIONS=$(curl -s https://pypi.org/pypi/optbinning/json | jq -r '.info.classifiers[] | select(startswith("Programming Language :: Python :: 3."))')
echo "$PYTHON_VERSIONS"

if echo "$PYTHON_VERSIONS" | grep -q "3.13"; then
    echo ""
    echo "✅ Python 3.13 IS SUPPORTED in optbinning $LATEST_VERSION!"
    echo ""
    echo "🎉 Ready to add Python 3.13 support to sliceline!"
    echo ""
    echo "Next steps:"
    echo "1. Review: docs/project_notes/python-3.13-readiness.md"
    echo "2. Update pyproject.toml to add Python 3.13 classifier"
    echo "3. Update .github/workflows/push-pull.yml to test Python 3.13"
    echo "4. Run: uv sync --all-extras && uv run pytest tests/ -v"
    echo "5. Create PR with the changes"
else
    echo ""
    echo "⏳ Python 3.13 NOT YET SUPPORTED in optbinning $LATEST_VERSION"
    echo ""
    echo "Status: Still waiting for PR #366 to be merged and released"
    echo "Manual check: https://github.com/guillermo-navas-palencia/optbinning/pull/366"
fi

echo ""
echo "================================================"
echo "Last checked: $(date)"
echo "================================================"
