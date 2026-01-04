#!/bin/bash

# Automated GitHub Push Script
# This script helps you push your local repository to GitHub

set -e  # Exit on error

echo ""
echo "============================================================"
echo "  Push Polymarket Bot to GitHub"
echo "============================================================"
echo ""
echo "This script will help you push your repository to GitHub."
echo ""
echo "Prerequisites:"
echo "  1. You have a GitHub account"
echo "  2. You've created a PRIVATE repository on GitHub"
echo "  3. You have your GitHub username and repository name"
echo ""
echo "If you haven't created a repository yet:"
echo "  Go to: https://github.com/new"
echo "  Create a PRIVATE repository (do NOT initialize with README)"
echo ""
read -p "Press Enter when you're ready to continue..."
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository. Run this from the project root."
    exit 1
fi

# Check if there's already a remote
if git remote get-url origin &>/dev/null; then
    echo "ℹ️  Remote 'origin' already exists:"
    git remote get-url origin
    echo ""
    read -p "Do you want to replace it? (y/n): " replace
    if [ "$replace" = "y" ] || [ "$replace" = "Y" ]; then
        git remote remove origin
        echo "✓ Removed existing remote"
    else
        echo "Keeping existing remote. Pushing..."
        git push -u origin main
        exit 0
    fi
fi

# Get GitHub username
echo "Enter your GitHub username:"
read -p "Username: " github_username

if [ -z "$github_username" ]; then
    echo "❌ Error: Username cannot be empty"
    exit 1
fi

# Get repository name
echo ""
echo "Enter your GitHub repository name:"
echo "(e.g., polymarket-auto-trading-bot)"
read -p "Repository name: " repo_name

if [ -z "$repo_name" ]; then
    echo "❌ Error: Repository name cannot be empty"
    exit 1
fi

# Construct GitHub URL
GITHUB_URL="https://github.com/$github_username/$repo_name.git"

echo ""
echo "============================================================"
echo "  Configuration Summary"
echo "============================================================"
echo "GitHub URL: $GITHUB_URL"
echo "Local branch: main"
echo ""
read -p "Is this correct? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancelled."
    exit 0
fi

# Add remote
echo ""
echo "Adding GitHub remote..."
git remote add origin "$GITHUB_URL"
echo "✓ Remote added"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo ""
echo "⚠️  You may be prompted for credentials:"
echo "   - Username: Your GitHub username"
echo "   - Password: Use a Personal Access Token (not your password!)"
echo "   - Get token at: https://github.com/settings/tokens"
echo ""
echo "Pushing..."
echo ""

if git push -u origin main; then
    echo ""
    echo "============================================================"
    echo "  🎉 SUCCESS! Repository pushed to GitHub"
    echo "============================================================"
    echo ""
    echo "Your repository is now available at:"
    echo "  https://github.com/$github_username/$repo_name"
    echo ""
    echo "Next steps:"
    echo "  1. Visit your repository on GitHub"
    echo "  2. Verify all files are present"
    echo "  3. Verify .env and .db files are NOT visible"
    echo "  4. Add a description and topics"
    echo ""
    echo "To push future changes:"
    echo "  git add ."
    echo "  git commit -m \"Your commit message\""
    echo "  git push"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "  ❌ Push failed"
    echo "============================================================"
    echo ""
    echo "Common issues:"
    echo ""
    echo "1. Authentication failed:"
    echo "   - Make sure you're using a Personal Access Token"
    echo "   - Get one at: https://github.com/settings/tokens"
    echo "   - Token needs 'repo' scope"
    echo ""
    echo "2. Repository doesn't exist:"
    echo "   - Create it first at: https://github.com/new"
    echo "   - Make sure the name matches exactly"
    echo ""
    echo "3. Repository not empty:"
    echo "   - If you initialized with README, you'll need to pull first:"
    echo "     git pull origin main --allow-unrelated-histories"
    echo "     git push -u origin main"
    echo ""
    exit 1
fi

