#!/bin/bash

echo ""
echo "========================================="
echo "RENAME GITHUB REPOSITORY"
echo "========================================="
echo ""
echo "Your local repository is ready with the new name:"
echo "  'Analytical-Probability-Experiment'"
echo ""
echo "To complete the rename on GitHub, choose one option:"
echo ""
echo "OPTION 1: Rename existing repository (Recommended - keeps all history)"
echo "  1. Opening GitHub settings page..."
echo "  2. In the 'Repository name' field, change to: Analytical-Probability-Experiment"
echo "  3. Click 'Rename' button"
echo ""
echo "Press Enter to open GitHub settings page..."
read

# Open GitHub settings page
open "https://github.com/BeamoINT/Polymarket-Auto-Trading-API/settings" 2>/dev/null || \
xdg-open "https://github.com/BeamoINT/Polymarket-Auto-Trading-API/settings" 2>/dev/null || \
echo "Please manually go to: https://github.com/BeamoINT/Polymarket-Auto-Trading-API/settings"

echo ""
echo "After renaming on GitHub, press Enter to push the changes..."
read

echo ""
echo "Pushing to GitHub..."
cd "/Users/HP/Desktop/Business/Polymarket Auto Trading API"

if git push -u origin main; then
    echo ""
    echo "========================================="
    echo "✅ SUCCESS!"
    echo "========================================="
    echo ""
    echo "Your repository has been renamed to:"
    echo "  'Analytical-Probability-Experiment'"
    echo ""
    echo "New URL:"
    echo "  https://github.com/BeamoINT/Analytical-Probability-Experiment"
    echo ""
    echo "GitHub automatically redirects the old URL, so existing clones still work!"
    echo ""
else
    echo ""
    echo "========================================="
    echo "❌ PUSH FAILED"
    echo "========================================="
    echo ""
    echo "Please make sure you've renamed the repository on GitHub."
    echo ""
    echo "If you haven't renamed it yet:"
    echo "  1. Go to: https://github.com/BeamoINT/Polymarket-Auto-Trading-API/settings"
    echo "  2. Change repository name to: Analytical-Probability-Experiment"
    echo "  3. Click 'Rename'"
    echo "  4. Run this script again"
    echo ""
fi

