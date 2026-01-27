#!/bin/bash
# GCP VM Quick Commands Reference
# Usage: Source this file or copy commands as needed

# =============================================================================
# CONNECTION
# =============================================================================

# SSH into VM
alias gcp-ssh='ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194'

# SSH with command execution helper
gcp() {
    ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "$@"
}

# =============================================================================
# COMMON COMMANDS (run via: gcp "command")
# =============================================================================

# Activate virtualenv prefix (use with other commands)
VENV="source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate"
PROJECT="cd ~/Analytical-Probability-Experiment"

# Bot status
alias gcp-status='gcp "$PROJECT && $VENV && polyb0t status"'

# System diagnostics
alias gcp-doctor='gcp "$PROJECT && $VENV && polyb0t doctor"'

# List pending intents
alias gcp-intents='gcp "$PROJECT && $VENV && polyb0t intents list"'

# View recent logs
alias gcp-logs='gcp "tail -100 ~/Analytical-Probability-Experiment/live_run.log"'

# Check running processes
alias gcp-ps='gcp "ps aux | grep polyb0t | grep -v grep"'

# Check disk space
alias gcp-disk='gcp "df -h"'

# Check memory
alias gcp-mem='gcp "free -h"'

# =============================================================================
# DIRECT SSH COMMANDS (copy-paste ready)
# =============================================================================

# Connect to VM
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194

# Check bot status
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t status"

# View logs
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "tail -100 ~/Analytical-Probability-Experiment/live_run.log"

# Run doctor
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t doctor"

# List intents
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t intents list"

# Show tradable markets
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate && polyb0t universe"

# =============================================================================
# KEY PATHS ON VM
# =============================================================================
# Project:     ~/Analytical-Probability-Experiment
# Config:      ~/Analytical-Probability-Experiment/.env
# Database:    ~/Analytical-Probability-Experiment/polybot.db
# Logs:        ~/Analytical-Probability-Experiment/live_run.log
# AI Models:   ~/Analytical-Probability-Experiment/data/ai_models/
# MoE Models:  ~/Analytical-Probability-Experiment/data/moe_models/
# Training DB: ~/Analytical-Probability-Experiment/data/ai_training.db
