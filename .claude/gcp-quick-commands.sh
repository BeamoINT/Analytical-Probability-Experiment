#!/bin/bash
# GCP VM Quick Commands Reference
# Usage: Source this file or copy commands as needed
#
# IMPORTANT: The bot runs as a systemd service!
# Always use systemctl commands, NOT manual process management.

# =============================================================================
# CONNECTION
# =============================================================================

SSH_KEY="/Users/HP/.ssh/gcp_vm"
SSH_USER="beamo_beamosupport_com"
SSH_HOST="34.2.57.194"
SSH_CMD="ssh -i $SSH_KEY $SSH_USER@$SSH_HOST"

# SSH into VM
alias gcp-ssh='ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194'

# SSH with command execution helper
gcp() {
    ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "$@"
}

# =============================================================================
# SYSTEMD SERVICE MANAGEMENT (USE THESE!)
# =============================================================================

# Restart service (USE THIS AFTER CODE CHANGES!)
alias gcp-restart='gcp "sudo systemctl restart polybot.service && sleep 3 && sudo systemctl status polybot.service"'

# Check service status
alias gcp-status='gcp "sudo systemctl status polybot.service"'

# Stop service
alias gcp-stop='gcp "sudo systemctl stop polybot.service"'

# Start service
alias gcp-start='gcp "sudo systemctl start polybot.service"'

# View systemd journal logs (last 50 lines)
alias gcp-journal='gcp "sudo journalctl -u polybot.service --no-pager -n 50"'

# Follow journal logs live
alias gcp-journal-follow='gcp "sudo journalctl -u polybot.service -f"'

# =============================================================================
# DEPLOYMENT (FULL WORKFLOW)
# =============================================================================

# Deploy changes: pull + restart service
alias gcp-deploy='gcp "cd ~/Analytical-Probability-Experiment && git pull && sudo systemctl restart polybot.service && sleep 3 && sudo systemctl status polybot.service"'

# Just pull (no restart)
alias gcp-pull='gcp "cd ~/Analytical-Probability-Experiment && git pull"'

# =============================================================================
# LOGS & MONITORING
# =============================================================================

# View recent application logs
alias gcp-logs='gcp "tail -100 ~/Analytical-Probability-Experiment/live_run.log"'

# Check for errors in logs
alias gcp-errors='gcp "grep -i \"error\\|exception\\|traceback\" ~/Analytical-Probability-Experiment/live_run.log | tail -30"'

# Check running processes
alias gcp-ps='gcp "ps aux | grep polyb0t | grep -v grep"'

# =============================================================================
# SYSTEM INFO
# =============================================================================

# Check disk space
alias gcp-disk='gcp "df -h"'

# Check memory
alias gcp-mem='gcp "free -h"'

# System uptime
alias gcp-uptime='gcp "uptime"'

# =============================================================================
# POLYBOT CLI COMMANDS
# =============================================================================

# Helper vars for CLI commands
VENV="source ~/.cache/pypoetry/virtualenvs/polyb0t-adUcXa5t-py3.12/bin/activate"
PROJECT="cd ~/Analytical-Probability-Experiment"

# Bot status (portfolio, balance, positions)
alias gcp-bot-status='gcp "$PROJECT && $VENV && polyb0t status"'

# System diagnostics
alias gcp-doctor='gcp "$PROJECT && $VENV && polyb0t doctor"'

# List pending intents
alias gcp-intents='gcp "$PROJECT && $VENV && polyb0t intents list"'

# Show tradable markets
alias gcp-universe='gcp "$PROJECT && $VENV && polyb0t universe"'

# =============================================================================
# KEY PATHS ON VM
# =============================================================================
# Project:     ~/Analytical-Probability-Experiment
# Config:      ~/Analytical-Probability-Experiment/.env
# Main DB:     ~/Analytical-Probability-Experiment/polybot.db (~12GB)
# Logs:        ~/Analytical-Probability-Experiment/live_run.log
# AI Models:   ~/Analytical-Probability-Experiment/data/ai_models/
# MoE Models:  ~/Analytical-Probability-Experiment/data/moe_models/
# Training DB: ~/Analytical-Probability-Experiment/data/ai_training.db (~800MB)
# Service:     /etc/systemd/system/polybot.service

# =============================================================================
# COPY-PASTE READY COMMANDS
# =============================================================================

# Restart service (MOST COMMON - use after deploying code!)
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "sudo systemctl restart polybot.service"

# Full deploy: pull + restart
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "cd ~/Analytical-Probability-Experiment && git pull && sudo systemctl restart polybot.service"

# Check service status
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "sudo systemctl status polybot.service"

# View systemd logs
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "sudo journalctl -u polybot.service --no-pager -n 50"

# View application logs
# ssh -i /Users/HP/.ssh/gcp_vm beamo_beamosupport_com@34.2.57.194 "tail -100 ~/Analytical-Probability-Experiment/live_run.log"
