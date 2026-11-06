#!/usr/bin/env bash
# Kill-switch hotkey (X11): Ctrl+Alt+K creates the STOP file -> the agent sends no more input and every goal stops.
# Runs xbindkeys in the foreground with a project-local config (no change to your system or ~/.xbindkeysrc).
# Re-arm: rm STOP
set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root
mkdir -p logs
cat > logs/xbindkeysrc <<RC
"touch $PWD/STOP"
  Control+Alt + k
RC
exec xbindkeys --nodaemon -f logs/xbindkeysrc
