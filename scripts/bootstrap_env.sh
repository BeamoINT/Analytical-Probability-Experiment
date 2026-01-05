#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".env" ]]; then
  echo ".env already exists - leaving it unchanged."
else
  if [[ -f "env.live" ]]; then
    cp env.live .env
    echo "Created .env from env.live"
  elif [[ -f "env.live.example" ]]; then
    cp env.live.example .env
    echo "Created .env from env.live.example"
  else
    echo "No env template found (env.live or env.live.example)."
    exit 1
  fi
fi

if [[ ! -f ".env.local" ]]; then
  cat > .env.local <<'EOF'
# Local secrets for this machine (NEVER COMMIT)
# Example:
# POLYBOT_POLYGON_PRIVATE_KEY=0x...
# POLYBOT_CLOB_API_KEY=...
# POLYBOT_CLOB_API_SECRET=...
# POLYBOT_CLOB_PASSPHRASE=...
EOF
  echo "Created .env.local template (add secrets here)"
else
  echo ".env.local already exists - leaving it unchanged."
fi

