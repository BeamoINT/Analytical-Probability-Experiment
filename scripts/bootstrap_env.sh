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

SECRETS_DIR="${HOME}/.polyb0t"
SECRETS_FILE="${SECRETS_DIR}/secrets.env"

mkdir -p "${SECRETS_DIR}"

if [[ ! -f "${SECRETS_FILE}" ]]; then
  cat > "${SECRETS_FILE}" <<'EOF'
# Machine-level secrets (NEVER COMMIT)
# The bot loads env files in this order:
#   env.live  -> .env -> .env.local
# Recommended: keep ALL secrets in this file and symlink .env.local to it.
#
# Required for live trading:
# POLYBOT_POLYGON_PRIVATE_KEY=0x...
# POLYBOT_CLOB_API_KEY=...
# POLYBOT_CLOB_API_SECRET=...
# POLYBOT_CLOB_PASSPHRASE=...
#
# Proxy mode (typical for Polymarket built-in wallet):
# POLYBOT_USER_ADDRESS=0x<proxy_wallet>
# POLYBOT_FUNDER_ADDRESS=0x<proxy_wallet>
# POLYBOT_SIGNATURE_TYPE=1
#
# Optional but recommended:
# POLYBOT_POLYGON_RPC_URL=https://polygon-rpc.com
EOF
  echo "Created ${SECRETS_FILE} template (add secrets here)"
else
  echo "${SECRETS_FILE} already exists - leaving it unchanged."
fi

if [[ -L ".env.local" ]]; then
  echo ".env.local symlink already exists."
elif [[ -f ".env.local" ]]; then
  echo ".env.local exists (file). Not overwriting. Consider symlinking it to ${SECRETS_FILE}."
else
  ln -s "${SECRETS_FILE}" .env.local
  echo "Symlinked .env.local -> ${SECRETS_FILE}"
fi

