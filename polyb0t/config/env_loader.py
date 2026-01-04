"""Environment loading and validation.

Goals:
- Ensure `.env` is loaded at runtime (not just examples).
- Fail fast with clear guidance if `.env` is missing or incomplete.
- Never print secrets.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class EnvStatus:
    env_path: str
    env_example_path: str
    loaded: bool


REQUIRED_ENV_VARS = [
    "POLYBOT_MODE",
    "POLYBOT_USER_ADDRESS",
    "POLYBOT_LOOP_INTERVAL_SECONDS",
    "POLYBOT_DRY_RUN",
]

OPTIONAL_AUTH_VARS = [
    "POLYBOT_CLOB_API_KEY",
    "POLYBOT_CLOB_API_SECRET",
    "POLYBOT_CLOB_PASSPHRASE",
]

# Recommended vars for live mode (warnings if missing)
RECOMMENDED_LIVE_VARS = [
    "POLYBOT_FUNDER_ADDRESS",
    "POLYBOT_SIGNATURE_TYPE",
]


def load_env_or_exit(env_path: str = ".env", env_example_path: str = "env.live.example") -> EnvStatus:
    """Load `.env` and validate required keys exist.

    This should be called at process start (CLI entrypoint, API startup).
    """
    if not os.path.exists(env_path):
        _print_env_missing(env_path=env_path, env_example_path=env_example_path)
        raise SystemExit(2)

    loaded = load_dotenv(dotenv_path=env_path, override=False)

    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        _print_env_incomplete(missing, env_path=env_path)
        raise SystemExit(2)

    # Optional auth vars: if any is set, require all three
    present = [k for k in OPTIONAL_AUTH_VARS if os.environ.get(k)]
    if present and len(present) != len(OPTIONAL_AUTH_VARS):
        _print_auth_incomplete(env_path=env_path)
        raise SystemExit(2)
    
    # Warn if live mode but missing recommended vars
    mode = os.environ.get("POLYBOT_MODE", "").lower()
    if mode == "live":
        missing_recommended = [k for k in RECOMMENDED_LIVE_VARS if not os.environ.get(k)]
        if missing_recommended:
            _print_recommended_warning(missing_recommended)

    return EnvStatus(env_path=env_path, env_example_path=env_example_path, loaded=loaded)


def _print_env_missing(*, env_path: str, env_example_path: str) -> None:
    sys.stderr.write("\nERROR: Missing .env file.\n")
    if os.path.exists(env_example_path):
        sys.stderr.write(
            f"Found `{env_example_path}`. Create your `.env` by copying it:\n\n"
            f"  cp {env_example_path} {env_path}\n"
            f"  # then edit {env_path}\n\n"
        )
    else:
        sys.stderr.write(
            "No `.env.example` found.\n"
            "Create a `.env` file with the required variables (see README).\n\n"
        )
    sys.stderr.write("Required vars:\n")
    for k in REQUIRED_ENV_VARS:
        sys.stderr.write(f"  - {k}\n")
    sys.stderr.write("\n")


def _print_env_incomplete(missing: list[str], *, env_path: str) -> None:
    sys.stderr.write("\nERROR: .env is missing required variables.\n")
    sys.stderr.write(f"File: {env_path}\n")
    sys.stderr.write("Missing:\n")
    for k in missing:
        sys.stderr.write(f"  - {k}\n")
    sys.stderr.write("\nFix: edit your .env and set the missing keys (see README).\n\n")


def _print_auth_incomplete(*, env_path: str) -> None:
    sys.stderr.write("\nERROR: Partial CLOB credentials detected.\n")
    sys.stderr.write(f"File: {env_path}\n")
    sys.stderr.write(
        "If you set any of these, you must set all of them:\n"
        "  - POLYBOT_CLOB_API_KEY\n"
        "  - POLYBOT_CLOB_API_SECRET\n"
        "  - POLYBOT_CLOB_PASSPHRASE\n\n"
        "To generate L2 credentials, see: docs/L2_CREDENTIALS_SETUP.md\n"
        "Or run: poetry run python scripts/generate_l2_creds.py\n\n"
    )


def _print_recommended_warning(missing: list[str]) -> None:
    """Print warning for missing recommended vars in live mode."""
    sys.stderr.write("\n⚠️  WARNING: Missing recommended live mode configuration.\n")
    sys.stderr.write("Missing:\n")
    for k in missing:
        sys.stderr.write(f"  - {k}\n")
    sys.stderr.write(
        "\nThese are optional but recommended for proper order routing.\n"
        "See docs/L2_CREDENTIALS_SETUP.md for details.\n\n"
    )

