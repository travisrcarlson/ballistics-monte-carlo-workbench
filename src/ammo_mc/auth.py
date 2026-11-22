"""Authentication helpers for gating the Streamlit Engineering Workbench."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import os
from typing import Any

_DEFAULT_SALT_HEX = "a5355f14127b8981f316c2ead20e7baa"
_DEFAULT_HASH_HEX = "7e45208552e762caa03b2a37df7cd12882112037128b4c3d0cf37e28454929a4"
_PBKDF2_ITERATIONS = 200_000


def verify_app_password(candidate: str, secrets_mapping: Mapping[str, Any] | None = None) -> bool:
    """Verify the access password against Streamlit secrets, environment override, or default digest."""
    if not isinstance(candidate, str) or not candidate:
        return False

    configured = ""
    if secrets_mapping is not None:
        try:
            secret_val = secrets_mapping.get("APP_PASSWORD")
            if isinstance(secret_val, str) and secret_val:
                configured = secret_val
        except Exception:
            configured = ""

    if not configured:
        env_val = os.environ.get("AMMO_MC_PASSWORD", "")
        if isinstance(env_val, str) and env_val:
            configured = env_val

    if configured:
        return hmac.compare_digest(candidate.encode("utf-8"), configured.encode("utf-8"))

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        candidate.encode("utf-8"),
        bytes.fromhex(_DEFAULT_SALT_HEX),
        _PBKDF2_ITERATIONS,
    ).hex()
    return hmac.compare_digest(derived, _DEFAULT_HASH_HEX)
