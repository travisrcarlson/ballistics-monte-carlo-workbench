"""Authentication helpers and rate-limiting for gating the Streamlit Engineering Workbench."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import hmac
import os
import time
from typing import Any

_DEFAULT_SALT_HEX = "a5355f14127b8981f316c2ead20e7baa"
_DEFAULT_HASH_HEX = "7e45208552e762caa03b2a37df7cd12882112037128b4c3d0cf37e28454929a4"
_PBKDF2_ITERATIONS = 200_000


@dataclass
class LoginRateLimiter:
    """In-session exponential-backoff rate limiter for failed login attempts."""

    max_attempts: int = 5
    base_cooldown_s: float = 15.0
    max_cooldown_s: float = 300.0
    failed_attempts: int = 0
    lockout_until: float = 0.0
    lockout_count: int = 0

    def is_locked(self, now: float | None = None) -> tuple[bool, float]:
        """Return (locked, remaining_seconds) at the current monotonic clock time."""
        t_now = time.monotonic() if now is None else float(now)
        if self.lockout_until > t_now:
            return True, float(self.lockout_until - t_now)
        if self.lockout_until > 0.0 and t_now >= self.lockout_until:
            self.lockout_until = 0.0
            self.failed_attempts = 0
        return False, 0.0

    def record_failure(self, now: float | None = None) -> float:
        """Record a failed attempt; returns active cooldown seconds (>0 when locked)."""
        t_now = time.monotonic() if now is None else float(now)
        locked, remaining = self.is_locked(now=t_now)
        if locked:
            return remaining
        self.failed_attempts += 1
        if self.failed_attempts >= self.max_attempts:
            cooldown = min(
                float(self.max_cooldown_s),
                float(self.base_cooldown_s) * (2 ** self.lockout_count),
            )
            self.lockout_count += 1
            self.lockout_until = t_now + cooldown
            self.failed_attempts = 0
            return float(cooldown)
        return 0.0

    def record_success(self) -> None:
        """Reset failure counters and lockout state upon successful login."""
        self.failed_attempts = 0
        self.lockout_until = 0.0
        self.lockout_count = 0

    def remaining_attempts(self) -> int:
        """Return remaining attempts allowed before triggering a cooldown."""
        return max(0, int(self.max_attempts) - int(self.failed_attempts))


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
