"""Unit tests for workbench password verification."""

from __future__ import annotations

import hashlib

from ammo_mc.auth import (
    _DEFAULT_HASH_HEX,
    _DEFAULT_SALT_HEX,
    _PBKDF2_ITERATIONS,
    LoginRateLimiter,
    verify_app_password,
)


class _BrokenSecrets:
    def get(self, key: str, default: str | None = None) -> str | None:
        raise RuntimeError("No secrets.toml configured")

    def __getitem__(self, key: str) -> str:
        raise RuntimeError("No secrets.toml configured")

    def __iter__(self):
        raise RuntimeError("No secrets.toml configured")

    def __len__(self) -> int:
        return 0


def test_default_hash_rejects_empty_and_wrong(monkeypatch) -> None:
    monkeypatch.delenv("AMMO_MC_PASSWORD", raising=False)
    assert verify_app_password("") is False
    assert verify_app_password("wrong-password") is False
    assert verify_app_password("123456") is False


def test_default_hash_integrity() -> None:
    assert len(_DEFAULT_SALT_HEX) == 32
    assert len(_DEFAULT_HASH_HEX) == 64
    assert _PBKDF2_ITERATIONS == 200_000
    assert hashlib.pbkdf2_hmac("sha256", b"nope", bytes.fromhex(_DEFAULT_SALT_HEX), _PBKDF2_ITERATIONS).hex() != _DEFAULT_HASH_HEX


def test_env_var_override(monkeypatch) -> None:
    monkeypatch.setenv("AMMO_MC_PASSWORD", "CustomTestKey!42")
    assert verify_app_password("CustomTestKey!42") is True
    assert verify_app_password("other") is False


def test_secrets_mapping_precedence_over_env(monkeypatch) -> None:
    monkeypatch.setenv("AMMO_MC_PASSWORD", "EnvSecret")
    secrets = {"APP_PASSWORD": "TomlSecretValue"}
    assert verify_app_password("TomlSecretValue", secrets_mapping=secrets) is True
    assert verify_app_password("EnvSecret", secrets_mapping=secrets) is False


def test_broken_secrets_falls_back_cleanly(monkeypatch) -> None:
    monkeypatch.setenv("AMMO_MC_PASSWORD", "FallbackEnv")
    assert verify_app_password("FallbackEnv", secrets_mapping=_BrokenSecrets()) is True  # type: ignore[arg-type]
    assert verify_app_password("bad", secrets_mapping=_BrokenSecrets()) is False  # type: ignore[arg-type]


def test_login_rate_limiter_lockout_backoff_and_reset() -> None:
    limiter = LoginRateLimiter(max_attempts=5, base_cooldown_s=15.0, max_cooldown_s=60.0)
    t0 = 100.0
    assert limiter.is_locked(now=t0) == (False, 0.0)
    assert limiter.remaining_attempts() == 5

    for idx in range(4):
        assert limiter.record_failure(now=t0 + idx) == 0.0
        assert limiter.remaining_attempts() == 4 - idx
        assert limiter.is_locked(now=t0 + idx)[0] is False

    # 5th failure triggers first 15s cooldown
    cooldown_1 = limiter.record_failure(now=t0 + 5.0)
    assert cooldown_1 == 15.0
    locked, rem = limiter.is_locked(now=t0 + 10.0)
    assert locked is True
    assert rem == 10.0

    # After cooldown expires, another 5 failures triggers 30s backoff
    assert limiter.is_locked(now=t0 + 21.0) == (False, 0.0)
    for idx in range(4):
        assert limiter.record_failure(now=t0 + 22.0 + idx) == 0.0
    cooldown_2 = limiter.record_failure(now=t0 + 26.0)
    assert cooldown_2 == 30.0

    # Reset on success clears lockout and restores remaining attempts
    limiter.record_success()
    assert limiter.is_locked(now=t0 + 27.0) == (False, 0.0)
    assert limiter.remaining_attempts() == 5
