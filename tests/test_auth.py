"""Unit tests for workbench password verification."""

from __future__ import annotations

import hashlib

from ammo_mc.auth import _DEFAULT_HASH_HEX, _DEFAULT_SALT_HEX, _PBKDF2_ITERATIONS, verify_app_password


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
