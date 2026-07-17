from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from .config import settings


def _build_fernet() -> Fernet:
    digest = hashlib.sha256(settings.credential_secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_secret(value: str) -> str:
    return _build_fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(value: str) -> str:
    return _build_fernet().decrypt(value.encode("utf-8")).decode("utf-8")
