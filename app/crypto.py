"""Encryption for the Gmail refresh tokens kept in the database.

A refresh token lets MeetPilot read someone's mail for as long as it is valid, so it is never stored as
plain text. The key lives in .env (TOKEN_ENCRYPTION_KEY), not in the database: a copy of the database
alone is not enough to use the tokens. If the key is lost, the tokens can't be read and every user
connects Gmail again."""
import os

from cryptography.fernet import Fernet, InvalidToken


class KeyProblem(Exception):
    """The encryption key is missing or wrong."""


def _box() -> Fernet:
    key = os.getenv("TOKEN_ENCRYPTION_KEY", "").strip()

    if not key:
        raise KeyProblem("TOKEN_ENCRYPTION_KEY is not set in .env.")

    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise KeyProblem("TOKEN_ENCRYPTION_KEY in .env is not a valid key.")


def available() -> bool:
    try:
        _box()
        return True
    except KeyProblem:
        return False


def encrypt(text: str) -> bytes:
    return _box().encrypt(text.encode("utf-8"))


def decrypt(token: bytes) -> str:
    try:
        return _box().decrypt(bytes(token)).decode("utf-8")
    except InvalidToken:
        raise KeyProblem("A saved Gmail token can't be read with the current TOKEN_ENCRYPTION_KEY. Connect Gmail again.")
