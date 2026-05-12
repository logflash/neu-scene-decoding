"""Vault utilities for the Anthropic API key.

Create vault (run once from terminal):
    /home/NEU480/environments/480s26/bin/python claude_api_key.py

Load in Jupyter:
    from claude_api_key import load_vault
    load_vault()
"""

# ATTRIBUTION: This file was entirely written by Claude Code.

import getpass, os, base64
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

VAULT_PATH = Path(__file__).parent / ".vault"

def main():
    api_key  = getpass.getpass("Paste your Anthropic API key: ")
    passcode = getpass.getpass("Choose a vault passcode: ")
    confirm  = getpass.getpass("Confirm passcode: ")

    assert passcode == confirm, "Passcodes do not match"
    assert api_key.startswith("sk-"), "That doesn't look like an Anthropic API key"

    salt   = os.urandom(16)
    kdf    = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
    fernet = Fernet(base64.urlsafe_b64encode(kdf.derive(passcode.encode())))
    token  = fernet.encrypt(api_key.encode())

    VAULT_PATH.write_bytes(salt + token)
    VAULT_PATH.chmod(0o600)

    del api_key, passcode, confirm, salt, token
    print(f"Vault written to {VAULT_PATH}  ({VAULT_PATH.stat().st_size} bytes)")

def load_vault():
    """Prompt for passcode, decrypt the vault, and set ANTHROPIC_API_KEY."""
    data     = VAULT_PATH.read_bytes()
    salt     = data[:16]
    token    = data[16:]
    passcode = getpass.getpass("Vault passcode: ")
    kdf      = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
    fernet   = Fernet(base64.urlsafe_b64encode(kdf.derive(passcode.encode())))
    try:
        api_key = fernet.decrypt(token).decode()
    except InvalidToken:
        raise ValueError("Wrong passcode or corrupted vault")
    os.environ["ANTHROPIC_API_KEY"] = api_key
    del api_key, passcode
    print("Vault unlocked — ANTHROPIC_API_KEY set.")


if __name__ == "__main__":
    main()