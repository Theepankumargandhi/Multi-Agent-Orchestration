"""Portable paths and credential names shared by file and policy boundaries."""

from pathlib import PurePath

SECRET_PARTS = frozenset({".ssh", ".aws", ".azure", ".gcloud", ".gnupg", "secrets", "credentials"})
SECRET_NAMES = frozenset({".env", ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials",
                          "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"})
SECRET_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".crt"})
WINDOWS_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}) | frozenset(
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in "123456789¹²³"
)


def relative_parts(value: str) -> tuple[str, ...]:
    """Reject aliases rather than normalize paths differently on each host OS."""
    parts = tuple(value.replace("\\", "/").split("/"))
    if any(
        not part or part in {".", ".."} or part.endswith((".", " "))
        or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
        or part.split(".", 1)[0].rstrip(" ").upper() in WINDOWS_DEVICE_NAMES
        for part in parts
    ):
        raise ValueError("a safe relative file path is required")
    return parts


def is_sensitive_path(path: PurePath) -> bool:
    return (any(part.lower() in SECRET_PARTS for part in path.parts)
            or path.name.lower() in SECRET_NAMES or path.name.lower().startswith(".env.")
            or path.suffix.lower() in SECRET_SUFFIXES)
