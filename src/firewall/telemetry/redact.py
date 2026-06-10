from __future__ import annotations

import hashlib
import re


EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
API_KEY_PATTERN = re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|api[_-]?key\s*[:=]\s*\S+)")


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact_pii(text: str, max_chars: int = 200) -> str:
    redacted = EMAIL_PATTERN.sub("<EMAIL>", text)
    redacted = PHONE_PATTERN.sub("<PHONE>", redacted)
    redacted = API_KEY_PATTERN.sub("<API_KEY>", redacted)
    if len(redacted) > max_chars:
        return redacted[:max_chars] + "..."
    return redacted
