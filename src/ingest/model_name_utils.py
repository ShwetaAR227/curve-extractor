"""Stage-1 part-number normalization (legacy-verified behavior).

Only DigiKey-specific packaging suffixes are stripped — they are meaningless
outside DigiKey. Manufacturer suffixes (``-TR``, ``-CT``, ``-PbF``, ``-E3``)
encode real part variants (reel, grade, compliance) and are kept.
"""
import re

_DIGIKEY_SUFFIXES = re.compile(r"-(TRDKR|DKR|ND)$", re.IGNORECASE)


def normalize_model_name(part_number: str) -> str:
    """Strip DigiKey-only suffixes (``-ND``, ``-DKR``, ``-TRDKR``) from a part number."""
    return _DIGIKEY_SUFFIXES.sub("", part_number.strip())


def sanitize_for_filesystem(name: str) -> str:
    """Replace path-breaking characters (``/ \\ ( )`` and whitespace) with ``_``."""
    sanitized = re.sub(r"[/\\()]+", "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")
