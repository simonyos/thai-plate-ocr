"""Mapping from the recognizer's anonymized Roboflow class codes (A01..A54)
to the Thai glyphs / Arabic digits that actually appear on Thai plates.

The upstream dataset (`card-detector/thai-license-plate-character-detect`)
labels classes as opaque `A##` strings. The mapping below was derived
empirically by (a) running the recognizer on held-out plates with known
ground-truth text, (b) cross-referencing Roboflow's published annotation
counts (A45..A54 have >800 annotations each, consistent with the 10 digits),
and (c) assigning the remaining high-frequency codes by elimination.

Consonants whose codes did not appear in our validation sample remain
unmapped and fall through to the raw `A##` label. This is acceptable for
reporting on this dataset's coverage; extending the mapping requires more
ground-truth plates containing those consonants.
"""

from __future__ import annotations

A_CODE_TO_GLYPH: dict[str, str] = {
    # digits (derived from class-count histogram + 5 user-labeled plates)
    "A45": "0",
    "A46": "1",
    "A47": "2",
    "A48": "3",
    "A49": "4",
    "A50": "5",
    "A51": "6",
    "A52": "7",
    "A53": "8",
    "A54": "9",
    # consonants observed in our validation sample
    "A01": "ก",
    "A06": "ฆ",
    "A07": "ง",
    "A14": "ฎ",
    "A16": "ฐ",
    "A30": "พ",
    "A34": "ย",
    "A37": "ว",
}


def to_glyph(code: str) -> str:
    """Return the Thai/Arabic glyph for an A## code, or the code itself if unmapped."""
    return A_CODE_TO_GLYPH.get(code, code)


def translate(codes: str | list[str]) -> str:
    """Translate a sequence of A## codes to a display string.

    Accepts either a concatenated string like "A53A01A30" or a list of codes.
    """
    if isinstance(codes, str):
        # split into A## tokens
        tokens: list[str] = []
        i = 0
        while i < len(codes):
            if codes[i] == "A" and i + 2 < len(codes):
                tokens.append(codes[i : i + 3])
                i += 3
            else:
                tokens.append(codes[i])
                i += 1
        codes = tokens
    return "".join(to_glyph(c) for c in codes)
