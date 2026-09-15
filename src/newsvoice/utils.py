"""Small shared helpers."""

from __future__ import annotations

import re

__all__ = ["WORD_RE", "word_count"]

#: A word is an alphanumeric run, optionally containing internal apostrophes or
#: hyphens. Counting on this rather than ``str.split`` keeps punctuation and
#: quotation marks out of the word counts reported for each segment.
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*")


def word_count(s: str) -> int:
    """Number of words in `s`, ignoring punctuation and quotation marks."""
    return len(WORD_RE.findall(s))
