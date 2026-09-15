"""Layer 1 — locating direct quotation by pattern matching.

Direct speech is delimited by quotation marks, so this layer is purely lexical
and needs no parse. Two decisions carry most of the weight:

**Per-line pairing.** Straight ``"`` characters are paired by alternation. A
single unbalanced ``"`` anywhere in a document flips the parity of every pair
after it and corrupts the remainder. Resetting the pairing state at each newline
contains the damage to one paragraph.

**Apostrophe disambiguation.** A single ``'`` is a quotation mark only in a
position where it cannot be a possessive or a contraction, hence the lookaround
pattern in :data:`_SGL_QUOTE_RE`.
"""

from __future__ import annotations

import re
from typing import List, Tuple

__all__ = ["find_quote_spans", "overlaps"]

Span = Tuple[int, int]

_CURLY_PAIRS = (("\u201c", "\u201d"), ("\u2018", "\u2019"))
_SGL_QUOTE_RE = re.compile(
    r"(?<![A-Za-z0-9])'(?=[A-Za-z\u201c\"])"
    r"|(?<=[A-Za-z0-9.,!?])'(?![A-Za-z0-9])"
)

_DOUBLE_RE = re.compile(r'["\u201c](?=[A-Za-z0-9])[^"\u201d\n]{1,1998}["\u201d]')

_TRAILING_ATTRIBUTION_RE = re.compile(
    r",\s+(?:(?:Mr|Mrs|Ms|Dr|Prof(?:essor)?|Senator|Minister)\.?\s+)?"
    r"[A-Z][A-Za-z\u2019'\-]+(?:\s+[A-Z][A-Za-z\u2019'\-]+){0,2}\s+"
    r"(?:said|says|stated|states|told|added|argued)\b"
)


def find_quote_spans(text: str) -> List[Span]:
    """Balanced quoted regions as ``[(start, end)]`` character offsets into `text`.

    Double quotation marks are treated as one compatible family: straight or
    curly opening marks may close with either a straight or curly closing mark.
    Requiring an alphanumeric character immediately after the opener prevents a
    straight closing mark from being reinterpreted as the start of a new quote.
    """
    spans: List[Span] = []

    for lm in re.finditer(r"[^\n]+", text):          # one line at a time
        base, line = lm.start(), lm.group()

        # Straight/curly/mixed double quotes, processed left-to-right.
        line_double_spans = []
        for m in _DOUBLE_RE.finditer(line):
            span = (base + m.start(), base + m.end())
            line_double_spans.append(span)
            spans.append(span)

        # News articles often open a quotation turn at the start of a paragraph
        # but omit a closing mark until the final paragraph. Preserve each such
        # intermediate paragraph as direct speech. Restricting this repair to a
        # line-initial opener avoids treating embedded scare quotes as dangling.
        lead = len(line) - len(line.lstrip())
        opening = base + lead
        if (lead < len(line) and line[lead] in ('"', "\u201c")
                and not any(a == opening for a, _ in line_double_spans)):
            end = base + len(line.rstrip())
            # Some source articles omit the closing quote before a trailing
            # attribution: `"This is ... manufacturing, Mr Walton said ...`.
            # End the direct span at that comma so the attribution stays visible
            # to the cue and speaker layers. The unmatched opener is retained,
            # faithfully marking the source as incomplete rather than inventing
            # punctuation.
            trailing_attribution = _TRAILING_ATTRIBUTION_RE.search(line[lead + 1:])
            if trailing_attribution:
                end = base + lead + 1 + trailing_attribution.start() + 1
            if end - opening > 2:
                spans.append((opening, end))

        # Curly single quotes remain unambiguous.
        depth, single_start = 0, None
        for m in re.finditer(r"[\u2018\u2019]", line):
            if m.group() == "\u2018":
                if depth == 0:
                    single_start = m.start()
                depth += 1
            elif depth > 0:
                depth -= 1
                if depth == 0 and single_start is not None:
                    spans.append((base + single_start, base + m.end()))

        # Straight singles, guarded against apostrophes.
        sidx = [m.start() for m in _SGL_QUOTE_RE.finditer(line)]
        spans += [(base + a, base + b + 1)
                  for a, b in zip(sidx[0::2], sidx[1::2])
                  if 3 < b - a < 600]

    spans = [(s, e) for s, e in spans
             if 1 < e - s < 2000 and "\n" not in text[s:e]]
    spans.sort()

    merged: List[Span] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def overlaps(a: Span, b: Span) -> bool:
    """True when two half-open character spans intersect."""
    return a[0] < b[1] and b[0] < a[1]
