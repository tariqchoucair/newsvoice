"""Input normalisation.

The pipeline treats line structure as authorial structure: paragraph boundaries
override dependency edges, and line boundaries bound quote pairing. That is the
right behaviour for clean news text, and the wrong behaviour for text that has
been hard-wrapped at a fixed column, because a newline in the middle of a
sentence is then read as a boundary that the author never wrote.

The effect is not cosmetic. A single mid-paragraph newline can shift sentence
segmentation enough to change which cue owns a quotation, which misattributes it
to a different speaker without any indication in the output.

Text from Factiva exports, PDF extraction and many scrapers is hard-wrapped.
:func:`normalise_paragraphs` repairs it. It is **not** applied automatically,
because doing so would silently change results for anyone whose input is already
clean.

Two conventions are in circulation and they need opposite treatment:

*Blank-line separated* — paragraphs divided by ``\\n\\n``, possibly hard-wrapped
within a paragraph. Single newlines are noise and are joined.

*Single-newline separated* — one paragraph per line, no blank lines. Here every
newline is meaningful and joining them would collapse the document into one
paragraph, which is far worse than the problem being fixed.

:func:`normalise_paragraphs` detects which it is given. Pass `style` explicitly
when you know, or when a document is too short for detection to be reliable.
"""

from __future__ import annotations

import re

__all__ = ["normalise_paragraphs", "detect_paragraph_style"]

_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def detect_paragraph_style(text: str) -> str:
    """Return ``"blank_line"`` or ``"single_newline"`` for `text`.

    A document containing any blank line is assumed to use blank lines as its
    paragraph separator, because mixing the two conventions deliberately is
    vanishingly rare while hard-wrapping inside blank-line-separated paragraphs
    is extremely common.
    """
    if _BLANK_LINE.search(text):
        return "blank_line"
    return "single_newline"


def normalise_paragraphs(
    text: str,
    style: str | None = None,
    join_hyphens: bool = False,
) -> str:
    """Normalise `text` to one line per paragraph, separated by blank lines.

    Parameters
    ----------
    text
        Raw document text.
    style
        ``"blank_line"``, ``"single_newline"``, or ``None`` to detect. Pass it
        explicitly for very short documents, where detection has little to go on.
    join_hyphens
        Rejoin words split across a line break by a hyphen (``govern-\\nment``).
        Off by default: it is common in PDF extraction and absent elsewhere, and
        applying it to text containing genuine line-final hyphens (``anti-\\ninflammatory``)
        silently corrupts them.

    Returns
    -------
    str
        Normalised text.

    Notes
    -----
    Character offsets in extraction output index the string that was passed to
    the extractor. If you normalise, **keep the normalised text** — offsets from
    a run on normalised input do not index the original.

    Examples
    --------
    >>> normalise_paragraphs("A sentence broken\\nacross lines.\\n\\nNext para.")
    'A sentence broken across lines.\\n\\nNext para.'

    >>> normalise_paragraphs("Para one.\\nPara two.")
    'Para one.\\n\\nPara two.'
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    style = style or detect_paragraph_style(text)

    if style not in ("blank_line", "single_newline"):
        raise ValueError(
            f"style must be 'blank_line', 'single_newline' or None, got {style!r}"
        )

    if join_hyphens:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)

    if style == "blank_line":
        # Mid-paragraph wraps are noise; blank lines are the real boundaries.
        text = _SINGLE_NEWLINE.sub(" ", text)
    else:
        # Every newline is a paragraph boundary; promote it to a blank line.
        text = _SINGLE_NEWLINE.sub("\n\n", text)

    text = _MANY_NEWLINES.sub("\n\n", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = "\n\n".join(part.strip() for part in text.split("\n\n") if part.strip())
    return text
