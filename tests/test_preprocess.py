"""Input normalisation, and the misattribution it exists to prevent.

The end-to-end test at the foot of this file is the reason the module exists: it
pins a case where a single mid-paragraph newline moves a quotation from one
speaker to another, with nothing in the output to indicate it happened.
"""

from __future__ import annotations

import json

import pytest

import newsvoice
from newsvoice.preprocess import detect_paragraph_style, normalise_paragraphs


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def test_blank_lines_detected():
    assert detect_paragraph_style("Para one.\n\nPara two.") == "blank_line"


def test_single_newlines_detected():
    assert detect_paragraph_style("Para one.\nPara two.") == "single_newline"


def test_blank_line_wins_when_both_are_present():
    """Hard-wrapping inside blank-line paragraphs is common; mixing is not."""
    text = "A paragraph broken\nacross two lines.\n\nA second paragraph."
    assert detect_paragraph_style(text) == "blank_line"


def test_text_without_newlines():
    assert detect_paragraph_style("One line, no breaks.") == "single_newline"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_hard_wrap_is_joined():
    text = "A sentence broken\nacross two lines.\n\nA second paragraph."
    assert normalise_paragraphs(text) == (
        "A sentence broken across two lines.\n\nA second paragraph.")


def test_single_newline_paragraphs_are_promoted_not_joined():
    """Joining these would collapse the document into one paragraph."""
    assert normalise_paragraphs("Para one.\nPara two.") == "Para one.\n\nPara two."


def test_clean_text_is_unchanged():
    text = "Para one.\n\nPara two.\n\nPara three."
    assert normalise_paragraphs(text) == text


def test_windows_line_endings():
    assert normalise_paragraphs("A line\r\nwrapped.\r\n\r\nNext.") == \
        "A line wrapped.\n\nNext."


def test_runs_of_blank_lines_collapse_to_one():
    assert normalise_paragraphs("Para one.\n\n\n\n\nPara two.") == \
        "Para one.\n\nPara two."


def test_repeated_spaces_collapse():
    assert normalise_paragraphs("Too    many   spaces.") == "Too many spaces."


def test_normalisation_is_idempotent():
    """Running it twice must not differ from running it once."""
    text = "A sentence broken\nacross lines.\n\n\nAnother   paragraph here.\n"
    once = normalise_paragraphs(text)
    assert normalise_paragraphs(once) == once


def test_style_can_be_forced():
    text = "Para one.\nPara two."
    assert normalise_paragraphs(text, style="blank_line") == "Para one. Para two."


def test_invalid_style_is_rejected():
    with pytest.raises(ValueError, match="style"):
        normalise_paragraphs("text", style="sideways")


def test_hyphen_rejoining_is_off_by_default():
    """On by default would corrupt genuine line-final hyphens.

    Uses blank-line style, since a lone hyphenated wrap in single-newline text
    is indistinguishable from a one-line paragraph ending in a hyphen.
    """
    text = "The govern-\nment responded.\n\nA second paragraph."
    assert "govern- ment" in normalise_paragraphs(text)
    assert normalise_paragraphs(text, join_hyphens=True).startswith(
        "The government responded.")


def test_empty_input():
    assert normalise_paragraphs("") == ""
    assert normalise_paragraphs("\n\n  \n") == ""


# --------------------------------------------------------------------------
# The misattribution this prevents
# --------------------------------------------------------------------------

QUOTE_TURN_PARAGRAPHS = [
    "Energy regulator warns of price rises",
    '"Families are already stretched to breaking point," said Ms Priya Raman, '
    "the regulator's chief executive.",
    '"We have looked at every alternative over the past eleven months.',
    '"There is no version of this decision that does not hurt someone."',
    "The AER declined to say whether the increase would be reviewed.",
]

CLEAN = "\n\n".join(QUOTE_TURN_PARAGRAPHS)

#: The same text with the attribution paragraph hard-wrapped, as a Factiva
#: export or PDF extraction would deliver it.
WRAPPED = CLEAN.replace(
    "said Ms Priya Raman, the regulator's",
    "said Ms Priya Raman, the\nregulator's",
)


def _quotes_by_actor(nlp, text):
    return {
        row["Actor Canonical Name"]: json.loads(row["Direct Segments (JSON)"])
        for row in newsvoice.extract_document("x", text, nlp)
    }


def test_clean_text_keeps_the_quotation_turn_with_one_speaker(nlp):
    by_actor = _quotes_by_actor(nlp, CLEAN)
    speaker = next(name for name in by_actor if name and "Raman" in name)
    assert len(by_actor[speaker]) == 3


def test_hard_wrap_misattributes_a_quotation(nlp):
    """The defect, pinned. One newline moves a quotation to another speaker.

    Nothing in the output marks this: the row is well-formed, the speaker is
    real and the quotation is real. Only the pairing is wrong.
    """
    by_actor = _quotes_by_actor(nlp, WRAPPED)
    speaker = next(name for name in by_actor if name and "Raman" in name)
    assert len(by_actor[speaker]) < 3, (
        "expected the hard wrap to cost the speaker quotations; if this fails "
        "the underlying sensitivity may have been fixed"
    )


def test_normalisation_restores_the_correct_attribution(nlp):
    """The whole point: normalise first and the output matches clean input."""
    repaired = normalise_paragraphs(WRAPPED)
    assert _quotes_by_actor(nlp, repaired) == _quotes_by_actor(nlp, CLEAN)


def test_offsets_index_the_normalised_text(nlp):
    """Normalised text is what you must keep; offsets do not map back."""
    repaired = normalise_paragraphs(WRAPPED)
    for row in newsvoice.extract_document("x", repaired, nlp):
        start = row["Evidence Start Character (0-based)"]
        end = row["Evidence End Character (exclusive)"]
        assert repaired[start:end] == row["Evidence Span"]
