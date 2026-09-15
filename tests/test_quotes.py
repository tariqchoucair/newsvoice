"""Layer 1 — quote span detection.

This layer is pure string processing with no parse, so every test here is
deterministic and model-independent.
"""

from __future__ import annotations

import pytest

from newsvoice.quotes import find_quote_spans, overlaps


def quoted(text: str) -> list[str]:
    """The substrings Layer 1 marks as direct speech."""
    return [text[s:e] for s, e in find_quote_spans(text)]


# --------------------------------------------------------------------------
# Basic pairing
# --------------------------------------------------------------------------

def test_straight_double_quotes():
    text = 'She said "the plan is dead" on Tuesday.'
    assert quoted(text) == ['"the plan is dead"']


def test_curly_double_quotes():
    text = "\u201cThe plan is dead,\u201d she said."
    assert quoted(text) == ["\u201cThe plan is dead,\u201d"]


def test_mixed_straight_opener_curly_closer():
    """Copy-paste from CMS routinely mixes the two families within one quote."""
    text = "\"The plan is dead,\u201d she said."
    assert quoted(text) == ["\"The plan is dead,\u201d"]


def test_curly_single_quotes():
    text = "She called it \u2018a disgrace\u2019 in the report."
    assert quoted(text) == ["\u2018a disgrace\u2019"]


def test_several_quotes_in_one_sentence_stay_separate():
    text = 'He said "one" but she said "two" and they said "three".'
    assert quoted(text) == ['"one"', '"two"', '"three"']


def test_offsets_index_the_source_string():
    """Offsets must be usable directly; downstream traceability depends on it."""
    text = 'She said "the plan is dead" on Tuesday.'
    (start, end), = find_quote_spans(text)
    assert text[start:end] == '"the plan is dead"'
    assert text[start] == '"' and text[end - 1] == '"'


# --------------------------------------------------------------------------
# Apostrophe disambiguation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The company's plan drew criticism.",
    "It wasn't costed and they didn't say why.",
    "The workers' entitlements were cut.",
    "She said the government's response wasn't adequate.",
])
def test_apostrophes_are_not_quote_marks(text):
    """Possessives and contractions must never open a quotation."""
    assert quoted(text) == []


def test_straight_single_quote_still_works_as_a_quote_mark():
    text = "The report called it 'a failure of governance' last year."
    assert quoted(text) == ["'a failure of governance'"]


# --------------------------------------------------------------------------
# Unbalanced marks
# --------------------------------------------------------------------------

def test_unbalanced_quote_damage_is_contained_to_its_line():
    """A stray opener must not flip the parity of every later pair.

    This is the whole reason pairing state resets at each newline. Without it a
    single unbalanced mark corrupts the remainder of the document.
    """
    text = ('"An unbalanced opener that never closes\n'
            'Clean line with "a proper quote" in it.\n'
            'Another clean line with "a second quote" here.')
    assert '"a proper quote"' in quoted(text)
    assert '"a second quote"' in quoted(text)


def test_line_initial_opener_without_closer_is_kept_as_direct_speech():
    """A paragraph opening a quotation turn is direct speech even unclosed."""
    text = '"We have been negotiating in good faith for eleven months'
    assert quoted(text) == [text]


def test_unclosed_quote_ends_at_a_trailing_attribution():
    """`"Quote, Mr Walton said ...` — the attribution stays outside the quote.

    The source omitted the closing mark. Ending the span at the comma keeps the
    cue and speaker visible to Layers 2 and 3 rather than swallowing them,
    without inventing punctuation that was not in the source.
    """
    text = '"This is real manufacturing, Mr Walton said on Tuesday.'
    assert quoted(text) == ['"This is real manufacturing,']


def test_no_span_crosses_a_newline():
    text = ('"An opener here\n'
            'and a closer down here"')
    assert all("\n" not in q for q in quoted(text))


# --------------------------------------------------------------------------
# Hygiene of the returned spans
# --------------------------------------------------------------------------

def test_spans_are_sorted_and_non_overlapping():
    text = ('"First quote" then prose then "second quote".\n'
            'A new line with \u201ca third\u201d and \'a fourth\' quote.')
    spans = find_quote_spans(text)
    assert spans == sorted(spans)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 <= b0, "spans overlap"


def test_empty_and_quoteless_text():
    assert find_quote_spans("") == []
    assert find_quote_spans("Nothing quoted here at all.") == []


def test_very_long_quote_is_rejected():
    """The 2,000-character ceiling guards against runaway spans."""
    text = '"' + "word " * 600 + '"'
    assert find_quote_spans(text) == []


# --------------------------------------------------------------------------
# overlaps()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ((0, 10), (5, 15), True),
    ((0, 10), (10, 20), False),    # half-open: touching is not overlapping
    ((5, 15), (0, 10), True),
    ((0, 10), (2, 4), True),       # containment
    ((0, 1), (1, 2), False),
])
def test_overlaps(a, b, expected):
    assert overlaps(a, b) is expected
