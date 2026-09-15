"""Layers 6 and 7 — categorical columns and indirect segment construction."""

from __future__ import annotations

import pytest

from newsvoice.cues import find_cues
from newsvoice.segments import (
    build_indirect_segments,
    classify_completeness,
    classify_explicitness,
    classify_position,
    is_contentful,
)
from newsvoice.syntax import cue_subject


# --------------------------------------------------------------------------
# Quote Completeness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("quote,expected", [
    ('"We will not be moving on this."', "FULL_SENTENCE"),
    ('"transcended language barriers"', "CLAUSE"),
    ('"a good tool"', "PHRASE"),
    ('"Extraordinary"', "SINGLE_WORD"),
])
def test_classify_completeness(nlp, quote, expected):
    assert classify_completeness([quote], nlp) == expected


def test_completeness_of_no_direct_speech(nlp):
    assert classify_completeness([], nlp) == "NOT_APPLICABLE"


def test_completeness_of_empty_quote_marks(nlp):
    assert classify_completeness(['""'], nlp) == "NOT_APPLICABLE"


def test_completeness_uses_the_longest_segment(nlp):
    """A row with several quotes is described by its most complete one."""
    segments = ['"Extraordinary"', '"We will not be moving on this."']
    assert classify_completeness(segments, nlp) == "FULL_SENTENCE"


# --------------------------------------------------------------------------
# Attribution Explicitness
# --------------------------------------------------------------------------

def test_explicitness_named(nlp):
    doc = nlp("Ms Priya Raman said costs would rise.")
    speaker = cue_subject(find_cues(doc)[0][0], doc)
    assert classify_explicitness(speaker, doc) == "NAMED"


def test_explicitness_pronoun(nlp):
    doc = nlp("She said costs would rise.")
    speaker = cue_subject(find_cues(doc)[0][0], doc)
    assert classify_explicitness(speaker, doc) == "PRONOUN"


def test_explicitness_implied_when_no_speaker(nlp):
    assert classify_explicitness(None, nlp("Anything at all.")) == "IMPLIED"


@pytest.mark.model
def test_explicitness_descriptive_reference(nlp):
    doc = nlp("The economist said costs would rise.")
    speaker = cue_subject(find_cues(doc)[0][0], doc)
    assert classify_explicitness(speaker, doc) == "DESCRIPTIVE_REFERENCE"


# --------------------------------------------------------------------------
# Attribution Position
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cue,content,expected", [
    ((0, 5), [(10, 30)], "BEFORE_CONTENT"),
    ((40, 45), [(10, 30)], "AFTER_CONTENT"),
    ((15, 20), [(10, 30)], "EMBEDDED"),
])
def test_classify_position(cue, content, expected):
    assert classify_position(cue[0], cue[1], content) == expected


def test_position_without_a_cue():
    assert classify_position(None, None, [(10, 30)]) == "NOT_EXPLICIT"


def test_position_without_content():
    assert classify_position(0, 5, []) == "NOT_EXPLICIT"


# --------------------------------------------------------------------------
# Indirect segments
# --------------------------------------------------------------------------

def test_complementizer_that_is_stripped(nlp):
    """`that` is semantically empty and is not part of the proposition."""
    text = "Ms Raman said that costs would rise sharply."
    doc = nlp(text)
    cue = find_cues(doc)[0][0]
    speaker = cue_subject(cue, doc)
    segments = build_indirect_segments(
        doc, (0, len(text)),
        [(cue.start_char, cue.end_char),
         (speaker.start_char, speaker.end_char)])
    assert segments, "expected an indirect segment"
    assert not segments[0][0].lower().startswith("that ")


@pytest.mark.model
def test_modal_complementizers_are_preserved(nlp):
    """`if` and `whether` carry the clause's modality and must survive.

    "predict if a market will move" is not "predict a market will move".
    """
    text = "The regulator asked whether costs would rise."
    doc = nlp(text)
    cue = find_cues(doc)[0][0]
    speaker = cue_subject(cue, doc)
    cuts = [(cue.start_char, cue.end_char)]
    if speaker is not None:
        cuts.append((speaker.start_char, speaker.end_char))
    segments = build_indirect_segments(doc, (0, len(text)), cuts)
    assert any("whether" in seg.lower() for seg, _, _ in segments)


def test_segments_carry_faithful_offsets(nlp):
    """Offsets must index the source exactly; traceability depends on it."""
    text = "Ms Raman said costs would rise sharply this year."
    doc = nlp(text)
    cue = find_cues(doc)[0][0]
    segments = build_indirect_segments(
        doc, (0, len(text)), [(cue.start_char, cue.end_char)])
    for seg_text, start, end in segments:
        assert text[start:end] == seg_text


def test_debris_is_not_a_segment(nlp):
    """A stranded determiner is not a proposition."""
    doc = nlp("The system works.")
    determiner = [t for t in doc if t.pos_ == "DET"]
    assert is_contentful(determiner) is False


def test_empty_token_list_is_not_contentful():
    assert is_contentful([]) is False
