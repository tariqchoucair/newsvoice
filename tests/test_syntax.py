"""Layer 3 — speaker and reported-content resolution from the parse.

The passive cases matter most. In news prose the grammatical subject of a
passive reporting verb is usually the *recipient* or the *target*, not the
source, and attributing to it puts words in the wrong mouth — a silent error
that no downstream check catches.
"""

from __future__ import annotations

import pytest

from newsvoice.cues import find_cues
from newsvoice.syntax import cue_content, cue_subject, is_finite_clause


def first_speaker(nlp, sentence: str):
    """Surface text of the speaker for the first cue, or None."""
    doc = nlp(sentence)
    cues = find_cues(doc)
    if not cues:
        return "(no cue)"
    speaker = cue_subject(cues[0][0], doc)
    return None if speaker is None else speaker.text


# --------------------------------------------------------------------------
# Ordinary subjects
# --------------------------------------------------------------------------

def test_canonical_subject(nlp):
    assert first_speaker(nlp, "Ms Raman said costs would rise.") == "Ms Raman"


def test_inverted_attribution(nlp):
    """`"Quote," said Ms Raman.` — post-verbal subject of a finite verb."""
    assert first_speaker(nlp, '"Costs will rise," said Ms Raman.') == "Ms Raman"


def test_honorific_is_kept_in_the_surface_mention(nlp):
    """`Speaker Mention` records what the text said; stripping happens later."""
    assert first_speaker(nlp, "Dr Alan Whitfield said the modelling was wrong.") \
        == "Dr Alan Whitfield"


@pytest.mark.model
def test_appositive_affiliation_does_not_displace_the_person(nlp):
    speaker = first_speaker(
        nlp, "Dr Alan Whitfield, of Northfield University, warned of blackouts.")
    assert speaker is not None and "Whitfield" in speaker


# --------------------------------------------------------------------------
# Passives — the error-prone cases
# --------------------------------------------------------------------------

def test_passive_told_without_agent_has_no_speaker(nlp):
    """`Marcus was told` names the recipient. Attributing to him is wrong."""
    assert first_speaker(nlp, "Marcus was told the plan had failed.") is None


@pytest.mark.model
def test_passive_told_with_by_agent_uses_the_agent(nlp):
    speaker = first_speaker(
        nlp, "Marcus was told by the regulator that the plan had failed.")
    assert speaker is not None and "regulator" in speaker


def test_fronted_passive_question_has_no_speaker(nlp):
    """`Asked about X, Mr Dale ...` — Dale answers, the asker is unnamed."""
    assert first_speaker(nlp, "Asked about the cost, Mr Dale remained silent.") \
        is None


def test_passive_stance_verb_without_agent_has_no_speaker(nlp):
    """`The unions were slammed` names the target of criticism, not its source."""
    assert first_speaker(nlp, "The unions were slammed over the claim.") is None


def test_quoted_as_saying_keeps_the_passive_subject_as_speaker(nlp):
    """A lexical exception: here the passive subject IS the source."""
    assert first_speaker(
        nlp, "Mr Farrow was quoted as saying the offer was an insult.") \
        == "Mr Farrow"


# --------------------------------------------------------------------------
# Inherited subjects
# --------------------------------------------------------------------------

@pytest.mark.model
def test_conjoined_cue_inherits_the_governing_subject(nlp):
    speaker = first_speaker(
        nlp, "The regulator released the report and urged consumers to switch.")
    assert speaker is not None and "regulator" in speaker


def test_according_to_takes_the_following_noun_phrase(nlp):
    speaker = first_speaker(
        nlp, "According to the Australian Energy Regulator, prices rose.")
    assert speaker is not None and "Australian Energy Regulator" in speaker


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

def test_reported_clause_is_the_content(nlp):
    doc = nlp("Ms Raman said costs would rise sharply.")
    content = cue_content(find_cues(doc)[0][0], doc)
    assert content is not None and "costs would rise" in content.text


@pytest.mark.model
def test_warn_of_frame_is_recognised(nlp):
    """`warned of blackouts` puts the content in a prepositional phrase."""
    doc = nlp("The entrepreneur warned of blackouts across the state.")
    content = cue_content(find_cues(doc)[0][0], doc, allow_np=True)
    assert content is not None and "blackouts" in content.text


def test_content_does_not_cross_a_paragraph_break(nlp):
    """Paragraph boundaries are authorial structure and override the parse."""
    text = "Ms Raman said costs would rise\n\nA separate paragraph follows here."
    doc = nlp(text)
    content = cue_content(find_cues(doc)[0][0], doc)
    if content is not None:
        assert "separate paragraph" not in content.text


# --------------------------------------------------------------------------
# is_finite_clause
# --------------------------------------------------------------------------

def test_is_finite_clause_requires_a_verb_and_a_subject(nlp):
    doc = nlp("Ms Raman said costs would rise.")
    content = cue_content(find_cues(doc)[0][0], doc)
    assert is_finite_clause(content) is True


def test_is_finite_clause_rejects_none(nlp):
    assert is_finite_clause(None) is False
