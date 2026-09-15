"""Layer 2 — attribution cue detection.

Most of this layer is guards: constructions that carry a reporting lemma but are
not attributions. Each guard gets a positive case (the cue fires) and a negative
case (it does not), because a guard that suppresses everything passes a
negative-only test.
"""

from __future__ import annotations

import pytest

from newsvoice.cues import find_cues


def cue_texts(nlp, sentence: str) -> list[str]:
    return [text for _, text, _ in find_cues(nlp(sentence))]


def cue_tiers(nlp, sentence: str) -> list[str]:
    return [tier for _, _, tier in find_cues(nlp(sentence))]


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------

def test_core_cue_is_detected(nlp):
    assert cue_texts(nlp, "Ms Raman said costs would rise.") == ["said"]
    assert cue_tiers(nlp, "Ms Raman said costs would rise.") == ["core"]


def test_multiword_phrase_cue(nlp):
    assert cue_tiers(nlp, "According to the regulator, prices rose.") == ["phrase"]


def test_overlapping_phrase_cues_collapse_to_one_maximal_match(nlp):
    """`was quoted`, `quoted as saying` and the full phrase all match.

    Emitting all three would duplicate the attribution and its content.
    """
    cues = cue_texts(nlp, "Mr Farrow was quoted as saying the offer was an insult.")
    assert cues == ["was quoted as saying"]


# --------------------------------------------------------------------------
# Negation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "The department did not reveal the modelling.",
    "He never said the plan was funded.",
    "The regulator didn't say what the cost would be.",
])
def test_negated_cue_is_not_an_attribution(nlp, sentence):
    """A non-disclosure gives no one voice."""
    assert cue_texts(nlp, sentence) == []


def test_positive_counterpart_of_negation_still_fires(nlp):
    assert cue_texts(nlp, "The department revealed the modelling.") == ["revealed"]


def test_refused_to_comment_produces_no_attribution(nlp):
    """Caught by the non-finite action-sense guard rather than by negation."""
    assert cue_texts(nlp, "The regulator refused to comment on the decision.") == []


@pytest.mark.xfail(
    reason="KNOWN LIMITATION: the negation guard looks for not/n't/never, so "
           "lexical refusals such as 'declined to say' are not suppressed and "
           "produce an attribution for a non-disclosure. 'refused to comment' "
           "is caught only incidentally, by the action-sense guard. See "
           "docs/KNOWN_ISSUES.md.",
    strict=True,
)
def test_declined_to_say_produces_no_attribution(nlp):
    assert cue_texts(
        nlp, "The regulator declined to say whether it would be reviewed.") == []


# --------------------------------------------------------------------------
# Lexical guards
# --------------------------------------------------------------------------

def test_unveil_is_event_narration_not_attribution(nlp):
    """Otherwise `unveiled the policy` steals a quotation from a nearby cue."""
    assert cue_texts(nlp, "The minister unveiled the policy on Monday.") == []


def test_hit_is_reportive_only_as_the_phrasal_stance_cue(nlp):
    assert cue_texts(nlp, "Ms Raman hit out at the plan.") == ["hit"]
    assert cue_texts(nlp, "The car hit the barrier.") == []


@pytest.mark.model
def test_continue_is_aspectual_before_a_verbal_complement(nlp):
    assert cue_texts(nlp, "The outage will continue getting worse.") == []
    assert cue_texts(nlp, "The outage will continue to worsen.") == []


@pytest.mark.model
def test_continue_is_reportive_when_the_quote_is_its_own_sentence(nlp):
    assert cue_texts(nlp, 'Mr Dale continued. "We are not finished."') == \
        ["continued"]


@pytest.mark.model
def test_continue_survives_inverted_attribution(nlp):
    """The aspectual guard only looks at complements that FOLLOW the cue.

    In `"...," Mr Dale continued.` the clause precedes the cue, so the guard
    correctly declines to fire.
    """
    assert cue_texts(nlp, '"We are not finished," Mr Dale continued.') == \
        ["continued"]


@pytest.mark.model
@pytest.mark.xfail(
    reason="KNOWN LIMITATION: a colon-introduced continuation attaches the "
           "quoted clause as a ccomp that follows the cue, which is exactly "
           "the shape the aspectual guard suppresses. Reported speech and "
           "aspectual complements are not distinguishable at that point. "
           "Widening the guard risks re-admitting 'continue getting worse'; "
           "see docs/KNOWN_ISSUES.md before changing it.",
    strict=False,
)
def test_continue_with_colon_introduced_quotation(nlp):
    assert cue_texts(nlp, 'Mr Dale continued: "We are not finished."') == \
        ["continued"]


@pytest.mark.model
def test_nonfinite_action_sense_is_not_a_reporting_event(nlp):
    assert cue_texts(nlp, "He would not be able to respond to the claims.") == []


def test_finite_respond_is_a_reporting_event(nlp):
    assert cue_texts(nlp, "She responded that the figures were wrong.") == \
        ["responded"]


@pytest.mark.model
def test_nominal_participle_is_not_a_cue(nlp):
    """`the Coalition's proposed seven reactors` modifies a noun."""
    assert cue_texts(nlp, "The Coalition's proposed seven reactors drew criticism.") \
        == []


# --------------------------------------------------------------------------
# Ordering and structure
# --------------------------------------------------------------------------

def test_cues_are_returned_in_document_order(nlp):
    sentence = ("Ms Raman said costs would rise, and Mr Dale added that "
                "bills would follow.")
    spans = [span.start_char for span, _, _ in find_cues(nlp(sentence))]
    assert spans == sorted(spans)


def test_no_cue_in_text_without_reporting(nlp):
    assert cue_texts(nlp, "Prices rose by nine per cent over the year.") == []
