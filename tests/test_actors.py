"""Layers 4 and 5 — actor typing, coreference and alias resolution.

The coreference tests are written against :func:`canonicalise` directly rather
than through the pipeline, because the resolver's contract is a list of mention
dicts in document order. Testing it in isolation keeps a parse failure upstream
from masquerading as a resolver failure.
"""

from __future__ import annotations

import pytest

from newsvoice import actors
from newsvoice.actors import canonicalise, entity_type, strip_titles
from newsvoice.cues import find_cues
from newsvoice.syntax import cue_subject


def speaker_type(nlp, phrase: str) -> str:
    doc = nlp(f"{phrase} said the results were clear.")
    cues = find_cues(doc)
    assert cues, f"no cue found for {phrase!r}"
    speaker = cue_subject(cues[0][0], doc)
    assert speaker is not None, f"no speaker found for {phrase!r}"
    return entity_type(speaker, doc)


# --------------------------------------------------------------------------
# Actor typing
# --------------------------------------------------------------------------

@pytest.mark.model
@pytest.mark.parametrize("phrase,expected", [
    ("Dr Alan Whitfield", "PERSON"),
    ("The Australian Energy Regulator", "ORGANISATION"),
    ("Researchers", "GROUP"),
    ("Students", "GROUP"),
    ("The committee", "ORGANISATION"),
])
def test_entity_type(nlp, phrase, expected):
    assert speaker_type(nlp, phrase) == expected


def test_plural_pronoun_is_a_group(nlp):
    assert speaker_type(nlp, "They") == "GROUP"


def test_singular_pronoun_is_a_person(nlp):
    assert speaker_type(nlp, "She") == "PERSON"


# --------------------------------------------------------------------------
# The divergent-table regression
# --------------------------------------------------------------------------

def test_reflexive_pronoun_is_typed_as_a_group(nlp):
    """Reflexives must be in the pronoun table for ``entity_type`` to work."""
    assert speaker_type(nlp, "Themselves") == "GROUP"


def test_layer_four_pronoun_table_would_have_mistyped_reflexives(nlp, monkeypatch):
    """Demonstrates that the two notebook tables produced different answers.

    This is the defect itself, reproduced. The Layer 4 cell's table omitted the
    four reflexive pronouns, so ``PRONOUN_GENDER.get("themselves")`` returned
    ``None`` and ``entity_type`` fell through to PERSON. The Layer 5 cell's table
    included them and returned GROUP. Which one ``entity_type`` saw depended on
    which cell had most recently been executed — a difference invisible in the
    output CSV.
    """
    layer_four_table = {
        "he": "m", "him": "m", "his": "m",
        "she": "f", "her": "f", "hers": "f",
        "they": "p", "them": "p", "their": "p",
        "it": "n", "its": "n",
    }
    monkeypatch.setattr(actors, "PRONOUN_GENDER", layer_four_table)
    assert speaker_type(nlp, "Themselves") == "PERSON", (
        "expected the old table to reproduce the bug; if this fails the "
        "divergence no longer has observable consequences and this test can go"
    )


# --------------------------------------------------------------------------
# Titles
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("Dr Alan Whitfield", "Alan Whitfield"),
    ("Ms Priya Raman", "Priya Raman"),
    ("Senator Dale", "Dale"),
    ("Alan Whitfield", "Alan Whitfield"),
])
def test_strip_titles(name, expected):
    assert strip_titles(name) == expected


def test_strip_titles_never_returns_empty(): 
    """A mention consisting only of a title must not collapse to nothing."""
    assert strip_titles("Dr") == "Dr"


# --------------------------------------------------------------------------
# Coreference and aliasing
# --------------------------------------------------------------------------

def test_definite_epithet_resolves_to_a_named_person():
    mentions = [
        {"text": "The 21-year-old actress", "type": "PERSON",
         "start": 0, "end": 23, "definite": True},
        {"text": "Jenna Ortega", "type": "PERSON", "start": 120, "end": 132},
    ]
    assert canonicalise(mentions) == ["Jenna Ortega", "Jenna Ortega"]


def test_indefinite_epithet_introduces_a_new_referent():
    """`A company spokesperson` is not anaphoric.

    Resolving it puts one person's words in another's mouth — the failure mode
    is crediting a denial to the person being denied.
    """
    mentions = [
        {"text": "Jenna Ortega", "type": "PERSON", "start": 0, "end": 12},
        {"text": "A company spokesperson", "type": "PERSON",
         "start": 100, "end": 122, "definite": False},
    ]
    resolved = canonicalise(mentions)
    assert resolved[1] != "Jenna Ortega"


def test_pronoun_resolves_to_the_nearest_matching_person():
    mentions = [
        {"text": "Jenna Ortega", "type": "PERSON", "start": 0, "end": 12},
        {"text": "she", "type": "PERSON", "start": 60, "end": 63},
    ]
    assert canonicalise(mentions)[1] == "Jenna Ortega"


def test_surname_expands_to_the_full_name():
    mentions = [
        {"text": "Jenna Ortega", "type": "PERSON", "start": 0, "end": 12},
        {"text": "Ms Ortega", "type": "PERSON", "start": 200, "end": 209},
    ]
    assert canonicalise(mentions) == ["Jenna Ortega", "Jenna Ortega"]


def test_one_speaker_gets_one_canonical_name():
    """Two canonical names for one speaker quietly breaks per-actor aggregation."""
    mentions = [
        {"text": "Dr Angel Zhong", "type": "PERSON", "start": 0, "end": 14},
        {"text": "Dr Zhong", "type": "PERSON", "start": 100, "end": 108},
        {"text": "Zhong", "type": "PERSON", "start": 200, "end": 205},
        {"text": "she", "type": "PERSON", "start": 300, "end": 303},
    ]
    assert len(set(canonicalise(mentions))) == 1


def test_definite_organisation_resolves_to_the_named_body():
    mentions = [
        {"text": "Northfield University", "type": "ORGANISATION",
         "start": 0, "end": 21},
        {"text": "The company", "type": "ORGANISATION",
         "start": 400, "end": 411},
    ]
    assert canonicalise(mentions)[1] == "Northfield University"


def test_acronym_expands_to_the_full_organisation_name():
    mentions = [
        {"text": "Australian Energy Regulator", "type": "ORGANISATION",
         "start": 0, "end": 27},
        {"text": "AER", "type": "ORGANISATION", "start": 200, "end": 203},
    ]
    assert canonicalise(mentions)[1] == "Australian Energy Regulator"


def test_acronym_generation_is_derailed_by_a_leading_determiner():
    """Documents the cause of the failure below. Pure string handling.

    ``_acronym_of`` takes the initial of every capitalised word, so a retained
    ``The`` contributes a ``T`` and the generated key becomes ``TAER``.
    """
    assert actors._acronym_of("Australian Energy Regulator") == "AER"
    assert actors._acronym_of("The Australian Energy Regulator") == "TAER"


@pytest.mark.xfail(
    reason="KNOWN LIMITATION: an organisation first named with a leading "
           "determiner generates the wrong acronym key, so later mentions of "
           "the acronym do not resolve. See docs/KNOWN_ISSUES.md.",
    strict=True,
)
def test_acronym_resolves_when_the_inventory_entry_keeps_its_determiner():
    mentions = [
        {"text": "The Australian Energy Regulator", "type": "ORGANISATION",
         "start": 0, "end": 31},
        {"text": "AER", "type": "ORGANISATION", "start": 200, "end": 203},
    ]
    assert canonicalise(mentions)[1] == "The Australian Energy Regulator"


def test_background_supplies_antecedents_never_seen_as_speakers():
    """A name appearing only as the OBJECT of a cue must still be available.

    Without the background inventory, "according to the study's lead author
    Dr Angel Zhong" never registers, and a later "Dr Zhong" cannot expand.
    """
    background = [{"text": "Angel Zhong", "type": "PERSON",
                   "start": 10, "end": 21}]
    mentions = [{"text": "Dr Zhong", "type": "PERSON", "start": 300, "end": 308}]
    assert canonicalise(mentions, background=background) == ["Angel Zhong"]


def test_resolver_returns_one_entry_per_mention():
    """Downstream code zips this with the row list; length must be exact."""
    mentions = [
        {"text": "Jenna Ortega", "type": "PERSON", "start": 0, "end": 12},
        {"text": "she", "type": "PERSON", "start": 60, "end": 63},
        {"text": "", "type": "PERSON", "start": None, "end": None},
    ]
    assert len(canonicalise(mentions)) == len(mentions)


def test_empty_mention_list():
    assert canonicalise([]) == []
