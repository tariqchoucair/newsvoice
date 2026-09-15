"""Layer 8 — global quote-to-cue assignment, and the sentence repairs it needs.

The assignment rules are the part of the pipeline most likely to be silently
wrong: a quotation attributed to the wrong cue still produces a well-formed row,
with a real speaker and a real quote, and nothing downstream flags it.
"""

from __future__ import annotations

import pytest

from newsvoice.assignment import assign_quotes, paragraph_units, sentence_units
from newsvoice.cues import find_cues
from newsvoice.quotes import find_quote_spans


def _setup(nlp, text):
    """Everything ``assign_quotes`` needs, derived from `text`."""
    doc = nlp(text)
    quote_spans = find_quote_spans(text)
    cues = find_cues(doc)
    cue_positions = [(i, c.start_char, c.end_char)
                     for i, (c, _, _) in enumerate(cues)]
    sents = sentence_units(doc, text, quote_spans)
    paras = paragraph_units(text)
    return doc, quote_spans, cues, cue_positions, sents, paras


# --------------------------------------------------------------------------
# Sentence unit repairs
# --------------------------------------------------------------------------

def test_sentence_boundary_inside_a_quotation_is_repaired(nlp):
    """A quoted utterance containing `?` or `.` is still one utterance."""
    text = '"Is it too late? No, I don\'t think so," she said. Prices rose.'
    doc = nlp(text)
    units = sentence_units(doc, text, find_quote_spans(text))
    first = text[units[0][0]:units[0][1]]
    assert "Is it too late?" in first and "I don't think so" in first


def test_trailing_attribution_is_reattached_to_its_quote(nlp):
    """`"Quote," he said.` is routinely split after the closing mark."""
    text = '"The plan is dead," he said. The minister disagreed.'
    doc = nlp(text)
    units = sentence_units(doc, text, find_quote_spans(text))
    first = text[units[0][0]:units[0][1]]
    assert "he said" in first


def test_sentence_units_never_empty(nlp):
    text = "No sentences here"
    assert sentence_units(nlp(text), text, []) != []


# --------------------------------------------------------------------------
# Paragraph units
# --------------------------------------------------------------------------

def test_paragraph_units_split_on_blank_lines():
    text = "Para one.\n\nPara two.\n\nPara three."
    assert [text[a:b] for a, b in paragraph_units(text)] == \
        ["Para one.", "Para two.", "Para three."]


def test_flat_text_has_no_paragraph_structure():
    """With no structure the sentence gap alone must bound continuation."""
    assert paragraph_units("One line only, no breaks at all.") == []


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------

def test_local_cue_owns_its_own_sentence_quote(nlp):
    """The rule that prevents greedy stealing across sentence boundaries."""
    text = ('Mr Dale said the review was complete. '
            'Ms Raman said "the numbers do not add up" in her submission.')
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assignment = assign_quotes(quotes, positions, sents, paras, text)

    assert len(quotes) == 1
    owner = cues[assignment[quotes[0]]][0]
    assert owner.text == "said"
    assert owner.start_char > text.index("Ms Raman")


def test_quote_only_sentence_continues_the_nearby_cue(nlp):
    text = ('Ms Raman said the offer was unacceptable.\n\n'
            '"This is an insult to every worker in the sector."')
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assignment = assign_quotes(quotes, positions, sents, paras, text)
    assert len(quotes) == 1
    assert quotes[0] in assignment


def test_a_quote_with_no_cue_anywhere_is_left_unassigned(nlp):
    """`reject` is not in the cue lexicon, so nothing here reports speech.

    Layer 8 assigns nothing. Attaching such quotations is the job of the
    pipeline's orphan-recovery pass, which works from established speakers
    rather than from cues.
    """
    text = ('Ms Raman rejected the offer outright.\n\n'
            '"This is an insult to every worker."')
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assert cues == []
    assert assign_quotes(quotes, positions, sents, paras, text) == {}


def test_paragraph_veto_blocks_a_distant_continuation(nlp):
    """A continuation may not cross more than one paragraph break."""
    text = ('Ms Raman rejected the offer.\n\n'
            'Prices rose by nine per cent.\n\n'
            'Costs continued to climb.\n\n'
            '"This is an insult to every worker."')
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assignment = assign_quotes(quotes, positions, sents, paras, text)
    assert quotes[0] not in assignment, (
        "a quotation three paragraphs from any cue should not be assigned"
    )


def test_every_quote_gets_at_most_one_owner(nlp, article_simple):
    """The output is a mapping, so this is structural — but worth pinning."""
    doc, quotes, cues, positions, sents, paras = _setup(nlp, article_simple)
    assignment = assign_quotes(quotes, positions, sents, paras, article_simple)
    assert set(assignment).issubset(set(quotes))
    assert all(0 <= index < len(cues) for index in assignment.values())


def test_a_cue_inside_a_quote_does_not_own_it(nlp):
    """In `told him to "come home"`, the quoted phrase is content, not a report."""
    text = 'She told him to "come home before dark" that evening.'
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assignment = assign_quotes(quotes, positions, sents, paras, text)
    for quote, cue_index in assignment.items():
        cue_start = positions[cue_index][1]
        assert not (quote[0] <= cue_start < quote[1])


def test_no_quotes_means_no_assignment(nlp):
    text = "Ms Raman said costs would rise sharply this year."
    doc, quotes, cues, positions, sents, paras = _setup(nlp, text)
    assert assign_quotes(quotes, positions, sents, paras, text) == {}
