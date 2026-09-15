"""Shared fixtures.

Every article used in the tests is synthetic. The notebook's regression cell
keyed its assertions to phrases from four specific copyrighted news articles in
a private corpus, so it could not run anywhere else and its failures could not be
reproduced. The fixtures below were written to exercise the same behaviours —
inverted attribution, passive reporting without an agent, epithet resolution,
acronym expansion, multi-paragraph quotation turns — using text that ships with
the package.

Tests that need a parse use whichever model is installed, preferring the
transformer model the pipeline was tuned on and falling back to the small model
so the suite runs on any machine. Assertions sensitive to parse quality carry
``@pytest.mark.model``; run ``pytest -m "not model"`` to skip them.
"""

from __future__ import annotations

import pytest

MODEL_PREFERENCE = ("en_core_web_trf", "en_core_web_sm")


@pytest.fixture(scope="session")
def model_name() -> str:
    """Name of the best installed spaCy model, skipping the suite if none is."""
    import spacy.util

    for candidate in MODEL_PREFERENCE:
        if spacy.util.is_package(candidate):
            return candidate
    pytest.skip(
        "no spaCy English model installed; run "
        "`python -m spacy download en_core_web_sm`"
    )


@pytest.fixture(scope="session")
def nlp(model_name):
    """A loaded pipeline, shared across the session because loading is slow."""
    import newsvoice

    return newsvoice.load_pipeline(model_name, prefer_gpu=False)


@pytest.fixture(scope="session")
def second_nlp(model_name):
    """A second, independently loaded pipeline with its own ``Vocab``.

    Used to prove the cue matcher is not cached against one global vocabulary,
    which is what the notebook's import-time ``Matcher(nlp.vocab)`` did.
    """
    import newsvoice

    return newsvoice.load_pipeline(model_name, prefer_gpu=False)


# --------------------------------------------------------------------------
# Synthetic articles
# --------------------------------------------------------------------------

#: Straightforward reporting: named speaker, direct quote, inverted attribution.
ARTICLE_SIMPLE = """Power costs set to rise, inquiry told

The Australian Energy Regulator said household bills would rise by nine per cent.

"Families are already stretched to breaking point," said Ms Priya Raman, the
regulator's chief executive. "We do not take this decision lightly."

Dr Alan Whitfield, of Northfield University, warned of further increases.
"""

#: Multi-paragraph quotation turn where the closing mark is omitted until the
#: final paragraph, plus a quote-only paragraph continuing the same speaker.
ARTICLE_TURN = """Union rejects wage offer

Ms Priya Raman rejected the offer outright.

"This is an insult to every worker in the sector.

"We have been negotiating in good faith for eleven months, and this is what we
are handed."

"We will be recommending our members vote no."

The company declined to comment.
"""

#: Epithets, acronyms and metonymy, for the coreference layer.
ARTICLE_ALIASES = """Regulator faces scrutiny over report

The Australian Energy Regulator released its findings on Tuesday. The report
said costs had been understated for a decade.

"The modelling was simply wrong," said Dr Alan Whitfield.

The Northfield University economist added that the error was avoidable.

The AER declined to say whether anyone would be disciplined.
"""

#: Constructions that must NOT produce an attribution.
ARTICLE_NEGATIVES = """Minister unveils plan as critics circle

The minister unveiled the policy on Monday.

The department did not reveal the modelling behind the forecast.

Asked about the cost, Mr Dale remained silent.

Think carefully about what this means for your bill.

The outage will continue getting worse through the evening.
"""


@pytest.fixture
def article_simple() -> str:
    return ARTICLE_SIMPLE


@pytest.fixture
def article_turn() -> str:
    return ARTICLE_TURN


@pytest.fixture
def article_aliases() -> str:
    return ARTICLE_ALIASES


@pytest.fixture
def article_negatives() -> str:
    return ARTICLE_NEGATIVES
