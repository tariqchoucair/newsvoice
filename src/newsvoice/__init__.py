"""newsvoice — voice and attribution extraction from news text.

Finds who is given voice in a news article, what they are reported as saying,
and how that attribution is made. Output is one row per attribution with a
22-column schema, including character offsets back into the source text so that
every extracted segment can be traced to the discourse it came from.

Quick start
-----------
::

    import pandas as pd
    import newsvoice

    nlp = newsvoice.load_pipeline()
    rows = newsvoice.extract_document("a1", article_text, nlp)

    articles = pd.read_csv("articles.csv")
    quotes = newsvoice.extract_corpus(
        articles, nlp, id_column="article_id", text_column="full_text"
    )

The eight layers are importable individually — :mod:`newsvoice.quotes`,
:mod:`newsvoice.cues`, :mod:`newsvoice.syntax`, :mod:`newsvoice.actors`,
:mod:`newsvoice.segments`, :mod:`newsvoice.assignment` — for inspecting or
replacing one stage of the pipeline.
"""

from __future__ import annotations

from typing import Optional

from .pipeline import (
    COLUMNS,
    DEFAULT_CONFIG,
    ExtractionConfig,
    extract_corpus,
    extract_document,
)

__all__ = [
    "COLUMNS",
    "DEFAULT_CONFIG",
    "ExtractionConfig",
    "extract_corpus",
    "extract_document",
    "load_pipeline",
    "DEFAULT_MODEL",
    "__version__",
]

__version__ = "1.0.0"

#: The model the pipeline was developed and validated against. The dependency
#: heuristics in :mod:`newsvoice.syntax` were tuned on transformer parses;
#: smaller models will run but attribution accuracy degrades, most visibly on
#: inverted attribution and on clausal complement selection.
DEFAULT_MODEL = "en_core_web_trf"


def load_pipeline(
    model: str = DEFAULT_MODEL,
    prefer_gpu: bool = True,
    max_length: int = 2_000_000,
):
    """Load and configure a spaCy pipeline for extraction.

    Parameters
    ----------
    model
        Name of an installed spaCy model. Defaults to :data:`DEFAULT_MODEL`.
    prefer_gpu
        Request GPU allocation if one is available. Falls back to CPU silently,
        as ``spacy.prefer_gpu`` does.
    max_length
        Raised well above the spaCy default so that long articles do not need
        chunking.

    Returns
    -------
    spacy.language.Language

    Raises
    ------
    OSError
        If `model` is not installed, with the download command in the message.
    """
    import spacy

    if prefer_gpu:
        spacy.prefer_gpu()
    try:
        nlp = spacy.load(model)
    except OSError as exc:
        raise OSError(
            f"spaCy model {model!r} is not installed. Install it with:\n"
            f"    python -m spacy download {model}"
        ) from exc
    nlp.max_length = max_length
    return nlp
