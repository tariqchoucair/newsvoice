"""End-to-end tests for the orchestrator and the corpus API.

These are the tests that would have caught the notebook's structural problems:
that the output schema is stable, that documents which raise are recorded rather
than lost, that the corpus runner takes a DataFrame rather than reading a
hard-coded path, and that every offset in the output indexes the source text.
"""

from __future__ import annotations

import json

import pytest

import newsvoice
from newsvoice import COLUMNS, ExtractionConfig, extract_corpus, extract_document


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_every_row_has_exactly_the_documented_columns(nlp, article_simple):
    rows = extract_document("a1", article_simple, nlp)
    assert rows
    for row in rows:
        assert list(row.keys()) == COLUMNS


def test_article_id_is_copied_to_every_row(nlp, article_simple):
    rows = extract_document("article-42", article_simple, nlp)
    assert {row["Article Id"] for row in rows} == {"article-42"}


def test_json_columns_are_valid_json(nlp, article_simple):
    rows = extract_document("a1", article_simple, nlp)
    for row in rows:
        for column in ("Direct Segments (JSON)", "Indirect Segments (JSON)",
                       "Attribution Cues (JSON)"):
            assert isinstance(json.loads(row[column]), list)


def test_voice_type_matches_the_segments_present(nlp, article_simple):
    for row in extract_document("a1", article_simple, nlp):
        direct = json.loads(row["Direct Segments (JSON)"])
        indirect = json.loads(row["Indirect Segments (JSON)"])
        expected = ("HYBRID" if direct and indirect
                    else "DIRECT" if direct else "INDIRECT")
        assert row["Voice Type"] == expected


def test_word_counts_are_consistent(nlp, article_simple):
    for row in extract_document("a1", article_simple, nlp):
        assert row["Total Voiced Word Count"] == (
            row["Direct Word Count"] + row["Indirect Word Count"])
        assert row["Total Voice Segments"] == (
            len(json.loads(row["Direct Segments (JSON)"]))
            + len(json.loads(row["Indirect Segments (JSON)"])))


def test_categorical_columns_use_documented_values(nlp, article_simple):
    completeness = {"FULL_SENTENCE", "CLAUSE", "PHRASE", "SINGLE_WORD",
                    "NOT_APPLICABLE"}
    explicitness = {"NAMED", "DESCRIPTIVE_REFERENCE", "PRONOUN", "IMPLIED"}
    position = {"BEFORE_CONTENT", "AFTER_CONTENT", "INTERRUPTED", "EMBEDDED",
                "NOT_EXPLICIT"}
    entity = {"PERSON", "ORGANISATION", "GROUP", None}

    for row in extract_document("a1", article_simple, nlp):
        assert row["Quote Completeness"] in completeness
        assert row["Attribution Explicitness"] in explicitness
        assert row["Attribution Position"] in position
        assert row["Actor Entity Type"] in entity
        assert row["Voice Type"] in {"DIRECT", "INDIRECT", "HYBRID"}


# --------------------------------------------------------------------------
# Traceability — the property the whole design exists to preserve
# --------------------------------------------------------------------------

def test_evidence_offsets_index_the_source_text(nlp, article_simple):
    """Every row must be locatable in the discourse it came from."""
    for row in extract_document("a1", article_simple, nlp):
        start = row["Evidence Start Character (0-based)"]
        end = row["Evidence End Character (exclusive)"]
        assert 0 <= start < end <= len(article_simple)
        assert article_simple[start:end] == row["Evidence Span"]


def test_direct_segments_are_verbatim_substrings(nlp, article_simple):
    """Direct speech is selected from the source, never paraphrased."""
    for row in extract_document("a1", article_simple, nlp):
        for segment in json.loads(row["Direct Segments (JSON)"]):
            assert segment in article_simple


def test_indirect_segments_are_verbatim_substrings(nlp, article_simple):
    for row in extract_document("a1", article_simple, nlp):
        for segment in json.loads(row["Indirect Segments (JSON)"]):
            assert segment in article_simple


def test_document_position_is_a_percentage(nlp, article_simple):
    for row in extract_document("a1", article_simple, nlp):
        assert 0.0 <= row["Document Position (%)"] <= 100.0


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------

def test_named_speakers_are_recovered(nlp, article_simple):
    actors = {row["Actor Canonical Name"] for row in
              extract_document("a1", article_simple, nlp)}
    assert any(name and "Raman" in name for name in actors)


def test_headline_is_not_treated_as_an_attribution(nlp):
    """A telegraphic first line often carries a reporting verb."""
    text = ("Power costs to rise, regulator says\n\n"
            "Ms Priya Raman said bills would climb by nine per cent.")
    rows = extract_document("a1", text, nlp)
    for row in rows:
        assert row["Evidence Start Character (0-based)"] >= text.index("\n")


def test_document_without_voice_yields_one_marked_row(nlp):
    text = "Prices rose by nine per cent over the year to June.\n\nBills followed."
    rows = extract_document("a1", text, nlp)
    assert len(rows) == 1
    assert rows[0]["Processing Status"] == "complete_no_voice"
    assert rows[0]["Total Voice Segments"] == 0


def test_empty_document(nlp):
    rows = extract_document("a1", "", nlp)
    assert len(rows) == 1
    assert rows[0]["Processing Status"] == "complete_no_voice"


def test_rows_are_ordered_by_position_in_the_document(nlp, article_simple):
    starts = [row["Evidence Start Character (0-based)"]
              for row in extract_document("a1", article_simple, nlp)]
    assert starts == sorted(starts)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def test_default_config_reproduces_the_notebook_constants():
    config = newsvoice.DEFAULT_CONFIG
    assert config.context_window == 130
    assert config.orphan_recovery is True
    assert config.max_orphan_gap == 1200
    assert config.max_sentence_gap == 2
    assert config.headline_max_chars == 200
    assert config.dedupe_min_overlap == 0.6


def test_config_is_immutable():
    """Configuration recorded alongside results must not drift mid-run."""
    with pytest.raises(Exception):
        newsvoice.DEFAULT_CONFIG.context_window = 500


def test_config_as_dict_is_serialisable():
    assert json.loads(json.dumps(newsvoice.DEFAULT_CONFIG.as_dict()))


def test_context_window_changes_the_snippet(nlp, article_simple):
    narrow = extract_document("a1", article_simple, nlp,
                              config=ExtractionConfig(context_window=0))
    wide = extract_document("a1", article_simple, nlp,
                            config=ExtractionConfig(context_window=200))
    assert len(narrow[0]["Context Snippet"]) < len(wide[0]["Context Snippet"])


def test_orphan_recovery_can_be_switched_off(nlp, article_turn):
    """Off is the stricter attribution standard; it must not gain rows."""
    on = extract_document("a1", article_turn, nlp,
                          config=ExtractionConfig(orphan_recovery=True))
    off = extract_document("a1", article_turn, nlp,
                           config=ExtractionConfig(orphan_recovery=False))
    assert len(off) <= len(on)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def test_supplying_a_parsed_doc_gives_the_same_result(nlp, article_simple):
    """``nlp.pipe`` batching must not change the output."""
    without = extract_document("a1", article_simple, nlp)
    with_doc = extract_document("a1", article_simple, nlp,
                                doc=nlp(article_simple))
    assert without == with_doc


# --------------------------------------------------------------------------
# Corpus API
# --------------------------------------------------------------------------

@pytest.fixture
def frame(article_simple, article_turn):
    import pandas as pd
    return pd.DataFrame({
        "article_id": ["a1", "a2"],
        "full_text": [article_simple, article_turn],
    })


def test_extract_corpus_returns_a_dataframe_with_the_schema(nlp, frame):
    out = extract_corpus(frame, nlp)
    assert list(out.columns) == COLUMNS
    assert set(out["Article Id"]) == {"a1", "a2"}


def test_extract_corpus_matches_per_document_extraction(nlp, frame):
    combined = extract_corpus(frame, nlp)
    separate = sum(
        (len(extract_document(i, t, nlp))
         for i, t in zip(frame["article_id"], frame["full_text"])),
        0,
    )
    assert len(combined) == separate


def test_extract_corpus_accepts_custom_column_names(nlp, article_simple):
    import pandas as pd
    renamed = pd.DataFrame({"doc": ["x1"], "body": [article_simple]})
    out = extract_corpus(renamed, nlp, id_column="doc", text_column="body")
    assert set(out["Article Id"]) == {"x1"}


def test_missing_column_raises_a_helpful_keyerror(nlp, frame):
    with pytest.raises(KeyError, match="text"):
        extract_corpus(frame, nlp, text_column="text")


def test_invalid_on_error_is_rejected(nlp, frame):
    with pytest.raises(ValueError, match="on_error"):
        extract_corpus(frame, nlp, on_error="explode")


def test_a_failing_document_is_recorded_not_lost(nlp, frame, monkeypatch):
    """The notebook printed a traceback and appended an error row; keep that."""
    import newsvoice.pipeline as pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic parse failure")

    monkeypatch.setattr(pipeline, "find_quote_spans", boom)
    out = extract_corpus(frame, nlp, on_error="record")

    assert len(out) == 2
    assert (out["Processing Status"] == "error").all()
    assert out["Context Snippet"].str.contains("synthetic parse failure").all()


def test_on_error_raise_propagates(nlp, frame, monkeypatch):
    """What you want in tests and when debugging a new corpus."""
    import newsvoice.pipeline as pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic parse failure")

    monkeypatch.setattr(pipeline, "find_quote_spans", boom)
    with pytest.raises(RuntimeError, match="synthetic parse failure"):
        extract_corpus(frame, nlp, on_error="raise")


def test_empty_frame_produces_an_empty_table(nlp):
    import pandas as pd
    empty = pd.DataFrame({"article_id": [], "full_text": []})
    out = extract_corpus(empty, nlp)
    assert len(out) == 0
    assert list(out.columns) == COLUMNS
