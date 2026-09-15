"""Regression tests for the notebook's global state.

Three defects are pinned here:

1. **A module-level ``nlp``.** Every layer closed over one pipeline object
   created in the setup cell. That made the layers impossible to test in
   isolation, impossible to run against two models in one process, and made the
   cue matcher permanently bound to one vocabulary.

2. **A global ``warnings.filterwarnings("ignore")``.** Acceptable in a notebook,
   not in a library: importing the package silently suppressed every warning in
   the host process, including warnings from unrelated code.

3. **Hard-coded Google Drive paths.** Input and output locations were literals
   inside a cell that also called ``drive.mount``, so the code could not run
   outside one Colab account.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import pytest

import newsvoice
from newsvoice import cues

SRC_DIR = pathlib.Path(newsvoice.__file__).parent
SOURCES = sorted(SRC_DIR.glob("*.py"))


def test_import_does_not_touch_global_warnings_filters():
    """Importing a library must not reconfigure the host process's warnings."""
    import importlib

    before = list(warnings.filters)
    for module in ("newsvoice", "newsvoice.pipeline", "newsvoice.cues",
                   "newsvoice.actors", "newsvoice.segments"):
        importlib.reload(importlib.import_module(module))
    assert list(warnings.filters) == before


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_module_level_pipeline(path):
    """No module may bind a loaded pipeline at import time."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("nlp", "_nlp"):
                    pytest.fail(
                        f"{path.name} binds a module-level pipeline; pass it as "
                        f"an argument instead"
                    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_hardcoded_paths_or_colab(path):
    """No absolute user paths, drive mounts, or notebook-only imports."""
    text = path.read_text()
    for needle in ("/content/drive", "google.colab", "drive.mount",
                   "MyDrive", "C:\\\\", "OneDrive"):
        assert needle not in text, f"{path.name} contains {needle!r}"


def test_cue_matcher_works_across_independent_pipelines(nlp, second_nlp):
    """The matcher must bind to the Doc's vocabulary, not a captured global.

    The notebook built ``Matcher(nlp.vocab)`` once at import time. Passing a Doc
    from any other pipeline would have matched against the wrong vocabulary.
    """
    sentence = "According to the regulator, prices rose sharply."

    first = cues.find_cues(nlp(sentence))
    second = cues.find_cues(second_nlp(sentence))

    assert [t for _, t, _ in first] == [t for _, t, _ in second]
    assert any(tier == "phrase" for _, _, tier in first), (
        "the multi-word cue 'according to' was not matched"
    )


def test_layers_accept_an_explicit_pipeline(nlp):
    """``classify_completeness`` takes the parser as an argument, not a global."""
    from newsvoice.segments import classify_completeness

    assert classify_completeness(['"We will not be moving on this."'], nlp) == \
        "FULL_SENTENCE"
    assert classify_completeness([], nlp) == "NOT_APPLICABLE"


def test_public_api_is_explicit():
    """The documented entry points exist and are exported."""
    for name in ("extract_document", "extract_corpus", "load_pipeline",
                 "COLUMNS", "ExtractionConfig"):
        assert name in newsvoice.__all__
        assert hasattr(newsvoice, name)
