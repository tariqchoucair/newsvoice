"""Command-line interface.

The notebook's input cell mounted Google Drive and read from a literal path
under one account. These tests pin the replacement: every location is an
argument, and a missing input fails with a message rather than a traceback.
"""

from __future__ import annotations

import pytest

from newsvoice.cli import build_parser, main


@pytest.fixture
def csv_input(tmp_path, article_simple, article_turn):
    import pandas as pd

    path = tmp_path / "articles.csv"
    pd.DataFrame({
        "article_id": ["a1", "a2"],
        "full_text": [article_simple, article_turn],
    }).to_csv(path, index=False)
    return path


def test_parser_requires_an_output(csv_input):
    with pytest.raises(SystemExit):
        build_parser().parse_args([str(csv_input)])


def test_parser_defaults_match_the_library():
    import newsvoice

    args = build_parser().parse_args(["in.csv", "-o", "out.csv"])
    assert args.id_column == "article_id"
    assert args.text_column == "full_text"
    assert args.model == newsvoice.DEFAULT_MODEL
    assert args.context_window == newsvoice.DEFAULT_CONFIG.context_window
    assert args.max_sentence_gap == newsvoice.DEFAULT_CONFIG.max_sentence_gap


def test_missing_input_file_exits_with_a_message(tmp_path, capsys):
    code = main([str(tmp_path / "nope.csv"), "-o", str(tmp_path / "out.csv")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_end_to_end_run_writes_a_csv(csv_input, tmp_path, model_name):
    import pandas as pd
    from newsvoice import COLUMNS

    out = tmp_path / "nested" / "quotes.csv"
    code = main([str(csv_input), "-o", str(out),
                 "--model", model_name, "--no-gpu", "--quiet"])

    assert code == 0
    assert out.exists(), "output directory should be created if absent"
    written = pd.read_csv(out)
    assert list(written.columns) == COLUMNS
    assert set(written["Article Id"]) == {"a1", "a2"}


def test_limit_truncates_the_input(csv_input, tmp_path, model_name):
    import pandas as pd

    out = tmp_path / "quotes.csv"
    main([str(csv_input), "-o", str(out), "--model", model_name,
          "--no-gpu", "--quiet", "--limit", "1"])
    assert set(pd.read_csv(out)["Article Id"]) == {"a1"}


def test_run_report_records_the_configuration(csv_input, tmp_path,
                                              model_name, capsys):
    """The configuration used must be recoverable from the run, not inferred."""
    out = tmp_path / "quotes.csv"
    main([str(csv_input), "-o", str(out), "--model", model_name, "--no-gpu"])
    err = capsys.readouterr().err
    assert "config=" in err
    assert "context_window" in err
    assert model_name in err
