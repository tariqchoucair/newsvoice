"""Command-line interface.

::

    python -m newsvoice articles.csv -o quotes.csv \\
        --id-column article_id --text-column full_text

Replaces the notebook's hard-coded Google Drive paths. Every path and tunable is
an argument, and the configuration actually used is echoed to stderr so it can be
pasted into a methods section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import DEFAULT_MODEL, __version__, load_pipeline
from .pipeline import ExtractionConfig, extract_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newsvoice",
        description="Extract voice and attribution from news text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path,
                        help="input CSV with one row per document")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="output CSV, one row per attribution")
    parser.add_argument("--id-column", default="article_id",
                        help="column holding the document identifier")
    parser.add_argument("--text-column", default="full_text",
                        help="column holding the document text")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="spaCy model to load")
    parser.add_argument("--no-gpu", action="store_true",
                        help="do not request GPU allocation")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N rows (for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--no-orphan-recovery", action="store_true",
                        help="do not carry a speaker onto quote-only paragraphs")
    parser.add_argument("--context-window", type=int, default=130,
                        help="characters of context kept either side of evidence")
    parser.add_argument("--max-sentence-gap", type=int, default=2,
                        help="sentences a quote may sit from its cue")
    parser.add_argument("--on-error", choices=("record", "raise"),
                        default="record",
                        help="record failures as rows, or stop on the first")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the progress bar and run report")
    parser.add_argument("--version", action="version",
                        version=f"newsvoice {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    import pandas as pd

    frame = pd.read_csv(args.input)
    if args.limit is not None:
        frame = frame.head(args.limit)

    config = ExtractionConfig(
        context_window=args.context_window,
        orphan_recovery=not args.no_orphan_recovery,
        max_sentence_gap=args.max_sentence_gap,
    )

    nlp = load_pipeline(args.model, prefer_gpu=not args.no_gpu)

    if not args.quiet:
        print(f"newsvoice {__version__}  model={args.model}", file=sys.stderr)
        print(f"config={json.dumps(config.as_dict())}", file=sys.stderr)
        print(f"documents={len(frame):,}", file=sys.stderr)

    quotes = extract_corpus(
        frame,
        nlp,
        id_column=args.id_column,
        text_column=args.text_column,
        config=config,
        batch_size=args.batch_size,
        n_process=args.n_process,
        progress=not args.quiet,
        on_error=args.on_error,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(args.output, index=False)

    if not args.quiet:
        n_err = int((quotes["Processing Status"] == "error").sum())
        print(f"wrote {args.output}  ({len(quotes):,} rows, {n_err} errors)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
