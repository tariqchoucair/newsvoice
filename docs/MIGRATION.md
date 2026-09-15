# Migration from the notebook

`voice_and_attribution_extraction.ipynb` (revision `2026-09-03-r9`) became this
package. This document maps the old cells to the new modules and records every
behavioural change, so results produced before and after can be compared.

## Where each cell went

| Notebook cell | Now |
|---|---|
| Setup / imports | `newsvoice.load_pipeline()` |
| Data input (Drive mount) | `extract_corpus()` arguments, or the CLI |
| Layer 1 — `find_quote_spans` | `newsvoice.quotes` |
| Layer 2 — `find_cues` | `newsvoice.cues` |
| Layer 3 — `expand_speaker`, `cue_subject`, `cue_content` | `newsvoice.syntax` |
| Layer 4 — `entity_type` | `newsvoice.actors` |
| Layer 5 — `canonicalise` | `newsvoice.actors` |
| Layer 6 — `classify_*` | `newsvoice.segments` |
| Layer 7 — `build_indirect_segments` | `newsvoice.segments` |
| Layer 8 — `assign_quotes` | `newsvoice.assignment` |
| Orchestration — `extract_document` | `newsvoice.pipeline` |
| Single-document check | `demo.ipynb` |
| Regression diagnostics | `tests/` |
| Corpus run + export | `extract_corpus()`, or the CLI |

Layers 4 and 5 share a module. They were merged to eliminate the duplicated
gazetteers described below; separating them again would reintroduce the risk.

## Behavioural changes

**One change affects output.** Everything else is structural.

### Reflexive pronouns are now typed consistently

`PRONOUN_GENDER` was defined in both the Layer 4 and Layer 5 cells, and the two
tables differed: Layer 4's omitted `himself`, `herself`, `themselves`, `itself`.
`entity_type` was written in Layer 4, but because cells share one namespace and
Layer 5 ran later, it saw Layer 5's table during a full corpus run — and Layer 4's
if that cell was re-executed afterwards while iterating.

The package keeps the fifteen-entry table. For a corpus processed by running the
notebook top to bottom, **this reproduces what you already have**: Layer 5's table
was in effect. Output differs only if a run had the Layer 4 cell executed after
Layer 5, in which case a speaker referred to by a reflexive pronoun was typed
`PERSON` and is now typed `GROUP`.

If you need to know whether a past run was affected, look for rows where
`Speaker Mention` is a reflexive pronoun. In most news corpora there will be very
few, and possibly none.

### Everything else is unchanged by default

`ExtractionConfig()` reproduces the notebook's constants exactly: `CTX = 130`,
`ORPHAN_RECOVERY = True`, orphan gap 1200, sentence gap 2, headline cutoff 200,
dedupe overlap 0.6. Calling `extract_document(doc_id, text, nlp)` without a config
is behaviourally identical to the notebook's `extract_document(doc_id, text)`.

The layer functions themselves were moved verbatim. Nothing in the extraction
logic was rewritten, tuned, or "improved" during the move — that was a deliberate
constraint, so that any difference in output has exactly one candidate
explanation.

## API changes

### The parser is passed explicitly

```python
# Notebook — nlp was a module-level global
rows = extract_document(doc_id, text)

# Package
nlp = newsvoice.load_pipeline()
rows = newsvoice.extract_document(doc_id, text, nlp)
```

`classify_completeness` likewise takes the pipeline as a second argument.

### Corpus runs take a DataFrame

```python
# Notebook — paths hard-coded in a cell, Drive mounted
BASE = Path('/content/drive/MyDrive/quotes_project')
data_input = pd.read_csv(BASE / "data/00_master_articles.csv")
# ... 25 lines of loop, error handling and export

# Package
quotes = newsvoice.extract_corpus(
    pd.read_csv("articles.csv"), nlp,
    id_column="article_id", text_column="full_text",
)
quotes.to_csv("quotes.csv", index=False)
```

Failures are still recorded as rows with `Processing Status = "error"` and the
exception in `Context Snippet`. Pass `on_error="raise"` to stop on the first one
instead, which is what you want when debugging a new corpus.

### Constants became configuration

`CTX` and `ORPHAN_RECOVERY` were module-level globals edited in place. They are
now fields on `ExtractionConfig`, along with three values that were buried as
default arguments (`max_gap=1200`, `min_overlap=0.6`, `max_sentence_gap=2`) and
one that was a literal in the headline check (`200`).

This matters beyond tidiness: these are researcher degrees of freedom. Recording
`config.as_dict()` alongside results makes a run reproducible by someone who does
not have your notebook.

## Reproducing a past run

```python
import newsvoice, pandas as pd

nlp = newsvoice.load_pipeline("en_core_web_trf")
quotes = newsvoice.extract_corpus(
    pd.read_csv("00_master_articles.csv"), nlp,
    id_column="article_id", text_column="full_text",
)
```

To compare against notebook output, join on `Article Id` and
`Evidence Start Character (0-based)`, which together identify a row. Any
difference beyond reflexive-pronoun typing is a bug — please report it.

## What was removed

- `warnings.filterwarnings("ignore")`. A library must not reconfigure the host
  process's warnings. If spaCy warnings are noisy for you, filter them in your
  own script.
- `from google.colab import drive` and the mount call.
- The `DEMO` blocks at the foot of each layer cell. Their content survives as
  tests and as the examples in `demo.ipynb`.
- The regression diagnostics cell. It selected one of four check profiles by
  string-matching against specific copyrighted news articles in a private corpus,
  so it could not run elsewhere. Its assertions were rewritten against synthetic
  fixtures in `tests/`.
