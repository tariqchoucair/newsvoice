# newsvoice

Voice and attribution extraction from news text.

`newsvoice` finds who is given voice in a news article, what they are reported as
saying, and how the attribution is made. It returns one row per attribution with
a 22-column schema, including character offsets back into the source text so
every extracted segment can be traced to the discourse it came from.

## Try it

Open [`demo.ipynb`](demo.ipynb) in Google Colab, it works through a complete example on a sample article.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tariqchoucair/newsvoice/blob/main/demo.ipynb)

## Install it

Via a notebook (e.g., Google Colab):

```bash
%pip install -q git+https://github.com/tariqchoucair/newsvoice.git
%pip install -q https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
```

Locally, you need Python 3.9 or newer. Open a terminal and run:

```bash
pip install git+https://github.com/tariqchoucair/newsvoice.git
python -m spacy download en_core_web_trf
```

The second line downloads the language model (`en_core_web_trf`). It's about 450 MB, so it takes a few minutes.
`python -m spacy download en_core_web_sm` is 12 MB and runs much
faster, but it gets attributions wrong more often. See [Choice of model](#choice-of-model).

## Usage

You need a dataset with one row per article and at least two columns - an ID and the text. For example:

| article_id | full_text |
|---|---|
| a1 | Energy regulator warns of price rises\n\nThe Australian... |
| a2 | Minister rejects claims\n\nThe minister said... |

Any other columns can exist, but they are ignored. Name your columns whatever you like; you'll tell the
code which two (ID and text) to use. 

Example of usage: 

```python
import pandas as pd
import newsvoice

# Load the language model
MODEL = "en_core_web_trf" 
nlp = newsvoice.load_pipeline(MODEL)

# Read your articles
articles = pd.read_csv("articles.csv")

# Extract. One row out per attribution found
quotes = newsvoice.extract_corpus(
    articles,
    nlp,
    id_column="article_id",     # the column holding your IDs
    text_column="full_text",    # the column holding your article text
)

quotes.to_csv("quotes.csv", index=False)
print(f"{len(quotes):,} attributions from {len(articles):,} articles")
```

Or you can run it straight from the terminal:

```bash
newsvoice articles.csv -o quotes.csv --id-column article_id --text-column full_text
```

You can also run one article at a time:

```python
import newsvoice

nlp = newsvoice.load_pipeline()
rows = newsvoice.extract_document("a1", article_text, nlp)
```

The individual run returns a list of dictionaries rather than a table, it is useful for checking a
single case or building your own loop.

## What newsvoice extracts

Given a news text, for example:

> Opposition Leader Peter Dutton has declared the plan would cost "a fraction"
> of Labor's. "Today we announce seven sites," Mr Dutton told reporters.
>
> The Australian Workers Union attacked the proposal. "This is a twentieth
> century ideology," said the union's national secretary Paul Farrow.

it returns four attributions: two for Dutton (one hybrid, one direct), one for
the union as an organisation, and one for Farrow - resolved from "the union's
national secretary" to his name, and typed as a PERSON distinct from the
ORGANISATION in the previous sentence.

## How it works

The pipeline works with eight layers, each importable and replaceable individually.

| Layer | Module | Does |
|---|---|---|
| 1 | `quotes` | Finds quoted regions by pattern matching. No parse required. |
| 2 | `cues` | Finds reporting expressions (`said`, `according to`) in two tiers. |
| 3 | `syntax` | Dependency parse: who spoke, and what they said. |
| 4 | `actors` | Types the actor as PERSON / ORGANISATION / GROUP. |
| 5 | `actors` | Resolves epithets, pronouns, acronyms and metonymy to a canonical actor. |
| 6 | `segments` | Assigns the three categorical columns. |
| 7 | `segments` | Builds indirect segments by cutting quotes and attribution out of the reported content. |
| 8 | `assignment` | Decides globally which cue owns which quotation. |

Layer 8 is global by design because if we resolve quote ownership locally, one cue at a
time, this will let whichever cue processed first take a quotation belonging to a later
one, and the resulting row is well-formed but wrong.

Three arbitration rules then run in `pipeline`: global ownership settles first,
then cues reporting one speaker in one sentence are merged, then a cue whose
evidence span sits wholly inside an already-claimed one is suppressed. Two
recovery passes pick up quotations that arbitration legitimately left unattached.

## Output schema

One row per attribution.

**Identity**

| Column | Values |
|---|---|
| `Article Id` | As supplied by the caller. |
| `Actor Entity Type` | `PERSON` / `ORGANISATION` / `GROUP`, or empty when no speaker was resolved. |
| `Actor Canonical Name` | Who the mention refers to, after coreference and alias resolution. |
| `Speaker Mention` | What the text actually said. Empty for recovered orphan quotations. |

`Speaker Mention` and `Actor Canonical Name` are deliberately separate. The
surface form is reproduced from the evidence while the canonical name is an inference.

**Content**

| Column | Values |
|---|---|
| `Voice Type` | `DIRECT` / `INDIRECT` / `HYBRID`. |
| `Direct Segments (JSON)` | JSON list of quoted strings, verbatim from the source. |
| `Indirect Segments (JSON)` | JSON list of reported-speech strings, verbatim from the source. |
| `Total Voice Segments` | Count of both lists. |
| `Direct Word Count`, `Indirect Word Count`, `Total Voiced Word Count` | Words, excluding punctuation and quotation marks. |
| `Total Voiced Character Count` | Characters across all segments. |
| `Quote Completeness` | `FULL_SENTENCE` / `CLAUSE` / `PHRASE` / `SINGLE_WORD` / `NOT_APPLICABLE`, from the longest direct segment. |

**Attribution**

| Column | Values |
|---|---|
| `Attribution Cues (JSON)` | JSON list of the reporting expressions on this row. |
| `Attribution Explicitness` | `NAMED` / `DESCRIPTIVE_REFERENCE` / `PRONOUN` / `IMPLIED`. |
| `Attribution Position` | `BEFORE_CONTENT` / `AFTER_CONTENT` / `INTERRUPTED` / `EMBEDDED` / `NOT_EXPLICIT`. |

**Provenance**

| Column | Values |
|---|---|
| `Evidence Span` | The stretch of source text this row was built from. |
| `Evidence Start Character (0-based)`, `Evidence End Character (exclusive)` | Offsets into the text passed in. `text[start:end] == Evidence Span` always holds. |
| `Document Position (%)` | Where the row sits in the document. |
| `Context Snippet` | Evidence plus surrounding context, for human checking. |
| `Processing Status` | `complete` / `complete_no_voice` / `error`. |

A document with no attributions yields exactly one row with
`complete_no_voice`, so documents never silently disappear from the input corpus.

## Configuration

Everything tunable is on `ExtractionConfig`, and the defaults reproduce the
values the pipeline was developed with.

```python
from newsvoice import ExtractionConfig, extract_corpus

config = ExtractionConfig(orphan_recovery=False, max_sentence_gap=1)
quotes = extract_corpus(articles, nlp, config=config)
```

| Parameter | Default | Effect |
|---|---|---|
| `context_window` | 130 | Characters of context either side of the evidence span. |
| `orphan_recovery` | `True` | Carry a speaker across a paragraph break onto a quote-only paragraph. |
| `max_orphan_gap` | 1200 | Furthest an orphan quotation may be from its speaker. |
| `max_sentence_gap` | 2 | Sentences a quotation may sit from its cue. |
| `headline_max_chars` | 200 | First lines shorter than this, without sentence punctuation, are headlines. |
| `dedupe_min_overlap` | 0.6 | Evidence overlap above which the shorter row is dropped. |

These are researcher degrees of freedom with real effects on the output.
`orphan_recovery` in particular sets an attribution
standard: on, it follows news convention that a reader carries attribution
across a paragraph break; off, it requires explicit attribution and yields
fewer and only more explicit rows. A study reporting results from this pipeline should
report the configuration alongside them.
`config.as_dict()` returns a JSON-serialisable record.

## Choice of model

The dependency heuristics in `syntax` were developed against
`en_core_web_trf`. Smaller models can be used here, but degrade in specific,
non-random ways: in our tests, inverted attribution (`"Quote," said Ms Chen`) and clausal
complement selection are affected, and both cause **misattribution**.
If you use anything other than `en_core_web_trf`, validate on your own material
before trusting the output, and say which model you used in your methods.

## Validation

Although we developed this tool via multiple tests, we recommend hand-coding a random sample of your own
corpus and reporting agreement against it.

## Known issues

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md), each entry describes a construction where the
output is wrong in a known way.

## Citation

Choucair, T. (2026). newsvoice: Voice and attribution extraction from news text (Version 1.0.0) [Computer software]

If you use this in published work, please cite it. See `CITATION.cff`, or the
"Cite this repository" button on GitHub.

## Licence

MIT. See [`LICENSE`](LICENSE).
