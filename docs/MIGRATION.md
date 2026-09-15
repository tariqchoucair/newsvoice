# Migration from the notebook

`newsvoice` began as a Jupyter notebook that was circulated privately before this
package existed. If you were given that notebook, or you have results produced
with it, this page tells you what changed.

If you are new here, you can skip this file entirely.

## The one behavioural change

**Reflexive pronoun typing.** The notebook defined its pronoun table twice, in
two different cells, and the two versions differed: one omitted `himself`,
`herself`, `themselves` and `itself`. Because notebook cells share a namespace,
which version was in effect depended on the order cells had been run in — a
difference invisible in the output.

The package keeps the complete table. For a corpus produced by running the
notebook top to bottom, this reproduces what you already have. Output differs
only where a speaker was referred to by a reflexive pronoun *and* the cells had
been re-run out of order, in which case that speaker was typed `PERSON` and is
now typed `GROUP`.

To check whether any of your rows were affected, look for `Speaker Mention`
values that are reflexive pronouns. In most news corpora there will be very few,
and often none.

Everything else is unchanged. The extraction functions were moved verbatim; none
of the logic was rewritten or retuned during the move, precisely so that any
difference in output has one candidate explanation.

## Before you compare results

Check whether your source text was hard-wrapped at a fixed column, as Factiva
exports and PDF extractions usually are. A newline in the middle of a paragraph
is read as a paragraph break, which can attach a quotation to the wrong speaker —
see [issue 1](KNOWN_ISSUES.md).

This affects the notebook and the package equally, so it is not a difference
between them, but it does mean results from either may need regenerating:

```python
import newsvoice

articles["full_text"] = articles["full_text"].map(newsvoice.normalise_paragraphs)
```

## What the API looks like now

The notebook held a pipeline in a global variable and read from hard-coded paths.
Both are now arguments.

```python
# Notebook
rows = extract_document(doc_id, text)

# Package
nlp = newsvoice.load_pipeline()
rows = newsvoice.extract_document(doc_id, text, nlp)
```

Corpus runs take a DataFrame rather than reading a fixed location:

```python
quotes = newsvoice.extract_corpus(
    pd.read_csv("articles.csv"), nlp,
    id_column="article_id", text_column="full_text",
)
```

The tunables that were loose constants in the notebook — the context window,
orphan recovery, the sentence gap — are now fields on `ExtractionConfig`, with
defaults identical to the notebook's values. They are researcher degrees of
freedom, so record `config.as_dict()` alongside your results.

## Reproducing a notebook run

```python
import newsvoice, pandas as pd

nlp = newsvoice.load_pipeline("en_core_web_trf")
articles = pd.read_csv("your_articles.csv")
quotes = newsvoice.extract_corpus(
    articles, nlp, id_column="article_id", text_column="full_text",
)
```

To compare against notebook output, join on `Article Id` and
`Evidence Start Character (0-based)`, which together identify a row. Any
difference beyond reflexive-pronoun typing is a bug — please open an issue.
