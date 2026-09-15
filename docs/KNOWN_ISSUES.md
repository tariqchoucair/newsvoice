# Known issues

Specific constructions where the output is wrong in a known way. Each entry says
what happens, why, and what it would cost to change. Where a test exists it is
named.

## 1. Hard-wrapped input misattributes quotations

**Symptom.** A document whose paragraphs are wrapped at a fixed column loses
quotations and attributes others to the wrong speaker. In one reproduced case a
three-paragraph quotation turn lost two paragraphs, one of which was attributed
to an organisation named in the *following* sentence. Every row involved was
well-formed: a real speaker, a real quotation, wrong pairing. Nothing in the
output indicates it happened.

**Cause.** The pipeline reads line structure as authorial structure — paragraph
boundaries override dependency edges, and line boundaries bound quote pairing.
That is correct for clean news text. A newline in the middle of a paragraph is
therefore taken for a boundary the author never wrote, which shifts sentence
segmentation enough to change the sentence-gap count in Layer 8 and hand a
quotation to a different cue.

**Scope.** Serious and common. Factiva exports, PDF extraction and many scrapers
hard-wrap. Reproduced on both `en_core_web_sm` and `en_core_web_trf`.

**Fix.** Normalise before extracting:

```python
articles["full_text"] = articles["full_text"].map(newsvoice.normalise_paragraphs)
```

`normalise_paragraphs` is deliberately **not** applied automatically, because
doing so would silently change results for anyone whose input is already clean.
It detects whether the document separates paragraphs with blank lines or with
single newlines and treats each correctly — joining every newline in
single-newline text would collapse the document into one paragraph, which is
worse than the problem being fixed.

Keep the normalised text: offsets index the string passed to the extractor.

**Tests.** `tests/test_preprocess.py::test_hard_wrap_misattributes_a_quotation`
pins the defect; `::test_normalisation_restores_the_correct_attribution` pins
the repair.

## 2. `continued:` before a quotation produces no attribution

**Symptom.** `Mr Dale continued: "We are not finished."` yields no cue, so the
quotation is left to orphan recovery or dropped. The full-stop form
(`Mr Dale continued. "..."`) and the inverted form (`"...," Mr Dale continued.`)
both work.

**Cause.** `continue` carries an aspectual guard, because *the outage will
continue getting worse* must not be read as reported speech. The guard suppresses
the cue when a clausal complement **follows** it. A colon-introduced quotation is
parsed as exactly that: a `ccomp` after the cue. At that point the two are not
distinguishable from the parse alone.

**Why it has not been changed.** Widening the guard to admit colon-introduced
complements risks re-admitting genuine aspectual uses, and the cost of a false
positive here is a misattributed quotation, which is worse than a missing one. A
fix probably needs to look at the punctuation between cue and complement rather
than at the dependency label. This is a judgement call about which error to
prefer, not an oversight.

**Test.** `tests/test_cues.py::test_continue_with_colon_introduced_quotation`,
marked `xfail`.

## 3. Accuracy degrades on non-transformer models, silently

**Symptom.** Running with `en_core_web_sm` produces output that looks
well-formed but misattributes more often, particularly for inverted attribution
and for quotations spanning a sentence boundary. Nothing in the output indicates
this.

**Cause.** The heuristics in `syntax` and `assignment` were developed against
`en_core_web_trf` parses. Smaller models assign different dependency labels to
the same constructions.

**Mitigation.** `load_pipeline` defaults to the transformer model. Tests
sensitive to parse quality carry the `model` marker. If you must use a smaller
model, validate on your own material first.

## 4. Test suite has not been run against `en_core_web_trf`

The suite was written and verified on `en_core_web_sm`. Assertions marked
`model` encode `sm` behaviour and some may need loosening — or may reveal genuine
differences — under `trf`. **Run the full suite once on the transformer model
before relying on it.**

## 5. Acronyms do not resolve when the full name kept its determiner

**Symptom.** In an article that first names *The Australian Energy Regulator* and
later says *The AER*, the second mention stays as `AER` instead of resolving.
One organisation ends up with two canonical names, which silently breaks
per-actor aggregation.

**Cause.** `_acronym_of` takes the initial of every capitalised word, so a
retained leading `The` contributes a `T` and the key generated is `TAER`. The
later `AER` finds no match. Whether the determiner is retained depends on how
`expand_speaker` trimmed the phrase, so the same organisation can resolve in one
article and not in another.

**Scope.** Pure string handling; nothing to do with the model. Affects any
organisation conventionally named with a definite article — which in Australian
and British news is most of them.

**Fix.** Strip a leading determiner in `_acronym_of` before taking initials, and
ideally normalise inventory keys the same way. Small and low-risk, but it changes
output, so it belongs in a version bump with a note rather than in a refactor.

**Tests.** `tests/test_actors.py::test_acronym_generation_is_derailed_by_a_leading_determiner`
pins the cause; `::test_acronym_resolves_when_the_inventory_entry_keeps_its_determiner`
is a strict `xfail` that will start failing the moment it is fixed.

## 6. Lexical refusals are read as attributions

**Symptom.** *The regulator declined to say whether it would be reviewed*
produces a row attributing content to the regulator. It is a non-disclosure: the
regulator said nothing.

**Cause.** The negation guard looks for `not`, `n't` and `never`. Refusal verbs
(`declined`, `refused`, `would not be drawn`) negate the reporting event
lexically rather than syntactically. `refused to comment` happens to be caught,
but only incidentally, by the non-finite action-sense guard that covers
`comment`/`reply`/`respond` — `say` is not in that set, so `declined to say`
slips through.

**Fix.** Suppress a cue governed by a refusal verb. The lemma set is short
(`decline`, `refuse`, `deny`, `withhold`) but the dependency test needs care so
that *declined, saying the matter was closed* still reports.

**Test.** `tests/test_cues.py::test_declined_to_say_produces_no_attribution`,
strict `xfail`.

## 7. Evaluative stance cues are treated as voice

Verbs such as `slam`, `attack`, `condemn`, `laud` are tier-2 cues, so
`The regulator slammed the proposal` produces an attribution with
`The regulator` as actor and `the proposal` as indirect content.

This is deliberate — stance attribution is voice for most content-analytic
purposes — but it is a **construct decision, not a neutral one**. A study
counting "who is quoted" will get a different answer from one counting "who is
given voice", and this pipeline implements the second. If you need the first,
filter on `Voice Type == "DIRECT"` or on the cue lemma, and say so in your
methods.

## 8. Claim counts index rhetorical style as well as prevalence

A speaker who argues in an enumerative register yields more rows than one making
the same argument once. Rows from the same article and speaker are not
independent, which bears on standard errors. Choosing the unit of analysis — the
attribution, the speaker–article pair, or the article — is a substantive decision
that changes what a distribution of rows describes.

Not a defect, but a property of the output that is easy to forget when the result
is a flat CSV.

## 9. Orphan recovery can carry a speaker too far

With `orphan_recovery=True` (the default), a quote-only paragraph inherits the
most recent prior speaker within `max_orphan_gap` characters. In articles that
interleave two speakers across short paragraphs without re-attributing, this will
sometimes attach a quotation to the wrong one.

The guard requires nothing but whitespace between the attributed material and the
quotation, which makes it conservative, but it is not exact. Set
`orphan_recovery=False` for a stricter standard at the cost of recall.

## 10. English only

The cue lexicons, honorifics, epithet heads and morphological rules are English
and largely Australian/British English (`emphasise` and `emphasize` are both
present; `defence` spellings are assumed in places). Other languages need new
lexicons throughout, not translation.

---

## Fixed

These were defects in the original notebook, diagnosed during the refactor.
Regression tests exist for each.

### `PRONOUN_GENDER` was defined twice, with diverging contents

The Layer 4 cell defined an eleven-entry table; the Layer 5 cell defined a
fifteen-entry one adding the reflexives. `entity_type` was written in Layer 4 but,
because Layer 5 executed later in a shared namespace, saw Layer 5's table during
a full run — and Layer 4's table if that cell was re-executed afterwards while
iterating. Reflexive pronouns were therefore typed `GROUP` or `PERSON` depending
on cell execution order, which is not recorded in the output.

Resolved by merging Layers 4 and 5 into one module and importing a single table
from `lexicons`. The fifteen-entry form was kept: it is both the one in effect
during corpus runs and the semantically correct one.

Tests: `tests/test_actors.py::test_layer_four_pronoun_table_would_have_mistyped_reflexives`
reproduces the defect by monkeypatching the old table back in;
`tests/test_lexicons.py::test_shared_constant_is_defined_exactly_once` fails if
any module rebinds a shared gazetteer.

### `TITLES` was defined twice

Identically, so it was harmless — but it was one edit away from repeating the
`PRONOUN_GENDER` failure. Same resolution, same test.

### Module-level `nlp`

Every layer closed over one pipeline created in the setup cell. The cue matcher
was built at import time against that pipeline's vocabulary, so a `Doc` from any
other pipeline would have matched against the wrong vocabulary. The parser is now
passed explicitly and the matcher binds lazily to `Doc.vocab`.

Tests: `tests/test_no_global_state.py`.

### Global `warnings.filterwarnings("ignore")`

Acceptable in a notebook; in a library it silently suppressed every warning in
the host process. Removed.

Test: `test_import_does_not_touch_global_warnings_filters`.

### Hard-coded Google Drive paths

Input and output were literals in a cell that also called `drive.mount`, so the
code ran in exactly one Colab account. Paths are now arguments to
`extract_corpus` and to the CLI.

Tests: `tests/test_cli.py`, `test_no_hardcoded_paths_or_colab`.

### Regression checks keyed to a private corpus

The notebook's diagnostic cell asserted against phrases from four specific
copyrighted news articles, selecting a check profile by string-matching the
article text. It could not run anywhere else, and its failures could not be
reproduced. Replaced with synthetic fixtures exercising the same behaviours, which
ship with the package.
