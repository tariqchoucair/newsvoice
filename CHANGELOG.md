# Changelog

Notable changes to `newsvoice`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[semantic versioning](https://semver.org/).

## [Unreleased]

### Planned
- Strip a leading determiner before generating an acronym key, so that an
  organisation first named as *The Australian Energy Regulator* resolves a later
  *AER* instead of producing a second canonical name
  ([known issue 5](docs/KNOWN_ISSUES.md)).
- Suppress cues governed by a refusal verb, so that *declined to say* stops
  producing an attribution for a non-disclosure
  ([known issue 6](docs/KNOWN_ISSUES.md)).

## [1.0.0] — 2026-09-15

First release. Reorganised from a Jupyter notebook into an installable package
with a test suite.

### Added
- `normalise_paragraphs` and `detect_paragraph_style` for repairing hard-wrapped
  input, which otherwise attributes quotations to the wrong speaker without
  raising an error.
- `ExtractionConfig`, replacing the notebook's loose constants. Defaults
  reproduce the notebook's values exactly; `as_dict()` gives a serialisable
  record for a methods section.
- `extract_corpus` for running over a DataFrame, with per-document error capture.
- Command-line interface: `newsvoice articles.csv -o quotes.csv`.
- Test suite of 218 tests plus 3 documented expected failures. All fixtures are
  synthetic; no news text ships with the package.
- `docs/KNOWN_ISSUES.md`, listing constructions where output is wrong in a known
  way.

### Changed
- Reflexive pronouns are now typed consistently. The notebook defined its pronoun
  table twice with diverging contents, so typing depended on cell execution
  order. This is the only behavioural difference from the notebook; all other
  output is unchanged.

### Fixed
- Removed a module-level spaCy pipeline that made the layers untestable in
  isolation and bound the cue matcher to a single vocabulary.
- Removed a global `warnings.filterwarnings("ignore")` that suppressed warnings
  across the host process on import.
- Replaced hard-coded Google Drive paths with arguments.
