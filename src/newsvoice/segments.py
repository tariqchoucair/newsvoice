"""Layers 6 and 7 — categorical columns, and indirect segment construction.

Layer 6 assigns the three descriptive columns:

``Quote Completeness``
    ``FULL_SENTENCE`` / ``CLAUSE`` / ``PHRASE`` / ``SINGLE_WORD`` /
    ``NOT_APPLICABLE``, from the longest direct segment.

``Attribution Explicitness``
    ``NAMED`` / ``DESCRIPTIVE_REFERENCE`` / ``PRONOUN`` / ``IMPLIED``.

``Attribution Position``
    ``BEFORE_CONTENT`` / ``AFTER_CONTENT`` / ``INTERRUPTED`` / ``EMBEDDED`` /
    ``NOT_EXPLICIT``.

Layer 7 works on tokens and decides what to keep by dependency role. It cuts the
quoted material and the attribution phrase out of the reported-content span, then
trims each remaining piece down to the proposition itself.

:func:`classify_completeness` needs to parse the quoted string, so it takes the
pipeline as an explicit argument. In the notebook it reached for a module-level
``nlp``, which is what made the layer untestable in isolation.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc

from .quotes import overlaps

__all__ = [
    "classify_completeness",
    "classify_explicitness",
    "classify_position",
    "clean_segment_tokens",
    "segment_text",
    "is_contentful",
    "build_indirect_segments",
]


_QUOTE_STRIP = "\u201c\u201d\"'\u2018\u2019 ,.;:"

def classify_completeness(direct_segments, nlp):
    if not direct_segments:
        return "NOT_APPLICABLE"
    txt = max(direct_segments, key=len).strip().strip(_QUOTE_STRIP).strip()
    if not txt:
        return "NOT_APPLICABLE"
    d = nlp(txt)
    if len([t for t in d if not t.is_punct]) <= 1:
        return "SINGLE_WORD"
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in d)
    has_subj = any(t.dep_ in ("nsubj","nsubjpass","expl","csubj") for t in d)
    if has_verb and has_subj:
        return "FULL_SENTENCE"
    return "CLAUSE" if has_verb else "PHRASE"


def classify_explicitness(speaker_span, doc):
    if speaker_span is None:
        return "IMPLIED"
    if speaker_span.root.pos_ == "PRON":
        return "PRONOUN"
    if any(t.pos_ == "PROPN" for t in speaker_span):
        return "NAMED"
    if any(e.label_ in ("PERSON","ORG","GPE","NORP") for e in doc.ents
           if overlaps((e.start_char, e.end_char),
                       (speaker_span.start_char, speaker_span.end_char))):
        return "NAMED"
    return "DESCRIPTIVE_REFERENCE"


def classify_position(cue_start, cue_end, content_spans):
    if cue_start is None or not content_spans:
        return "NOT_EXPLICIT"
    lo = min(s for s, _ in content_spans)
    hi = max(e for _, e in content_spans)
    if any(s < cue_start and cue_end < e for s, e in content_spans):
        return "EMBEDDED"
    if cue_start < lo:
        return "BEFORE_CONTENT"
    if cue_end > hi:
        return "AFTER_CONTENT"
    return "INTERRUPTED"


# Dependency labels / lemmas that introduce reported speech and are not part of it.
# Only "that" is semantically empty. "if" / "whether" carry the clause's
# modality -- "predict if a stock market will move" is not "predict a stock
# market will move" -- so they stay.
_COMPLEMENTIZERS = {"that"}
# "mark" is NOT blanket-dropped: it covers "if"/"while"/"because", which carry
# the clause's meaning. Only the empty complementizer "that" is removed.
_LEADING_DROP_DEPS = {"cc", "punct", "intj", "discourse"}
_TRAILING_DROP_DEPS = {"cc", "punct"}
# Function words left dangling when the phrase they governed was cut out:
# "The system works, the" (the speaker NP "study" was removed).
# Prepositions are NOT dropped: "predict with" is truncated but informative,
# whereas a stranded determiner ("The system works, the") is pure debris.
_TRAILING_DROP_POS = {"DET", "CCONJ", "PUNCT", "SPACE"}

def _token_range(doc, start_char, end_char):
    """Tokens fully inside a character span."""
    return [t for t in doc
            if t.idx >= start_char and t.idx + len(t.text) <= end_char]


def clean_segment_tokens(toks):
    """Trim a token list down to the reported content itself.

    Removes, from the left: coordinating conjunctions ("And"), the complementizer
    ("that when people are in a bad mood" -> "when people are in a bad mood"),
    discourse markers, and punctuation. From the right: punctuation and dangling
    conjunctions.
    """
    i, j = 0, len(toks)

    while i < j:
        t = toks[i]
        if t.is_punct or t.is_space:
            i += 1
            continue
        if t.dep_ in _LEADING_DROP_DEPS and t.lemma_.lower() != "not":
            i += 1
            continue
        if t.lower_ in _COMPLEMENTIZERS and t.dep_ in ("mark", "det", "dep", "nsubj"):
            i += 1
            continue
        if t.lower_ in ("and", "but", "so", "however", "meanwhile") and i == 0:
            i += 1
            continue
        break

    while j > i:
        t = toks[j - 1]
        if (t.is_punct or t.is_space or t.dep_ in _TRAILING_DROP_DEPS
                or t.pos_ in _TRAILING_DROP_POS):
            j -= 1
            continue
        break

    return toks[i:j]


def segment_text(doc, toks):
    """Exact source substring for a token list, so offsets stay faithful."""
    if not toks:
        return ""
    return doc.text[toks[0].idx: toks[-1].idx + len(toks[-1].text)]


def is_contentful(toks):
    """Keep a segment only if it carries actual propositional content."""
    if not toks:
        return False
    if not any(t.pos_ in ("VERB", "AUX", "NOUN", "PROPN", "PRON", "ADJ", "NUM")
               for t in toks):
        return False
    # a lone determiner/preposition fragment is debris, not content
    if len(toks) == 1 and toks[0].pos_ in ("DET", "ADP", "PART", "CCONJ", "SCONJ"):
        return False
    return True


def build_indirect_segments(doc, content_span, cut_spans, min_chars=3):
    """Split `content_span` around `cut_spans`, then clean each piece linguistically.

    `cut_spans` are character ranges to remove: quoted material (which belongs in
    Direct Segments) and the attribution phrase itself (cue + speaker).
    """
    a, b = content_span
    cuts = sorted(s for s in cut_spans if s[1] > a and s[0] < b)

    pieces, cur = [], a
    for cs, ce in cuts:
        if cs > cur:
            pieces.append((cur, min(cs, b)))
        cur = max(cur, ce)
    if cur < b:
        pieces.append((cur, b))

    out = []
    for pa, pb in pieces:
        if pb - pa < min_chars:
            continue
        toks = clean_segment_tokens(_token_range(doc, pa, pb))
        if not is_contentful(toks):
            continue
        txt = segment_text(doc, toks).strip()
        if re.search(r"\bof being$", txt, re.I):
            continue
        # Cutting a quote out of passive or relative syntax can leave fragments
        # such as `Two unions have been` or a bare `which`. They carry no voiced
        # proposition on their own.
        txt = re.sub(r"\s+(?:(?:has|have|had|is|are|was|were)\s+)?been$",
                     "", txt, flags=re.I).strip()
        if txt.lower() in {"which", "that", "who", "whom", "whose", "where"}:
            continue
        if len(txt) >= min_chars:
            out.append((txt, toks[0].idx, toks[-1].idx + len(toks[-1].text)))
    return out
