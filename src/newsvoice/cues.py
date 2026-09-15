"""Layer 2 — lexicon plus POS/lemma evidence for attribution cues.

A cue is the reporting expression that signals an attribution: ``said``,
``told``, ``according to``. Detection is a two-tier lexicon filtered by a set of
morphological and dependency guards, each of which suppresses a construction
that looks reportive but is not.

The notebook built its :class:`~spacy.matcher.Matcher` once, at import time,
against a module-level ``nlp``. Here the matcher is built lazily from the
vocabulary of whichever ``Doc`` is passed in, so the layer has no global state
and works with any loaded pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab

from .lexicons import (
    CORE_CUE_LEMMAS,
    CUE_LEMMAS,
    MULTIWORD_CUES,
)

__all__ = ["find_cues"]

Cue = Tuple[Span, str, str]

# Cached matchers, keyed by ``id`` of the vocabulary they were built against.
# The vocabulary itself is stored alongside so that it cannot be garbage
# collected and have its id reused by a different object.
_MATCHER_CACHE: Dict[int, Tuple[Vocab, Matcher]] = {}

#: Noun-modifying participles that are nonetheless genuinely reportive, as in
#: "posts stating that ...". Everything else in ``amod``/``acl`` position is a
#: nominal modifier ("the proposed reactors") and not an attribution.
_REPORTIVE_ACL = frozenset({
    "state", "say", "argue", "claim", "allege", "describe", "recommend",
    "advocate",
})

#: Cues whose non-finite action sense is not a reporting event.
_ACTION_SENSE_CUES = frozenset({"respond", "reply", "comment"})


def _get_matcher(vocab: Vocab) -> Matcher:
    """Return a phrase matcher for `vocab`, building and caching it on first use."""
    key = id(vocab)
    cached = _MATCHER_CACHE.get(key)
    if cached is not None and cached[0] is vocab:
        return cached[1]
    matcher = Matcher(vocab)
    for phrase in MULTIWORD_CUES:
        matcher.add("CUE", [[{"LOWER": word} for word in phrase.split()]])
    _MATCHER_CACHE[key] = (vocab, matcher)
    return matcher


def find_cues(doc: Doc) -> List[Cue]:
    """Yield ``(span, surface_text, tier)`` for every attribution trigger in `doc`.

    ``tier`` is one of ``"phrase"`` (a multi-token cue such as *according to*),
    ``"core"`` (tier-1 reporting verb) or ``"peripheral"`` (tier-2 verb, which
    the orchestrator gates on the presence of a quotation or finite clause).
    """
    matcher = _get_matcher(doc.vocab)

    # Matcher phrases may overlap (`was quoted`, `quoted as saying`, and the
    # complete `was quoted as saying`). Retain one maximal cue rather than
    # emitting duplicate attribution labels and duplicated content.
    phrase_candidates = [(doc[s:e], doc[s:e].text, "phrase")
                         for _, s, e in matcher(doc)]
    phrase_candidates.sort(key=lambda item: (item[0].start, -len(item[0])))

    out: List[Cue] = []
    for candidate in phrase_candidates:
        sp = candidate[0]
        if not any(sp.start < kept[0].end and kept[0].start < sp.end
                   for kept in out):
            out.append(candidate)

    covered = {t.i for sp, _, _ in out for t in sp}
    for tok in doc:
        if tok.i in covered or tok.pos_ not in ("VERB", "AUX"):
            continue
        lem = tok.lemma_.lower()

        # `unveil` is event narration, not a voice-attribution cue. Keeping
        # it lets `unveiled the policy` steal a nearby quotation from `declared`.
        if lem == "unveil":
            continue

        # Nominal participles modify a noun rather than report anyone's words:
        # `the Coalition's proposed seven reactors`. They must be removed here,
        # before global quote ownership, rather than only after a row is built.
        nominal_participle = (
            {"Part", "Ger"} & set(tok.morph.get("VerbForm"))
            and tok.dep_ in ("amod", "acl")
            and not any(ch.dep_ in ("nsubj", "nsubjpass") for ch in tok.children)
        )
        if nominal_participle and lem not in _REPORTIVE_ACL:
            continue

        # `hit` is reportive only in the phrasal stance cue `hit out`.
        if lem == "hit":
            next_token = doc[tok.i + 1].lower_ if tok.i + 1 < len(doc) else ""
            has_out_particle = any(ch.dep_ == "prt" and ch.lower_ == "out"
                                   for ch in tok.children)
            if next_token != "out" and not has_out_particle:
                continue

        # `continue` is reportive in "Kenny continued", but aspectual in
        # "the crisis will continue getting worse" / "continue to sound".
        # Use both dependency and immediate morphology because transformer parses
        # do not assign the same complement label consistently in every sentence.
        if lem == "continue":
            verbal_complement = any(
                ch.i > tok.i and ch.pos_ in ("VERB", "AUX", "ADJ")
                and ch.dep_ in ("xcomp", "ccomp", "advcl", "acl")
                for ch in tok.children)
            following = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
            aspectual_surface = following is not None and (
                following.lower_ == "to" or following.tag_ == "VBG")
            if verbal_complement or aspectual_surface:
                continue

        # Non-finite action senses such as "would not be able to respond" are not
        # reporting events and must not steal a nearby quotation from `said`.
        if lem in _ACTION_SENSE_CUES:
            has_own_subject = any(ch.dep_ in ("nsubj", "nsubjpass")
                                  for ch in tok.children)
            has_reported_clause = any(ch.dep_ == "ccomp" for ch in tok.children)
            nonfinite = "Inf" in tok.morph.get("VerbForm") or tok.tag_ == "VB"
            if nonfinite and not has_own_subject and not has_reported_clause:
                continue

        if lem in CUE_LEMMAS:
            # `did not reveal the costings` describes a non-disclosure; it does
            # not give voice to the actor. The neighbouring positive `said`
            # remains an independent attribution.
            negated = any(ch.dep_ == "neg" for ch in tok.children)
            if not negated:
                left = doc[max(tok.sent.start, tok.i - 3):tok.i]
                negated = any(t.lower_ in ("not", "n't", "never") for t in left)
            if negated:
                continue

            tier = "core" if lem in CORE_CUE_LEMMAS else "peripheral"
            out.append((doc[tok.i:tok.i + 1], tok.text, tier))

    out.sort(key=lambda x: x[0].start)
    return out
