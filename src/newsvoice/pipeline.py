"""Orchestration — running all eight layers over a document or a corpus.

Three arbitration rules keep the layers from over-generating, and they run in a
fixed order because each depends on the previous one having settled:

1. **Global quote ownership** (Layer 8) is decided before any row is built.
2. **Same-sentence cue merging** collapses several cues reporting one speaker in
   one sentence into a single row.
3. **Nested-cue suppression** drops a cue whose evidence span is wholly inside
   an already-claimed one, except for additive continuations (``adding that``).

Recovery passes then run for quotations that arbitration legitimately left
unattached: embedded quotations behind a weak pronominal cue, and quote-only
paragraphs continuing a speaker introduced earlier.

All tunable values live in :class:`ExtractionConfig` rather than as module-level
constants. They are researcher degrees of freedom with real effects on the
output, and a study reporting results from this pipeline should report the
configuration it used alongside them.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from spacy.tokens import Doc

from .actors import canonicalise, entity_type, strip_titles
from .assignment import assign_quotes, paragraph_units, sentence_units
from .cues import find_cues
from .lexicons import NP_CUES
from .quotes import find_quote_spans
from .segments import (
    build_indirect_segments,
    classify_completeness,
    classify_explicitness,
    classify_position,
)
from .syntax import attribution_span, cue_content, cue_subject, is_finite_clause
from .utils import word_count

__all__ = [
    "COLUMNS",
    "ExtractionConfig",
    "DEFAULT_CONFIG",
    "extract_document",
    "extract_corpus",
]


#: Output schema, in order. Every row returned by :func:`extract_document` has
#: exactly these keys.
COLUMNS = [
    "Article Id",
    "Actor Entity Type",
    "Actor Canonical Name",
    "Voice Type",
    "Direct Segments (JSON)",
    "Indirect Segments (JSON)",
    "Total Voice Segments",
    "Direct Word Count",
    "Indirect Word Count",
    "Total Voiced Word Count",
    "Total Voiced Character Count",
    "Quote Completeness",
    "Speaker Mention",
    "Attribution Cues (JSON)",
    "Attribution Explicitness",
    "Attribution Position",
    "Evidence Span",
    "Evidence Start Character (0-based)",
    "Evidence End Character (exclusive)",
    "Document Position (%)",
    "Context Snippet",
    "Processing Status",
]


@dataclass(frozen=True)
class ExtractionConfig:
    """Tunable parameters governing extraction.

    Defaults reproduce the values hard-coded in the original notebook, so
    ``extract_document(..., config=None)`` is behaviourally identical to it.

    Attributes
    ----------
    context_window
        Characters of surrounding text kept either side of the evidence span in
        ``Context Snippet``.
    orphan_recovery
        Carry a speaker across a paragraph break onto an otherwise unattributed
        quotation. News prose relies on the reader doing this; switching it off
        trades recall for a stricter attribution standard.
    max_orphan_gap
        Maximum characters between the attributed material and an orphan
        quotation for recovery to apply.
    max_sentence_gap
        How many sentences a quotation may be from its cue when its own sentence
        contains no cue.
    headline_max_chars
        A first line shorter than this and not ending in sentence punctuation is
        treated as a headline and excluded from attribution.
    dedupe_min_overlap
        Proportion of evidence-span overlap above which the shorter of two rows
        is discarded.
    """

    context_window: int = 130
    orphan_recovery: bool = True
    max_orphan_gap: int = 1200
    max_sentence_gap: int = 2
    headline_max_chars: int = 200
    dedupe_min_overlap: float = 0.6

    def as_dict(self) -> Dict[str, Any]:
        """Configuration as a plain dict, for recording alongside results."""
        return asdict(self)


#: The configuration used when none is supplied.
DEFAULT_CONFIG = ExtractionConfig()


def _attachable_quotes(qspans, used_q, cs, ce, sstart, send, text, cue_sent_starts,
                       speaker=None, all_cue_starts=frozenset()):
    out = []
    for q in qspans:
        if q in used_q or q[0] <= cs < q[1]:
            continue
        if q[0] >= sstart and q[1] <= send:                       # (a) same sentence
            out.append(q); continue
        gap = text[q[1]:cs] if q[1] <= cs else text[ce:q[0]]      # (b) adjacent
        if len(gap) <= 4 and not gap.strip(" \t,.:;-\u2014\u2013\n"):
            out.append(q); continue
        # (b2) Classic inversion: `"Quote," Dr Lim says.` The SPEAKER sits between the
        # quote and the cue, so the gap is not empty and rule (b) misses it -- and
        # spaCy usually ends a sentence at the closing quote, so rule (a) misses it
        # too. The quote was therefore dropped and the row mislabelled INDIRECT with
        # a span starting after the opening quote mark.
        if speaker is not None and q[1] <= cs:
            between = text[q[1]:cs]
            if speaker.text and speaker.text in between:
                rest = between.replace(speaker.text, "", 1)
                if not rest.strip(" \t,.:;-\u2014\u2013\n"):
                    out.append(q); continue
        if q[0] >= send and q[0] - send < 400:                    # (c) continuation
            # A quote carrying its OWN cue immediately after it belongs to that cue --
            # `"Is it too late? ..." he says` -- not to the previous paragraph's
            # speaker. Without this guard the earlier cue claims it first and the
            # quote is filed against the wrong attribution.
            owns_cue = any(q[1] <= c0 <= q[1] + 60 for c0 in all_cue_starts)
            if (not text[send:q[0]].strip(" \n\t")
                    and q[0] not in cue_sent_starts and not owns_cue):
                out.append(q)
    return out


def _merge_same_sentence(raw, text):
    """Collapse multiple cues reporting the same speaker in the same sentence."""
    out = []
    for r in raw:
        for p in out:
            same = ((p["speaker"] is None) == (r["speaker"] is None)) and (
                p["speaker"] is None or p["speaker"].text.lower() == r["speaker"].text.lower())
            between_start = min(p["cs"], r["cs"])
            between_end = max(p["ce"], r["ce"])
            same_paragraph = not re.search(r"\n\s*\n", text[between_start:between_end])
            if (same and same_paragraph
                    and p["cue_span"].root.sent == r["cue_span"].root.sent):
                p["direct"]   += [d for d in r["direct"]   if d not in p["direct"]]
                p["indirect"] += [d for d in r["indirect"] if d not in p["indirect"]]
                p["cue_list"] += [r["cue_text"]]
                p["lo"], p["hi"] = min(p["lo"], r["lo"]), max(p["hi"], r["hi"])
                p["content_spans"] += r["content_spans"]
                p["vt"] = ("HYBRID" if p["direct"] and p["indirect"]
                           else "DIRECT" if p["direct"] else "INDIRECT")
                break
        else:
            r["cue_list"] = [r["cue_text"]]
            out.append(r)
    return out


def _dedupe(rows, min_overlap):
    keep = []
    for r in sorted(rows, key=lambda z: -z["Total Voiced Character Count"]):
        a0, a1 = r["Evidence Start Character (0-based)"], r["Evidence End Character (exclusive)"]
        if not any(max(0, min(a1, k["Evidence End Character (exclusive)"])
                       - max(a0, k["Evidence Start Character (0-based)"]))
                   / min(max(1, a1 - a0),
                         max(1, k["Evidence End Character (exclusive)"]
                                - k["Evidence Start Character (0-based)"])) >= min_overlap
                   for k in keep):
            keep.append(r)
    keep.sort(key=lambda z: z["Evidence Start Character (0-based)"])
    return keep


def _recover_pronominal_show_quotes(raw, qspans, used_q, text):
    """Recover quotes lost behind a weak nested `it/this/that showed` cue.

    Ownership arbitration can prefer the nearer `showed`, then suppress that
    candidate because its subject is only a discourse pronoun. Run after
    same-sentence cue merging and attach a still-unrepresented quote to the
    narrowest surviving `tell` record that encloses it. The textual guard is
    deliberately strict, so named nested sources are never reassigned.
    """
    represented = {segment for record in raw for segment in record["direct"]}
    tell_form = re.compile(r"^(?:tell(?:s|ing)?|told)$", re.I)
    weak_show = re.compile(
        r"\b(?:it|this|that)\s+show(?:ed|s|n)?\b[^\n]{0,140}$", re.I)

    for q in qspans:
        quote_text = text[q[0]:q[1]]
        if quote_text in represented:
            continue
        candidates = [
            record for record in raw
            if record["lo"] <= q[0] and q[1] <= record["hi"]
            and any(tell_form.fullmatch(str(cue or "").strip())
                    for cue in record.get("cue_list", []))
            and weak_show.search(text[max(record["lo"], q[0] - 180):q[0]])
        ]
        if not candidates:
            continue
        record = min(candidates, key=lambda item: item["hi"] - item["lo"])
        record["direct"].append(quote_text)
        record["content_spans"].append(q)
        record["vt"] = "HYBRID" if record["indirect"] else "DIRECT"
        used_q.add(q)
        represented.add(quote_text)
    return raw


def _recover_orphan_quotes(raw, qspans, used_q, text, max_gap):
    """Attach quotes that no cue claimed to the most recent prior speaker.

    A news paragraph consisting only of quotation continues the speaker introduced
    earlier -- the reader carries the attribution across the break. Requires the quote
    to open its own paragraph with nothing but whitespace between it and the attributed
    material, so genuinely new reporting is not swept up.
    """
    if not raw:
        return raw
    anchored, extra = sorted(raw, key=lambda r: r["lo"]), []
    for q in qspans:
        if q in used_q or len(text[q[0]:q[1]].strip()) < 15:
            continue
        prior = [r for r in anchored if r["hi"] <= q[0] and r["speaker"] is not None]
        if not prior:
            continue
        src = prior[-1]
        if q[0] - src["hi"] > max_gap:
            continue
        line_start = text.rfind("\n", 0, q[0]) + 1
        if text[line_start:q[0]].strip() or text[src["hi"]:q[0]].strip():
            continue
        recovered = dict(cue_span=None, cue_text=None, speaker=src["speaker"],

                         direct=[text[q[0]:q[1]]], indirect=[], cue_list=[],

                         lo=q[0], hi=q[1], cs=None, ce=None,

                         content_spans=[q], vt="DIRECT", orphan=True)

        extra.append(recovered)

        # Make this recovered quote an anchor for the next consecutive quote-only
        # paragraph. Without this, recovery always stops after the first orphan.
        anchored.append(recovered)
        anchored.sort(key=lambda r: r["lo"])

        used_q.add(q)
    return sorted(raw + extra, key=lambda r: r["lo"])



def _empty_row(doc_id, status):
    r = {c: None for c in COLUMNS}
    r.update({"Article Id": doc_id,
              "Direct Segments (JSON)": "[]", "Indirect Segments (JSON)": "[]",
              "Total Voice Segments": 0, "Direct Word Count": 0, "Indirect Word Count": 0,
              "Total Voiced Word Count": 0, "Total Voiced Character Count": 0,
              "Quote Completeness": "NOT_APPLICABLE",
              "Attribution Explicitness": "IMPLIED",
              "Attribution Position": "NOT_EXPLICIT",
              "Processing Status": status})
    return r

def _is_definite(sp, text):
    """Definite ("the ...") vs indefinite ("a ...") NP, read from the left context."""
    lead = text[max(0, sp.start_char - 6):sp.start_char].lower().rstrip()
    if lead.endswith(("a", "an")):
        return False
    if lead.endswith("the"):
        return True
    return None          # bare NP -- leave the resolver's default behaviour


def _background(doc):
    """Every named entity in the document, for the resolver's pass-1 inventory."""
    return [{"text": ent.text, "start": ent.start_char, "end": ent.end_char,
             "type": ("PERSON" if ent.label_ == "PERSON"
                      else "ORGANISATION" if ent.label_ in ("ORG", "GPE") else "GROUP")}
            for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "NORP")]


def extract_document(doc_id, text, nlp, doc=None, config=None):
    """Run all layers over one document and return a list of row dicts.

    Parameters
    ----------
    doc_id
        Identifier copied into the ``Article Id`` column of every row.
    text
        The document text. Character offsets in the output index into this
        exact string, so it must be the same string the caller stores.
    nlp
        A loaded spaCy pipeline, used to parse `text` when `doc` is not
        supplied and to parse quoted strings for ``Quote Completeness``.
    doc
        An already-parsed ``Doc`` for `text`. Supply this when batching with
        ``nlp.pipe`` to avoid parsing twice.
    config
        An :class:`ExtractionConfig`. Defaults to :data:`DEFAULT_CONFIG`,
        which reproduces the notebook's hard-coded values exactly.

    Returns
    -------
    list of dict
        One dict per attribution, keyed by :data:`COLUMNS`. A document with
        no attributions yields a single row with ``Processing Status`` set
        to ``"complete_no_voice"``.
    """
    config = config or DEFAULT_CONFIG
    text = str(text)
    if doc is None:
        doc = nlp(text)

    # Headlines are not attributions. The first line of an article is its headline:
    # telegraphic, unpunctuated, and often carrying a reporting verb ("... Australian
    # study claims").
    _nl    = text.find("\n")
    _first = text[:_nl] if _nl > 0 else ""
    _headline_end = _nl if (0 < _nl < config.headline_max_chars
                            and not _first.rstrip().endswith((".", "!", "?"))) else 0

    qspans = find_quote_spans(text)

    # Remove cues inside quoted speech before global ownership. Otherwise an internal
    # verb can claim a quote and lose it when that cue is suppressed later.
    cues = [(sp, cue_text, tier) for sp, cue_text, tier in find_cues(doc)
            if not any(a <= sp.start_char < b for a, b in qspans)]
    cue_sent_starts = {c.root.sent.start_char for c, _, _ in cues}
    all_cue_starts  = {c.start_char for c, _, _ in cues}
    # Decide quote ownership globally, before any row is built (§9a).
    _sents  = sentence_units(doc, text, qspans)
    _paras  = paragraph_units(text)
    _cuepos = [(k, c.start_char, c.end_char) for k, (c, _, _) in enumerate(cues)]
    _assign = assign_quotes(qspans, _cuepos, _sents, _paras, text,
                            max_sentence_gap=config.max_sentence_gap)
    used_q, claimed, raw = set(), set(), []

    for _cue_i, (cue_span, cue_text, tier) in enumerate(cues):
        cs, ce = cue_span.start_char, cue_span.end_char
        sent   = cue_span.root.sent
        sstart, send = sent.start_char, sent.end_char

        # A cue *inside* quoted speech is part of what was said, not an attribution:
        # in `told him to "come home"`, the quoted phrase is content, not a new report.
        if any(a <= cs < b for a, b in qspans):
            continue
        if cs < _headline_end:
            continue

        # CMS promo / teaser lines dropped mid-article: "EXPLAINED: What are 'bomb
        # cyclones'". An all-caps reporting verb is never running prose.
        if cue_text.isupper() and len(cue_text) > 2:
            continue

        # Subjectless participles are modifiers, not attributions: "..., describing a
        # kind of view which gamers understand" elaborates the previous clause. Left
        # in, the pipeline invents an actor out of the participle's object.
        # The exception is a participle in a sentence that carries a direct quote --
        # those are real attributions, and excluding them costs ~0.02 recall.
        _v = cue_span.root
        inherited_reporter = (
            _v.lemma_.lower() in {"claim", "allege", "argue", "say", "state", "report", "reveal",
                                  "slam", "revile", "express", "back", "laud", "warn"}
            and cue_subject(cue_span, doc) is not None)
        if (tier != "phrase"
                and {"Part", "Ger"} & set(_v.morph.get("VerbForm"))
                and not any(ch.dep_ in ("nsubj", "nsubjpass") for ch in _v.children)
                and not any(a <= send and sstart <= b for a, b in qspans)
                and not inherited_reporter):
            continue

        # Imperative editorial voice -- "Think lighting that shifts to suit your mood",
        # "Explain what data is collected" -- is the writer addressing the reader, not
        # a speaker being quoted. A subjectless sentence-initial verb is the giveaway.
        v = cue_span.root
        if tier != "phrase" and not any(c.dep_ in ("nsubj", "nsubjpass") for c in v.children):
            inherited_subject = cue_subject(cue_span, doc)
            if ((v.i == v.sent.start or "Inf" in v.morph.get("VerbForm"))
                    and inherited_subject is None
                    and not any(a <= v.sent.start_char < b for a, b in qspans)):
                continue

        speaker = cue_subject(cue_span, doc)                       # Layer 3
        # Reject degenerate speakers: bare punctuation, determiners, stray possessive
        # clitics ("'s" left behind when the parse attaches the cue to the wrong token),
        # and single stray characters. These otherwise surface as actor names.
        if speaker is not None:
            bare = re.sub(r"[^A-Za-z0-9 ]", "", speaker.text).strip()
            if (speaker.root.pos_ in ("DET", "PUNCT", "SPACE")
                    or speaker.root.dep_ == "case"
                    or len(bare) < 2
                    or re.fullmatch(r"['\u2019]?s", speaker.text.strip(), re.I)):
                speaker = None

        # Suppress passive `was told` when no by-agent identifies the source,
        # and source-less fronted questions such as `Asked about X, Dutton ...`.
        passive_tell = (cue_span.root.lemma_.lower() == "tell"
                        and any(ch.dep_ == "nsubjpass" for ch in cue_span.root.children))
        passive_ask = (cue_span.root.lemma_.lower() == "ask"
                       and cue_span.root.tag_ == "VBN"
                       and not any(ch.dep_ in ("agent", "nsubj")
                                   for ch in cue_span.root.children))
        if speaker is None and (passive_tell or passive_ask):
            continue

        direct, indirect, content_spans, owned_q = [], [], [], []

        # Ownership was decided globally in §9a; this cue takes only its own quotes.
        # Do not mark a quote as consumed until the candidate row survives arbitration.
        for q in [q for q in qspans if _assign.get(q) == _cue_i and q not in used_q]:

            direct.append(text[q[0]:q[1]])                          # Layer 1

            content_spans.append(q)
            owned_q.append(q)

        allow_np = cue_span.root.lemma_.lower() in NP_CUES or tier == "phrase"
        content  = cue_content(cue_span, doc, allow_np=allow_np)

        # A quotation wholly inside this cue's semantic complement belongs here
        # when its provisional owner lies outside that complement (or no owner was
        # found). This recovers embedded phrases such as `“do anything”` without
        # stealing quotations from a genuinely nested reporting cue.
        if content is not None and _v.lemma_.lower() == "tell":
            for q in qspans:
                if q in used_q or q in owned_q:
                    continue
                # Dependency subtrees often stop before the closing full
                # stop and quotation mark, so permit only that tiny punctuation
                # overhang—not arbitrary text beyond the complement.
                if not (content.start_char <= q[0] and q[1] <= content.end_char + 3):
                    continue
                owner = _assign.get(q)
                owner_start = (cues[owner][0].start_char
                               if owner is not None and 0 <= owner < len(cues) else None)
                owner_speaker = (cue_subject(cues[owner][0], doc)
                                 if owner is not None and 0 <= owner < len(cues)
                                 else None)
                speaker_name = (strip_titles(speaker.text).lower().strip()
                                if speaker is not None else "")
                owner_name = (strip_titles(owner_speaker.text).lower().strip()
                              if owner_speaker is not None else "")
                same_speaker = bool(
                    speaker_name and owner_name
                    and (speaker_name == owner_name
                         or speaker_name.split()[-1] == owner_name.split()[-1]))
                # `telling X it showed Y would “...”` contains a syntactically
                # nested `show` cue, but its pronominal subject is not a new
                # quoted source. Treat that owner as weak so the governing
                # `tell` attribution keeps the embedded direct words. A named
                # nested source (`Smith told X Jones said “...”`) remains intact.
                owner_lemma = (cues[owner][0].root.lemma_.lower()
                               if owner is not None and 0 <= owner < len(cues)
                               else "")
                weak_nested_owner = (
                    owner_lemma == "show" and owner_name in {"it", "this", "that"})
                if (owner_start is None
                        or not (content.start_char <= owner_start < content.end_char)
                        or same_speaker or weak_nested_owner):
                    direct.append(text[q[0]:q[1]])
                    content_spans.append(q)
                    owned_q.append(q)

        # `was quoted as saying` already expresses attribution; when it owns a
        # direct quotation, do not also reproduce that quotation as indirect text.
        if tier == "phrase" and "quoted" in cue_text.lower() and direct:
            content = None

        if content is not None and not allow_np and not direct and not is_finite_clause(content):
            content = None
        # first-person peripheral cues ("I think", "we believe") are the text's own
        # producer reasoning aloud, not an attribution to a third party. This matters
        # enormously in parliament and submission text, written in the first person.
        if (tier == "peripheral" and not direct and speaker is not None
                and speaker.text.lower().strip() in ("i", "we", "you", "me", "us")):
            continue
        if (tier == "peripheral" and not direct and not allow_np
                and not is_finite_clause(content)):
            content = None                                          # tier-2 gate

        if content is not None:                                     # indirect = content − quotes
            a, b, cur, pieces = content.start_char, content.end_char, content.start_char, []
            # For phrase cues ("according to X") the content is the whole sentence,
            # so the attribution phrase sits inside it and must be excised whole.
            cuts = list(content_spans)
            cuts.extend(q for q in qspans
                        if q[0] < content.end_char and content.start_char < q[1])
            cuts.append(attribution_span(cue_span, doc) if tier == "phrase"
                        else (cs, ce))
            if speaker is not None:
                cuts.append((speaker.start_char, speaker.end_char))
            for seg_txt, sa, sb in build_indirect_segments(
                    doc, (content.start_char, content.end_char), cuts):
                indirect.append(seg_txt)
                content_spans.append((sa, sb))

        if not direct and not indirect:
            continue
        nested = any(c0 <= cs and ce <= c1 for c0, c1 in claimed)
        additive_continuation = (_v.lemma_.lower() == "add"
                                 and {"Part", "Ger"} & set(_v.morph.get("VerbForm"))
                                 and speaker is not None)
        if nested and not additive_continuation:                  # nested-cue suppression
            continue

        used_q.update(owned_q)
        claimed.add((min(s for s, _ in content_spans), max(e for _, e in content_spans)))

        starts, ends = [cs] + [s for s, _ in content_spans], [ce] + [e for _, e in content_spans]
        if speaker is not None:
            starts.append(speaker.start_char); ends.append(speaker.end_char)
        lo, hi = min(starts), max(ends)
        while hi < len(text) and text[hi] in '.!?\u201d\u2019"\' ':  # absorb trailing punctuation
            hi += 1
            if text[hi - 1] in '.!?':
                break

        raw.append(dict(cue_span=cue_span, cue_text=cue_text, speaker=speaker,
                        direct=direct, indirect=indirect, lo=lo, hi=min(hi, len(text)),
                        cs=cs, ce=ce, content_spans=content_spans,
                        vt="HYBRID" if direct and indirect else ("DIRECT" if direct else "INDIRECT")))

    raw = _merge_same_sentence(raw, text)
    raw = _recover_pronominal_show_quotes(raw, qspans, used_q, text)

    # If an additive participle's content was already absorbed by the governing
    # attribution, nested arbitration may correctly avoid a second row. Preserve
    # the observable cue itself on that governing row.
    for add_span, add_text, add_tier in cues:
        if add_span.root.lemma_.lower() != "add":
            continue
        for record in raw:
            same_sentence = record["cue_span"].root.sent == add_span.root.sent
            contains_cue = record["lo"] <= add_span.start_char <= record["hi"]
            if same_sentence and contains_cue and add_text not in record["cue_list"]:
                record["cue_list"].append(add_text)
                break

    if config.orphan_recovery:
        raw = _recover_orphan_quotes(raw, qspans, used_q, text,
                                     max_gap=config.max_orphan_gap)

    # A parser may select the possessor in `Farrow's predecessor Dan Walton`
    # or `Kelly's predecessor Geoff Dyke` as the cue subject. The named person
    # following `predecessor/successor` is the actual speaker.
    for _r in raw:
        # Orphan quotation continuations intentionally have no explicit cue.
        # They must bypass cue-dependent predecessor correction.
        if _r.get("cue_span") is None:
            continue
        cue = _r["cue_span"].root
        sent = cue.sent
        role_tokens = [t for t in doc[sent.start:cue.i]
                       if t.lemma_.lower() in {"predecessor", "successor"}]
        if not role_tokens:
            continue
        role = role_tokens[-1]
        candidates = [ent for ent in doc.ents
                      if ent.label_ == "PERSON"
                      and role.idx < ent.start_char < cue.idx]
        if candidates:
            _r["speaker"] = min(candidates, key=lambda ent: cue.idx - ent.end_char)

    # --- apposition: "Atlassian CEO Mike Cannon-Brookes" -> the PERSON entity.
    # A role epithet immediately followed by a named person is an apposition; the
    # name is the referent. Checking adjacency beats any document-wide "nearest
    # person" guess, which picks whoever spoke last and is wrong as often as not.
    for _r in raw:
        sp = _r["speaker"]
        if sp is None:
            continue
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            adjacent = 0 <= ent.start_char - sp.end_char <= 1
            nested   = (sp.start_char <= ent.start_char and ent.end_char <= sp.end_char
                        and len(ent.text) < len(sp.text))
            if adjacent or nested:
                _r["speaker"] = ent
                break

    mentions = [{"text":  r["speaker"].text if r["speaker"] is not None else "",
                 "definite": (_is_definite(r["speaker"], text)
                              if r["speaker"] is not None else None),
                 "type":  entity_type(r["speaker"], doc) if r["speaker"] is not None else "PERSON",
                 "start": r["speaker"].start_char if r["speaker"] is not None else None,
                 "end":   r["speaker"].end_char   if r["speaker"] is not None else None}
                for r in raw]
    canon = canonicalise(mentions, background=_background(doc))

    rows = []
    for i, r in enumerate(raw):
        sp = r["speaker"]
        lo, hi = r["lo"], r["hi"]
        a, b = (max(0, lo - config.context_window),
                min(len(text), hi + config.context_window))
        dwc = sum(map(word_count, r["direct"]))                     # Layer 8
        iwc = sum(map(word_count, r["indirect"]))
        rows.append({
            "Article Id": doc_id,
            "Actor Entity Type": entity_type(sp, doc) if sp is not None else None,
            "Actor Canonical Name": canon[i] if sp is not None else None,
            "Voice Type": r["vt"],
            "Direct Segments (JSON)":   json.dumps(r["direct"],   ensure_ascii=False),
            "Indirect Segments (JSON)": json.dumps(r["indirect"], ensure_ascii=False),
            "Total Voice Segments": len(r["direct"]) + len(r["indirect"]),
            "Direct Word Count": dwc, "Indirect Word Count": iwc,
            "Total Voiced Word Count": dwc + iwc,
            "Total Voiced Character Count": sum(len(x) for x in r["direct"] + r["indirect"]),
            "Quote Completeness": classify_completeness(r["direct"], nlp),
            "Speaker Mention": None if r.get("orphan") else (sp.text if sp is not None else None),
            "Attribution Cues (JSON)": json.dumps(r["cue_list"], ensure_ascii=False),
            "Attribution Explicitness": ("IMPLIED" if r.get("orphan")
                                        else classify_explicitness(sp, doc)),
            "Attribution Position": classify_position(r["cs"], r["ce"], r["content_spans"]),
            "Evidence Span": text[lo:hi],
            "Evidence Start Character (0-based)": lo,
            "Evidence End Character (exclusive)": hi,
            "Document Position (%)": round(lo / max(1, len(text)) * 100, 2),
            "Context Snippet": ("\u2026" if a > 0 else "") + text[a:b] + ("\u2026" if b < len(text) else ""),
            "Processing Status": "complete",
        })

    # `Actor Entity Type` must follow the RESOLVED actor. Apposition and the
    # background inventory can rename "the study's lead author" to "Angel Zhong";
    # without this the row keeps the surface phrase's type and reads ORGANISATION
    # for a person.
    _person_names = {b["text"] for b in _background(doc) if b["type"] == "PERSON"}
    _person_names |= {strip_titles(n) for n in _person_names}
    for row in rows:
        nm = row.get("Actor Canonical Name")
        if nm and nm in _person_names and row["Actor Entity Type"] != "PERSON":
            row["Actor Entity Type"] = "PERSON"
        elif nm and re.search(r"\bunions?\b", nm, re.I):
            row["Actor Entity Type"] = "ORGANISATION"

    rows = _dedupe(rows, min_overlap=config.dedupe_min_overlap)
    if not rows:
        rows = [_empty_row(doc_id, "complete_no_voice")]
    return rows


def extract_corpus(
    frame,
    nlp,
    id_column: str = "article_id",
    text_column: str = "full_text",
    config: Optional[ExtractionConfig] = None,
    batch_size: int = 32,
    n_process: int = 1,
    progress: bool = False,
    on_error: str = "record",
):
    """Run :func:`extract_document` over a table of documents.

    Parameters
    ----------
    frame
        A ``pandas.DataFrame`` with at least `id_column` and `text_column`.
        No other columns are read.
    nlp
        A loaded spaCy pipeline, e.g. from :func:`newsvoice.load_pipeline`.
    config
        An :class:`ExtractionConfig`; defaults to :data:`DEFAULT_CONFIG`.
    batch_size, n_process
        Passed through to ``nlp.pipe``.
    progress
        Show a ``tqdm`` progress bar if ``tqdm`` is installed.
    on_error
        ``"record"`` writes a row with ``Processing Status = "error"`` and the
        exception in ``Context Snippet``, then continues. ``"raise"`` propagates,
        which is what you want in tests and when debugging a new corpus.

    Returns
    -------
    pandas.DataFrame
        One row per attribution, with the columns of :data:`COLUMNS`.

    Raises
    ------
    KeyError
        If `id_column` or `text_column` is not present in `frame`.
    ValueError
        If `on_error` is not one of ``"record"`` or ``"raise"``.
    """
    import pandas as pd

    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    missing = [c for c in (id_column, text_column) if c not in frame.columns]
    if missing:
        raise KeyError(
            f"column(s) {missing} not in frame; available: {list(frame.columns)}"
        )

    config = config or DEFAULT_CONFIG
    texts = [str(t) for t in frame[text_column]]
    ids = [str(t) for t in frame[id_column]]

    pipe = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    stream = zip(ids, texts, pipe)
    if progress:
        try:
            from tqdm.auto import tqdm
            stream = tqdm(stream, total=len(texts), desc="extracting")
        except ImportError:
            pass

    rows: List[Dict[str, Any]] = []
    for doc_id, text, doc in stream:
        try:
            rows.extend(extract_document(doc_id, text, nlp, doc=doc, config=config))
        except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
            if on_error == "raise":
                raise
            error_row = _empty_row(doc_id, "error")
            error_row["Context Snippet"] = (
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            rows.append(error_row)

    return pd.DataFrame(rows, columns=COLUMNS)
