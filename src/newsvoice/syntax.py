"""Layer 3 — dependency parsing: who spoke, and what they said.

Given a cue verb, the parse tree answers both questions. The speaker is found by
walking the cue's subject edges, with explicit handling for passives (where the
grammatical subject is usually the recipient, not the source), for inverted
attribution (``"Quote," said Ms Chen``), and for conjoined or subordinate cues
that inherit their subject from a governing clause.

Content selection is frame-aware: core complements (``ccomp``) outrank
contextual subordinate clauses (``advcl``), and a dozen common reporting frames
(``warn of``, ``call on``, ``accuse of``, ``hit out at``) are handled explicitly
because their reported content is not in a clausal complement at all.

Paragraph boundaries override dependency edges throughout. They are authorial
structure, and a reported-content span must never jump across one.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc, Span, Token

from .lexicons import CLAUSAL, NP_KEEP, TITLES

__all__ = [
    "expand_speaker",
    "cue_subject",
    "cue_content",
    "is_finite_clause",
    "attribution_span",
]


def expand_speaker(tok, doc):
    """Grow a token into the full speaker noun phrase."""
    # relative pronouns point back to the noun they modify
    if tok.tag_ in ("WDT", "WP", "WP$") or tok.text.lower() in ("which","who","that","whom"):
        # In "a review of universities, which recommended ...", parsers often
        # attach `which` to `universities`. Prefer a nearby document-source noun.
        source_heads = {"review", "report", "study", "survey", "paper", "analysis",
                        "inquiry", "audit", "investigation", "submission", "statement"}
        sent_start = tok.sent.start
        source_left = [t for t in doc[max(sent_start, tok.i - 20):tok.i]
                       if t.lemma_.lower() in source_heads]
        if source_left:
            source = source_left[-1]
            return doc[source.i:source.i + 1]

        anchor = tok.head
        while anchor is not None and anchor.pos_ in ("VERB","AUX") and anchor != anchor.head:
            anchor = anchor.head
        if anchor is not None and anchor.pos_ in ("NOUN", "PROPN"):
            tok = anchor
        else:
            return None

    # Preserve coordinated named subjects (`both the AWU and MEU`) as one
    # source mention. A tight NER lookup would otherwise retain only one union.
    coord = [tok]
    if tok.dep_ == "conj" and tok.head.pos_ in ("NOUN", "PROPN"):
        coord.append(tok.head)
    coord.extend(ch for ch in tok.children
                 if ch.dep_ == "conj" and ch.pos_ in ("NOUN", "PROPN"))
    if len({t.i for t in coord}) > 1:
        members = sorted({t for root in coord for t in root.subtree}, key=lambda t: t.i)
        start, end = members[0].i, members[-1].i + 1
        while start > tok.sent.start and doc[start - 1].lower_ in ("both", "either", "neither"):
            start -= 1
        return doc[start:end]

    for ent in doc.ents:                       # prefer a tight NER span
        if ent.start <= tok.i < ent.end and ent.label_ in ("PERSON","ORG","GPE","NORP"):
            lead = ent.start
            while lead > 0 and doc[lead - 1].text.lower().strip(".") in TITLES:
                lead -= 1                      # keep the honorific in the *mention*
            return doc[lead:ent.end]

    toks = [tok]
    for ch in tok.lefts:
        if ch.dep_ in NP_KEEP:
            toks += list(ch.subtree)
    for ch in tok.rights:
        if ch.dep_ in ("compound", "flat"):
            toks += list(ch.subtree)
        if ch.dep_ == "prep" and ch.text.lower() in ("of", "from", "at"):
            toks += list(ch.subtree)
    toks = sorted(set(toks), key=lambda t: t.i)
    while toks and (toks[0].is_punct or toks[0].pos_ == "DET"):
        toks = toks[1:]
    while toks and toks[-1].is_punct:
        toks = toks[:-1]
    return doc[toks[0].i:toks[-1].i + 1] if toks else doc[tok.i:tok.i + 1]


def cue_subject(cue_span, doc):
    """Dependency-based speaker for a cue span."""
    if cue_span.text.lower().startswith("according to"):
        # Walk past determiners and pre-modifiers to the head of the noun phrase
        i = cue_span.end
        while i < len(doc) and doc[i].pos_ in ("DET", "ADV", "ADJ", "PUNCT", "NUM"):
            i += 1
        if i >= len(doc):
            return None
        head = doc[i]
        while (head.head is not head and head.head.i > cue_span.end
               and head.dep_ in ("compound", "amod", "nmod", "poss", "det")):
            head = head.head
        return expand_speaker(head, doc)

    v = cue_span.root

    # In `Paul Farrow was quoted as saying`, the passive subject is still the
    # source of the quoted words. This is a lexical exception to ordinary
    # passives, where the grammatical subject is the target, not the speaker.
    if "quoted" in cue_span.text.lower():
        for ch in v.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                return expand_speaker(ch, doc)

    # In fronted passive questions (`Asked about X, Mr Dutton remained ...`),
    # the following named person is the respondent, not the unidentified asker.
    # Without a by-agent there is no source actor to attribute.
    if (v.lemma_.lower() == "ask" and v.tag_ == "VBN"
            and not any(ch.dep_ in ("agent", "nsubj") for ch in v.children)):
        return None

    # In passive "Marcus was told", the grammatical subject is the recipient,
    # not the source. Use an explicit by-agent when available; otherwise leave
    # the speaker unresolved rather than attributing the information to Marcus.
    if v.lemma_.lower() == "tell" and any(ch.dep_ == "nsubjpass" for ch in v.children):
        for ch in v.children:
            if ch.dep_ == "agent":
                for g in ch.children:
                    if g.dep_ == "pobj":
                        return expand_speaker(g, doc)
        return None

    # For ordinary passives, an explicit by-agent is the speaker. With no
    # agent, leave the speaker unresolved; `unions were slammed/accused` names
    # the target of criticism, not whoever voiced it.
    passive_subject = any(ch.dep_ == "nsubjpass" for ch in v.children)
    if passive_subject:
        for ch in v.children:
            if ch.dep_ == "agent":
                for g in ch.children:
                    if g.dep_ == "pobj":
                        return expand_speaker(g, doc)
        return None

    for ch in v.children:                                   # canonical subject
        if ch.dep_ == "nsubj":
            return expand_speaker(ch, doc)
    for ch in v.children:                                   # passive: "reported by X"
        if ch.dep_ == "agent":
            for g in ch.children:
                if g.dep_ == "pobj":
                    return expand_speaker(g, doc)
    # Conjoined / subordinate cue: "Meta released X and urged people to ...".
    # `urged` is a conj of `released`, whose nsubj is the real speaker.
    h = v.head
    seen = 0
    while h is not None and seen < 5:
        for ch in h.children:
            if ch.dep_ in ("nsubj", "nsubjpass"):
                return expand_speaker(ch, doc)
        if h.head is h:
            break
        h = h.head
        seen += 1
    # Inverted subject: `"...", said Ms Chen`. This happens only with a FINITE verb.
    # For a participle -- "..., describing a kind of view and informational
    # perspective" -- the post-verbal dependent is the verb's OBJECT, and taking it
    # as the speaker invents an actor out of whatever was being described.
    if "Fin" in v.morph.get("VerbForm") or v.tag_ in ("VBD", "VBZ", "VBP"):
        for ch in v.rights:
            if ch.dep_ in ("dobj","attr","oprd") and ch.pos_ in ("PROPN","NOUN","PRON"):
                return expand_speaker(ch, doc)
    return None


def cue_content(cue_span, doc, allow_np=False):
    """Return the reported content associated with an attribution cue.

    Selection is frame-aware. Core complements (`ccomp`) outrank contextual
    subordinate clauses (`advcl`), and common prepositional reporting frames are
    handled explicitly.
    """
    v = cue_span.root

    if cue_span.text.lower().startswith(("according to", "in the words of", "as stated by")):
        return v.sent

    # Parenthetical attribution: "The system works, the study said, through ...".
    if v.dep_ in ("parataxis", "dep") and v.head is not v:
        return v.sent

    # Hard paragraph boundaries override dependency edges. They are
    # authorial structure; a reported-content span must never jump across one.
    before = doc.text[:v.idx]
    left_breaks = list(re.finditer(r"\n\s*\n", before))
    paragraph_start = left_breaks[-1].end() if left_breaks else 0
    right_break = re.search(r"\n\s*\n", doc.text[v.idx:])
    paragraph_end = (v.idx + right_break.start()) if right_break else len(doc.text)

    def child_span(child):
        subtree = [t for t in child.subtree
                   if paragraph_start <= t.idx < paragraph_end]
        if not subtree:
            return None
        subtree.sort(key=lambda t: t.i)

        # `urging Australia to invest in submarines, which both parties opposed`:
        # the comma-relative is journalist narration, not part of what was urged.
        for j, tok in enumerate(subtree[:-1]):
            if tok.text != ",":
                continue
            following = next((t for t in subtree[j + 1:] if not t.is_space), None)
            if following is not None and following.lower_ in {
                    "which", "who", "whom", "whose", "where"}:
                subtree = subtree[:j]
                break

        if not subtree:
            return None
        return doc[subtree[0].i:subtree[-1].i + 1]

    lemma = v.lemma_.lower()

    # Participial frame: "..., claiming to have been bullied, unlawfully
    # dismissed, and witnessing tampering ...". Parsers frequently leave the
    # coordinated predicates outside the first xcomp subtree, so retain the
    # complete remainder of the reporting clause.
    if lemma == "claim" and (v.tag_ == "VBG" or "Ger" in v.morph.get("VerbForm")):
        local = [t for t in doc[v.i + 1:v.sent.end] if t.idx < paragraph_end]
        if local:
            return doc[local[0].i:local[-1].i + 1]

    # Frame: "Dutton flagged further announcements were coming about sites ...".
    # Parser attachment varies for the trailing `about` phrase, so use the
    # complete post-cue remainder of this reporting sentence.
    if lemma == "flag" and v.i + 1 < v.sent.end:
        return doc[v.i + 1:v.sent.end]

    # Frame: `the entrepreneur warned of blackouts`.
    if lemma == "warn":
        args = [ch for ch in v.children
                if (ch.dep_ == "prep" and ch.lower_ in ("of", "about", "against"))]
        if args:
            toks = [t for ch in args for t in ch.subtree
                    if paragraph_start <= t.idx < paragraph_end]
            if toks:
                return doc[min(t.i for t in toks):max(t.i for t in toks) + 1]

    # Evaluative stance frames. In active voice, keep the target object and
    # any charge/reason complement. In passive voice, the grammatical subject is
    # part of the attributed proposition, while a by-agent (if present) is the
    # actual source.
    if lemma in {"slam", "attack", "condemn", "criticise", "criticize"}:
        args = [ch for ch in v.children
                if ch.dep_ in ("dobj", "obj", "nsubjpass")
                or (ch.dep_ == "prep" and ch.lower_ in
                    ("for", "over", "about", "at"))]
        if args:
            toks = [t for ch in args for t in ch.subtree
                    if paragraph_start <= t.idx < paragraph_end]
            return doc[min(t.i for t in toks):max(t.i for t in toks) + 1]

    # Frame: "Kenny accused [the government] [of being ... and ignoring ...]".
    # Keep the direct/passive target and the `of` complement as one faithful span.
    if lemma == "accuse":
        args = [ch for ch in v.children
                if ch.dep_ in ("dobj", "obj", "nsubjpass")
                or (ch.dep_ == "prep" and ch.lower_ == "of")]
        if args:
            toks = [t for ch in args for t in ch.subtree]
            return doc[min(t.i for t in toks):max(t.i for t in toks) + 1]

    # `hit out at the plans`, `advocated for nuclear`, `lobbied the
    # government to lift the ban`, and `pitched the plan as a policy`.
    if lemma in {"hit", "advocate", "lobby", "pitch"}:
        wanted_preps = {"hit": {"at", "over"},
                        "advocate": {"for"},
                        "lobby": {"for"},
                        "pitch": {"as"}}[lemma]
        args = [ch for ch in v.children
                if ch.dep_ in ("dobj", "obj", "xcomp", "ccomp")
                or (ch.dep_ == "prep" and ch.lower_ in wanted_preps)]
        if args:
            toks = [t for ch in args for t in ch.subtree
                    if paragraph_start <= t.idx < paragraph_end]
            return doc[min(t.i for t in toks):max(t.i for t in toks) + 1]

    # Frame: "BHP called on the government to remove bans".
    if lemma == "call":
        args = [ch for ch in v.children
                if (ch.dep_ == "prep" and ch.lower_ in ("on", "for"))
                or (ch.dep_ in ("xcomp", "ccomp", "advcl", "acl")
                    and ch.i > v.i)]
        if args:
            toks = [t for ch in args for t in ch.subtree]
            return doc[min(t.i for t in toks):max(t.i for t in toks) + 1]

    allowed = set(CLAUSAL)
    if allow_np:
        allowed.update(("dobj", "obj", "attr", "oprd", "pobj"))

    candidates = []
    priority = {
        "ccomp": 0,
        "xcomp": 1,
        "dobj": 2,
        "obj": 2,
        "attr": 2,
        "oprd": 2,
        "pobj": 2,
        "parataxis": 3,
        "acl": 4,
        "advcl": 5,
    }

    for ch in v.children:
        if ch.dep_ not in allowed:
            continue
        if ch.dep_ == "xcomp" and ch.pos_ not in ("VERB", "AUX", "ADJ"):
            continue
        span = child_span(ch)
        if span is not None:
            candidates.append((priority.get(ch.dep_, 99), -len(span), span))

    if not candidates:
        return None

    # Prefer the semantic complement. Length breaks ties only within one role.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]

def is_finite_clause(span):
    """A finite verb plus its own subject — the test for real reported speech."""
    if span is None:
        return False
    return (any(t.pos_ in ("VERB","AUX") and "Fin" in t.morph.get("VerbForm") for t in span)
            and any(t.dep_ in ("nsubj","nsubjpass","expl","csubj") for t in span))

def attribution_span(cue_span, doc):
    """Character extent of a phrase cue plus its object: "according to <NP>".

    Excising the NER span alone is not enough: for "according to a recently
    published RMIT University study" the entity is only "RMIT University", so
    cutting it leaves "a", "study" and other shards in the reported content.
    The whole noun phrase has to be removed as one contiguous region.
    """
    i = cue_span.end
    while i < len(doc) and doc[i].pos_ in ("DET", "ADV", "ADJ", "PUNCT", "NUM"):
        i += 1
    if i >= len(doc):
        return (cue_span.start_char, cue_span.end_char)
    head = doc[i]
    while (head.head is not head and head.head.i > cue_span.end
           and head.dep_ in ("compound", "amod", "nmod", "poss", "det")):
        head = head.head
    sub = [t for t in sorted(head.subtree, key=lambda t: t.i) if t.i >= cue_span.end]
    end = sub[-1].idx + len(sub[-1]) if sub else cue_span.end_char
    return (cue_span.start_char, end)
