"""Layer 8 — global quote-to-cue assignment.

Which cue owns a given quotation is a **global** decision. Resolving it locally,
one cue at a time, lets whichever cue is processed first take a quotation that
belongs to a later one, and the error is unrecoverable downstream. This layer
therefore runs over the whole document before any output row is built.

Decision order, strongest evidence first:

1. **Continuation of an open turn.** A quotation opening a paragraph that
   directly follows an unclosed quotation belongs to the same speaker.
2. **Same sentence.** A cue in the quotation's own sentence owns it. This is the
   rule that prevents greedy stealing across sentence boundaries.
3. **Nearest cue across a boundary**, but only when the quotation's own sentence
   has no cue of its own and contains nothing but the quotation.
4. **Paragraph veto.** Where paragraph structure exists, a continuation may not
   cross more than one paragraph break.

Sentence boundaries need two repairs before they can carry any of this, both
handled by :func:`sentence_units`.
"""

from __future__ import annotations

import re

from spacy.tokens import Doc

__all__ = ["sentence_units", "paragraph_units", "assign_quotes"]


def _spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def sentence_units(doc, text, quote_spans):
    """Sentence spans that are safe to reason about, as [(start_char, end_char)].

    spaCy's boundaries are the starting point but need two repairs before they can
    carry an assignment decision:

      1. A boundary landing INSIDE a quotation is spurious -- "Is it too late? No,
         I don't think it's too late," is one quoted utterance, not two sentences.
      2. `"Quote," he said.` is routinely split after the closing quote, which
         separates a quote from the cue that introduces it. Any sentence that begins
         with attribution leftovers (a short fragment ending in a reporting verb) is
         merged back into the preceding one.
    """
    bounds = [(s.start_char, s.end_char) for s in doc.sents]
    if not bounds:
        return [(0, len(text))]

    # --- repair 1: never break inside a quotation
    merged = []
    for b in bounds:
        if merged and any(q[0] < merged[-1][1] < q[1] for q in quote_spans):
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)

    # --- repair 2: reattach a trailing attribution fragment to its quote
    out = []
    for s0, s1 in merged:
        frag = text[s0:s1].strip()
        prev_ends_quote = out and text[out[-1][0]:out[-1][1]].rstrip().endswith(
            ('"', '\u201d', "'", '\u2019'))
        short_attrib = len(frag) < 90 and re.match(
            r'^[A-Z]?[^.!?]{0,80}\b(said|says|added|explained|noted|told|asked|'
            r'replied|continued|wrote|warned|argued|insisted|explains|adds)\b', frag)
        if prev_ends_quote and short_attrib:
            out[-1] = (out[-1][0], s1)
        else:
            out.append((s0, s1))
    return out


def paragraph_units(text):
    """Paragraph spans from blank lines. Returns [] when the text has no structure."""
    if "\n\n" not in text and text.count("\n") < 2:
        return []
    spans, pos = [], 0
    for m in re.finditer(r"\n\s*\n|\n", text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


def _index_of(spans, pos):
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    return -1

def assign_quotes(quote_spans, cue_positions, sents, paras, text,
                  max_sentence_gap=2):
    """Assign each quote to exactly one cue.

    `cue_positions` :: [(cue_index, start_char, end_char)].
    Returns {quote_span: cue_index}; quotes with no owner are absent.

    Decision order, strongest evidence first:

      1. **Same sentence.** A cue in the quote's own sentence owns it. This is the
         rule that fixes greedy stealing: the quote never escapes to another
         sentence's cue while a local one exists.
      2. **Nearest cue across a boundary**, but only when the quote's own sentence
         has NO cue of its own. A quote that lives in a bare sentence is a
         continuation, and belongs to the closest cue within `max_sentence_gap`
         sentences.
      3. **Paragraph veto.** When paragraph structure exists, a continuation may not
         cross more than one paragraph break. Without paragraphs the sentence gap
         alone bounds it, which is why punctuation is the primary signal.

    Ties inside a sentence go to the nearer cue, measured edge-to-edge.
    """
    sent_of_quote = {q: _index_of(sents, q[0]) for q in quote_spans}
    sent_of_cue = {ci: _index_of(sents, c0) for ci, c0, _ in cue_positions}

    cues_in_sent = {}
    for ci, c0, c1 in cue_positions:
        cues_in_sent.setdefault(sent_of_cue[ci], []).append((ci, c0, c1))

    assignment = {}
    for q in quote_spans:
        si = sent_of_quote[q]

        # --- rule 1: continue an already established open quotation turn
        # Run this before local-cue matching because sentence-boundary repair can
        # combine a closed quotation with the reporter paragraph that follows it.
        prior_quotes = [p for p in quote_spans
                        if p[1] <= q[0] and p in assignment]
        if prior_quotes:
            previous = max(prior_quotes, key=lambda p: p[1])
            only_breaks = not text[previous[1]:q[0]].strip()
            previous_is_open = not text[previous[0]:previous[1]].rstrip().endswith(
                ('"', '”', "'", '’'))
            if paras:
                pp = _index_of(paras, previous[0])
                qp = _index_of(paras, q[0])
                adjacent_paragraph = pp >= 0 and qp == pp + 1
            else:
                adjacent_paragraph = only_breaks
            if previous_is_open and only_breaks and adjacent_paragraph:
                assignment[q] = assignment[previous]
                continue

        # --- rule 2: a cue in the quote's own sentence, outside the quote itself
        local = [(ci, c0, c1) for ci, c0, c1 in cues_in_sent.get(si, [])
                 if not (q[0] <= c0 < q[1])]
        if local:
            def local_key(item):
                ci, c0, c1 = item
                distance = min(abs(c0 - q[1]), abs(q[0] - c1))
                between = text[q[1]:c0].lower() if c0 >= q[1] else ""
                nested_after_quote = bool(re.search(
                    r"\b(?:when|because|although|while|whereas|if|since)\b",
                    between))
                return (nested_after_quote, distance)

            ci = min(local, key=local_key)[0]
            assignment[q] = ci
            continue

        # --- rule 3: continuation -- only for a bare quote sentence with no cue
        # Embedded fragments such as `the service operates through “our values”.`
        # must not be pulled into an attribution in the next sentence.
        if si < 0:
            continue
        sa, sb = sents[si]
        sentence_quotes = sorted(
            (x for x in quote_spans if sa <= x[0] and x[1] <= sb),
            key=lambda x: x[0])
        outside, cursor = [], sa
        for qa, qb in sentence_quotes:
            outside.append(text[cursor:qa])
            cursor = qb
        outside.append(text[cursor:sb])
        residue = "".join(outside).strip(" \t\r\n,.:;!?—–-()[]{}\"'“”‘’")
        if residue:
            continue

        best, best_cost = None, None
        for ci, c0, c1 in cue_positions:
            if q[0] <= c0 < q[1]:
                continue
            cj = sent_of_cue[ci]
            if cj < 0 or si < 0:
                continue
            gap = abs(cj - si)
            if gap == 0 or gap > max_sentence_gap:
                continue
            # --- rule 3: paragraph veto
            if paras:
                pi, pj = _index_of(paras, q[0]), _index_of(paras, c0)
                if pi >= 0 and pj >= 0 and abs(pi - pj) > 1:
                    continue
            between = text[min(c1, q[0]):max(c1, q[0])]
            cost = (gap, len(between.strip()))
            if best_cost is None or cost < best_cost:
                best, best_cost = ci, cost
        if best is not None:
            assignment[q] = best
    return assignment
