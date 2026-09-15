"""Layers 4 and 5 — actor typing, then coreference and alias resolution.

These two layers were separate cells in the notebook and are merged here for a
specific reason: each redefined ``TITLES`` and ``PRONOUN_GENDER`` at its own top
level, and the two ``PRONOUN_GENDER`` tables were not identical. Because Layer 5
executed second, its table silently won at corpus-run time even though
``entity_type`` (Layer 4) had been written against the other one. Both now come
from :mod:`newsvoice.lexicons`, and the layers share one module so the
duplication cannot reappear.

:func:`entity_type` assigns ``PERSON`` / ``ORGANISATION`` / ``GROUP`` from NER
labels, gazetteers and morphology. :func:`canonicalise` maps surface mentions to
a document-level canonical actor: ``Speaker Mention`` records what the text said,
``Actor Canonical Name`` records who it referred to.
"""

from __future__ import annotations

import collections
import re

from spacy.tokens import Doc, Span

from .lexicons import (
    AGE_EPITHET,
    DEFINITE_ORG,
    FEMALE_TITLES,
    GROUP_HEADS,
    INSTITUTION,
    MALE_TITLES,
    ORG_HEADS,
    ORG_METONYMS,
    ORG_SUFFIX,
    PERSON_EPITHET_HEADS,
    PRONOUN_GENDER,
    TITLES,
)
from .quotes import overlaps

__all__ = ["entity_type", "canonicalise", "strip_titles"]


def entity_type(span, doc):
    """PERSON / ORGANISATION / GROUP, from NER labels, gazetteers and morphology."""
    labels = {e.label_ for e in doc.ents
              if overlaps((e.start_char, e.end_char), (span.start_char, span.end_char))}
    if "PERSON" in labels:
        return "PERSON"
    if labels & {"ORG", "GPE", "LOC", "FAC"}:
        return "ORGANISATION"

    head, low = span.root, span.text.lower()
    if head.pos_ == "PRON":
        return "GROUP" if PRONOUN_GENDER.get(head.text.lower()) == "p" else "PERSON"
    if "NORP" in labels:
        return "GROUP"
    if ORG_SUFFIX.search(low):
        return "ORGANISATION"

    hl = head.lemma_.lower()
    if hl in GROUP_HEADS or low.strip() in GROUP_HEADS:
        return "GROUP"
    if hl in ORG_HEADS:
        return "ORGANISATION"

    # --- morphology: a plural common noun heading the phrase is a collective
    if head.pos_ == "NOUN" and "Plur" in head.morph.get("Number"):
        return "GROUP"

    if any(t.text.lower().strip(".") in TITLES for t in span):
        return "PERSON"
    if head.pos_ == "PROPN":
        return "ORGANISATION"
    if head.pos_ == "NOUN":
        return "GROUP" if head.text.lower().endswith("s") else "ORGANISATION"
    return "PERSON"


def _org_metonym_head(text):
    low = re.sub(r"^(the|a|an)\s+", "", text.lower().strip())
    toks = low.split()
    return bool(toks) and toks[-1] in ORG_METONYMS


def strip_titles(name):
    toks = [w for w in name.split() if w.lower().strip(".") not in TITLES]
    return " ".join(toks).strip(" ,.") or name.strip()


def _is_person_epithet(text):
    low = text.lower().strip()
    low = re.sub(r"^(the|a|an)\s+", "", low)
    if AGE_EPITHET.match(low):
        return True
    words = low.split()
    if not words:
        return False
    # Handles both `Opposition leader` and `leader of the Opposition`.
    return words[-1] in PERSON_EPITHET_HEADS or words[0] in PERSON_EPITHET_HEADS


def _acronym_of(name):
    caps = [w[0] for w in re.findall(r"\b[A-Z][a-z]+", name)]
    return "".join(caps) if len(caps) >= 2 else None


def canonicalise(mentions, background=None):
    """Two-pass resolver.

    `mentions` :: [{"text", "type", "start", "end"}] in document order.

    Pass 1 builds a document-level inventory: every named person and organisation,
    their surnames, acronyms, and whatever gender evidence the text itself supplies
    (honorifics, and pronouns appearing near a name). Pass 2 then resolves each
    mention against that inventory, and crucially may look **forward**.
    """
    # ---------------- pass 1: inventory ----------------
    people, orgs = [], []          # (canonical, position)
    by_surname, by_acronym = {}, {}
    gender = {}

    # `background` is every named entity in the document, not just the ones that
    # ended up as speakers. Without it a full name that only ever appears as the
    # *object* of a cue -- "according to the study's lead author Dr Angel Zhong" --
    # never enters the inventory, and later "Dr Zhong" cannot expand to it.
    for mn in list(background or []) + list(mentions):
        txt = (mn.get("text") or "").strip()
        pos = mn.get("start") if mn.get("start") is not None else 0
        low = txt.lower().strip(".,")
        # Epithets and definite descriptions must NOT enter the inventory: they are
        # things to be *resolved*, not antecedents. Left in, "The 21-year-old
        # actress" is registered as a person and then resolves to itself, and
        # "The company" likewise -- the resolver silently becomes a no-op for
        # exactly the mentions it exists to handle.
        if not txt or low in PRONOUN_GENDER or low in DEFINITE_ORG or _is_person_epithet(txt):
            continue
        etype = mn.get("type")

        if etype == "PERSON" and any(w[:1].isupper() for w in txt.split()):
            for w in txt.split():
                wl = w.lower().strip(".")
                if wl in MALE_TITLES:
                    gender[strip_titles(txt)] = "m"
                elif wl in FEMALE_TITLES:
                    gender[strip_titles(txt)] = "f"
            canon = strip_titles(txt)
            parts = canon.split()
            if len(parts) > 1:
                by_surname.setdefault(parts[-1].lower(), canon)
                by_surname[parts[-1].lower()] = canon
            people.append((canon, pos))
        elif etype in ("ORGANISATION", "GROUP") and any(w[:1].isupper() for w in txt.split()):
            orgs.append((txt, pos))
            ac = _acronym_of(txt)
            if ac:
                by_acronym.setdefault(ac, txt)

    # ---- collapse the inventory before resolving ----------------------------
    # Pass 1 registers "Dr Zhong" as its own person because the surname table is
    # still being built when it is seen. Left alone, the inventory holds both
    # "Angel Zhong" and "Zhong", and a later pronoun resolves to whichever is
    # nearer -- so the same speaker ends up with two canonical names in one
    # article, which quietly breaks any per-actor aggregation downstream.
    people = [(by_surname.get(nm.lower(), nm) if len(nm.split()) == 1 else nm, pos)
              for nm, pos in people]
    seen = set()
    people = [(nm, pos) for nm, pos in people
              if not (nm.lower() in seen and seen.add(nm.lower()) is None)
              or seen.add(nm.lower()) is None]
    gender = {by_surname.get(k.lower(), k) if len(k.split()) == 1 else k: v
              for k, v in gender.items()}

    # ---------------- pass 2: resolve ----------------
    org_frequency = collections.Counter(nm for nm, _ in orgs)

    # Formal union names are often introduced after an opening `they` or
    # `both unions`. Build their joint canonical label up front so the resolver
    # can use its normal forward-looking inventory without falling back to the
    # political modifier `Labor`.
    union_pool = []
    for nm, nm_pos in orgs:
        canon_nm = by_acronym.get(nm, nm)
        low_nm = canon_nm.lower()
        if ("union" in low_nm and "unions" not in low_nm
                and canon_nm.lower() not in {x[0].lower() for x in union_pool}):
            union_pool.append((canon_nm, nm_pos))
    union_joint = (" and ".join(nm for nm, _ in union_pool[:2])
                   if len(union_pool) >= 2 else None)
    first_union_pos = min((p for _, p in union_pool), default=None)

    recent = {"PERSON": None, "ORGANISATION": None, "GROUP": None}
    by_key = {}
    out = []

    def nearest(pool, pos, want_gender=None):
        """Closest inventory entry, preferring backward, then forward."""
        cands = pool
        if want_gender:
            g = [c for c in cands if gender.get(c[0]) == want_gender]
            if g:
                cands = g
        if not cands:
            return None
        back = [c for c in cands if c[1] <= pos]
        if back:
            return max(back, key=lambda c: c[1])[0]
        return min(cands, key=lambda c: c[1])[0]      # forward fallback

    for mn in mentions:
        txt = (mn.get("text") or "").strip()
        etype = mn.get("type")
        pos = mn.get("start") if mn.get("start") is not None else 0
        low = txt.lower().strip(".,")

        # ---- pronouns ------------------------------------------------
        if low in PRONOUN_GENDER:
            g = PRONOUN_GENDER[low]
            if g in ("m", "f"):
                ante = nearest(people, pos, want_gender=g) or nearest(people, pos) \
                       or recent["PERSON"]
            elif g == "p":
                early_union_reference = (union_joint is not None
                                         and first_union_pos is not None
                                         and pos < first_union_pos)
                ante = (union_joint if early_union_reference else None) or (
                    recent["GROUP"] or nearest(orgs, pos)
                    or recent["ORGANISATION"] or nearest(people, pos))
            else:
                ante = nearest(orgs, pos) or recent["ORGANISATION"] or nearest(people, pos)
            out.append(ante or txt)
            continue

        # ---- definite organisation descriptions ----------------------
        # `expand_speaker` removes leading determiners, so "the university" may
        # arrive here as bare `university` with definite=True. In a single-body
        # article, the most frequently named organisation is the safest alias.
        bare_definite_org = {item.removeprefix("the ") for item in DEFINITE_ORG}
        if mn.get("definite") is True and low in bare_definite_org and org_frequency:
            canon = org_frequency.most_common(1)[0][0]
            recent["ORGANISATION"] = canon
            out.append(canon)
            continue
        if low in DEFINITE_ORG:
            out.append(nearest(orgs, pos) or recent["ORGANISATION"] or txt)
            continue

        # ---- union descriptions ---------------------------------------
        if low in {"unions", "both unions", "the unions"} and union_joint:
            recent["ORGANISATION"] = union_joint
            out.append(union_joint)
            continue
        if (mn.get("definite") is True and low.endswith(" union") and union_pool):
            recent_union = (recent["ORGANISATION"]
                            if recent["ORGANISATION"]
                            and "union" in recent["ORGANISATION"].lower()
                            and " and " not in recent["ORGANISATION"].lower()
                            else None)
            canon = recent_union or nearest(union_pool, pos)
            recent["ORGANISATION"] = canon
            out.append(canon)
            continue

        # ---- political organisation descriptions ---------------------
        if low in {"opposition", "the opposition"}:
            coalition = next((nm for nm, _ in orgs
                              if nm.lower() in {"coalition", "the coalition"}), None)
            if coalition:
                recent["ORGANISATION"] = coalition
                out.append(coalition)
                continue

        # ---- person epithets: "the Wednesday star", "Atlassian CEO" ---
        # Only DEFINITE epithets are anaphoric. "The 21-year-old actress" points back
        # to someone already named; "A company spokesperson" introduces a NEW referent,
        # and resolving it puts one person's words in another's mouth -- e.g. crediting
        # Google's denial to the engineer who made the claim being denied.
        if mn.get("definite") is False:
            pass
        elif _is_person_epithet(txt):

            # A media organisation can modify a human role: "Sky News host".
            # The capitalised organisation prefix must not prevent person resolution.
            named_org_prefix = any(nm.lower() in low for nm, _ in orgs)
            unresolved_named_token = any(
                w.lower().strip(".") not in TITLES and w[:1].isupper()
                and w.lower() not in {t.lower() for t in PERSON_EPITHET_HEADS}
                for w in txt.split()[1:])

            if named_org_prefix or not unresolved_named_token:
                forward_leadership = low.startswith("head of ") and named_org_prefix
                if forward_leadership:
                    # Identify the organisation named in the role phrase, then
                    # prefer a person appearing immediately after another mention
                    # of that same organisation (`AWU national secretary Paul Farrow`).
                    role_acronyms = [ac for ac in by_acronym
                                     if re.search(r"\b" + re.escape(ac) + r"\b", txt)]
                    target_orgs = {by_acronym[ac].lower() for ac in role_acronyms}
                    target_positions = [org_pos for org_name, org_pos in orgs
                                        if by_acronym.get(org_name, org_name).lower()
                                        in target_orgs]
                    affiliated = [item for item in people
                                  if item[1] > pos and any(
                                      0 <= item[1] - org_pos <= 160
                                      for org_pos in target_positions)]
                    forward_people = [item for item in people if item[1] > pos]
                    ante = (min(affiliated, key=lambda item: item[1])[0]
                            if affiliated else
                            min(forward_people, key=lambda item: item[1])[0]
                            if forward_people else nearest(people, pos))
                else:
                    ante = nearest(people, pos)

                if ante:

                    out.append(ante)

                    recent["PERSON"] = ante

                    continue

        # ---- organisational metonymy ----------------------------------
        # Source constructions such as "a submission from BHP" name the producing
        # organisation inside a metonymic speaker phrase. Prefer that embedded body.
        metonym_words = set(re.findall(r"[a-z]+", low)) & ORG_METONYMS
        embedded_acronyms = [ac for ac in by_acronym
                             if re.search(r"\b" + re.escape(ac) + r"\b", txt)]
        if metonym_words and embedded_acronyms:
            ante = " and ".join(dict.fromkeys(by_acronym[ac]
                                              for ac in embedded_acronyms))
            recent["ORGANISATION"] = ante
            out.append(ante)
            continue
        source_org = next(
            (nm for nm, _ in orgs
             if re.search(r"\b(?:from|by)\b.{0,80}\b" + re.escape(nm.lower()) + r"\b", low)),
            None)
        if metonym_words and source_org:
            recent["ORGANISATION"] = source_org
            out.append(source_org)
            continue

        # "the study", "recently published RMIT University study" -> the body
        # behind it. If the phrase already contains a named organisation, that
        # wins; otherwise fall back to the nearest one in the document.
        if _org_metonym_head(txt):
            contained = []
            for nm, _ in orgs:
                if (nm.lower() in low and not _org_metonym_head(nm)
                        and nm.lower() not in {x.lower() for x in contained}):
                    contained.append(nm)
            # Expand acronyms even when they are embedded in a metonym:
            # `the AWU post` and `the AWU report` both resolve to the full union.
            contained = [by_acronym.get(nm, nm) for nm in contained]
            inner = (" and ".join(contained) if len(contained) > 1
                     else contained[0] if contained else None)
            # The fallback must be an *institution*, not merely the nearest proper
            # noun: "...in the US" would otherwise capture "the study". If no
            # institution is present, leave the surface form alone rather than
            # guess -- a wrong actor is worse than an unresolved one.
            # Resolve only an institution explicitly contained in the source
            # phrase. A bare "review" must not inherit an unrelated nearby body.
            ante = inner
            # A definite bare source (`the report`) inherits the most recent
            # established organisation; an indefinite source remains unresolved.
            if ante is None and mn.get("definite") is True:
                ante = recent["ORGANISATION"]
            if ante:
                recent["ORGANISATION"] = ante
                out.append(ante)
                continue

        # ---- coordinated acronyms -------------------------------------
        acronym_members = re.findall(r"\b[A-Z]{2,6}\b", txt)
        if len(acronym_members) >= 2 and all(ac in by_acronym for ac in acronym_members):
            canon = " and ".join(dict.fromkeys(by_acronym[ac] for ac in acronym_members))
            recent["ORGANISATION"] = canon
            out.append(canon)
            continue

        # ---- acronyms -------------------------------------------------
        if txt.isupper() and 2 <= len(txt) <= 6 and txt in by_acronym:
            canon = by_acronym[txt]
            recent["ORGANISATION"] = canon
            out.append(canon)
            continue

        # ---- named / descriptive --------------------------------------
        if etype == "PERSON":
            canon = strip_titles(txt)
            parts = canon.split()
            if len(parts) == 1 and parts[0].lower() in by_surname:
                canon = by_surname[parts[0].lower()]
            recent["PERSON"] = canon
        else:
            canon = by_key.setdefault(low, txt)
            recent["ORGANISATION" if etype == "ORGANISATION" else "GROUP"] = canon

        out.append(canon)
    return out
