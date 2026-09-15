"""Gazetteers and lexical resources shared across layers.

Every constant used by more than one layer lives here and is imported, never
redefined. In the notebook this module's contents were spread across four cells
and two of them were defined twice, with ``PRONOUN_GENDER`` diverging between
its two definitions (see ``docs/MIGRATION.md``). Centralising them makes that
class of bug impossible rather than merely unlikely.
"""

from __future__ import annotations

import re

__all__ = [
    "TITLES",
    "MALE_TITLES",
    "FEMALE_TITLES",
    "PRONOUN_GENDER",
    "CORE_CUE_LEMMAS",
    "PERIPHERAL_CUE_LEMMAS",
    "CUE_LEMMAS",
    "MULTIWORD_CUES",
    "NP_CUES",
    "ORG_HEADS",
    "GROUP_HEADS",
    "ORG_SUFFIX",
    "DEFINITE_ORG",
    "PERSON_EPITHET_HEADS",
    "AGE_EPITHET",
    "ORG_METONYMS",
    "INSTITUTION",
    "CLAUSAL",
    "NP_KEEP",
]

# --------------------------------------------------------------------------
# Honorifics
# --------------------------------------------------------------------------

#: Honorifics stripped from a canonical name but retained in the surface mention.
TITLES = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "dame",
    "lord", "rev", "senator", "minister", "president", "premier", "mayor",
    "judge", "justice", "cr",
})

#: Honorifics that supply gender evidence for coreference.
MALE_TITLES = frozenset({"mr", "sir", "lord"})
FEMALE_TITLES = frozenset({"mrs", "ms", "miss", "dame"})

# --------------------------------------------------------------------------
# Pronouns
# --------------------------------------------------------------------------

#: Pronoun -> gender/number code: ``m``ale, ``f``emale, ``p``lural, ``n``euter.
#:
#: Reflexives are included. The notebook defined this table twice — once in the
#: Layer 4 cell without reflexives and once in the Layer 5 cell with them — so
#: whether ``entity_type`` classified "themselves" as GROUP or PERSON depended
#: on which cell had been executed most recently. The Layer 5 form is both the
#: one that was in effect during full corpus runs (its cell ran later) and the
#: semantically correct one, so it is the form kept here.
PRONOUN_GENDER = {
    "he": "m", "him": "m", "his": "m", "himself": "m",
    "she": "f", "her": "f", "hers": "f", "herself": "f",
    "they": "p", "them": "p", "their": "p", "themselves": "p",
    "it": "n", "its": "n", "itself": "n",
}

# --------------------------------------------------------------------------
# Attribution cues (Layer 2)
# --------------------------------------------------------------------------

#: Tier 1 — verbs that report speech in essentially every context.
CORE_CUE_LEMMAS = frozenset({
    "say", "tell", "add", "write", "note", "explain", "claim", "report",
    "announce", "warn", "argue", "state", "acknowledge", "stress", "admit",
    "reveal", "insist", "declare", "reply", "respond", "comment", "quip",
    "testify", "concede", "assert", "reiterate", "emphasise", "emphasize",
    "allege", "contend", "maintain", "complain", "lament", "unveil",
    "identify", "deny", "continue", "recount", "detail", "outline", "confirm",
    "post", "tweet", "publish", "brief", "attribute",
})

#: Tier 2 — reportive only when accompanied by a quotation or a finite clause.
PERIPHERAL_CUE_LEMMAS = frozenset({
    "believe", "find", "describe", "recommend", "support", "call", "ask",
    "urge", "agree", "observe", "conclude", "predict", "estimate", "welcome",
    "consider", "accuse", "deny", "promise", "point", "highlight", "suggest",
    "criticise", "criticize", "praise", "flag", "question", "caution", "hint",
    "propose", "recall", "echo", "hope", "think", "fear", "slam", "revile",
    "express", "back", "laud", "attack", "condemn", "advocate", "pitch",
    "lobby", "hit",
})

CUE_LEMMAS = CORE_CUE_LEMMAS | PERIPHERAL_CUE_LEMMAS

#: Multi-token attribution phrases, matched with spaCy's ``Matcher``.
MULTIWORD_CUES = (
    "according to",
    "in the words of",
    "as stated by",
    "in a statement",
    "was quoted as saying",
    "quoted as saying",
    "was quoted",
    "told reporters",
)

#: Cues whose reported content may legitimately be a noun phrase, not a clause.
NP_CUES = frozenset({
    "call", "claim", "reveal", "accuse", "urge", "support", "welcome",
    "recommend", "announce", "back", "oppose", "propose", "describe",
    "criticise", "criticize", "praise", "flag", "warn", "slam", "revile",
    "express", "laud", "attack", "condemn", "advocate", "pitch", "lobby",
    "hit",
})

# --------------------------------------------------------------------------
# Actor typing (Layer 4)
# --------------------------------------------------------------------------

#: Common-noun heads that indicate an institutional actor.
ORG_HEADS = frozenset({
    "company", "firm", "organisation", "organization", "agency", "department",
    "government", "commission", "council", "committee", "report", "study",
    "review", "inquiry", "tribunal", "court", "bank", "school", "university",
    "hospital", "ministry", "body", "regulator", "authority", "institute",
    "startup", "business", "team", "board", "union", "party", "coalition",
    "bureau", "office", "paper", "submission", "statement", "spokesperson",
    "spokesman", "spokeswoman",
})

#: Common-noun heads that indicate an unincorporated collective actor.
GROUP_HEADS = frozenset({
    "people", "students", "researchers", "experts", "australians", "fans",
    "viewers", "teachers", "workers", "customers", "artists", "users",
    "parents", "critics", "scientists", "doctors", "farmers", "voters",
    "academics", "campaigners", "advocates", "observers", "analysts",
    "officials", "locals", "residents", "staff", "authors", "respondents",
    "participants", "members",
})

ORG_SUFFIX = re.compile(
    r"\b(inc|ltd|plc|corp|corporation|company|commission|council|"
    r"university|institute|department|ministry|agency|association|"
    r"foundation|group|party|union|bureau|authority|board|centre|"
    r"center|society|committee|office|bank|court)\b",
    re.I,
)

# --------------------------------------------------------------------------
# Coreference and alias resolution (Layer 5)
# --------------------------------------------------------------------------

#: Definite descriptions that stand in for an organisation already named.
DEFINITE_ORG = frozenset({
    "the company", "the firm", "the organisation", "the organization",
    "the group", "the agency", "the department", "the government body",
    "the council", "the commission", "the bank", "the university",
    "the club", "the team", "the party", "the ministry", "the board",
})

#: Role nouns that can stand in for a person: "the Wednesday star", "Atlassian CEO".
PERSON_EPITHET_HEADS = frozenset({
    "star", "actress", "actor", "singer", "player", "author", "director",
    "ceo", "cfo", "cto", "founder", "chief", "boss", "leader", "chairman",
    "chairwoman", "president", "spokesperson", "spokesman", "spokeswoman",
    "professor", "researcher", "scientist", "doctor", "lawyer", "coach",
    "captain", "manager", "minister", "premier", "mayor", "senator",
    "politician", "musician", "artist", "writer", "journalist", "economist",
    "analyst", "expert", "veteran", "host", "presenter", "anchor",
    "broadcaster", "commentator", "secretary", "official", "representative",
    "advocate", "head", "entrepreneur", "businessman", "businesswoman",
    "olympian", "champion", "winner", "survivor", "victim", "mother",
    "father", "widow",
})

AGE_EPITHET = re.compile(r"^\d{1,3}[- ]year[- ]old\b", re.I)

#: Nouns standing by metonymy for the body that produced the thing:
#: "the study found ...", "the report warns ..." -> the university or agency.
ORG_METONYMS = frozenset({
    "study", "report", "research", "review", "survey", "paper", "analysis",
    "inquiry", "audit", "investigation", "findings", "data", "submission",
    "statement", "release", "post", "message", "documents", "filing",
    "ruling", "judgment",
})

INSTITUTION = re.compile(
    r"\b(univers|institut|college|school|commission|council|department|ministry|"
    r"agency|authority|bureau|board|centre|center|foundation|society|association|"
    r"laborator|hospital|company|corp|inc|ltd|plc|group|union|party|committee|"
    r"office|bank|court|tribunal|academy|observatory|network|team|unit)",
    re.I,
)

# --------------------------------------------------------------------------
# Dependency labels (Layer 3)
# --------------------------------------------------------------------------

#: Clausal dependency relations that can carry reported content.
CLAUSAL = ("ccomp", "advcl", "acl", "xcomp", "parataxis")

#: Dependency relations kept when growing a token into its full noun phrase.
NP_KEEP = frozenset({
    "compound", "amod", "nmod", "poss", "det", "flat", "appos", "nummod",
})
