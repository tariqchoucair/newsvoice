"""Regression tests for the duplicated-gazetteer defect.

The notebook defined ``TITLES`` in both the Layer 3 and Layer 5 cells, and
``PRONOUN_GENDER`` in both the Layer 4 and Layer 5 cells. The two
``PRONOUN_GENDER`` tables were **not** identical: the Layer 4 form omitted the
four reflexive pronouns. Because Python cells share one namespace and Layer 5
executed later, ``entity_type`` — written in Layer 4 — silently saw Layer 5's
table during a full run, but saw its own table if the Layer 4 cell was re-run
afterwards. The behaviour of the pipeline therefore depended on cell execution
order, which is not recorded anywhere in the output.

These tests pin the resolution: one definition, in one module, imported
everywhere, including the reflexives.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from newsvoice import actors, lexicons

SRC_DIR = pathlib.Path(actors.__file__).parent

#: Names that must exist in exactly one module.
SHARED_NAMES = [
    "TITLES",
    "PRONOUN_GENDER",
    "MALE_TITLES",
    "FEMALE_TITLES",
    "ORG_HEADS",
    "GROUP_HEADS",
    "ORG_SUFFIX",
    "DEFINITE_ORG",
    "PERSON_EPITHET_HEADS",
    "ORG_METONYMS",
    "CUE_LEMMAS",
    "CORE_CUE_LEMMAS",
    "NP_CUES",
]


def _module_level_assignments(path: pathlib.Path) -> set[str]:
    """Names bound at module level in `path`, ignoring imports."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


@pytest.mark.parametrize("name", SHARED_NAMES)
def test_shared_constant_is_defined_exactly_once(name):
    """No module other than lexicons.py may define a shared gazetteer."""
    offenders = [
        path.name
        for path in sorted(SRC_DIR.glob("*.py"))
        if path.name != "lexicons.py" and name in _module_level_assignments(path)
    ]
    assert offenders == [], (
        f"{name} is redefined in {offenders}; it must be imported from "
        f"lexicons.py so the two copies cannot diverge"
    )


def test_pronoun_gender_includes_reflexives():
    """The Layer 4 table omitted these four; the merged table must not."""
    for pronoun, code in (("himself", "m"), ("herself", "f"),
                          ("themselves", "p"), ("itself", "n")):
        assert lexicons.PRONOUN_GENDER.get(pronoun) == code


def test_actors_uses_the_shared_pronoun_table():
    """Identity, not equality — two equal copies would still be two copies."""
    assert actors.PRONOUN_GENDER is lexicons.PRONOUN_GENDER
    assert actors.TITLES is lexicons.TITLES


def test_cue_tiers_are_disjoint_where_it_matters():
    """A lemma in both tiers resolves to 'core', so overlap must be deliberate."""
    both = lexicons.CORE_CUE_LEMMAS & lexicons.PERIPHERAL_CUE_LEMMAS
    assert both == {"deny"}, (
        f"unexpected overlap between cue tiers: {sorted(both)}. A lemma in both "
        f"is always treated as core; add it to one tier only."
    )


def test_cue_lemmas_is_the_union_of_its_tiers():
    assert lexicons.CUE_LEMMAS == (
        lexicons.CORE_CUE_LEMMAS | lexicons.PERIPHERAL_CUE_LEMMAS
    )


def test_gazetteers_are_lowercase():
    """Lookups lowercase the surface form, so entries must be lowercase too."""
    for name in ("TITLES", "ORG_HEADS", "GROUP_HEADS", "PERSON_EPITHET_HEADS",
                 "ORG_METONYMS", "CUE_LEMMAS", "NP_CUES"):
        entries = getattr(lexicons, name)
        bad = [e for e in entries if e != e.lower()]
        assert bad == [], f"{name} contains non-lowercase entries: {bad}"


def test_immutable_gazetteers_are_frozen():
    """Frozen sets stop a caller mutating a table other layers depend on."""
    for name in ("TITLES", "CUE_LEMMAS", "NP_CUES", "ORG_HEADS", "GROUP_HEADS"):
        assert isinstance(getattr(lexicons, name), frozenset), (
            f"{name} should be a frozenset"
        )
