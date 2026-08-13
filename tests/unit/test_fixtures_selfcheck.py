"""The fixtures are asserted, not just described.

Every distribution the build prompt specifies is checked here. If the generator
drifts — a topic added, an outcome rebalanced, a trap quietly softened — these
fail. A specification that lives only in a docstring stops being true the first
time someone edits the data.

Entity resolution here runs in-memory against the registry CSVs using the same
`normalise_name` the graph uses. The Cypher resolver is exercised over the same
fixtures in tests/integration/test_graph_queries.py; keeping this half
stack-free means a fresh clone can still prove its own fixtures are coherent.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from src.graph.driver import normalise_name

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
REGISTRY = FIXTURES / "registry"

#: Commercial content is process-only (build prompt §8). Shared by the corpus and
#: paraphrase checks so one definition of "names a figure" governs both.
PRICE = re.compile(r"(?:USD|EUR|GBP|INR|\$|€|£|₹)\s?\d|\bday rate\b", re.IGNORECASE)


def _load_csv(name: str) -> list[dict[str, str]]:
    with (REGISTRY / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(name: str) -> Any:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def pairs() -> list[dict[str, Any]]:
    return list(_load_json("qa_pairs.json"))


@pytest.fixture(scope="module")
def paraphrases() -> list[dict[str, Any]]:
    return list(_load_json("question_paraphrases.json"))


@pytest.fixture(scope="module")
def answer_key() -> dict[str, Any]:
    return dict(_load_json("answer_key.json"))


@pytest.fixture(scope="module")
def golden_source() -> dict[str, Any]:
    return dict(_load_json("golden_rfp_source.json"))


@pytest.fixture(scope="module")
def registry_index() -> dict[str, dict[str, str]]:
    """Normalised name and code -> code, per entity kind."""
    index: dict[str, dict[str, str]] = {}
    for kind, filename, code_field, name_field in (
        ("vendor", "vendors.csv", "code", "name"),
        ("product", "products.csv", "code", "name"),
        ("certification", "certifications.csv", "code", "name"),
        ("client", "customers.csv", "id", "name"),
        ("location", "locations.csv", "code", "name"),
    ):
        lookup: dict[str, str] = {}
        for row in _load_csv(filename):
            lookup[row[code_field]] = row[code_field]
            lookup[normalise_name(row[name_field])] = row[code_field]
        index[kind] = lookup
    # Tools are our own accelerators, so they resolve against the product list.
    index["tool"] = dict(index["product"])
    return index


def resolve(index: dict[str, dict[str, str]], needle: str) -> str | None:
    """Resolve across every entity kind, the way the guardrail will."""
    for kind in sorted(index):
        lookup = index[kind]
        if needle in lookup:
            return lookup[needle]
        normalised = normalise_name(needle)
        if normalised in lookup:
            return lookup[normalised]
    return None


class TestRegistryShape:
    """Counts come straight from build prompt §8."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("vendors.csv", 12),
            ("products.csv", 8),
            ("certifications.csv", 7),
            ("case_studies.csv", 6),
            ("customers.csv", 8),
            ("smes.csv", 6),
            ("capabilities.csv", 6),
        ],
    )
    def test_row_counts(self, filename: str, expected: int) -> None:
        assert len(_load_csv(filename)) == expected

    def test_vendor_codes_are_unique(self) -> None:
        codes = [row["code"] for row in _load_csv("vendors.csv")]
        assert len(set(codes)) == len(codes)

    def test_four_case_studies_are_publicly_usable(self) -> None:
        """Two are deliberately not, so the public-usability guardrail has teeth."""
        flags = Counter(row["publicly_usable"] for row in _load_csv("case_studies.csv"))
        assert flags == {"true": 4, "false": 2}

    def test_every_sme_owns_at_least_one_capability(self) -> None:
        capability_ids = {row["id"] for row in _load_csv("capabilities.csv")}
        for sme in _load_csv("smes.csv"):
            owned = sme["capabilities"].split("|")
            assert 1 <= len(owned) <= 2
            assert set(owned) <= capability_ids

    def test_every_capability_has_an_owner(self) -> None:
        """A capability nobody owns makes escalation routing impossible."""
        owned: set[str] = set()
        for sme in _load_csv("smes.csv"):
            owned.update(sme["capabilities"].split("|"))
        assert {row["id"] for row in _load_csv("capabilities.csv")} <= owned


class TestCorpusDistributions:
    def test_forty_pairs(self, pairs: list[dict[str, Any]]) -> None:
        assert len(pairs) == 40

    def test_question_type_distribution(self, pairs: list[dict[str, Any]]) -> None:
        assert dict(sorted(Counter(p["question_type"] for p in pairs).items())) == {
            "commercial": 6,
            "company_info": 8,
            "compliance": 8,
            "technical": 18,
        }

    def test_outcome_distribution(self, pairs: list[dict[str, Any]]) -> None:
        assert dict(sorted(Counter(p["outcome"] for p in pairs).items())) == {
            "lost": 10,
            "unknown": 15,
            "won": 15,
        }

    def test_answers_are_150_to_300_words(self, pairs: list[dict[str, Any]]) -> None:
        offenders = [(p["id"], p["word_count"]) for p in pairs if not 150 <= p["word_count"] <= 300]
        assert offenders == []

    def test_word_count_matches_the_answer(self, pairs: list[dict[str, Any]]) -> None:
        """A stale count would make the word-limit eval measure nothing."""
        offenders = [p["id"] for p in pairs if len(p["answer"].split()) != p["word_count"]]
        assert offenders == []

    def test_dates_span_the_required_window(self, pairs: list[dict[str, Any]]) -> None:
        dates = sorted(p["answer_date"] for p in pairs)
        assert dates[0] < "2023-06-01", "corpus should reach back to early 2023"
        assert dates[-1] > "2026-01-01", "corpus should reach into 2026"

    def test_ids_are_unique(self, pairs: list[dict[str, Any]]) -> None:
        for field in ("id", "question_id", "answer_id", "topic_key"):
            values = [p[field] for p in pairs]
            assert len(set(values)) == len(values), f"duplicate {field}"


class TestSupersession:
    def test_exactly_four_chains(self, pairs: list[dict[str, Any]]) -> None:
        assert sum(1 for p in pairs if p["superseded_by"]) == 4

    def test_every_chain_points_at_a_real_answer(self, pairs: list[dict[str, Any]]) -> None:
        answer_ids = {p["answer_id"] for p in pairs}
        targets = [p["superseded_by"] for p in pairs if p["superseded_by"]]
        assert set(targets) <= answer_ids

    def test_the_superseded_answer_is_always_older(self, pairs: list[dict[str, Any]]) -> None:
        """Otherwise recency decay and the staleness eval contradict each other."""
        by_answer = {p["answer_id"]: p for p in pairs}
        for pair in pairs:
            if not pair["superseded_by"]:
                continue
            successor = by_answer[pair["superseded_by"]]
            assert pair["answer_date"] < successor["answer_date"], pair["id"]

    def test_no_chain_is_circular(self, pairs: list[dict[str, Any]]) -> None:
        by_answer = {p["answer_id"]: p for p in pairs}
        for pair in pairs:
            seen = {pair["answer_id"]}
            cursor = pair["superseded_by"]
            while cursor:
                assert cursor not in seen, f"cycle through {cursor}"
                seen.add(cursor)
                cursor = by_answer[cursor]["superseded_by"]


class TestConfidentiality:
    def test_exactly_one_confidential_pair(self, pairs: list[dict[str, Any]]) -> None:
        assert sum(1 for p in pairs if p["confidential"]) == 1

    def test_it_is_bound_to_bluepine(self, pairs: list[dict[str, Any]]) -> None:
        confidential = next(p for p in pairs if p["confidential"])
        assert confidential["customer"] == "Bluepine Health Systems"

    def test_no_other_pair_belongs_to_bluepine(self, pairs: list[dict[str, Any]]) -> None:
        """So 'confidential content never leaks' is testable without ambiguity."""
        bluepine = [p for p in pairs if p["customer"] == "Bluepine Health Systems"]
        assert len(bluepine) == 1
        assert bluepine[0]["confidential"] is True


class TestTopicFamilies:
    """`topic_key` identifies a record; `topic_family` names a subject.

    Calibration groups on the family, so these two must not be allowed to
    collapse into each other. They did once — every family was a singleton,
    which left the calibration corpus with zero same-subject pairs and made the
    anchor uncomputable.
    """

    def test_topic_keys_are_unique(self, pairs: list[dict[str, Any]]) -> None:
        """The golden source joins to a corpus answer through this, so it is an id."""
        keys = [p["topic_key"] for p in pairs]
        assert len(set(keys)) == len(keys)

    def test_exactly_four_families_hold_two_records(self, pairs: list[dict[str, Any]]) -> None:
        """The four supersession chains, and nothing else."""
        sizes = Counter(p["topic_family"] for p in pairs)
        assert sorted(k for k, v in sizes.items() if v == 2) == [
            "cutover-rollback",
            "database-migration",
            "iso-soc-scope",
            "landing-zone",
        ]
        assert sum(1 for v in sizes.values() if v > 2) == 0

    def test_a_family_of_two_is_exactly_a_supersession_chain(
        self, pairs: list[dict[str, Any]]
    ) -> None:
        by_family: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            by_family.setdefault(pair["topic_family"], []).append(pair)
        for family, members in by_family.items():
            if len(members) == 1:
                continue
            chained = sum(1 for m in members if m["superseded_by"])
            assert chained == 1, f"{family} has {len(members)} records but {chained} chain links"


class TestParaphrases:
    """Alternate phrasings exist so calibration has a real anchor.

    Before them the corpus could only measure "the same question" on the four
    supersession chains, whose question text is byte-identical — an anchor made
    of exact duplicates, which puts the derived floor above genuine matches.
    """

    def test_two_per_family(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        families = {p["topic_family"] for p in pairs}
        counts = Counter(p["topic_family"] for p in paraphrases)
        assert set(counts) == families
        assert set(counts.values()) == {2}, (
            "an uneven count would weight some subjects more heavily in the statistics"
        )

    def test_ids_are_unique(self, paraphrases: list[dict[str, Any]]) -> None:
        for field in ("id", "question_id", "topic_key", "question"):
            values = [p[field] for p in paraphrases]
            assert len(set(values)) == len(values), f"duplicate {field}"

    def test_none_restates_an_original_verbatim(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        """A paraphrase that copies the original measures nothing."""
        originals = {p["question"] for p in pairs}
        assert not originals.intersection(p["question"] for p in paraphrases)

    def test_each_carries_no_answer_of_its_own(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        """The corpus still holds 40 answers, so nothing new competes for a rank."""
        answers = {p["answer_id"] for p in pairs}
        assert {p["answer_id"] for p in paraphrases} <= answers
        assert all("answer" not in p for p in paraphrases)

    def test_none_points_at_a_superseded_answer(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        """Otherwise a phrasing's only match would be something retrieval suppresses."""
        superseded = {p["answer_id"] for p in pairs if p["superseded_by"]}
        assert not superseded.intersection(p["answer_id"] for p in paraphrases)

    def test_paraphrase_of_names_a_real_question(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        question_ids = {p["question_id"] for p in pairs}
        assert {p["paraphrase_of"] for p in paraphrases} <= question_ids

    def test_the_confidential_question_keeps_its_customer(
        self, paraphrases: list[dict[str, Any]], pairs: list[dict[str, Any]]
    ) -> None:
        """The leak this would open is the whole reason the customer is carried.

        `find_similar_questions` decides confidentiality by walking from the
        question to the RFP that asked it. A paraphrase of the confidential
        question hung off another customer's RFP would make ANS-0014 reachable
        through a synonym.
        """
        confidential = next(p for p in pairs if p["confidential"])
        rephrased = [p for p in paraphrases if p["answer_id"] == confidential["answer_id"]]
        assert len(rephrased) == 2
        assert all(p["customer"] == confidential["customer"] for p in rephrased)

    def test_no_pricing_figure_appears(self, paraphrases: list[dict[str, Any]]) -> None:
        for row in paraphrases:
            assert not PRICE.search(row["question"]), f"{row['id']} names a figure"


class TestGrounding:
    """Every entity named in a synthetic answer must resolve in the registry."""

    def test_every_cited_entity_resolves(
        self, pairs: list[dict[str, Any]], registry_index: dict[str, dict[str, str]]
    ) -> None:
        unresolved = [
            (pair["id"], entity)
            for pair in pairs
            for entity in pair["cited_entities"]
            if resolve(registry_index, entity) is None
        ]
        assert unresolved == []

    def test_registry_names_appearing_in_text_are_all_recorded(
        self, pairs: list[dict[str, Any]], registry_index: dict[str, dict[str, str]]
    ) -> None:
        """Catches an answer naming something it never declared it cited.

        Without this, cited_entities could quietly under-report and the
        grounding check above would pass while the prose named a stranger.
        """
        known_names = sorted(
            {
                row["name"]
                for filename in ("vendors.csv", "products.csv", "certifications.csv")
                for row in _load_csv(filename)
            }
        )
        missing: list[tuple[str, str]] = []
        for pair in pairs:
            declared = set(pair["cited_entities"])
            for name in known_names:
                if name in pair["answer"] and name not in declared:
                    missing.append((pair["id"], name))
        assert missing == []

    def test_no_pricing_figure_anywhere_in_the_corpus(self, pairs: list[dict[str, Any]]) -> None:
        """Commercial topics are process-only (build prompt §8)."""
        offenders = [p["id"] for p in pairs if PRICE.search(p["answer"])]
        assert offenders == []


class TestGoldenRfp:
    def test_twenty_questions(self, golden_source: dict[str, Any]) -> None:
        assert len(golden_source["questions"]) == 20

    def test_trap_distribution(self, golden_source: dict[str, Any]) -> None:
        """One trap per question. `forbidden_term` split out of the injection
        question so a regression in either cannot hide behind the other."""
        assert dict(sorted(Counter(q["trap"] for q in golden_source["questions"]).items())) == {
            "forbidden_term": 1,
            "injection": 1,
            "legal": 1,
            "none": 13,
            "pricing": 1,
            "unanswerable": 3,
        }

    def test_fifteen_questions_are_answerable_from_the_corpus(
        self, golden_source: dict[str, Any]
    ) -> None:
        """13 clean, plus the injection carrier and the warranty question — both
        of which are ordinary answerable questions that also trip a guardrail."""
        answerable = [q for q in golden_source["questions"] if q["expects_match"]]
        assert len(answerable) == 15

    def test_twelve_mandatory_and_five_word_limits(self, golden_source: dict[str, Any]) -> None:
        questions = golden_source["questions"]
        assert sum(1 for q in questions if q["mandatory"]) == 12
        assert sum(1 for q in questions if q["word_limit"] is not None) == 5

    def test_orders_are_sequential(self, golden_source: dict[str, Any]) -> None:
        orders = [q["order"] for q in golden_source["questions"]]
        assert orders == list(range(len(orders)))

    def test_the_injection_lives_on_exactly_the_designated_carrier(
        self, golden_source: dict[str, Any]
    ) -> None:
        needle = golden_source["injection_string"]
        carriers = [q["id"] for q in golden_source["questions"] if needle in q["text"]]
        assert carriers == [golden_source["injection_carrier_question_id"]]

    def test_the_carrier_is_clean_answerable_and_not_mandatory(
        self, golden_source: dict[str, Any]
    ) -> None:
        """Traps must be attributable: the carrier trips injection and nothing else.

        Answerable matters most — the trap's point is that an ordinary question
        can be tampered with. Non-mandatory keeps the injection eval and the
        mandatory-coverage eval from contending over the same question.
        """
        carrier = next(
            q
            for q in golden_source["questions"]
            if q["id"] == golden_source["injection_carrier_question_id"]
        )
        assert carrier["trap"] == "injection"
        assert carrier["expects_match"], "the carrier must still be answerable"
        assert carrier["mandatory"] is False

    def test_the_warranty_question_carries_only_the_term_trap(
        self, golden_source: dict[str, Any]
    ) -> None:
        """It used to carry the injection too, which masked injection regressions."""
        warranty = next(
            q for q in golden_source["questions"] if q["expects_match"] == "warranty-process"
        )
        assert warranty["trap"] == "forbidden_term"
        assert golden_source["injection_string"] not in warranty["text"]

    def test_at_least_three_questions_expect_a_supersession_chain_head(
        self, answer_key: dict[str, Any]
    ) -> None:
        """Build prompt §8 asks for at least three; the corpus supplies four.

        The fourth falls out of the compliance chain (certification scope) also
        being the best match for a golden question. That is a stronger test of
        the same property, not a weaker one, so it is kept.
        """
        heads = answer_key["supersession_head_question_ids"]
        assert len(heads) >= 3
        assert {"GQ-006", "GQ-007", "GQ-008"} <= set(heads)


class TestAnswerKeyConsistency:
    """The key is only useful if it agrees with the corpus it describes."""

    def test_every_expected_match_exists_in_the_corpus(
        self, answer_key: dict[str, Any], pairs: list[dict[str, Any]]
    ) -> None:
        answer_ids = {p["answer_id"] for p in pairs}
        expected = [
            q["expected_best_match_answer_id"]
            for q in answer_key["questions"]
            if q["expected_best_match_answer_id"]
        ]
        assert expected, "the key should expect at least one match"
        assert set(expected) <= answer_ids

    def test_no_expected_match_is_a_superseded_answer(
        self, answer_key: dict[str, Any], pairs: list[dict[str, Any]]
    ) -> None:
        """Retrieval must surface the replacement, never the thing it replaced."""
        superseded = {p["answer_id"] for p in pairs if p["superseded_by"]}
        expected = {
            q["expected_best_match_answer_id"]
            for q in answer_key["questions"]
            if q["expected_best_match_answer_id"]
        }
        assert expected & superseded == set()

    def test_every_expected_escalation_maps_to_a_designed_trap(
        self, answer_key: dict[str, Any]
    ) -> None:
        for question in answer_key["questions"]:
            if not question["expected_escalation"]:
                continue
            assert question["trap"] != "none" or question["expected_status"] == "NO_MATCH"
            assert question["expected_guardrail"] is not None

    def test_no_match_questions_have_no_expected_answer(self, answer_key: dict[str, Any]) -> None:
        for question in answer_key["questions"]:
            if question["expected_status"] == "NO_MATCH":
                assert question["expected_best_match_answer_id"] is None

    def test_counts_agree_with_the_question_list(self, answer_key: dict[str, Any]) -> None:
        questions = answer_key["questions"]
        assert answer_key["question_count"] == len(questions)
        assert answer_key["counts"]["unanswerable"] == 3
        assert answer_key["counts"]["mandatory"] == 12
        assert answer_key["counts"]["with_word_limit"] == 5

    def test_the_three_unanswerables_are_the_three_no_matches(
        self, answer_key: dict[str, Any]
    ) -> None:
        no_match = set(answer_key["expected_no_match_question_ids"])
        unanswerable = {
            q["question_id"] for q in answer_key["questions"] if q["trap"] == "unanswerable"
        }
        assert no_match >= unanswerable
