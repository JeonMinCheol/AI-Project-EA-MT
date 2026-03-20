import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EAMT_DIR = os.path.join(SRC_DIR, "eamt")

for path in [SRC_DIR, EAMT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from DTOlist import CandidateEntity, ScoredCandidate
from memory.builder import (
    build_entity_memory_block,
    render_entity_memory_text,
)


def test_builder_with_candidate():
    source_sentence = "Did Spirited Away win an Academy Award?"

    candidate = CandidateEntity(
        qid="Q22092344",
        candidate_source="anchored_qid",
        target_label="센과 치히로의 행방불명",
        target_aliases=["센치행", "센과 치히로"],
        entity_type="film",
        description="2001 Japanese animated film",
        popularity_score=10.0,
        alias_count=2,
        ambiguity_count=1,
        source_span="Spirited Away",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=1,
    )
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q22092344"
    assert memory.canonical_target == "센과 치히로의 행방불명"
    assert memory.alias_candidates == ["센치행"]
    assert memory.entity_type == "film"
    assert memory.source_sentence == source_sentence

    assert "Source Sentence: Did Spirited Away win an Academy Award?" in text
    assert "Spirited Away" in text
    assert "Q22092344" in text
    assert "센과 치히로의 행방불명" in text
    assert "film" in text


def test_builder_with_scored_candidate():
    source_sentence = "What is the seventh tallest mountain in North America?"

    candidate = CandidateEntity(
        qid="Q49",
        candidate_source="surface_search",
        target_label="북아메리카",
        target_aliases=["북미"],
        entity_type="continent",
        description="continent in the Northern Hemisphere",
        popularity_score=9.0,
        alias_count=1,
        ambiguity_count=1,
        source_span="North America",
        span_match=None,
    )

    scored = ScoredCandidate(
        candidate=candidate,
        reranker_score=0.9,
        prior_bonus=0.1,
        final_score=1.0,
        margin_to_next=0.5,
        feature_dump=None,
    )

    memory = build_entity_memory_block(
        [scored],
        source_sentence=source_sentence,
        alias_limit=1,
    )
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q49"
    assert memory.canonical_target == "북아메리카"
    assert memory.alias_candidates == ["북미"]
    assert memory.entity_type == "continent"
    assert memory.source_sentence == source_sentence

    assert "Source Sentence: What is the seventh tallest mountain in North America?" in text
    assert "North America" in text
    assert "Q49" in text
    assert "북아메리카" in text
    assert "continent" in text


def test_builder_with_no_alias():
    source_sentence = "Example sentence for entity memory."

    candidate = CandidateEntity(
        qid="Q100",
        candidate_source="anchored_qid",
        target_label="예시엔티티",
        target_aliases=[],
        entity_type="example_type",
        description="example description",
        popularity_score=1.0,
        alias_count=0,
        ambiguity_count=1,
        source_span="Example Entity",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=1,
    )
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q100"
    assert memory.canonical_target == "예시엔티티"
    assert memory.alias_candidates == []
    assert memory.source_sentence == source_sentence
    assert "예시엔티티" in text
    assert "Aliases:" not in text


def test_builder_with_no_description():
    source_sentence = "This sentence has no description."

    candidate = CandidateEntity(
        qid="Q101",
        candidate_source="anchored_qid",
        target_label="설명없는엔티티",
        target_aliases=["설명없음별칭"],
        entity_type="unknown_type",
        description="",
        popularity_score=1.0,
        alias_count=1,
        ambiguity_count=1,
        source_span="No Description Entity",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=1,
    )
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q101"
    assert memory.description == ""
    assert memory.source_sentence == source_sentence
    assert "설명없는엔티티" in text
    assert "Description:" not in text


def test_builder_with_empty_candidates():
    source_sentence = "No entity candidate found in this sentence."

    memory = build_entity_memory_block(
        [],
        source_sentence=source_sentence,
        alias_limit=1,
    )
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.entries == []
    assert memory.source_sentence == source_sentence
    assert text == "[ENTITY MEMORY]\n- None"


def test_alias_limit_is_applied():
    source_sentence = "Alias limit test sentence."

    candidate = CandidateEntity(
        qid="Q102",
        candidate_source="anchored_qid",
        target_label="다중별칭엔티티",
        target_aliases=["별칭1", "별칭2", "별칭3"],
        entity_type="test_type",
        description="alias limit test",
        popularity_score=1.0,
        alias_count=3,
        ambiguity_count=1,
        source_span="Alias Test Entity",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=2,
    )

    assert memory.alias_candidates == ["별칭1", "별칭2"]
    assert memory.source_sentence == source_sentence


def test_duplicate_aliases_are_removed():
    source_sentence = "Duplicate alias test sentence."

    candidate = CandidateEntity(
        qid="Q103",
        candidate_source="anchored_qid",
        target_label="중복엔티티",
        target_aliases=["중복별칭", "중복별칭", "다른별칭"],
        entity_type="test_type",
        description="duplicate alias test",
        popularity_score=1.0,
        alias_count=3,
        ambiguity_count=1,
        source_span="Duplicate Alias Entity",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=10,
    )

    assert memory.alias_candidates == ["중복별칭", "다른별칭"]
    assert memory.source_sentence == source_sentence


def test_canonical_target_is_removed_from_aliases():
    source_sentence = "Canonical alias removal test sentence."

    candidate = CandidateEntity(
        qid="Q104",
        candidate_source="anchored_qid",
        target_label="정식이름",
        target_aliases=["정식이름", "별칭A", "별칭B"],
        entity_type="test_type",
        description="canonical alias removal test",
        popularity_score=1.0,
        alias_count=3,
        ambiguity_count=1,
        source_span="Canonical Alias Entity",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source_sentence,
        alias_limit=10,
    )

    assert memory.alias_candidates == ["별칭A", "별칭B"]
    assert memory.source_sentence == source_sentence


if __name__ == "__main__":
    test_builder_with_candidate()
    test_builder_with_scored_candidate()
    test_builder_with_no_alias()
    test_builder_with_no_description()
    test_builder_with_empty_candidates()
    test_alias_limit_is_applied()
    test_duplicate_aliases_are_removed()
    test_canonical_target_is_removed_from_aliases()
    print("All builder tests passed.")