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

    memory = build_entity_memory_block([candidate], alias_limit=1)
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q22092344"
    assert memory.canonical_target == "센과 치히로의 행방불명"
    assert memory.alias_candidates == ["센치행"]
    assert memory.entity_type == "film"
    assert "Spirited Away" in text
    assert "Q22092344" in text
    assert "센과 치히로의 행방불명" in text
    assert "film" in text


def test_builder_with_scored_candidate():
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

    memory = build_entity_memory_block([scored], alias_limit=1)
    text = render_entity_memory_text(memory, target_lang="ko")

    assert memory.qid == "Q49"
    assert memory.canonical_target == "북아메리카"
    assert memory.alias_candidates == ["북미"]
    assert memory.entity_type == "continent"
    assert "North America" in text
    assert "Q49" in text
    assert "북아메리카" in text
    assert "continent" in text


if __name__ == "__main__":
    test_builder_with_candidate()
    test_builder_with_scored_candidate()
    print("All builder tests passed.")