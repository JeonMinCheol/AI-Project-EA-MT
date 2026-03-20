import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EAMT_DIR = os.path.join(SRC_DIR, "eamt")

for path in [SRC_DIR, EAMT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from DTOlist import CandidateEntity
from memory.builder import (
    build_entity_memory_block,
    render_entity_memory_text,
)
from translation.prompting import build_translation_prompt


def test_plain_prompt():
    source = "Did Spirited Away win an Academy Award?"

    prompt = build_translation_prompt(
        source=source,
        target_lang="ko",
        memory=None,
        mode="plain",
    )

    assert "[TASK]" in prompt
    assert "[SOURCE]" in prompt
    assert "Korean" in prompt
    assert source in prompt
    assert "[ENTITY MEMORY]" not in prompt


def test_entity_aware_prompt():
    source = "Did Spirited Away win an Academy Award?"

    candidate = CandidateEntity(
        qid="Q22092344",
        candidate_source="anchored_qid",
        target_label="센과 치히로의 행방불명",
        target_aliases=["센치행"],
        entity_type="film",
        description="2001 Japanese animated film",
        popularity_score=10.0,
        alias_count=1,
        ambiguity_count=1,
        source_span="Spirited Away",
        span_match=None,
    )

    memory = build_entity_memory_block(
        [candidate],
        source_sentence=source,
        alias_limit=1,
    )
    render_entity_memory_text(memory, target_lang="ko")

    prompt = build_translation_prompt(
        source=source,
        target_lang="ko",
        memory=memory,
        mode="entity-aware",
    )

    assert "[TASK]" in prompt
    assert "[ENTITY MEMORY]" in prompt
    assert "[SOURCE]" in prompt
    assert source in prompt
    assert "센과 치히로의 행방불명" in prompt
    assert "Q22092344" in prompt


if __name__ == "__main__":
    test_plain_prompt()
    test_entity_aware_prompt()
    print("All prompting tests passed.")