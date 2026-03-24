from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Sequence

from ..kb.index import generate_surface_ngrams, normalize_surface


@dataclass
class SourceSpanMatch:
    start: int
    end: int
    matched_text: str
    normalized_text: str
    match_kind: str
    match_score: float


_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9가-힣_]")


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_text_for_match(text: str) -> str:
    return normalize_surface(_safe_text(text))


def is_valid_candidate_span(span: str, min_char_len: int = 2) -> bool:
    text = _safe_text(span)
    if not text:
        return False

    normalized = normalize_text_for_match(text)
    if len(normalized) < min_char_len:
        return False

    if normalized.isdigit():
        return False

    if not any(ch.isalpha() for ch in normalized):
        return False

    return True


def _find_raw_match(source: str, variant: str) -> Optional[SourceSpanMatch]:
    source_text = _safe_text(source)
    variant_text = _safe_text(variant)

    if not source_text or not variant_text:
        return None

    escaped = re.escape(variant_text)
    boundary_pattern = rf"(?<!\w){escaped}(?!\w)"

    match = re.search(boundary_pattern, source_text, flags=re.IGNORECASE)
    if match:
        matched_text = source_text[match.start():match.end()]
        return SourceSpanMatch(
            start=match.start(),
            end=match.end(),
            matched_text=matched_text,
            normalized_text=normalize_text_for_match(matched_text),
            match_kind="boundary_exact",
            match_score=1.0,
        )

    lower_source = source_text.lower()
    lower_variant = variant_text.lower()
    start = lower_source.find(lower_variant)
    if start >= 0:
        end = start + len(variant_text)
        matched_text = source_text[start:end]
        return SourceSpanMatch(
            start=start,
            end=end,
            matched_text=matched_text,
            normalized_text=normalize_text_for_match(matched_text),
            match_kind="substring",
            match_score=0.85,
        )

    return None


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    results: List[str] = []
    seen = set()

    for item in items:
        text = _safe_text(item)
        key = normalize_text_for_match(text)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(text)

    return results


def align_source_span(
    source: str,
    label: str,
    aliases: Optional[Sequence[str]] = None,
    min_char_len: int = 2,
) -> Optional[SourceSpanMatch]:
    """
    source 안에서 label/alias와 가장 잘 맞는 mention span을 찾습니다.
    - longest match 우선
    - raw source 기준 boundary exact > substring
    """
    source_text = _safe_text(source)
    label_text = _safe_text(label)
    alias_list = list(aliases or [])

    variants = _dedupe_preserve_order([label_text] + alias_list)
    variants = [v for v in variants if is_valid_candidate_span(v, min_char_len=min_char_len)]

    if not source_text or not variants:
        return None

    variants.sort(key=lambda x: len(normalize_text_for_match(x)), reverse=True)

    best_match: Optional[SourceSpanMatch] = None

    for variant in variants:
        match = _find_raw_match(source_text, variant)
        if match is None:
            continue

        if best_match is None:
            best_match = match
            continue

        current_len = best_match.end - best_match.start
        new_len = match.end - match.start

        if match.match_score > best_match.match_score:
            best_match = match
        elif match.match_score == best_match.match_score and new_len > current_len:
            best_match = match
        elif (
            match.match_score == best_match.match_score
            and new_len == current_len
            and match.start < best_match.start
        ):
            best_match = match

    return best_match


def extract_candidate_spans_from_source(
    source: str,
    min_n: int = 1,
    max_n: int = 5,
    min_char_len: int = 2,
) -> List[str]:
    """
    SALT 스타일로 source 문장에서 n-gram/span 후보를 생성합니다.
    기존 kb/index.py 의 generate_surface_ngrams(...)를 재사용합니다.
    """
    source_text = _safe_text(source)
    if not source_text:
        return []

    raw_spans = generate_surface_ngrams(source_text, min_n=min_n, max_n=max_n)
    results: List[str] = []
    seen = set()

    for span in raw_spans:
        text = _safe_text(span)
        if not is_valid_candidate_span(text, min_char_len=min_char_len):
            continue

        normalized = normalize_text_for_match(text)
        if normalized in seen:
            continue

        seen.add(normalized)
        results.append(text)

    return results


__all__ = [
    "SourceSpanMatch",
    "normalize_text_for_match",
    "is_valid_candidate_span",
    "align_source_span",
    "extract_candidate_spans_from_source",
]