import re
from collections import defaultdict
from DTOlist import *

"""
surface 뜻: surface는 문장에 실제로 보이는 문자열을 뜻함

예: I visited New York last year. 라는 문장에서 entity의 surface는 New York이 된다. (new york, new york city 등 x)
"""


def _iter_nonempty_strings(values: list[str] | None) -> list[str]:
    """
    None이 들어와도 안전하게 순회할 수 있도록 비어 있지 않은 문자열만 반환한다.
    """
    if not values:
        return []
    return [value for value in values if value]


def _surface_priority(surface: str) -> tuple[int, int, str]:
    """
    더 구체적인 surface를 먼저 처리하기 위한 안정적인 정렬 키.
    """
    return (-len(surface.split()), -len(surface), surface)

def normalize_surface(text: str) -> str:
    """
     동작:
        1. 특수문자 제거
        2. 연속 공백 정리
        3. 소문자화

    입력:
        text: 원본 문자열

    출력:
        정규화된 문자열

    예시:
        "Hello, World! " -> "hello world"
    """
    normalized_text = re.sub(r"[^\w\s]", "", text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return normalized_text.lower()


def build_qid_index(records: list[KBEntityRecord]) -> dict[str, list[KBEntityRecord]]:
    """
    동작: KB 엔티티 레코드 목록으로부터 QID direct lookup 인덱스를 생성한다.

    입력:
        records: KBEntityRecord 객체 리스트

    출력:
        qid를 key로 하고, 해당 qid를 가지는 KBEntityRecord 리스트를 value로 갖는 사전

    설명:
        - 하나의 qid에 대해 target_lang별 레코드가 여러 개 있을 수 있으므로
          value를 list로 유지한다.
        - 이 인덱스는 공식 평가 경로에서 focal wikidata_id가 주어졌을 때
          빠르게 direct lookup하기 위한 주 인덱스다.

    예:
        {
            "Q123": [record_ja, record_ko],
            "Q456": [record_ja]
        }
    """
    qid_index = defaultdict(list)
    for record in records:
        qid_index[record.qid].append(record)
    return qid_index


def build_surface_index(records: list[KBEntityRecord]) -> dict[str, list[KBEntityRecord]]:
    """
    동작: KB 엔티티 레코드 목록으로부터 surface 역색인을 생성한다.

    입력:
        records: KBEntityRecord 객체 리스트

    출력:
        정규화된 surface 문자열을 key로 하고,
        해당 surface와 연결되는 KBEntityRecord 리스트를 value로 갖는 사전

    설명:
        - source 쪽 영어 label / alias / 미리 계산된 normalized_surfaces를 이용해
          역색인을 구성한다.
        - 하나의 surface가 여러 QID를 가리킬 수 있으므로 value는 list다.
        - 같은 (qid, target_lang) 조합이 중복 삽입될 수 있으므로 중복 제거 후
          popularity_score 기준 내림차순 정렬한다.

    예:
        {
            "new york": [kb1, kb2, ...],
            "titanic": [kbm, kbn, ...]
        }
    """
    surface_index = defaultdict(list)

    for record in records:
        surfaces = {
            normalize_surface(surface)
            for surface in _iter_nonempty_strings(record.normalized_surfaces)
        }

        if record.label_en:
            surfaces.add(normalize_surface(record.label_en))
        for alias in _iter_nonempty_strings(record.aliases_en):
            surfaces.add(normalize_surface(alias))

        for surface in surfaces:
            if surface:
                surface_index[surface].append(record)

    deduped_index = {}
    for surface, recs in surface_index.items():
        unique = {}
        for r in recs:
            unique[(r.qid, r.target_lang)] = r
        deduped_index[surface] = sorted(
            unique.values(),
            key=lambda x: x.popularity_score,
            reverse=True
        )

    return deduped_index


def lookup_entity_by_qid(
    qid_index: dict[str, list[KBEntityRecord]],
    wikidata_id: str,
    target_lang: str
) -> list[KBEntityRecord] | None:
    """
    동작: QID direct lookup 인덱스에서 특정 wikidata_id와 target_lang에 맞는 엔티티를 조회한다.

    입력:
        qid_index: build_qid_index()로 생성한 qid 인덱스
        wikidata_id: 조회할 Wikidata QID
        target_lang: 목표 언어 코드

    출력:
        조건을 만족하는 KBEntityRecord 리스트
        존재하지 않으면 None 반환

    필터 조건:
        - language_available == True
        - target_label(또는 target entity name)이 존재함
        - record.target_lang == target_lang

    설명:
        - 하나의 qid에 대해 여러 언어 레코드가 존재할 수 있으므로 list를 반환한다.
        - 공식 평가처럼 focal wikidata_id가 주어진 경우 가장 먼저 호출되는 핵심 함수다.
    """
    if wikidata_id not in qid_index:
        return None

    results = [
        record for record in qid_index[wikidata_id]
        if record.language_available
        and record.target_label is not None
        and len(record.target_label) > 0
        and record.target_lang == target_lang
    ]

    return results if results else None


def generate_surface_ngrams(source: str, max_n: int = 5) -> list[str]:
    """
    source 문장에서 surface search에 사용할 n-gram 후보를 생성한다.

    입력:
        source: 원문 문장
        max_n: 생성할 최대 n-gram 길이

    출력:
        정규화된 n-gram 문자열 리스트 (중복 제거)

    처리:
        1. source를 normalize_surface()로 정규화
        2. whitespace 기준 토큰화
        3. 1-gram부터 max_n-gram까지 모두 생성
        4. set으로 중복 제거

    예:
        source = "I visited New York City"
        max_n = 3
        출력 예시:
            ["i", "visited", "new", "york", "city",
             "i visited", "visited new", "new york", "york city",
             "i visited new", "visited new york", "new york city"]
    """
    tokens = normalize_surface(source).split()
    ngrams = set()

    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i:i+n]))

    return sorted(ngrams, key=_surface_priority)


def search_surface_candidates(
    surface_index: dict[str, list[KBEntityRecord]],
    surfaces: list[str],
    target_lang: str,
    max_candidates: int = 10
) -> list[CandidateEntity]:
    """
    surface 역색인을 이용해 후보 엔티티를 검색한다.

    입력:
        surface_index: build_surface_index()로 생성한 surface 역색인
        surfaces: 검색할 normalized surface 문자열 리스트
        target_lang: 목표 언어 코드
        max_candidates: 반환할 최대 후보 수

    출력:
        CandidateEntity 리스트
        (popularity_score 기준 내림차순 정렬 후 max_candidates개까지만 반환)

    처리:
        1. 각 surface가 surface_index에 존재하는지 확인
        2. 해당 surface와 연결된 KBEntityRecord 후보를 가져옴
        3. target_lang, label 존재 여부, 사용 가능 여부를 기준으로 필터링
        4. CandidateEntity로 변환
        5. 동일 QID 후보는 마지막 값으로 덮어쓰며 중복 제거
        6. popularity_score 기준 정렬 후 상위 max_candidates개 반환

    ambiguity_count:
        - 하나의 surface가 몇 개의 서로 다른 QID와 연결되는지를 의미한다.
        - 예를 들어 "titanic"이 영화와 배를 모두 가리키면 ambiguity_count = 2

    비고:
        - 이 함수는 QID direct lookup이 실패했거나,
          auxiliary retrieval이 필요한 경우에 보조 경로로 사용된다.
    """

    candidates = {}
    ordered_surfaces = sorted(
        {surface for surface in surfaces if surface},
        key=_surface_priority
    )

    for surface in ordered_surfaces:
        if surface not in surface_index:
            continue

        candidate_records = surface_index[surface]
        ambiguity_count = len({r.qid for r in candidate_records})

        for record in candidate_records:
            if (
                record.language_available
                and record.target_label is not None
                and len(record.target_label) > 0
                and record.target_lang == target_lang
            ):
                if record.qid in candidates:
                    continue

                candidates[record.qid] = CandidateEntity(
                    qid=record.qid,
                    candidate_source="surface_search",
                    target_label=record.target_label,
                    target_aliases=record.target_aliases,
                    entity_type=record.entity_type,
                    description=record.description,
                    popularity_score=record.popularity_score,
                    alias_count=len(record.target_aliases),
                    ambiguity_count=ambiguity_count,
                    source_span=surface,
                    span_match=None
                )

    # popularity_score 기준으로 내림차순 정렬
    results = sorted(
        candidates.values(),
        key=lambda x: x.popularity_score,
        reverse=True
    )
    return results[:max_candidates] # 후보 개수 제한
