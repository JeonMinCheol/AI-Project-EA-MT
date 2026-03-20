from DTOlist import *


"""
prompting.py 역할:
source 문장, target language, optional entity memory를 받아
번역 모델에 넣을 최종 prompt 문자열을 생성한다.

지원 모드:
- plain
- entity-aware
"""


def _safe_str(value):
    """
    None이면 빈 문자열, 아니면 문자열로 바꿔 공백 제거.
    """
    if value is None:
        return ""
    return str(value).strip()


def _normalize_target_lang(target_lang):
    """
    목표 언어 코드를 사람이 읽기 쉬운 이름으로 바꾼다.
    """
    target_lang = _safe_str(target_lang).lower()

    lang_map = {
        "ar": "Arabic",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "th": "Thai",
        "tr": "Turkish",
        "zh": "Chinese",
        "en": "English",
    }

    return lang_map.get(target_lang, target_lang if target_lang else "target language")


def _get_memory_text(memory):
    """
    EntityMemoryBlock에서 rendered_text를 안전하게 꺼낸다.
    없으면 빈 문자열 반환.
    """
    if memory is None:
        return ""

    rendered_text = getattr(memory, "rendered_text", "")
    return _safe_str(rendered_text)


def build_translation_prompt(source, target_lang, memory=None, mode="entity-aware"):
    """
    번역 모델에 넣을 최종 prompt 문자열 생성.

    입력:
        source: 번역할 원문 문장
        target_lang: 목표 언어 코드 (예: ko, ja, zh)
        memory: optional EntityMemoryBlock
        mode: "plain" 또는 "entity-aware"

    출력:
        prompt_text: 모델 입력 문자열
    """
    source = _safe_str(source)
    target_lang_name = _normalize_target_lang(target_lang)
    mode = _safe_str(mode).lower()

    task_text = (
        f"Translate the following sentence into {target_lang_name} naturally and accurately."
    )

    # plain prompt
    if mode == "plain":
        prompt_lines = [
            "[TASK]",
            task_text,
            "",
            "[SOURCE]",
            source,
        ]
        return "\n".join(prompt_lines)

    # entity-aware prompt
    memory_text = _get_memory_text(memory)

    if memory_text:
        prompt_lines = [
            "[TASK]",
            task_text,
            "",
            "[ENTITY MEMORY]",
            memory_text,
            "",
            "[SOURCE]",
            source,
        ]
    else:
        # memory가 없으면 fallback으로 plain 구조 사용
        prompt_lines = [
            "[TASK]",
            task_text,
            "",
            "[SOURCE]",
            source,
        ]

    return "\n".join(prompt_lines)