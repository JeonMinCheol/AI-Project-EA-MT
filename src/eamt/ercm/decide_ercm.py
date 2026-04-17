import re
import json
from typing import Any

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:  # pragma: no cover - fallback for non-vLLM test/runtime environments
    LLM = Any

    class SamplingParams:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.kwargs = kwargs

from DTOlist import TranslationDraft, EntityMemoryBlock, ERCMDecision, ScoredCandidate


_ERROR_TO_REASON = {
    "Omission": "missing_canonical",
    "Residue": "english_residue",
    "Wrong Alias": "wrong_alias",
    "Grammar": "grammar_issue",
    "Normal": "normal",
}
_VALID_ERROR_TYPES = {"Omission", "Residue", "Wrong Alias", "Grammar", "Normal"}

_LANG_NAME: dict[str, str] = {
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "ar": "Arabic",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "th": "Thai",
    "tr": "Turkish",
}

_SYSTEM_PROMPT = """You are an expert translation quality evaluator specializing in entity translation.
Your task is to identify errors in a machine-translated sentence, focusing specifically on how a named entity was handled.
You must respond only in JSON format with no additional text."""


def _build_user_prompt(
    source: str,
    draft_text: str,
    target_lang: str,
    source_span: str | None,
    alias_set: list[str],
) -> str:
    lang_name = _LANG_NAME.get(target_lang, target_lang)
    alias_str = ", ".join(f'"{a}"' for a in alias_set) if alias_set else "N/A"
    span_str = f'"{source_span}"' if source_span else "N/A"

    return f"""You are an expert linguistic evaluator focusing on entity translation accuracy in {lang_name}.
The entity {span_str} is expected to be present in the translation. Evaluate based on the hierarchy below.

## Given Information
- Source (English): "{source}"
- Entity in source: {span_str}
- Acceptable translations for this entity: [{alias_str}]
- Target language: {lang_name}
- Translation to evaluate: "{draft_text}"

## Step-by-step Evaluation (Choose the FIRST that applies)

### Step 1 — Entity Presence Check
- Search for the core name (stem) from [{alias_str}] in the translation.
- **If the entity is COMPLETELY missing → mark Omission.**
- **If the core entity is present (even with particles or articles) → proceed to Step 2.**

### Step 2 — Check for Residue
- Does the translation contain untranslated source text (English) or other foreign scripts (e.g., Chinese in a Korean sentence)?
- **Exceptions**: Technical terms (USB, x86, DNA) or brand names.
- **Error**: If the entity or sentence remains in English without reason → mark **Residue**.

### Step 3 — Check for Grammar (Linguistic Integration)
- Mark **Grammar** ONLY if the entity is present but its integration is incorrect for {lang_name}:
  1. **Particles (ko, ja)**: Mismatch with batchim (e.g., "꽃는" instead of "꽃은").
  2. **Gender/Number/Article (fr, de, es, it)**: Incorrect gender agreement or wrong definite/indefinite articles.
  3. **Case (ar, tr, de)**: Incorrect case declension for the sentence structure.
- **CRITICAL**: If the form is natural (e.g., "꽃에는", "꽃을"), it is NOT a Grammar error.

### Step 4 — Check for Wrong Alias (Semantics)
- Mark **Wrong Alias** ONLY if the core meaning is wrong or a term not in the list is used (e.g., "Hospital" instead of "Temple").
- **CRITICAL**: If the core stem matches anything in [{alias_str}], do NOT mark as Wrong Alias.

### Step 5 — Final Decision
- If no errors are found → mark **Normal**.

## Examples for Reference
- **Normal**: (Korean) Alias: "칼과 꽃" / Draft: "칼과 꽃에는..." -> Result: Normal (natural particle).
- **Grammar**: (Korean) Alias: "칼과 꽃" / Draft: "칼과 꽃는..." -> Result: Grammar (batchim mismatch).
- **Residue**: (German) Source: "The Blade" / Draft: "Wie 많은 'The Blade'..." -> Result: Residue (untranslated).
- **Wrong Alias**: (Korean) Alias: "티엔무 사원" / Draft: "티엔무 병원" -> Result: Wrong Alias (semantic error).

## Decision Logic
1. Extract the **core entity name** from "{draft_text}" (e.g., from "칼과 꽃에는", extract "칼과 꽃").
2. Is this core name in the [{alias_set}]? -> 'is_stem_in_alias_list'
3. **CRITICAL RULE**: If 'is_stem_in_alias_list' is TRUE, you **CANNOT** mark it as Omission or Wrong Alias. Any awkwardness MUST be **Grammar**.

## Response Format (JSON only)
{{
  "error_type": "Omission" | "Residue" | "Wrong Alias" | "Grammar" | "Normal",
  "reason": "Explain why in one clear sentence."
}}
"""


COMMON_ALLOWED = r"a-zA-Z0-9\s.,!?\"'()\-:;/@#%&" + r"\u3000-\u303F\uFF00-\uFFEF"

patterns = {
    "ko": rf"[^ㄱ-ㅎㅏ-ㅣ가-힣{COMMON_ALLOWED}]",
    "ja": rf"[^ぁ-んァ-ン一-龥々{COMMON_ALLOWED}]",
    "zh": rf"[^一-龥{COMMON_ALLOWED}]",
    "ar": rf"[^\u0600-\u06FF{COMMON_ALLOWED}]",
    "th": rf"[^\u0e00-\u0e7f{COMMON_ALLOWED}]",
    "de": rf"[^a-zA-Z0-9äöüÄÖÜß\s.,!?\"'()\-:;{COMMON_ALLOWED}]",
    "es": rf"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ¡¿\s.,!?\"'()\-:;{COMMON_ALLOWED}]",
    "fr": rf"[^a-zA-Z0-9àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ\s.,!?\"'()\-:;{COMMON_ALLOWED}]",
    "it": rf"[^a-zA-Z0-9àèéìòóùÀÈÉÌÒÓÙ\s.,!?\"'()\-:;{COMMON_ALLOWED}]",
    "tr": rf"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s.,!?\"'()\-:;{COMMON_ALLOWED}]",
}


def pre_check_errors(draft_text, alias_set, target_lang):
    if not any(alias in draft_text for alias in alias_set):
        return "Omission", "The entity core is completely missing from the translation."

    pattern = patterns.get(target_lang)
    illegal_chars = re.findall(pattern, draft_text)

    if illegal_chars:
        return "Residue", f"Contains illegal characters for {target_lang}: {''.join(set(illegal_chars))}"

    return None, None


def _to_float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _extract_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def build_alias_set(memory: EntityMemoryBlock) -> list[str]:
    candidates = []
    seen = set()

    def _add(s: str):
        normalized = s.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            candidates.append(normalized)

    if memory.canonical_target:
        _add(memory.canonical_target)
    if memory.alias_candidates:
        for a in memory.alias_candidates:
            if a:
                _add(a)

    return candidates


def should_trigger_ercm(
    draft: TranslationDraft,
    memory: EntityMemoryBlock,
    top_candidate: ScoredCandidate | None,
    threshold: float,
) -> ERCMDecision:
    """
    Rule-first ERCM trigger function aligned with v2 I/O spec:
    should_trigger_ercm(draft, memory, top_candidate, threshold) -> ERCMDecision
    """
    reasons: list[str] = []
    error_types: list[str] = []

    alias_set = build_alias_set(memory) if memory else []
    draft_text = draft.draft_text if draft and draft.draft_text else ""

    # 1) canonical/alias omission signal
    if alias_set and not any(alias in draft_text for alias in alias_set):
        reasons.append("missing_canonical")
        error_types.append("Omission")

    # 2) english residue signal (source_span survives in english script)
    source_span = memory.source_span if memory else None
    if source_span and re.search(r"[A-Za-z]", source_span):
        source_tokens = [tok.lower() for tok in re.findall(r"[A-Za-z]+", source_span) if len(tok) >= 4]
        draft_lower = draft_text.lower()
        if source_span.lower() in draft_lower or any(tok in draft_lower for tok in source_tokens):
            reasons.append("english_residue")
            error_types.append("Residue")

    # 3) low-margin signal from reranker uncertainty
    if top_candidate and top_candidate.margin_to_next is not None and top_candidate.margin_to_next < threshold:
        reasons.append("low_margin")

    if not reasons:
        return ERCMDecision(
            should_run=False,
            reasons=["normal"],
            confidence=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
            error_types=None,
            verifier_score=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
        )

    return ERCMDecision(
        should_run=True,
        reasons=list(dict.fromkeys(reasons)),
        confidence=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
        error_types=list(dict.fromkeys(error_types)) if error_types else None,
        verifier_score=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
    )


def run_llm_judge(
    source: str,
    draft: TranslationDraft,
    target_lang: str,
    alias_set: list[str],
    llm: LLM,
) -> ERCMDecision:
    memory = draft.used_memory
    source_span = memory.source_span if memory else None

    pre_error_type, _ = pre_check_errors(draft.draft_text, alias_set, target_lang)

    if pre_error_type:
        return ERCMDecision(
            should_run=True,
            reasons=[_ERROR_TO_REASON.get(pre_error_type, "ercm_needed")],
            confidence=1.0,
            error_types=[pre_error_type],
        )

    user_prompt = _build_user_prompt(
        source=source,
        draft_text=draft.draft_text,
        target_lang=target_lang,
        source_span=source_span,
        alias_set=alias_set,
    )

    conversation = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    outputs = llm.chat(conversation, sampling_params=sampling_params)
    raw = outputs[0].outputs[0].text.strip()
    json_text = _extract_json_text(raw)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return ERCMDecision(
            should_run=False,
            reasons=["judge_parse_error"],
            confidence=0.0,
            error_types=None,
        )

    parsed_error_types = parsed.get("error_types")
    if isinstance(parsed_error_types, list):
        error_types = [et for et in parsed_error_types if et in _VALID_ERROR_TYPES and et != "Normal"]
    else:
        single_type = parsed.get("error_type", "Normal")
        error_types = [single_type] if single_type in _VALID_ERROR_TYPES and single_type != "Normal" else []

    if not error_types:
        return ERCMDecision(
            should_run=False,
            reasons=[_ERROR_TO_REASON["Normal"]],
            confidence=_to_float_or_none(parsed.get("confidence", 0.8)),
            error_types=None,
            verifier_score=_to_float_or_none(parsed.get("verifier_score")),
        )

    reason_codes = [_ERROR_TO_REASON.get(error_type, "ercm_needed") for error_type in error_types]

    return ERCMDecision(
        should_run=True,
        reasons=list(dict.fromkeys(reason_codes)),
        confidence=_to_float_or_none(parsed.get("confidence", 0.8)),
        error_types=error_types,
        verifier_score=_to_float_or_none(parsed.get("verifier_score")),
    )
