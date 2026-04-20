import re
import json
from typing import Any

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:  # pragma: no cover
    LLM = Any

    class SamplingParams:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.kwargs = kwargs

from DTOlist import TranslationDraft, EntityMemoryBlock, ERCMDecision, ScoredCandidate


# ============================================================
# Constants / label mappings
# ============================================================

_ERROR_TO_REASON = {
    "Omission": "missing_canonical",
    "Residue": "foreign_script_residue",
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

_POST_EDIT_SYSTEM_PROMPT = """You are a conservative entity-aware translation post-editor.

Your rules:
1) Perform the smallest safe edit only.
2) Do not paraphrase unless it is strictly necessary.
3) Do not introduce a new entity name unless it is the preferred canonical translation.
4) Do not mix the source entity name and the canonical translation in the same repaired sentence unless the original draft already contained both and keeping both is necessary.
5) If the draft is already acceptable after simple cleanup, keep it unchanged.
6) If a safe repair is uncertain, return the minimally cleaned draft unchanged.

You must respond only in JSON format with no additional text.
"""


# ============================================================
# Script filters / patterns
# ============================================================

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


# ============================================================
# Prompt builders
# ============================================================

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
- Does the translation contain untranslated source text (English) or other foreign scripts?
- **Exceptions**: Technical terms (USB, x86, DNA) or brand names.
- **Error**: If the entity or sentence remains in another script without reason → mark Residue.

### Step 3 — Check for Grammar (Linguistic Integration)
- Mark Grammar ONLY if the entity is present but its integration is incorrect for {lang_name}:
  1. Particles (ko, ja): mismatch with batchim or inappropriate local attachment.
  2. Gender/Number/Article (fr, de, es, it): incorrect agreement or wrong articles.
  3. Case (ar, tr, de): incorrect case/attachment for the sentence structure.
- If the form is natural, it is NOT a Grammar error.

### Step 4 — Check for Wrong Alias (Semantics)
- Mark Wrong Alias ONLY if the core meaning is wrong or a term not in the list is used.
- If the core stem matches anything in [{alias_str}], do NOT mark as Wrong Alias.

### Step 5 — Final Decision
- If no errors are found → mark Normal.

## Response Format (JSON only)
{{
  "error_type": "Omission" | "Residue" | "Wrong Alias" | "Grammar" | "Normal",
  "reason": "Explain why in one clear sentence."
}}
"""


def _build_post_edit_user_prompt(
    source: str,
    draft_text: str,
    target_lang: str,
    memory: EntityMemoryBlock | None,
    alias_set: list[str],
    error_types: list[str] | None,
    reasons: list[str] | None,
) -> str:
    lang_name = _LANG_NAME.get(target_lang, target_lang)
    source_span = memory.source_span if memory else None
    canonical = memory.canonical_target if memory else None

    alias_str = ", ".join(f'"{a}"' for a in alias_set) if alias_set else "N/A"
    error_str = ", ".join(error_types) if error_types else "Normal"
    reason_str = ", ".join(reasons) if reasons else "normal"
    span_str = f'"{source_span}"' if source_span else "N/A"
    canonical_str = f'"{canonical}"' if canonical else "N/A"

    return f"""Repair the translation below with the smallest safe edit.

## Context
- Source (English): "{source}"
- Target language: {lang_name}
- Source entity: {span_str}
- Preferred canonical translation: {canonical_str}
- Acceptable entity aliases: [{alias_str}]
- Current draft: "{draft_text}"
- Detected error types: {error_str}
- Detected reasons: {reason_str}

## Strict repair policy
1. Preserve the original meaning.
2. Keep the sentence in {lang_name}.
3. Do NOT paraphrase unless strictly necessary.
4. Do NOT introduce a new entity name unless it is the preferred canonical translation.
5. Do NOT mix the English source entity and the target canonical translation in one repaired sentence.
6. If the current draft is already safe after minimal cleanup, keep it unchanged.
7. If uncertain, return the current draft unchanged.

## Output JSON format
{{
  "revised_text": "final repaired translation",
  "applied": true,
  "edit_summary": "one short sentence describing what was fixed"
}}
"""


# ============================================================
# Utility helpers
# ============================================================

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


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:؟。！？])", r"\1", text)
    text = re.sub(r"([¿¡])\s+", r"\1", text)
    return text.strip()


def _remove_illegal_chars(text: str, target_lang: str) -> str:
    pattern = patterns.get(target_lang)
    if not pattern:
        return _normalize_text(text)
    cleaned = re.sub(pattern, "", text)
    return _normalize_text(cleaned)


def _contains_illegal_chars(text: str, target_lang: str) -> bool:
    pattern = patterns.get(target_lang)
    if not pattern:
        return False
    return bool(re.search(pattern, text))


def _has_alias_match(text: str, alias_set: list[str]) -> bool:
    return any(alias and alias in text for alias in alias_set)


def _entity_present_alias(text: str, alias_set: list[str]) -> str | None:
    for alias in sorted(alias_set, key=len, reverse=True):
        if alias and alias in text:
            return alias
    return None


def _preferred_entity_form(memory: EntityMemoryBlock | None, alias_set: list[str]) -> str:
    if memory and memory.canonical_target and memory.canonical_target.strip():
        return memory.canonical_target.strip()
    for alias in alias_set:
        if alias and alias.strip():
            return alias.strip()
    return ""


def _hangul_has_batchim(ch: str) -> bool | None:
    if not ch or not ("가" <= ch <= "힣"):
        return None
    return (ord(ch) - ord("가")) % 28 != 0


def _detect_korean_particle_error(text: str, alias_set: list[str]) -> str | None:
    particle_rules = [
        ("은", "는"),
        ("이", "가"),
        ("을", "를"),
        ("과", "와"),
    ]

    for alias in sorted(alias_set, key=len, reverse=True):
        alias = alias.strip()
        if not alias:
            continue

        last_char = alias[-1]
        has_batchim = _hangul_has_batchim(last_char)
        if has_batchim is None:
            continue

        for batchim_form, no_batchim_form in particle_rules:
            pattern = rf"({re.escape(alias)})([{batchim_form}{no_batchim_form}])"
            for match in re.finditer(pattern, text):
                actual = match.group(2)
                expected = batchim_form if has_batchim else no_batchim_form
                if actual != expected:
                    return (
                        f'Korean particle mismatch after "{alias}": '
                        f'expected "{expected}", got "{actual}".'
                    )
    return None


def _fix_korean_particles(text: str, alias_set: list[str]) -> str:
    particle_rules = [
        ("은", "는"),
        ("이", "가"),
        ("을", "를"),
        ("과", "와"),
    ]

    fixed = text

    for alias in sorted(alias_set, key=len, reverse=True):
        alias = alias.strip()
        if not alias:
            continue

        last_char = alias[-1]
        has_batchim = _hangul_has_batchim(last_char)
        if has_batchim is None:
            continue

        for batchim_form, no_batchim_form in particle_rules:
            correct = batchim_form if has_batchim else no_batchim_form
            fixed = re.sub(
                rf"({re.escape(alias)})([{batchim_form}{no_batchim_form}])",
                rf"\1{correct}",
                fixed,
            )

    return _normalize_text(fixed)


def _insert_entity_before_final_punct(text: str, entity: str) -> str:
    text = text.strip()
    entity = entity.strip()
    if not entity:
        return _normalize_text(text)

    m = re.search(r"([?.!؟。！？])\s*$", text)
    if m:
        punct = m.group(1)
        body = text[:m.start()].rstrip()
        if body:
            return _normalize_text(f"{body} {entity}{punct}")
        return _normalize_text(f"{entity}{punct}")

    return _normalize_text(f"{text} {entity}")


def _fill_common_entity_slot(text: str, canonical: str, target_lang: str) -> str:
    if not canonical:
        return text

    if target_lang == "zh":
        text = re.sub(r"(電影)\s*([？?])", rf"\1《{canonical}》\2", text)

    text = re.sub(r"\b(film)\s+([?])", rf"\1 {canonical}\2", text)
    text = re.sub(r"\b(das|der|die|le|la|el|il)\s+([?؟])", rf"\1 {canonical}\2", text)

    if target_lang == "tr":
        text = re.sub(r"^\s*adlı\b", f"{canonical} adlı", text)

    return _normalize_text(text)


def _repair_leading_numeric_entity(text: str, canonical: str) -> str:
    if not canonical:
        return text

    m_text = re.match(r"^\s*(\d+)\b", text)
    m_canon = re.match(r"^\s*(\d+)\b", canonical)
    if not m_text or not m_canon:
        return text

    if m_text.group(1) != m_canon.group(1):
        return text

    return re.sub(r"^\s*\d+\b", canonical, text, count=1)


def _repair_source_question_frame(
    source: str,
    text: str,
    target_lang: str,
    alias_set: list[str],
) -> str:
    low = source.lower().strip()
    alias = _entity_present_alias(text, alias_set)

    if not alias:
        return text

    if target_lang == "ko" and low.startswith("where "):
        text = re.sub(
            rf"({re.escape(alias)})(은|는)\s*있습니까\?$",
            rf"\1\2 어디에 있습니까?",
            text,
        )

    if target_lang == "ja" and low.startswith("what genre"):
        if re.search(r"ジャンルは[?？]$", text):
            text = f"{alias}のジャンルは何ですか?"

    return _normalize_text(text)


def _sanitize_error_types_from_rules(
    draft_text: str,
    target_lang: str,
    alias_set: list[str],
    error_types: list[str],
) -> list[str]:
    cleaned = list(dict.fromkeys(error_types))

    # alias already exists -> Omission / Wrong Alias impossible
    if _has_alias_match(draft_text, alias_set):
        cleaned = [et for et in cleaned if et not in {"Omission", "Wrong Alias"}]

    # trust Korean grammar only when rule-based particle mismatch is found
    if target_lang == "ko" and "Grammar" in cleaned:
        if _detect_korean_particle_error(draft_text, alias_set) is None:
            cleaned = [et for et in cleaned if et != "Grammar"]

    return cleaned


def _should_use_llm_post_edit(
    decision: ERCMDecision,
    always_post_edit: bool = False,
) -> bool:
    if always_post_edit:
        return True
    error_types = decision.error_types or []
    return "Wrong Alias" in error_types


def _is_safe_post_edit_candidate(
    candidate: str,
    fallback_text: str,
    target_lang: str,
    memory: EntityMemoryBlock | None,
    alias_set: list[str],
    error_types: list[str] | None,
) -> bool:
    candidate = _normalize_text(candidate)
    if not candidate:
        return False

    errs = error_types or []
    canonical = _preferred_entity_form(memory, alias_set)
    source_span = (memory.source_span or "").strip() if memory and memory.source_span else ""

    pattern = patterns.get(target_lang)
    if pattern and re.findall(pattern, candidate):
        return False

    if any(err in errs for err in ["Omission", "Residue", "Wrong Alias"]):
        if canonical and not _has_alias_match(candidate, alias_set):
            return False

    if target_lang == "ko" and "Grammar" in errs:
        if _detect_korean_particle_error(candidate, alias_set) is not None:
            return False

    fallback_len = max(len(fallback_text.strip()), 1)
    cand_len = len(candidate.strip())
    if cand_len > fallback_len * 2.5 + 20:
        return False

    if "Residue" in errs and source_span and canonical and re.search(r"[A-Za-z]", source_span):
        if source_span.lower() in candidate.lower() and canonical.lower() not in candidate.lower():
            return False

    return True


def _build_default_edit_summary(
    original_text: str,
    revised_text: str,
    error_types: list[str] | None,
) -> str:
    if revised_text == original_text:
        return "No post-edit applied."
    if error_types:
        return f"Applied deterministic repair for {', '.join(error_types)}."
    return "Applied deterministic repair."


# ============================================================
# Alias / memory helpers
# ============================================================

def build_alias_set(memory: EntityMemoryBlock) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        normalized = s.strip()
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            candidates.append(normalized)

    if memory and memory.canonical_target:
        _add(memory.canonical_target)

    if memory and memory.alias_candidates:
        for a in memory.alias_candidates:
            if a:
                _add(a)

    return candidates


# ============================================================
# Pre-check
# ============================================================

def pre_check_errors(
    draft_text: str,
    alias_set: list[str],
    target_lang: str,
) -> tuple[str | None, str | None]:
    if alias_set and not _has_alias_match(draft_text, alias_set):
        return "Omission", "The entity core is completely missing from the translation."

    pattern = patterns.get(target_lang)
    if pattern:
        illegal_chars = re.findall(pattern, draft_text)
        if illegal_chars:
            return "Residue", f"Contains illegal characters for {target_lang}: {''.join(sorted(set(illegal_chars)))}"

    if target_lang == "ko":
        grammar_reason = _detect_korean_particle_error(draft_text, alias_set)
        if grammar_reason:
            return "Grammar", grammar_reason

    return None, None


# ============================================================
# Internal LLM decision helper (private)
# ============================================================

def _run_llm_judge_decision(
    source: str,
    draft: TranslationDraft,
    target_lang: str,
    alias_set: list[str],
    llm: LLM,
) -> ERCMDecision:
    memory = draft.used_memory
    source_span = memory.source_span if memory else None

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

    error_types = _sanitize_error_types_from_rules(
        draft_text=draft.draft_text,
        target_lang=target_lang,
        alias_set=alias_set,
        error_types=error_types,
    )

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


# ============================================================
# Public API 1: main decision entrypoint
# ============================================================

def should_trigger_ercm(
    draft: TranslationDraft,
    memory: EntityMemoryBlock,
    top_candidate: ScoredCandidate | None,
    threshold: float,
    source: str | None = None,
    target_lang: str | None = None,
    llm: LLM | None = None,
) -> ERCMDecision:
    """
    Main decision entrypoint.

    External callers should use this function.
    It combines:
    - rule-based checks
    - low-margin uncertainty check
    - optional LLM judge
    """
    reasons: list[str] = []
    error_types: list[str] = []

    alias_set = build_alias_set(memory) if memory else []
    draft_text = draft.draft_text if draft and draft.draft_text else ""

    # 1) rule-based pre-check
    if target_lang:
        pre_error_type, _ = pre_check_errors(draft_text, alias_set, target_lang)
        if pre_error_type:
            reasons.append(_ERROR_TO_REASON.get(pre_error_type, "ercm_needed"))
            error_types.append(pre_error_type)
    else:
        if alias_set and not _has_alias_match(draft_text, alias_set):
            reasons.append("missing_canonical")
            error_types.append("Omission")

    # 2) source English residue heuristic
    source_span = memory.source_span if memory else None
    if source_span and re.search(r"[A-Za-z]", source_span):
        draft_lower = draft_text.lower()
        source_tokens = [tok.lower() for tok in re.findall(r"[A-Za-z]+", source_span) if len(tok) >= 4]
        if source_span.lower() in draft_lower or any(tok in draft_lower for tok in source_tokens):
            if "Residue" not in error_types:
                reasons.append("foreign_script_residue")
                error_types.append("Residue")

    # 3) reranker uncertainty
    if top_candidate and top_candidate.margin_to_next is not None and top_candidate.margin_to_next < threshold:
        reasons.append("low_margin")

    # 4) explicit rule-based signal -> trigger immediately
    if error_types:
        return ERCMDecision(
            should_run=True,
            reasons=list(dict.fromkeys(reasons)),
            confidence=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
            error_types=list(dict.fromkeys(error_types)),
            verifier_score=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
        )

    # 5) low-margin only -> trigger without explicit error type
    if reasons:
        return ERCMDecision(
            should_run=True,
            reasons=list(dict.fromkeys(reasons)),
            confidence=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
            error_types=None,
            verifier_score=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
        )

    # 6) optional LLM decision
    if source is not None and target_lang is not None and llm is not None:
        llm_decision = _run_llm_judge_decision(
            source=source,
            draft=draft,
            target_lang=target_lang,
            alias_set=alias_set,
            llm=llm,
        )

        if top_candidate is not None:
            if llm_decision.confidence is None:
                llm_decision.confidence = _to_float_or_none(top_candidate.final_score)
            if getattr(llm_decision, "verifier_score", None) is None:
                llm_decision.verifier_score = _to_float_or_none(top_candidate.final_score)

        return llm_decision

    # 7) no issue detected
    return ERCMDecision(
        should_run=False,
        reasons=["normal"],
        confidence=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
        error_types=None,
        verifier_score=_to_float_or_none(top_candidate.final_score) if top_candidate else None,
    )


# ============================================================
# Public API 2: correction
# ============================================================

def _rule_based_correction(
    source: str,
    draft_text: str,
    target_lang: str,
    memory: EntityMemoryBlock | None,
    alias_set: list[str],
    error_types: list[str] | None,
) -> str:
    text = _normalize_text(draft_text or "")
    if not memory:
        return text

    canonical = _preferred_entity_form(memory, alias_set)
    source_span = (memory.source_span or "").strip()
    error_types = error_types or []

    if "Residue" in error_types:
        if source_span and canonical and source_span in text:
            text = text.replace(source_span, canonical)

        text = _remove_illegal_chars(text, target_lang)

        text = _repair_source_question_frame(
            source=source,
            text=text,
            target_lang=target_lang,
            alias_set=alias_set,
        )

        if canonical and not _has_alias_match(text, alias_set):
            text = _fill_common_entity_slot(text, canonical, target_lang)

            if target_lang == "th":
                text = _repair_leading_numeric_entity(text, canonical)

            if not _has_alias_match(text, alias_set):
                text = _insert_entity_before_final_punct(text, canonical)

    if "Omission" in error_types and canonical:
        if source_span and source_span in text:
            text = text.replace(source_span, canonical)

        if not _has_alias_match(text, alias_set) and _contains_illegal_chars(text, target_lang):
            text = _remove_illegal_chars(text, target_lang)

        if not _has_alias_match(text, alias_set):
            text = _fill_common_entity_slot(text, canonical, target_lang)

        if not _has_alias_match(text, alias_set) and target_lang == "th":
            text = _repair_leading_numeric_entity(text, canonical)

        if not _has_alias_match(text, alias_set) and target_lang == "tr":
            text = re.sub(r"^\s*adlı\b", f"{canonical} adlı", text)

        if not _has_alias_match(text, alias_set):
            text = _insert_entity_before_final_punct(text, canonical)

    if "Wrong Alias" in error_types and canonical:
        if not _has_alias_match(text, alias_set):
            if source_span and source_span in text:
                text = text.replace(source_span, canonical)
            else:
                text = _insert_entity_before_final_punct(text, canonical)

    if target_lang == "ko":
        text = _fix_korean_particles(text, alias_set)

    text = _repair_source_question_frame(
        source=source,
        text=text,
        target_lang=target_lang,
        alias_set=alias_set,
    )

    return _normalize_text(text)


def apply_ercm_correction(
    source: str,
    draft: TranslationDraft,
    target_lang: str,
    decision: ERCMDecision,
    llm: LLM | None = None,
    always_use_llm: bool = False,
) -> dict[str, Any]:
    """
    Public correction entrypoint.
    """
    memory = draft.used_memory
    alias_set = build_alias_set(memory) if memory else []
    original_text = draft.draft_text or ""

    if not decision.should_run and not always_use_llm:
        return {
            "original_text": original_text,
            "revised_text": original_text,
            "applied": False,
            "edit_summary": "No post-edit applied.",
            "error_types": decision.error_types,
            "reasons": decision.reasons,
        }

    fallback_text = _rule_based_correction(
        source=source,
        draft_text=original_text,
        target_lang=target_lang,
        memory=memory,
        alias_set=alias_set,
        error_types=decision.error_types,
    )

    fallback_summary = _build_default_edit_summary(
        original_text=original_text,
        revised_text=fallback_text,
        error_types=decision.error_types,
    )

    if not llm or (not always_use_llm and not _should_use_llm_post_edit(decision, always_post_edit=False)):
        return {
            "original_text": original_text,
            "revised_text": fallback_text,
            "applied": fallback_text != original_text,
            "edit_summary": fallback_summary,
            "error_types": decision.error_types,
            "reasons": decision.reasons,
        }

    user_prompt = _build_post_edit_user_prompt(
        source=source,
        draft_text=fallback_text,
        target_lang=target_lang,
        memory=memory,
        alias_set=alias_set,
        error_types=decision.error_types,
        reasons=decision.reasons,
    )

    conversation = [
        {"role": "system", "content": _POST_EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    revised_text = fallback_text
    edit_summary = fallback_summary
    applied = fallback_text != original_text

    try:
        outputs = llm.chat(conversation, sampling_params=sampling_params)
        raw = outputs[0].outputs[0].text.strip()
        json_text = _extract_json_text(raw)
        parsed = json.loads(json_text)

        candidate = str(parsed.get("revised_text", "")).strip()
        candidate_summary = str(parsed.get("edit_summary", "")).strip() or "Applied LLM post-edit."

        if candidate:
            candidate = _normalize_text(candidate)

            if target_lang in patterns:
                candidate = _remove_illegal_chars(candidate, target_lang)

            if target_lang == "ko":
                candidate = _fix_korean_particles(candidate, alias_set)

            candidate = _repair_source_question_frame(
                source=source,
                text=candidate,
                target_lang=target_lang,
                alias_set=alias_set,
            )

            if _is_safe_post_edit_candidate(
                candidate=candidate,
                fallback_text=fallback_text,
                target_lang=target_lang,
                memory=memory,
                alias_set=alias_set,
                error_types=decision.error_types,
            ):
                revised_text = candidate
                edit_summary = candidate_summary
                applied = candidate != original_text
    except Exception:
        pass

    canonical = _preferred_entity_form(memory, alias_set)
    if canonical and decision.error_types:
        if any(err in decision.error_types for err in ["Omission", "Residue", "Wrong Alias"]):
            if not _has_alias_match(revised_text, alias_set):
                revised_text = _insert_entity_before_final_punct(fallback_text, canonical)
                revised_text = _normalize_text(revised_text)
                applied = revised_text != original_text
                edit_summary = _build_default_edit_summary(
                    original_text=original_text,
                    revised_text=revised_text,
                    error_types=decision.error_types,
                )

    return {
        "original_text": original_text,
        "revised_text": revised_text,
        "applied": applied,
        "edit_summary": edit_summary,
        "error_types": decision.error_types,
        "reasons": decision.reasons,
    }


# ============================================================
# Public API 3: end-to-end helper
# ============================================================

def run_ercm(
    source: str,
    draft: TranslationDraft,
    memory: EntityMemoryBlock,
    target_lang: str,
    top_candidate: ScoredCandidate | None,
    threshold: float,
    judge_llm: LLM | None = None,
    correction_llm: LLM | None = None,
) -> dict[str, Any]:
    """
    End-to-end helper:
    1) decision via should_trigger_ercm
    2) correction via apply_ercm_correction
    """
    decision = should_trigger_ercm(
        draft=draft,
        memory=memory,
        top_candidate=top_candidate,
        threshold=threshold,
        source=source,
        target_lang=target_lang,
        llm=judge_llm,
    )

    correction = apply_ercm_correction(
        source=source,
        draft=draft,
        target_lang=target_lang,
        decision=decision,
        llm=correction_llm,
    )

    return {
        "decision": decision,
        "correction": correction,
        "final_text": correction["revised_text"],
    }
