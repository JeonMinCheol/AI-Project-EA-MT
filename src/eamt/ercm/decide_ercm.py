import re
import json
from vllm import LLM, SamplingParams
from DTOlist import TranslationDraft, EntityMemoryBlock, ERCMDecision

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
    "tr": rf"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s.,!?\"'()\-:;{COMMON_ALLOWED}]"
}

def pre_check_errors(draft_text, alias_set, target_lang):
    if not any(alias in draft_text for alias in alias_set):
        return "Omission", "The entity core is completely missing from the translation."

    pattern = patterns.get(target_lang)
    illegal_chars = re.findall(pattern, draft_text)

    if illegal_chars:
        return "Residue", f"Contains illegal characters for {target_lang}: {''.join(set(illegal_chars))}"
    
    return None, None

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

def run_llm_judge(
        source: str,
        draft: TranslationDraft,
        target_lang: str,
        alias_set: list[str],
        llm: LLM,
) -> ERCMDecision:
    memory = draft.used_memory
    source_span = memory.source_span if memory else None

    pre_error_type, pre_reason = pre_check_errors(draft.draft_text, alias_set, target_lang)

    if pre_error_type:
        return ERCMDecision(
            should_run=True,
            reasons=[pre_reason],
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

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ERCMDecision(
            should_run=False,
            reasons=["judge_parse_error"],
            error_types=None,
        )
    
    res_type = parsed.get("error_type", "Normal")
    reason = parsed.get("reason", "")

    error_types = [res_type]
    
    if res_type == "Normal":
        return ERCMDecision(
            should_run=False,
            reasons=["normal"],
            error_types=None,
        )
    
    return ERCMDecision(
        should_run=True,
        reasons=[reason],
        error_types=error_types,
    )