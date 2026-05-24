"""
LLM-as-judge evaluation for final LakeQA answers.

The judge compares the final produced answer against the task's expected
answer while also checking answer-format constraints from the final question.
"""

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from .llm.llm_factory import LLMFactory


JUDGE_SYSTEM_PROMPT = """You are a strict LakeQA final-answer judge.

Return exactly one JSON object and no extra text. The JSON schema is:
{
  "passed": true or false,
  "answer_type": "number" | "entity" | "list" | "date" | "other",
  "semantic_match": true or false,
  "format_ok": true or false,
  "numeric_match": true or false or null,
  "reason": "brief explanation"
}

Judge only the final answer, not the reasoning process.

Rules:
1. The expected answer is the authoritative answer.
2. Use the question to enforce explicit output constraints. Examples:
   - If the question requires lexical/alphabetical/sorted order, the produced list must be in that order.
   - If the question disallows acronyms, an acronym answer is incorrect.
   - If the question says "Only report the name", extra explanation or extra fields make the answer incorrect.
   - Treat "Your response should be in the format [Answer]" as requiring a concise answer value, not literal square brackets.
3. If the expected answer is numeric, the numeric value must match exactly. Do not accept approximations, rounded values, ranges, unit conversions, or extra numbers. Ignore only cosmetic commas and whitespace.
4. If the expected answer is an entity, accept semantically equivalent names unless the question's format constraints forbid that form.
   Examples that can pass when not forbidden:
   - "XX" vs "XX County"
   - "John F. Kennedy" vs "John Franklin Kennedy"
   - a U.S. state name vs its postal abbreviation
5. For lists or sets, all required elements must be present and no extra answer elements should be present. Preserve order when the question asks for order, ranking, top-k order, lexical order, or alphabetical order.
6. If the produced answer is empty or the answer is ambiguous, mark it incorrect.
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "pass", "passed"}:
            return True
        if normalized in {"false", "no", "0", "fail", "failed"}:
            return False
    return default


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return _as_bool(value, default=False)


def _extract_numbers(value: Any) -> list[str]:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return [str(value)]
    text = str(value)
    return re.findall(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?", text)


def _to_decimal(number_text: str) -> Optional[Decimal]:
    try:
        return Decimal(number_text.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def _is_scalar_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Decimal)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    return re.fullmatch(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?", text) is not None


def _scalar_number_matches(produced_answer: Any, expected_answer: Any) -> bool:
    expected_numbers = _extract_numbers(expected_answer)
    produced_numbers = _extract_numbers(produced_answer)
    if len(expected_numbers) != 1 or len(produced_numbers) != 1:
        return False

    expected_value = _to_decimal(expected_numbers[0])
    produced_value = _to_decimal(produced_numbers[0])
    if expected_value is None or produced_value is None:
        return False
    return expected_value == produced_value


def _empty_judge_result(
    judge_model: str,
    passed: bool,
    reason: str,
    answer_type: str = "",
    semantic_match: bool = False,
    format_ok: bool = False,
    numeric_match: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "llm_judge_model": judge_model,
        "llm_judge_pass": passed,
        "llm_judge_score": 1.0 if passed else 0.0,
        "llm_judge_answer_type": answer_type,
        "llm_judge_semantic_match": semantic_match,
        "llm_judge_format_ok": format_ok,
        "llm_judge_numeric_match": numeric_match,
        "llm_judge_reason": reason,
        "llm_judge_raw": "",
        "llm_judge_input_tokens": 0,
        "llm_judge_cached_input_tokens": 0,
        "llm_judge_cache_write_input_tokens": 0,
        "llm_judge_output_tokens": 0,
        "llm_judge_total_tokens": 0,
        "llm_judge_cost_usd": 0.0,
        "llm_judge_cost_source": "not_called",
        "llm_judge_error": "",
    }


def judge_task_result(
    task: Dict[str, Any],
    result: Dict[str, Any],
    judge_model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Judge one task result and return CSV-friendly judge fields."""
    expected_answer = task.get("answer", "")
    produced_answer = result.get("predicted_answer", "")
    question = task.get("final_question") or task.get("question") or result.get("question", "")

    if expected_answer in (None, ""):
        return _empty_judge_result(
            judge_model,
            passed=False,
            reason="Missing expected answer; judge was not called.",
        )

    if not str(produced_answer).strip():
        return _empty_judge_result(
            judge_model,
            passed=False,
            reason="Produced answer is empty; judge was not called.",
            answer_type="number" if _is_scalar_number(expected_answer) else "",
            numeric_match=False if _is_scalar_number(expected_answer) else None,
        )

    user_payload = {
        "question": question,
        "expected_answer": expected_answer,
        "produced_answer": produced_answer,
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Grade this LakeQA final answer.\n\n"
                f"{_json_dumps(user_payload)}"
            ),
        },
    ]

    try:
        llm = LLMFactory.create(judge_model, reasoning_effort=reasoning_effort)
        response = llm.generate(
            messages=messages,
            expect_json=True,
            temperature=0.0,
            max_tokens=max_tokens,
        )

        parsed = response.metadata.get("parsed_json")
        if not isinstance(parsed, dict):
            parsed = {}

        passed = _as_bool(parsed.get("passed"), default=False)
        semantic_match = _as_bool(parsed.get("semantic_match"), default=False)
        format_ok = _as_bool(parsed.get("format_ok"), default=False)
        numeric_match = _as_optional_bool(parsed.get("numeric_match"))
        answer_type = str(parsed.get("answer_type", "") or "")
        reason = str(parsed.get("reason", "") or "")

        if _is_scalar_number(expected_answer):
            numeric_match = _scalar_number_matches(produced_answer, expected_answer)
            answer_type = answer_type or "number"
            if passed and not numeric_match:
                passed = False
                reason = (reason + " " if reason else "") + (
                    "Numeric expected answer did not match exactly after deterministic check."
                )

        if passed and not format_ok:
            passed = False
            reason = (reason + " " if reason else "") + (
                "Judge marked the answer format as invalid."
            )
        if passed and not semantic_match and not _is_scalar_number(expected_answer):
            passed = False
            reason = (reason + " " if reason else "") + (
                "Judge did not mark the answer as semantically matching."
            )

        input_tokens = response.usage.get("input_tokens", 0)
        cached_input_tokens = response.usage.get("cached_input_tokens", 0)
        cache_write_input_tokens = response.usage.get("cache_write_input_tokens", 0)
        output_tokens = response.usage.get("output_tokens", 0)
        cost_info = LLMFactory.calculate_cost_breakdown(
            judge_model,
            input_tokens,
            output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
        )

        return {
            "llm_judge_model": judge_model,
            "llm_judge_pass": passed,
            "llm_judge_score": 1.0 if passed else 0.0,
            "llm_judge_answer_type": answer_type,
            "llm_judge_semantic_match": semantic_match,
            "llm_judge_format_ok": format_ok,
            "llm_judge_numeric_match": numeric_match,
            "llm_judge_reason": reason[:1000],
            "llm_judge_raw": response.content[:2000],
            "llm_judge_input_tokens": input_tokens,
            "llm_judge_cached_input_tokens": cached_input_tokens,
            "llm_judge_cache_write_input_tokens": cache_write_input_tokens,
            "llm_judge_output_tokens": output_tokens,
            "llm_judge_total_tokens": input_tokens + output_tokens,
            "llm_judge_cost_usd": cost_info["cost_usd"],
            "llm_judge_cost_source": cost_info["source"],
            "llm_judge_error": "" if response.metadata.get("json_valid") else response.metadata.get("json_error", ""),
        }
    except Exception as exc:
        return _empty_judge_result(
            judge_model,
            passed=False,
            reason="Judge call failed.",
        ) | {
            "llm_judge_cost_source": "judge_error",
            "llm_judge_error": f"{type(exc).__name__}: {str(exc)}",
        }
