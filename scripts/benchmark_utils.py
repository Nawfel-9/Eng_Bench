import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from prompts.task_prompts import USER_PROMPT, FORMAT_SUFFIXES


def infer_answer_format(question: str) -> str:
    text = question.lower()
    if "correct option letter" in text:
        return "multiple_choice"
    if "floating-point number with three decimal" in text:
        return "float_3dp"
    if "floating-point number with two decimal" in text:
        return "float_2dp"
    if "floating-point number with one decimal" in text:
        return "float_1dp"
    if "requiring an integer" in text:
        return "integer"
    if "data array" in text or "array or tuple" in text or "araary" in text:
        return "array"
    return "other"


def build_prompt(question: str, answer_format: str) -> str:
    """Build the user-turn text: question body + format-specific suffix."""
    suffix = FORMAT_SUFFIXES.get(answer_format, FORMAT_SUFFIXES["other"])
    return USER_PROMPT.format(question=question.strip()) + suffix


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s\.\-\(\),]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def quantize_decimal(value: str, places: int) -> str:
    quant = Decimal("1").scaleb(-places)
    return format(Decimal(value).quantize(quant, rounding=ROUND_HALF_UP), "f")


def _extract_last_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else ""


def extract_final_answer(prediction: str, answer_format: str) -> str:
    if not prediction or prediction.startswith("ERROR:"):
        return ""

    lines = [line.strip() for line in str(prediction).splitlines() if line.strip()]
    tail = lines[-1] if lines else str(prediction).strip()
    full = str(prediction).strip()

    if answer_format == "multiple_choice":
        letter_matches = re.findall(r"\b([A-Z])\b", full.upper())
        return letter_matches[-1] if letter_matches else ""

    if answer_format in {"float_1dp", "float_2dp", "float_3dp"}:
        places = {"float_1dp": 1, "float_2dp": 2, "float_3dp": 3}[answer_format]
        candidate = _extract_last_number(tail) or _extract_last_number(full)
        if not candidate:
            return ""
        try:
            return quantize_decimal(candidate, places)
        except (InvalidOperation, ValueError):
            return candidate

    if answer_format == "integer":
        int_matches = re.findall(r"[-+]?\d+", tail)
        if int_matches:
            return int_matches[-1]
        int_matches = re.findall(r"[-+]?\d+", full)
        return int_matches[-1] if int_matches else ""

    if answer_format == "array":
        tuple_matches = re.findall(r"(\([^)]*\))", tail)
        if tuple_matches:
            return tuple_matches[-1]
        tuple_matches = re.findall(r"(\([^)]*\))", full)
        return tuple_matches[-1] if tuple_matches else tail

    return tail
