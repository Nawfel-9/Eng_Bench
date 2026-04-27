BASE_PROMPT = (
    "You are an expert electrical and electronics engineer solving a problem from a "
    "technical diagram. Read the image carefully, use only the information visible "
    "in the diagram and question, and avoid guessing beyond the evidence.\n\n"
    "{question}\n\n"
)

FORMAT_SUFFIXES = {
    "multiple_choice": (
        "Give a brief technical explanation. On the last line, write only the final "
        "option letter."
    ),
    "float_1dp": (
        "Give a brief technical explanation. On the last line, write only the final "
        "numeric answer with exactly one decimal place."
    ),
    "float_2dp": (
        "Give a brief technical explanation. On the last line, write only the final "
        "numeric answer with exactly two decimal places."
    ),
    "float_3dp": (
        "Give a brief technical explanation. On the last line, write only the final "
        "numeric answer with exactly three decimal places."
    ),
    "integer": (
        "Give a brief technical explanation. On the last line, write only the final "
        "integer value."
    ),
    "array": (
        "Follow the requested answer format exactly. On the last line, write only the "
        "final answer in the requested array or tuple style."
    ),
    "other": (
        "Follow the requested answer format exactly. On the last line, write only the "
        "final answer."
    ),
}
