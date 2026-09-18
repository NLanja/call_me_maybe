"""Utilities for constrained token generation.

This module filters valid tokens and controls model generation for
closed choices, numbers, and strings.
"""


from typing import Any

MAX_TOKENS = 12
MIN_STRING_TOKENS = 16
MAX_STRING_TOKENS = 256
STRING_TOKEN_MARGIN = 4
ALLOWED_NUMBER_CHARS = set("0123456789.-")
ALLOWED_INTEGER_CHARS = set("0123456789-")
FORBIDDEN_STRING_CHARS = set('"\n<>Ċ')
CLOSED_CHOICE_PREFIX = "fun_"


def filter_closed(
    prefix: str,
    candidates: list[str],
    id_to_token: dict[int, str],
    special_ids: set[int],
) -> list[int]:
    """Filter tokens matching the available choices.

    Args:
        prefix: Text generated.
        candidates: List of choices.
        id_to_token: Mapping of token IDs to token.
        special_ids: Token IDs to exclude.

    Returns:
        List of valid token IDs.
    """
    valid_ids = []

    for token_id, token in id_to_token.items():
        if token_id in special_ids:
            continue

        new_text = prefix + token

        for candidate in candidates:
            if candidate.startswith(new_text):
                valid_ids.append(token_id)
                break

    return valid_ids


def is_complete_choice(text: str, candidates: list[str]) -> bool:
    """Check whether the text constitutes a complete choice.

    Args:
        text: Generated text.
        candidate: List of choices.

    Returns:
        True if the text matches a choice.
    """
    return text in candidates


def filter_number(
    prefix: str,
    id_to_token: dict[int, str],
    special_ids: set[int],
    allow_decimal: bool = True,
) -> list[int]:
    """Filter tokens that can form a number.

    Args:
        prefix: Number generated so far.
        id_to_token: Mapping of token IDs to token.
        special_ids: Token IDs to exclude.
        allow_decimal: Whether a decimal point is allowed. True for
            "number" or "float" parameters, False for "integer"
            parameters.

    Returns:
        List of valid token IDs.
    """
    valid_ids = []
    is_first_token = prefix == ""
    has_dot = "." in prefix

    for token_id, token in id_to_token.items():
        if token_id in special_ids:
            continue

        if not token:
            continue

        candidate = token
        if is_first_token and candidate.startswith("Ġ"):
            candidate = candidate[1:]

        if not candidate:
            continue

        if not allow_decimal and "." in candidate:
            continue

        if not all(char in ALLOWED_NUMBER_CHARS for char in candidate):
            continue

        if allow_decimal and has_dot and "." in candidate:
            continue

        if "-" in candidate and not is_first_token:
            continue

        valid_ids.append(token_id)

    return valid_ids


def is_valid_number(text: str) -> bool:
    """Check if a text is a valid number.

    Args:
        text: Text to validate.

    Returns:
        True if the text is a valid number.

    """
    try:
        float(text)
        return True
    except ValueError:
        return False


def is_valid_integer(text: str) -> bool:
    """Check if a text is a valid integer (no decimal point).

    Args:
        text: Text to validate.

    Returns:
        True if the text is a valid integer.

    """

    try:
        int(text)
        return True
    except ValueError:
        return False


def filter_string(
    prefix: str,
    id_to_token: dict[int, str],
    special_ids: set[int],
    end_token_id: int,
) -> list[int]:
    """Filter tokens allowed in a string.

    Args:
        prefix: Text generated.
        id_to_token: Mapping of token IDs to token.
        special_ids: Token IDs to exclude.
        end_token_id: String termination token ID.

    Returns:
        List of valid token IDs.
    """
    valid_ids = []
    is_first_token = prefix == ""

    for token_id, token in id_to_token.items():
        if token_id in special_ids:
            continue

        if token_id == end_token_id:
            if not is_first_token:
                valid_ids.append(token_id)
            continue

        if any(char in FORBIDDEN_STRING_CHARS for char in token):
            continue

        valid_ids.append(token_id)

    return valid_ids


def compute_string_max_tokens(model: Any, source_text: str) -> int:
    """Compute a token budget for string extraction, sized to the input.

    An extracted string parameter can never be longer than the text it
    was extracted from. Sizing the generation budget on the length of
    `source_text` (instead of using a fixed constant) avoids truncating
    long, legitimate values while avoiding wasted forward passes -
    unlike closed/number/integer modes, string mode has no cheap
    early-stop signal other than reaching the closing quote, and the
    SDK does not cache past key values, so every wasted step recomputes
    attention over the whole growing sequence.

    Args:
        model: Model used to tokenize `source_text`.
        source_text: Text the string parameter is expected to be
            extracted from (typically the user prompt).

    Returns:
        Token budget, clamped between `MIN_STRING_TOKENS` and
        `MAX_STRING_TOKENS`.
    """
    try:
        token_count = len(model.encode(source_text).tolist()[0])
    except Exception:
        token_count = MIN_STRING_TOKENS

    budget = token_count + STRING_TOKEN_MARGIN
    return max(MIN_STRING_TOKENS, min(budget, MAX_STRING_TOKENS))


def get_best_token(logits: list[float], valid_ids: list[int]) -> int | None:
    """Get the highest-scoring valid token.

    Args:
        logits: Token scores from the model.
        valid_ids: Allowed token IDs.

    Returns:
        Highest-scoring token ID, or None if no token is valid.

    """
    if not valid_ids:
        return None

    return max(valid_ids, key=lambda token_id: logits[token_id])


def generate_constrained(
    model: Any,
    input_ids: list[int],
    id_to_token: dict[int, str],
    special_ids: set[int],
    mode: str,
    candidates: list[str] | None = None,
    end_token_id: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> tuple[str, list[int]]:
    """Generate text with constrained token selection.

    Args:
        model: Model used for token generation.
        input_ids: Initial input token IDs.
        id_to_token: Mapping of token IDs to tokens.
        special_ids: Token IDs to exclude.
        mode: Generation mode: closed, number, integer or string.
        candidates: List of choices for closed mode.
        end_token_id: Termination token for string mode.
        max_tokens: Maximum number of tokens to generate. Defaults to
            `MAX_TOKENS`; pass a value from `compute_string_max_tokens`
            for string mode to size the budget to the input.

    Returns:
        Generated text and complete token IDs.

    """
    if mode not in {"closed", "number", "integer", "string"}:
        raise ValueError("Unknown mode.")

    if mode == "closed" and candidates is None:
        raise ValueError("'candidates' is mandatory for closed mode.")

    if mode == "string" and end_token_id is None:
        raise ValueError("'End_token_id' is mandatory for string mode.")

    if mode == "closed":
        assert candidates is not None
        candidates = [CLOSED_CHOICE_PREFIX + c for c in candidates]

    raw_text = ""
    start_len = len(input_ids)
    string_terminated = False

    for step in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        if mode == "closed":
            assert candidates is not None
            valid_ids = filter_closed(
                raw_text,
                candidates,
                id_to_token,
                special_ids,
            )
        elif mode in {"number", "integer"}:
            allow_decimal = mode == "number"
            valid_ids = filter_number(
                raw_text,
                id_to_token,
                special_ids,
                allow_decimal=allow_decimal,
            )

            is_complete = (
                is_valid_number(raw_text) if allow_decimal
                else is_valid_integer(raw_text)
            )

            if is_complete:
                unconstrained_best = max(
                    range(len(logits)), key=lambda i: logits[i]
                )
                best_token = id_to_token.get(unconstrained_best, "")
                best_char = (
                    best_token.removeprefix("Ġ")
                )
                allowed_chars = (
                    ALLOWED_NUMBER_CHARS if allow_decimal
                    else ALLOWED_INTEGER_CHARS
                )
                if not best_char or not all(
                    char in allowed_chars for char in best_char
                ):
                    break

        else:
            assert end_token_id is not None
            valid_ids = filter_string(
                raw_text, id_to_token, special_ids, end_token_id
            )

        if not valid_ids:
            break

        next_id = get_best_token(logits, valid_ids)

        if next_id is None:
            break

        if mode == "string" and next_id == end_token_id:
            string_terminated = True
            break

        raw_text += id_to_token[next_id]
        input_ids = input_ids + [next_id]

        if mode == "closed":
            assert candidates is not None
            if is_complete_choice(raw_text, candidates):
                break

    if mode == "string" and not string_terminated:
        raise ValueError(
            "String generation did not reach the closing quote within "
            f"the {max_tokens}-token budget; refusing to return a "
            "truncated value."
        )

    new_ids = input_ids[start_len:]

    if mode == "closed":
        generated_text = raw_text.removeprefix(CLOSED_CHOICE_PREFIX)
    else:
        generated_text = model.decode(new_ids) if len(new_ids) > 0 else ""

    return generated_text, input_ids
