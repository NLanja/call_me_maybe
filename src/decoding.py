from typing import Any

MAX_TOKENS = 30
ALLOWED_NUMBER_CHARS = set("0123456789.-")


def filter_closed(
    prefix: str,
    candidates: list[str],
    id_to_token: dict[int, str],
    special_ids: set[int],
) -> list[int]:
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
    return text in candidates


def filter_number(
    id_to_token: dict[int, str],
    special_ids: set[int],
) -> list[int]:
    valid_ids = []

    for token_id, token in id_to_token.items():
        if token_id in special_ids:
            continue

        if not token:
            continue

        if all(char in ALLOWED_NUMBER_CHARS for char in token):
            valid_ids.append(token_id)

    return valid_ids


def is_valid_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def filter_string(
    id_to_token: dict[int, str],
    special_ids: set[int],
    end_token_id: int,
) -> list[int]:
    valid_ids = []

    for token_id, token in id_to_token.items():
        if token_id in special_ids:
            continue

        if token_id == end_token_id:
            valid_ids.append(token_id)
            continue

        if '"' not in token:
            valid_ids.append(token_id)

    return valid_ids


def get_best_token(logits: list[float], valid_ids: list[int]) -> int | None:
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
    end_token_id: int | None = None
) -> tuple[str, list[int]]:
    if mode not in {"closed", "number", "string"}:
        raise ValueError("Unknown mode.")

    if mode == "closed" and candidates is None:
        raise ValueError("'candidates' is mandatory for closed mode.")

    if mode == "string" and end_token_id is None:
        raise ValueError("'End_token_id' is mandatory for string mode.")

    raw_text = ""
    start_len = len(input_ids)

    for _ in range(MAX_TOKENS):
        logits = model.get_logits_from_input_ids(input_ids)

        if mode == "closed":
            assert candidates is not None
            valid_ids = filter_closed(
                raw_text,
                candidates,
                id_to_token,
                special_ids,
            )
        elif mode == "number":
            valid_ids = filter_number(id_to_token, special_ids)
        else:
            assert end_token_id is not None
            valid_ids = filter_string(id_to_token, special_ids, end_token_id)

        if not valid_ids:
            break

        next_id = get_best_token(logits, valid_ids)

        if next_id is None:
            break

        if mode == "string" and next_id == end_token_id:
            break

        raw_text += id_to_token[next_id]
        input_ids = input_ids + [next_id]

        if mode == "closed":
            assert candidates is not None
            if is_complete_choice(raw_text, candidates):
                break

    new_ids = input_ids[start_len]

    if mode == "closed":
        generated_text = raw_text
    else:
        generated_text = model.decode(new_ids) if new_ids else ""

    return generated_text, input_ids
