"""Utilities for loading model vocabulary and tokenizer data."""

from typing import Any
from .config_parse import load_json_file


def load_vocabulary(model: Any) -> tuple[dict[int, str], set[int]]:
    """Load the model vocabulary and special token ID.

    Args:
        model: Model used to locate vocabulary and tokenizer files.

    Returns:
        A mapping of token ID to tokens and a set of special token ID.
    """
    vocab_path = model.get_path_to_vocab_file()
    vocab = load_json_file(vocab_path)
    id_to_token = {token_id: token for token, token_id in vocab.items()}

    tokenizer_path = model.get_path_to_tokenizer_file()
    tokenizer = load_json_file(tokenizer_path)
    added_tokens = tokenizer["added_tokens"]
    special_ids: set[int] = set()

    for token in added_tokens:
        id_to_token[token["id"]] = token["content"]
        if token["special"]:
            special_ids.add(token["id"])

    return id_to_token, special_ids
