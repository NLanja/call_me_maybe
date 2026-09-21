"""Command-line entry point for the function calling tool.

This module parses command-line arguments and runs the function calling
pipeline using constrained decoding.
"""

import argparse
import sys
from pathlib import Path

from .config_parse import (
    JsonFileError,
    load_function_definitions,
    load_function_test,
    save_output,
)
from .generator import run_pipeline
from .vocabulary import load_vocabulary


def validate_json_path(path: str) -> str:
    """Validate that a file path has a JSON extension.

    Args:
        path: File path to validate.

    Returns:
        The validated file path.

    Raises:
        argparse.ArgumentTypeError: If the file is not a JSON file.
    """
    if Path(path).suffix.lower() != ".json":
        raise argparse.ArgumentTypeError(
            f"Invalid file format: {path}. Expected a .json file."
        )
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        help="Path to the function definitions JSON file."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to the prompts JSON File"
    )

    parser.add_argument(
        "--output",
        type=validate_json_path,
        help="Path to the output JSON File to generate"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model identifier to use for inference. Default is Qwen/Qwen3-0.6B."
    )

    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline.

    Loads input data, processes prompts, and saves the generated results.
    Any keyboard interruption is caught here so the program
    always exits cleanly instead of printing a raw traceback.
    """
    args = parse_args()

    try:
        _run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)


def _run(args: argparse.Namespace) -> None:
    """Load inputs, run the pipeline, and save the results.

    Args:
        args: Parsed command-line argumenrts
    """
    try:
        list_fun = load_function_definitions(args.functions_definition)
        list_prompt = load_function_test(args.input)
    except JsonFileError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model(model_name=args.model)
    except Exception as e:
        print(f"Error: failed to load the LLM model '{args.model}': {e}")
        sys.exit(1)

    try:
        id_to_token, special_ids = load_vocabulary(model)
    except JsonFileError as e:
        print(f"Error: failed to load model vocabulary: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: unexpected failure while loading vocabulary: {e}")
        sys.exit(1)

    quote_ids = [
        token_id for token_id, token in id_to_token.items()
        if token == '"'
    ]

    if len(quote_ids) != 1:
        print(
            "Error: expected the double-quote character to map to a "
            f"single token, got {len(quote_ids)} tokens: {quote_ids}. "
            "The 'string' generation mode relies on this assumption."
        )
        sys.exit(1)

    quote_token_id = quote_ids[0]

    try:
        results = run_pipeline(
            model=model,
            functions=list_fun,
            prompts=list_prompt,
            id_to_token=id_to_token,
            special_ids=special_ids,
            quote_token_id=quote_token_id,
        )
    except Exception as e:
        print(f"Error: unexpected failure while processing prompts: {e}")
        sys.exit(1)

    try:
        save_output(results, args.output)
    except JsonFileError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Done: {len(results)}/{len(list_prompt)} "
          "prompts processed successfully.")


if __name__ == "__main__":
    main()
