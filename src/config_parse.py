"""Utilities for loading and validating JSON data."""


import json
from pydantic import BaseModel, ValidationError
from pathlib import Path
from typing import Any

DATA_DEFAULTS: dict[str, str] = {
    "functions_definition": "data/input/functions_definition.json",
    "function_calling_tests": "data/input/function_calling_tests.json",
    "output": "data/output/function_calling_results.json",
}


class JsonFileError(Exception):
    """Raised when a JSON file cannot be loaded."""


class DataType(BaseModel):
    """Represent a function parameter or return type."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent a function definition and its parameters."""

    name: str
    description: str
    parameters: dict[str, DataType]
    returns: DataType


class Prompt(BaseModel):
    """Represent a function-calling test prompt."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Represent the result of a function-calling test."""

    prompt: str
    name: str
    parameters: dict[str, Any]


def load_json_file(
    path: str | None = None,
    default_key_name: str | None = None,
) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file. If not provided, the default path
            associated with `default_key_name` is used.
        default_key_name: Key used to retrieve the default file path
            from `DATA_DEFAULTS`.

    Returns:
        The parsed JSON data.

    Raises:
        JsonFileError: If the file is invalid, the file does not
            exist, access is denied, or JSON content is invalid.
    """
    if not path:
        if default_key_name is None:
            raise JsonFileError("No file path or default key provided.")
        path_to_open = DATA_DEFAULTS[default_key_name]
    else:
        path_to_open = path

    if Path(path_to_open).suffix.lower() != ".json":
        raise JsonFileError(
            f"Invalid file format: {path_to_open}. Expected a .json file."
        )

    try:
        with open(path_to_open, "r") as file:
            return json.load(file)

    except FileNotFoundError as error:
        raise JsonFileError(
            f"File not found: {path_to_open}"
        ) from error

    except PermissionError as error:
        raise JsonFileError(
            f"Permission denied: {path_to_open}"
        ) from error

    except json.JSONDecodeError as error:
        raise JsonFileError(
            f"Invalid JSON: {path_to_open}"
        ) from error


def display_validation_errors(error: ValidationError) -> None:
    """Display Pydantic validation errors in a readable format.

    Args:
        error: Pydantic validation error containing the invalid fields
            and their corresponding error messages.

    """
    for err in error.errors():
        loc = ".".join(str(value) for value in err["loc"])
        msg = err["msg"]

        print(f"Invalid field '{loc}': {msg}")


def load_function_definitions(
    path: str | None = None
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file. If not provided, the default
            function definition file is used.

    Returns:
        A list of validated `FunctionDefinition` objects.

    Raises:
        JsonFileError: If the JSON file cannot be loaded or contains
            invalid function definition data.
    """
    data = load_json_file(
        path,
        "functions_definition"
    )

    try:
        functions = [
            FunctionDefinition(**item)
            for item in data
        ]
        return functions
    except ValidationError as error:
        display_validation_errors(error)
        raise JsonFileError("Invalid function definition data.")


def load_function_test(
    path: str | None = None
) -> list[Prompt]:
    """Load and validate function-calling tests from a JSON file.

    Args:
        path: Path to the JSON file. If not provided, the default
            function-calling test file is used.

    Returns:
        A list of validated `Prompt` objects.

    Raises:
        JsonFileError: If the JSON file cannot be loaded or contains
            invalid test data.
    """
    data = load_json_file(
        path,
        "function_calling_tests"
    )

    try:
        functions = [
            Prompt(**item)
            for item in data
        ]

        return functions

    except ValidationError as error:
        display_validation_errors(error)
        raise JsonFileError("Invalid input data.")


def save_output(results: list[FunctionCallResult], path: str | None) -> None:
    """Save function-calling results to a JSON file.

    Args:
        results: List of function-calling results to save.
        path: Path to the output JSON file. If not provided, the default
            output path is used.

    Raises:
        JsonFileError: If the output file cannot be written because
            permission is denied.
    """
    if not path:
        path_to_write = DATA_DEFAULTS["output"]
    else:
        path_to_write = path

    output_path = Path(path_to_write)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
    except PermissionError as error:
        raise JsonFileError(
            f"Permission denied: {output_path}"
        ) from error
