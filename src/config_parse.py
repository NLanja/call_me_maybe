import json
from pydantic import BaseModel, ValidationError
from pathlib import Path

DATA_DEFAULTS: dict = {
    "functions_definition": "data/input/functions_definition.json",
    "function_calling_tests": "data/input/function_calling_tests.json",
    "output": "data/output/output.json",
}


class JsonFileError(Exception):
    """Raised when a JSON file cannot be loaded."""


class DataType(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, DataType]
    returns: DataType


class Prompt(BaseModel):
    prompt: str


def load_json_file(path, default_key_name=None):
    if not path:
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


def display_validation_errors(error):
    for err in error.errors():
        loc = ".".join(str(value) for value in err["loc"])
        msg = err["msg"]

        print(f"Invalid field '{loc}': {msg}")


def load_function_definitions(path=None):
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
        raise JsonFileError("Invalid function definition data")


def load_function_test(path=None):
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
        raise JsonFileError("Invalid function definition data")
