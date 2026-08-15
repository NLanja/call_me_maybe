import sys
import os
os.environ['HF_HOME'] = '/goinfre/lanasain/.cache/huggingface'

from llm_sdk import Small_LLM_Model
from .config_parse import (
    load_function_definitions,
    load_function_test,
    JsonFileError)


def main():
    check = Small_LLM_Model()
    try:
        functions = load_function_definitions(
            "data/input/functions_definition.json"
            )
        print(functions)
        print()
        test = load_function_test("")
        print(test)
        for i in test:
            print(i)

    except JsonFileError as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
