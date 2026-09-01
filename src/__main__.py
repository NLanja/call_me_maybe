import sys
import os
import getpass
import argparse

# os.environ['HF_HOME'] = '/goinfre/lanasain/.cache/huggingface'

# os.environ.setdefault(
#     "HF_HOME",
#     f"/goinfre/{getpass.getuser()}/.cache/huggingface",
# )

from llm_sdk import Small_LLM_Model
from .vocabulary import load_vocabulary
from .generator import run_pipeline
from .config_parse import (
    load_function_definitions,
    load_function_test,
    JsonFileError,
    save_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Function calling tool using constrained decodimg."
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
        type=str,
        help="Path to the output JSON File to generate"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    check = Small_LLM_Model()
    list_fun = load_function_definitions(args.functions_definition)
    list_prompt = load_function_test(args.input)

    id_to_token, special_ids = load_vocabulary(check)
    
    # quote_ids = check.encode('"').tolist()[0]
    quote_ids = [
        token_id for token_id, token in id_to_token.items()
        if token == '"'
    ]

    if len(quote_ids) != 1:
        raise RuntimeError(
            f"Expected the double-quote character to map to a single "
            f"token, got {len(quote_ids)} tokens: {quote_ids}. "
            "The 'string' generation mode relies on this assumption."
        )
    
    quote_token_id = quote_ids[0]

    results = run_pipeline(
        model=check,
        functions=list_fun,
        prompts=list_prompt,
        id_to_token=id_to_token,
        special_ids=special_ids,
        quote_token_id=quote_token_id,
    )

    save_output(results, args.output)

    print(f"Done: {len(results)}/{len(list_prompt)} prompts processed successfully.")
#     lists = []
#     for a in range(len(list_fun)):
#         lists.append(list_fun[a].name)
#     # print(lists)

#     for i in range(len(list_prompt)):
#         a = check.encode(f"Here is my prompt '{list_prompt[i].prompt}' and the list of functions '{lists}'; just give me the name of the function that corresponds to my prompt in the list of functions.")
#         encode = a[0].tolist()
#         lenght = len(encode)

#         num_tokens_to_generate = 10

#         for _ in range(num_tokens_to_generate):
#             logit = check.get_logits_from_input_ids(encode)
#             response = max(range(len(logit)), key=lambda i: logit[i])
#             encode.append(response)

#         deco = check.decode(encode[lenght:])
#         print(deco)
#         print(list_prompt[i].prompt)

if __name__ == "__main__":
    main()
    # print(getpass.getuser())
