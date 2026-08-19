import sys
import os
import argparse

os.environ['HF_HOME'] = '/goinfre/lanasain/.cache/huggingface'

from llm_sdk import Small_LLM_Model
from .config_parse import (
    load_function_definitions,
    load_function_test,
    JsonFileError)


def main():
    parser = argparse.ArgumentParser(description = "data")
    parser.add_argument("--functions_definition", type=str, required=False, help="Function definition")
    parser.add_argument("--prompt", type=str, required=False, help="Prompt")
    args = parser.parse_args()
    print(args.functions_definition)

#     check = Small_LLM_Model()
#     list_fun = load_function_definitions("")
#     list_prompt = load_function_test("")
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
