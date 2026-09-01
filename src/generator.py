from typing import Any

from .config_parse import FunctionDefinition, Prompt, FunctionCallResult
from .decoding import generate_constrained, is_valid_number


def build_functions_index(
    functions: list[FunctionDefinition],
) -> dict[str, FunctionDefinition]:
    """Index function definitions by name for fast lookup."""
    return {function.name: function for function in functions}


def build_chat_prompt(
    system_message: str,
    user_message: str,
    assistant_prefix: str = "",
) -> str:
    return (
        f"<|im_start|>system\n"
        f"{system_message} /no_think<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{assistant_prefix}"
    )


def choose_function_name(
    model: Any,
    prompt_text: str,
    functions_by_name: dict[str, FunctionDefinition],
    id_to_token: dict[int, str],
    special_ids: set[int],
) -> str:
    function_names = list(functions_by_name.keys())

    function_listing = "\n".join(
        f"- {name}: {function_def.description}"
        for name, function_def in functions_by_name.items()
    )

    system_message = (
        "You are a function-calling assistant. Respond with only the "
        "name of the function that matches the user's request, nothing "
        "else. Base your choice on each function's description, not "
        "just its name - the name alone can be misleading."
    )
    user_message = (
        f"Available functions:\n{function_listing}\n\n"
        f"User request: \"{prompt_text}\""
    )

    instruction = build_chat_prompt(system_message, user_message)
    input_ids = model.encode(instruction).tolist()[0]

    name, _ = generate_constrained(
        model=model,
        input_ids=input_ids,
        id_to_token=id_to_token,
        special_ids=special_ids,
        mode="closed",
        candidates=function_names,
    )

    return name


def generate_parameter_value(
    model: Any,
    prompt_text: str,
    param_name: str,
    param_type: str,
    function_name: str,
    function_description: str,
    already_extracted: dict[str, Any],
    all_param_names: list[str],
    id_to_token: dict[int, str],
    special_ids: set[int],
    quote_token_id: int,
) -> Any:
    system_message = (
        "You are a function-calling assistant. The function to call has "
        "already been chosen; your only job now is to extract its raw "
        "input parameters from the user's request. "
        f"Chosen function: \"{function_name}\" - {function_description}. "
        "This function will perform its own computation once called - "
        "you must NOT pre-compute or answer the user's question "
        "yourself. Just copy the relevant raw value(s) as they appear "
        "in the request. Respond with only the value, nothing else - "
        "no explanation."
    ) 

    context_lines = [f"All parameters needed: {all_param_names}"]
    if already_extracted:
        context_lines.append(f"Already extracted so far: {already_extracted}")
    context_lines.append(
        f"Now extract the value for parameter \"{param_name}\" "
        f"(type: {param_type})"
    )

    user_message = (
        f"User request: \"{prompt_text}\"\n" + "\n".join(context_lines)
    )

    if param_type == "string":
        instruction = build_chat_prompt(system_message, user_message)
        input_ids = model.encode(instruction).tolist()[0]
        input_ids = input_ids + [quote_token_id]

        text, _ = generate_constrained(
            model=model,
            input_ids=input_ids,
            id_to_token=id_to_token,
            special_ids=special_ids,
            mode="string",
            end_token_id=quote_token_id,
        )
        return text

    instruction = build_chat_prompt(
        system_message, user_message, assistant_prefix="Value: "
    )
    input_ids = model.encode(instruction).tolist()[0]

    if param_type == "boolean":
        text, _ = generate_constrained(
            model=model,
            input_ids=input_ids,
            id_to_token=id_to_token,
            special_ids=special_ids,
            mode="closed",
            candidates=["true", "false"],
        )
        return text == "true"
    
    if param_type == "number":
        text, _ = generate_constrained(
            model=model,
            input_ids=input_ids,
            id_to_token=id_to_token,
            special_ids=special_ids,
            mode="number",
        )
        if not is_valid_number(text):
            raise ValueError(
                f"Model produced an invalid number for '{param_name}': "
                f"{text!r}"
            )
        return float(text)

    raise ValueError(f"Unsupported parameter type: {param_type!r}")


def process_prompt(
    model: Any,
    prompt: Prompt,
    functions_by_name: dict[str, FunctionDefinition],
    id_to_token: dict[int, str],
    special_ids: set[int],
    quote_token_id: int,
) -> FunctionCallResult:
    chosen_name = choose_function_name(
        model, prompt.prompt, functions_by_name, id_to_token, special_ids
    )

    function_def = functions_by_name[chosen_name]

    parameters: dict[str, Any] = {}
    for param_name, param_schema in function_def.parameters.items():
        parameters[param_name] = generate_parameter_value(
            model=model,
            prompt_text=prompt.prompt,
            param_name=param_name,
            param_type=param_schema.type,
            function_name=chosen_name,
            function_description=function_def.description,
            already_extracted=parameters,
            all_param_names=list(function_def.parameters.keys()),
            id_to_token=id_to_token,
            special_ids=special_ids,
            quote_token_id=quote_token_id,
        )
    
    return FunctionCallResult(
        prompt=prompt.prompt,
        name=chosen_name,
        parameters=parameters,
    )


def run_pipeline(
    model: Any,
    functions: list[FunctionDefinition],
    prompts: list[Prompt],
    id_to_token: dict[int, str],
    special_ids: set[int],
    quote_token_id: int,
) -> list[FunctionCallResult]:
    functions_by_name = build_functions_index(functions)
    results: list[FunctionCallResult] = []

    for prompt in prompts:
        try:
            result = process_prompt(
                model=model,
                prompt=prompt,
                functions_by_name=functions_by_name,
                id_to_token=id_to_token,
                special_ids=special_ids,
                quote_token_id=quote_token_id,
            )
            results.append(result)
        except Exception as error:
            print(
                f"Warning: skipping prompt {prompt.prompt!r} "
                f"due to an error: {error}"
            )
    
    return results
