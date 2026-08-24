from typing import Any
from .config_parse import FunctionDefinition, Prompt, FunctionCallResult
from .decoding import generate_constrained, is_valid_number


def build_functions_index(functions: list[FunctionDefinition]) -> dict[str, FunctionCallResult]:
    return {function.name: function for function in functions}


def choose_function_name(model: Any, prompt_text, function_names, id_to_token, special_ids) -> str:
    instruction = (
        "You are a function-calling assistant. Given the user request below, respond with only the name of the matching function.\n"
        f"Available functions: {function_names}\n"
        f"User request: \"{prompt_text}\"\n"
        f"Function name: "
    )

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
    id_to_token: dict[int, str],
    special_ids: set[int],
    quote_token_id: int,
) -> Any:
    instruction = (
        f"User request: \"{prompt_text}\"\n"
        f"Give the value of the parameter \"{param_name}\""
        f"(type: {param_type}) needed to answer this request. \n"
        "Value: "
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
                f"Model produced an invalid number for '{param_name}"
                f"{text!r}"
            )
        return float(text)

    if param_type == "string":
        text, _ = generate_constrained(
            model=model,
            input_ids=input_ids,
            id_to_token=id_to_token, 
            special_ids=special_ids,
            mode="string",
            end_token_id=quote_token_id,
        )
        return text
    
    raise ValueError(f"Unsupported parameter type: {param_type!r}")


def process_prompt(
    model: Any,
    prompt: Prompt,
    functions_by_name: dict[str, FunctionDefinition],
    id_to_token: dict[int, str],
    special_ids: set[int, str],
    quote_token_id: int,
) -> FunctionCallResult:
    function_names = list(functions_by_name.keys())

    chosen_name = choose_function_name(
        model, prompt.prompt, function_names, id_to_token, special_ids
    )

    function_def = functions_by_name[chosen_name]

    parameters: dict[str, Any] = {}
    for param_name, param_schema in function_def.parameters.items():
        parameters[param_name] = generate_parameter_value(
            model=model,
            prompt_text=prompt.prompt,
            param_name=param_name,
            param_type=param_schema.type,
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
