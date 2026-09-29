*This project has been created as part of the 42 curriculum by lanasain.*

---

# Call me maybe - Function calling

---

## Description

**Call Me Maybe** is a **function calling** tool designed to transform natural language requests into structured function calls made of a function name and typed arguments.

By default, the project uses the small local language model **Qwen/Qwen3-0.6B**, but it also allows using other compatible models. The model is guided using a **constrained decoding** approach: at each step of the generation, allowed tokens are filtered to ensure the model outputs data that follows the expected format.

Instead of asking the model to generate raw JSON directly and validating it afterwards, **Call Me Maybe constrains the generation process itself**.
This guarantees that the output is **syntactically valid and strictly follows the defined schema**.

### How is works

By default, the project uses:
* **Model:** `Qwen/Qwen3-0.6B`
* **Function definitions:** `data/input/functions_definition.json`
* **Prompts:** `data/input/function_calling_tests.json`
* **Results :** `data/output/function_calling_results.json`

These parameters can be customized when running the program.

The model analyzes each natural language request, selects the corresponding function, and extracts its arguments while adhering to the project's decoding constraints.


## Instructions

The project uses **uv** to manage the Python environment and dependencies.

Installation requires approximately **10 GB of free disk space**, mainly for Python dependencies, the `uv` cache, and downloading the model from Hugging Face.

To avoid using up the limited space in your home directory, the `Makefile` automatically configures the cache paths in **`/goinfre`**:

```make
export UV_CACHE_DIR=/goinfre/$(USER)/.cache/uv
export HF_HOME=/goinfre/$(USER)/.cache/huggingface
```

No manual setup is required before running `make`.

> **NOt enough space on `/goinfre`**
>
> If `/goinfre` does not have enough free space, you can replace `/goinfre` with `/sgoinfre` in the `Makefile`:
> ```make
> export UV_CACHE_DIR=/sgoinfre/$(USER)/.cache/uv
> export HF_HOME=/sgoinfre/$(USER)/.cache/huggingface
> ```

### Installation

To install the dependencies:

```bash
make install
```

This command runs `uv sync` and prepares the environment needed for the project.


### Running the program

To run the project with default settings:

```bash
make run
```


You can pass extra arguments to the program using `ARGS`.


For example, to use a different model:
```bash
make run ARGS="--model Qwen/Qwen3-0.6B"
```

You can also customize the input and output file paths:

```bash
make run ARGS="--functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json"
```

### Debugging

To run the program with the Python debugger:

```bash
make debug
```

Arguments can also be passed here:

```bash
make debug ARGS="--model Qwen/Qwen3-0.6B"
```

### Code Quality Check

To check the code quality with **Flake8** and **Mypy**:

```bash
make lint
```

To run a strict check with `mypy --strict`:

```bash
make lint-strict
```


### Cleanup

To remove generated files and caches:

```bash
make clean
```

### Available Commands

| Command            | Description
| ------------------ | ------------------------------------------------- |
| `make install`     | Installs dependencies using `uv`                  |
| `make run`         | Runs the program                                  |
| `make debug`       | Runs the program with the Python debugger         |
| `make lint`        | Checks the code with Flake8 and mypy              |
| `make lint-strict` | Checks the code with Flake8 and `mypy --strict`   |
| `make clean`       | Removes Python caches and generated files         |
| `make`             | Installs dependencies and then runs the program   |


## Resources

### Documentation and References

* **[W3Schools - Python & JSON](https://www.w3schools.com/python/)** - Reference for Python data structures, file handling, and JSON parsing.
* **[Prompt Enginnering Guide](https://www.promptingguide.ai/fr)** - Guide on prompt design and prompting techniques.
* **[Hugging Face - Generation Strategies](https://huggingface.co/docs/transformers/en/generation_strategies)** - Reference for generation strategies and decoding mechanisms.
* **[Python Documentation - argparse](https://docs.python.org/3/library/argparse.html)** - Official documentation for handling command-line arguments.
* **[uv Documentation](https://docs.astral.sh/uv/)** - Documentation for the Python project and dependency manager used in this project.


### Use of AI

AI (Claude) was used as a **development assistant**, specifically for:

* **Planning:** Breaking down features into tasks and organizing development steps.
* **Understanding concepts:** Deepen your knowledge of LLM, finction calling, prompting.
* **Design:** Exploring different strategies for parameter extraction, constrained generation.
* **Debugging:** Analyzing errors and finding fixes.

All implementation, technical decisions, testing, and validation were carried out during development.


## Algorithm explanation

### General Principle
The JSON structure is never generated freely by the model. The JSON skeleton (`{`, `"`, `:`, `,`, `}`) is written directly in Python; the model only fills inthe blanks: the function name, followed by each parameter value. This approach avoids building a complex JSON parser or grammar while ensuring 100% valid JSON output.

### Constrained decoding modes

* **`closed`**: Used for function names and booleans (`true`/`false`). At each step, only tokens that maintain a valid prefix matching at least one candidate in the list are allowed, checked via `str.startswith`.

* **`number`**: Used for `number` or `float` values. Only characters `0-9`, `.` and `-` are allowed. A specific mechanism handles the initial space token (`Ġ`) used by BPE tokenizers, while a stopping rule detects when the number is complete.

* **`integer`**: Only characters `0-9`, and `-` characters are permitted. Similar to numbers, it handles the initial space token (`Ġ`) and includes a stop condition to detect the end of the integer value.

* **`string`**: Tokens are allowed unless they contain restricted characters, such as quotes (`"`), `<`, or newlines. The closing token (end quote) is only allowed after actual content has been generated to avoid empty strings.

### Vocabulary building steps

The tokenizer's vocabulary (`vocab.json`) is inverted (`id -> token`) and extended with special tokens from `tokenizer.json` (e.g., `<|im_start|>`). A list of forbidden tokens (`special: true`) is created and excluded from generation.

### Prompt formatting (ChatML)

Each prompt sent to the model follows the ChatML template expected by Qwen (`<|im_start|>system ... <|im_end|> <|im_start|>user ... <|im_end|> <|im_start|>assistant`), using the `/no_think` directive to disable Qwen3's internal reasoning mode, which would otherwise clutter generation with `<think>...</think>` tags.


## Design decisions

* **Independent call per field:** Rather than building a growing context window, each step (function name, individual arguments) starts with a fresh prompt. This makes debugging easier and prevents token alignment issues when manually stitching text fragments together.
* **Assistant prefix anchoring (`assistant_prefix`):** Prefixes like `"Value: "` or directly inserting an opening quote (`"`) for strings prevent the model from drifting into long explanations instead of providing direct answers.
* **Decoding generated token IDs (`model.decode`):** Used instead of raw string concatenation due to BPE byte-level encoding (e.g., `Ġ` for space, `Ã©` for `é`), which would otherwise produce malformed characters in the final JSON.
* **Regex parameter guidance via prompting:** Regex-producing parameters are not detected automatically from the parameter name; they are still generated with the generic `string` constrained-decoding mode. To improve quality, the system prompt sent to the model includes an explicit instruction ("For regex, provide a single, valid, generic, and reusable regex pattern.") whenever a parameter value is requested. This is a prompting-level hint, not a structural heuristic, so it does not guarantee correctness for complex patterns (see *Performance analysis*).
* **Per-prompt error handling:** If one prompt fails, the rest of the batch continues processing (`try/except` in `run_pipeline`), categorizing error types for clearer diagnostics.


## Performance analysis

* Each forward pass to the model (`get_logits_from_input_ids`) takes approximately 0.5 to 0.9 seconds.
* Token filtering overhead (after precomputation) is negligible (< 1ms).
* `MAX_TOKENS` was reduced to limit the worst-case generation time for individual fields.
* For the provided test suite (11 prompts), total execution time remains well under the 5-minute limit required by the assignment.
* Observed accuracy for standard types (number extraction, function names, short strings) is high and reliable. Accuracy for complex parameters like regex patterns is lower, making it the most challenging case.


## Challenges faced

### Managing prefixes in constrained decoding

One of the main difficulties occurred while testing the **constrained decoding** system.

The constraint mechanism for `closed` mode relies on a prefix (`fun_`) to identify and control allowed choices during token filtering and candidate matching.

During testing, removing the `fn_` prefix from function names in `function_definitions.json` degraded performance significantly: the model could no longer reliably select correct functions. This experiment showed that the constrained decoding logic depended heavily on prefix presence.

Two solutions were considered:

1. **Refactor constrained decoding** to work independently of any prefix.
2. **Keep the current logic** and add an internal prefix dedicated to generation, separate from actual function names.

The second solution was chosen to minimize changes to the existing, tested codebase.

An internal prefix `fun_` is added to candidate choices during constrained decoding:

```python
CLOSED_CHOICE_PREFIX = "fun_"

if mode == "closed":
    candidates = [CLOSED_CHOICE_PREFIX + c for c in candidates]
```

The model generates choice outputs containing the internal prefix, which is then stripped from the final string:

```python
generated_text = raw_text.removeprefix(CLOSED_CHOICE_PREFIX)
```

This ensures token filtering works correctly while leaving actual function names independent of internal generation prefixes.

This challenge highlighted an important takeaway: in **constrained decoding**, choice representations, tokenization patterns, and filtering rules must strictly align with the generator's expected format.


## Testing strategy

* Manual iterative testing using the provided prompt dataset (`data/input/function_calling_tests.json`), inspecting JSON output after code changes.
* Systematic validation of JSON formatting (`json.loads` succeeds by design since the structural JSON frame is managed by Python rather than generated by the model).
* Manual checks for semantic accuracy (correct function name selection and argument mapping) across test cases.
* Boundary cases identified and fixed: empty strings, reasoning token leaks, parameter confusion, and number truncation.


## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example output:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  },
]
```


## Bonus

### Support for multiple LLMs

The project is not restricted to **Qwen/Qwen3-0.6B**. The inference model can be selected dynamically using the `--model` flag.

Default setting:

```bash
Qwen/Qwen3-0.6B
```

You can test another compatible model:

```bash
make run ARGS="--model Qwen/Qwen3-1.7B"
```

This feature makes it easy to compare constrained decoding behavior across different language models.

### Visualizing the generation process

The `--trace` option prints a **step-by-step trace of token generation** directly in the terminal.

```bash
make run ARGS="--trace"
```

For each generation step, the output details:

* Step number
* Generated prefix so far
* Number of generated tokens
* Number of tokens allowed by constraints
* Top valid tokens with their scores
* Token selected by the model

Example during function selection:

```text
[TRACE] step=0 prefix='' tokens_generated=0
[TRACE] valid_ids_count=3
[TRACE] top_valid_tokens=['f(8.34)', 'fu(4.68)', 'fun(1.65)']

[TRACE] step=1 prefix='f' tokens_generated=1
[TRACE] valid_ids_count=2
[TRACE] top_valid_tokens=['un(4.67)', 'u(-3.05)']

[TRACE] step=2 prefix='fun' tokens_generated=2
[TRACE] valid_ids_count=3
[TRACE] top_valid_tokens=['_fn(13.64)', '_(13.55)', '_f(12.95)']

[TRACE] step=3 prefix='fun_fn' tokens_generated=3
[TRACE] valid_ids_count=16
[TRACE] top_valid_tokens=[
    '_add(30.10)',
    '_g(20.81)',
    '_sub(20.41)',
    '_get(18.35)',
    '_ad(16.54)'
]
```

This trace demonstrates how **constrained decoding narrows the search space**: at each step, only candidate-compatible tokens are retained.

The same logic applies to numeric arguments. For instance, generating `3.0` narrows down candidates progressively:

```text
[TRACE] step=0 prefix='' tokens_generated=0
[TRACE] valid_ids_count=90
[TRACE] top_valid_tokens=['3(26.25)', '2(20.95)', '0(18.62)', ...]

[TRACE] step=1 prefix='3' tokens_generated=1
[TRACE] valid_ids_count=25
[TRACE] top_valid_tokens=['.(24.61)', '0(13.77)', '2(9.59)', ...]

[TRACE] step=2 prefix='3.' tokens_generated=2
[TRACE] valid_ids_count=10
[TRACE] top_valid_tokens=['0(29.78)', '5(17.40)', '2(16.44)', ...]
```

The final output remains structured and valid:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

This logging feature is especially useful for **debugging**, analyzing model decisions, and understanding constrained decoding behavior.


