from concurrent.futures import ThreadPoolExecutor
import json
import os
from typing import Generator, Iterator, List
import time
from rich import console

from custom_types import *
from utils import open_ai_chat_completion, num_tokens_from_string
from validations import get_validated_data


console = console.Console()


def generate_completions(
    prompts: List[str],
    model: OpenAiChatModels,
    decoding_args: OpenAIDecodingArguments,
    use_max_tokens: bool = False,
    num_workers: int = 1
) -> Generator[Iterator[Optional[OpenAiChatCompletionResponse]], None, None]:
    # Make sure the number of prompts is greater than the number of workers
    assert len(prompts) >= num_workers, f"Number of workers ({num_workers}) must be less than or equal to the number of prompts ({len(prompts)})."

    # Create a thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Generate the completions
        completion = executor.map(
            open_ai_chat_completion,
            prompts,
            [model] * len(prompts),
            [decoding_args] * len(prompts),
            [use_max_tokens] * len(prompts),
        )
        # Return the completions
        yield completion

def write_subject_lines_to_file(file_name: str, prompt: str, subject_lines: List[str]) -> bool:
    try:
        written_completions = open(file_name, "r").read()
        written_completions = written_completions.strip() or "[]"
        written_completions = eval(written_completions)
        written_completions.append(
            {
                "prompt": prompt,
                "subject_lines": subject_lines
            }
        )
        with open(file_name, "w") as f:
            if file_name.endswith("json"):
                f.write(json.dumps(written_completions, indent=4, default=vars))
            else:
                f.write(str(written_completions))
    except Exception as e:
        console.log(f"[bold red]Error writing to file: {e}")
        return False

    return True


def generate_prompts(prompt_template: str, num_examples: int = 5) -> str:
    prompt = prompt_template.replace("{{examples}}", "")
    return prompt


if __name__ == "__main__":
    print()

    current_path = os.getcwd()
    prompt_file_path = os.path.join(current_path, "prompt.txt")
    with open(prompt_file_path) as f:
        prompt = f.read()

    prompts = [prompt] * 25

    completions = generate_completions(
        prompts,
        OpenAiChatModels.GPT_4,
        OpenAIDecodingArguments(temperature=1.0, max_tokens=4000),
        use_max_tokens=False,
        num_workers=5,
    )

    total_subject_lines = 0
    completions_file_path = os.path.join(current_path, "completions.json")
    failed_completions_file_path = os.path.join(current_path, "failed_completions.txt")
    with console.status("[bold green]Generating completions...") as status:
        for iterator in completions:
            for completion in iterator:
                if completion is None:
                    continue
                output = completion.choices[0].message.content
                try:
                    output = eval(output)
                except Exception as e:
                    console.log(f"[bold red]Error evaluating output: {e}")
                    with open(failed_completions_file_path, "a") as f:
                        f.write(output)
                        f.write("\n")
                    continue

                # check if output is a list
                if not isinstance(output, list):
                    console.log("[bold red]Completion is not a list")
                    continue

                # check if the elements are good
                for elem in output:
                    try:
                        # elem = eval(elem)
                        if not isinstance(elem, dict):
                            console.log("[bold red]Completion element is not a dict")
                            continue
                        prompt = elem.get("prompt")
                        if not prompt:
                            console.log("[bold red]Completion element does not have a prompt")
                            continue
                        subject_lines = elem.get("subject_lines")
                        if not isinstance(subject_lines, list):
                            console.log("[bold red]Completion element subject lines is not a list")
                            continue
                        valid_data = get_validated_data(prompt, subject_lines)
                        if len(valid_data) == 0:
                            console.log("[bold red]Completion element has no valid data")
                            continue

                        # Write the completion to a file
                        if write_subject_lines_to_file(completions_file_path, prompt, valid_data):
                            console.log(f"[bold green]Completions written: {len(valid_data)}")
                            total_subject_lines += len(valid_data)
                    except TypeError as type_error:
                        console.log(f"[bold red]Type Error: {type_error}, element: {elem}")
                    except Exception as e:
                        console.log(f"[bold red]Error: {e}")

    console.log(f"Total successful completions: {total_subject_lines}")
