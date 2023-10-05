from typing import List, Optional, Sequence, Tuple, Union
from rich import console


console = console.Console()


def validate_subject_lines(prompt: str, subject_lines: str, log: bool = False) -> bool:
    # Validations -
    # 1. Subject lines should contain the following things, provided they are present in the prompt:
    #   - "{{contact.first_name|default:hey}}"
    #   - "{{contact.last_name|default:there}}"
    #   - "{{shop_name}}"

    # get the parts which are in between {{ and }}
    parts = []
    for i in range(len(prompt)):
        if prompt[i] == "{" and prompt[i + 1] == "{":
            i += 2
            part = "{{"
            while prompt[i] != "}":
                part += prompt[i]
                i += 1
            part += "}}"
            parts.append(part)

    # check if all the required parts are present in the subject lines
    for part in parts:
        if part not in subject_lines:
            if log:
                console.log(f"[bold red]Error: {part} not present in subject lines")
            return False
        if log:
            console.log(f"[bold green]Success: {part} present in subject lines")
    if log:
        console.log(f"[bold green]All validations passed")
    return True


def get_validated_data(
    prompt: str, subject_lines: List[str], log: bool = False
) -> List[str]:
    valid_data = []
    for i, subject_line in enumerate(subject_lines):
        check = validate_subject_lines(prompt, subject_line)
        if not check:
            if log:
                console.log(f"[bold red]Error: Subject {i+1} is not valid")
            # continue
        else: 
            valid_data.append(subject_line)
        if log:
            console.log(f"[bold green]Success: Subject {i+1} is valid")
    return valid_data
