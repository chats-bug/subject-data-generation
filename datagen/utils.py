from constants import *
from datetime import datetime
import dotenv
import os
import openai
from rich import console
import tiktoken
import time
from typing import Optional, Union

from custom_types import (
    OpenAiChatModels,
    OpenAIDecodingArguments,
    OpenAiChatCompletionResponse,
)


console = console.Console()

# ---------------------------------------------------------------------------
# Loading the environment variables
# ---------------------------------------------------------------------------
dotenv_path = os.path.join(".env")
dotenv.load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def open_ai_chat_completion(
    prompt: str,
    model: OpenAiChatModels = OpenAiChatModels.GPT_3_5_TURBO_16K,
    decoding_args: OpenAIDecodingArguments = OpenAIDecodingArguments(),
    use_max_tokens_override: bool = False,
    num_tries: int = 0,
) -> Optional[OpenAiChatCompletionResponse]:
    """
    Generate a completion using the OpenAI chat API.

    Params
    ------
    prompt: str
            The prompt to generate a completion for.
    model: OpenAiChatModels
            The OpenAI chat model to use. Defaults to OpenAiChatModels.GPT_3_5_TURBO.
    decoding_args: OpenAIDecodingArguments
            The decoding arguments to use. Defaults to OpenAIDecodingArguments().

    Returns
    -------
    OpenAiChatCompletionResponse or None: The completion response from the OpenAI chat API. Returns None if the completion fails.
    """
    # Calculate the number of tokens in the prompt
    num_tokens_prompt = num_tokens_from_string(prompt, model)
    token_limit = model.token_limit
    assert (
        num_tokens_prompt + decoding_args.max_tokens <= token_limit
    ), f"The prompt has {num_tokens_prompt} tokens and requested completion tokens are {decoding_args.max_tokens}. \
    The total is {num_tokens_prompt+decoding_args.max_tokens} but the model only supports {token_limit} tokens."

    # Override the max tokens if necessary
    if use_max_tokens_override:
        decoding_args.max_tokens = token_limit - num_tokens_prompt

    # Generate the completion
    try:
        completion = openai.ChatCompletion.create(
            model=model.value,
            messages=[
                {"role": "system", "content": "You are a super helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            **decoding_args.__dict__,
        )
        completion = OpenAiChatCompletionResponse(**completion)
    except openai.error.RateLimitError as rle:
        console.log(f"[bold red]Rate limit error: {rle}")
        console.log(f"[bold red]Waiting 60 seconds...")
        time.sleep(60)
        completion = None
    except Exception as e:
        console.log(f"[bold red]Error while generation: {e}")
        completion = None
    return completion


def num_tokens_from_string(
    string: str, encoding_name: Union[str, OpenAiChatModels]
) -> int:
    """
    Returns the number of tokens in a text string.

    Params
    ------
    string: str
            The string to count the number of tokens in.
    encoding_name: Union[str, OpenAiChatModels]
            The name of the encoding to use. Can be a string or an OpenAiChatModels enum.

    Returns
    -------
    int: The number of tokens in the string.
    """
    if isinstance(encoding_name, OpenAiChatModels):
        encoding = tiktoken.encoding_for_model(model_name=encoding_name.value)
    else:
        encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == "__main__":
    prompt = """Create a subject line for a promotional email campaign announcing a promotional offer on clothes during this Diwali. Include {{contact.first_name|default:there}}, {{contact.last_name|default:there}} and {{shop_name}}."""
    completion = open_ai_chat_completion(
        prompt=prompt,
        model=OpenAiChatModels.GPT_4,
        decoding_args=OpenAIDecodingArguments(max_tokens=300),
    )
    if completion is None:
        print("Completion failed")
        exit(1)

    print(completion.choices[0].message.content)
