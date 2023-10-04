from dataclasses import dataclass
import pydantic
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union, TypeVar, Generic


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    Value: Optional[T]
    Error: Optional[Exception]

    def __bool__(self):
        return self.Value is not None


class OpenAiChatModels(Enum):
    GPT_4 = "gpt-4"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_32K_0613 = "gpt-4-32k-0613"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"

    def __str__(self):
        return f"OpenAiChatModels(Model={self.value}, Token Limit={self.token_limit})"

    @property
    def token_limit(self) -> int:
        if "gpt-4" in self.value:
            # Model belongs to GPT-4 family
            if "32k" in self.value:
                return 32_768
            return 8_192
        else:
            # Model belongs to GPT-3.5 family
            if "16k" in self.value:
                return 16_384
            return 4_096

    @property
    def cost_per_1000(self) -> Tuple[float, float]:
        if "gpt-4" in self.value:
            # Model belongs to GPT-4 family
            if "32k" in self.value:
                return (0.06, 0.12)
            return (0.03, 0.06)
        else:
            # Model belongs to GPT-3.5 family
            if "16k" in self.value:
                return (0.003, 0.004)
            return (0.0015, 0.002)


@dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 4000
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class OpenAiChatCompletionMessage(pydantic.BaseModel):
    role: str
    content: str

    class Config:
        orm_mode = True


class OpenAiChatCompletionChoice(pydantic.BaseModel):
    index: int
    message: OpenAiChatCompletionMessage
    finish_reason: str

    class Config:
        orm_mode = True


class OpenAiChatCompletionUsage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    class Config:
        orm_mode = True


class OpenAiChatCompletionResponse(pydantic.BaseModel):
    """
    OpenAI chat completion response. (https://platform.openai.com/docs/api-reference/chat/object)
    """

    id: str
    object: str
    created: int
    choices: List[OpenAiChatCompletionChoice]
    usage: OpenAiChatCompletionUsage

    class Config:
        orm_mode = True
