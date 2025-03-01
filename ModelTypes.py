from abc import abstractmethod, ABC
from enum import Enum
from typing import Final


# Enum for all models available
class MODEL_TYPES(Enum):
    TEST_IMPLEMENTATION = "test_implementation"
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"

    @staticmethod
    def is_member(member_string: str) -> bool:
        return member_string in MODEL_TYPES._value2member_map_


class ModelType(ABC):
    def __init__(self, model_id: str) -> None:
        if MODEL_TYPES.is_member(model_id):
            self.MODEL_ID: Final[MODEL_TYPES] = MODEL_TYPES(model_id)
        else:
            raise ValueError(f"'{model_id}' is not a valid MODEL_TYPE")

    def get_id(self) -> str:
    	# no longer necessary
        return self.MODEL_ID.value

    async def generate_responses(
        self,
        formatted_prompts: list[str],
        run_amount: int = 1
    ) -> list[str]:

        responses: list[str] = await self.request_api_responses(formatted_prompts, run_amount)
        return responses


    @abstractmethod
    async def request_api_responses(self, prompts: list[str], run_amount: int) -> list[str]:
        pass
