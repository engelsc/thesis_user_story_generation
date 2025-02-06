from abc import abstractmethod, ABC
from enum import Enum
from typing import Final, cast
import pandas as pd


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
        return self.MODEL_ID.value

    def create_final_prompts(
        self, data_set: pd.DataFrame, model_prompt: str
    ) -> list[str]:
        final_prompts: list[str] = []
        for idx, _ in data_set.iterrows():
            idx = cast(int, idx)
            final_prompts.append(
                model_prompt
                + "\n\nREQUIREMENT:\n"
                + str(data_set.iloc[idx]["text_description"])
            )
        # print("Final prompts: " + str(final_prompts))
        return final_prompts

    def add_stories_to_raw_data(
        self, responses: list[str], raw_data: pd.DataFrame, run_amount: int
    ) -> pd.DataFrame:
        response_data: pd.DataFrame = raw_data.copy()

        story_index: int = 0
        for id, _ in response_data.iterrows():
            id = cast(int, id)
            for i in range(1, run_amount + 1):
                response_data.at[id, f"story_{i}"] = responses[story_index]
                story_index += 1
        return response_data

    @abstractmethod
    async def generate_responses(
        self, raw_data: pd.DataFrame, model_prompt: str, run_amount: int
    ) -> pd.DataFrame:
        pass
