from abc import abstractmethod, ABC
from enum import Enum
from typing import Final, override, cast
import pandas as pd
import random


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


# TEST IMPLEMENTATION
class TestImplementation(ModelType):
    user_stories: list[str] = [
        "As a customer, I want to filter products by price range, brand, and customer ratings so that I can find the products that best match my preferences and budget.",
        "As a social media user, I want to update my profile picture and bio so that my profile accurately reflects my current appearance and interests.",
        "As a patient, I want to book, reschedule, or cancel my medical appointments online so that I can manage my healthcare schedule conveniently.",
        "As a fitness enthusiast, I want my fitness tracker data to sync automatically with my health app so that I can track my workouts and progress in one place.",
        "As a home cook, I want to share my recipes with a community of food enthusiasts so that others can try my dishes and provide feedback.",
        "As an event organizer, I want to create and manage events, send invitations, and track RSVPs so that I can ensure the event runs smoothly and attendees are informed.",
        "As a student, I want to track my progress and receive a certificate upon course completion so that I can showcase my achievements and knowledge gained.",
        "As a bank customer, I want to receive notifications for transactions and low balance alerts so that I can monitor my account activity and avoid overdrafts.",
        "As a team member, I want to assign tasks, set deadlines, and track progress in a shared workspace so that our team can collaborate effectively and meet project goals.",
        "As a movie lover, I want to create and manage a watchlist and receive recommendations so that I can easily find and enjoy new content that matches my interests.",
    ]

    def __init__(self) -> None:
        super().__init__("test_implementation")

    @override
    async def generate_responses(
        self, raw_data: pd.DataFrame, model_prompt: str, run_amount: int
    ) -> pd.DataFrame:
        if run_amount < 1 or run_amount > 3:
            raise ValueError("The value for run_amount must be between 1 and 3!")

        response_data: pd.DataFrame = raw_data.copy()

        for id, _ in response_data.iterrows():
            if run_amount >= 1:
                response_data.at[id, "story_1"] = random.choice(self.user_stories)
            if run_amount >= 2:
                response_data.at[id, "story_2"] = random.choice(self.user_stories)
            if run_amount == 3:
                response_data.at[id, "story_3"] = random.choice(self.user_stories)

        return response_data
