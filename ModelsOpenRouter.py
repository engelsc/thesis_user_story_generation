from openai.types.chat import ChatCompletion
from ModelTypes import ModelType
from os import getenv
import pandas as pd
from typing import override, cast, Any
from openai import AsyncOpenAI
import asyncio


# Retrieve the API key from the environment variable
OPENROUTER_API_KEY = getenv("OPEN_ROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError(
        "OpenRouter API key not found. Please set the OPEN_ROUTER_API_KEY environment variable."
    )


class Llama31Free(ModelType):
    # Free model provided on OpenRouter with openrouter api
    def __init__(self) -> None:
        super().__init__("llama")

    @override
    async def generate_responses(
        self, raw_data: pd.DataFrame, model_prompt: str, run_amount: int
    ) -> pd.DataFrame:
        async with AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        ) as client:
            final_prompts: list[str] = self.create_final_prompts(raw_data, model_prompt)

            operations: list[Any] = []
            for prompt in final_prompts:
                for _ in range(run_amount):
                    operations.append(self.post_request(prompt, client))
            # operations = [self.post_request(prompt, client) for prompt in final_prompts]
            responses: list[str] = await asyncio.gather(*operations)

        response_data: pd.DataFrame = raw_data.copy()

        story_index: int = 0
        for id, _ in response_data.iterrows():
            id = cast(int, id)
            for i in range(1, run_amount + 1):
                response_data.at[id, f"story_{i}"] = responses[story_index]
                story_index += 1

        print("LLM RESPONSES:\n" + str(responses))
        return response_data

    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        completion: ChatCompletion = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return str(completion.choices[0].message.content)
