from openai.types.chat import ChatCompletion
from ModelTypes import ModelType
from os import getenv
import pandas as pd
from typing import override, Any
from openai import AsyncOpenAI
import asyncio


# Retrieve the API key from the environment variable
OPENROUTER_API_KEY = getenv("OPEN_ROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError(
        "OpenRouter API key not found. Please set the OPEN_ROUTER_API_KEY environment variable."
    )


class Mistral7BFree(ModelType):
    # Free model provided on OpenRouter via openrouter api and OpenAI package
    def __init__(self) -> None:
        super().__init__("mistral")

    @override
    async def generate_responses(self, raw_data: pd.DataFrame, model_prompt: str, run_amount: int) -> pd.DataFrame:

        final_prompts: list[str] = self.create_final_prompts(raw_data, model_prompt)

        responses: list[str] = await self.request_api_responses(final_prompts, run_amount)
        # print("LLM RESPONSES:\n" + str(responses))

        response_df: pd.DataFrame = self.add_stories_to_raw_data(responses, raw_data, run_amount)

        return response_df


    async def request_api_responses(self, prompts: list[str], run_amount: int) -> list[str]:

        async with AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        ) as client:
            requests: list[Any] = []
            for prompt in prompts:
                for _ in range(run_amount):
                    requests.append(self.post_request(prompt, client))
            responses: list[str] = await asyncio.gather(*requests)
        return responses


    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        completion: ChatCompletion = await client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        return str(completion.choices[0].message.content)
