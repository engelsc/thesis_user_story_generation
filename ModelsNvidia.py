from os import getenv
from typing import Any, override
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from ModelTypes import ModelType
import pandas as pd
import asyncio


# Retrieve the API key from the environment variable
NVIDIA_API_KEY = getenv("NVIDIA_API_KEY_1")
if NVIDIA_API_KEY is None:
    raise ValueError(
        "NVidia NIM API key not found. Please set the OPEN_ROUTER_API_KEY environment variable."
    )

class Llama31Nvidia(ModelType):
	# Model on Nvidia NIM test using OpenAI library

	def __init__(self) -> None:
		super().__init__("llama")

	@override
	async def generate_responses(self, raw_data: pd.DataFrame, model_prompt: str, run_amount: int) -> pd.DataFrame:

		final_prompts: list[str] = self.create_final_prompts(raw_data, model_prompt)

		responses: list[str] = await self.request_api_responses(final_prompts, run_amount)

		response_df: pd.DataFrame = self.add_stories_to_raw_data(responses, raw_data, run_amount)

		return response_df


	async def request_api_responses(self, prompts: list[str], run_amount: int) -> list[str]:

		async with AsyncOpenAI(
			base_url = "https://integrate.api.nvidia.com/v1",
			api_key = NVIDIA_API_KEY
		) as client:
			requests: list[Any] = []
			for prompt in prompts:
				for _ in range(run_amount):
					requests.append(self.post_request(prompt, client))
			responses: list[str] = await asyncio.gather(*requests)
		return responses

	async def post_request(self, prompt: str, client: AsyncOpenAI)-> str:
		completion: ChatCompletion = await client.chat.completions.create(
			model="meta/llama-3.1-8b-instruct",
			messages=[{"role":"user","content":prompt}],
			temperature=0.2,
			top_p=0.7,
			max_tokens=1024,
			stream=False
		)

		return str(completion.choices[0].message.content)
