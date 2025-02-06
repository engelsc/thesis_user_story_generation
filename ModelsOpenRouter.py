from openai.types.chat import ChatCompletion
from ModelTypes import ModelType
from abc import abstractmethod
from os import getenv
import pandas as pd
from typing import override, Any
from openai import AsyncOpenAI
import asyncio


# Maximum amount of simultaneous requests to the API
MAX_SEMAPHORES: int = 10

# Retrieve the API key from the environment variable
OPENROUTER_API_KEY = getenv("OPEN_ROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError(
        "OpenRouter API key not found. Please set the OPEN_ROUTER_API_KEY environment variable."
    )


class OpenRouterModelType(ModelType):
# Base class for all async handling of multiple prompts via the OpenRouterAPI
# Builds prompts, instantiates OpenAI client

    # Add semaphore to limit concurrent API requests
    # Rate limits on OpenRouter are dependant on model type and credits remaining.
    # Source: https://openrouter.ai/docs/api-reference/limits
    _semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_SEMAPHORES)

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
                    requests.append(asyncio.create_task(self._bounded_post_request(prompt, client)))

            responses: list[str] = await asyncio.gather(*requests)
        return responses


    async def _bounded_post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        async with self._semaphore:
            try:
                return await self.post_request(prompt, client)
            except Exception as e:
                print(f"Error sending request: {e}")
                return "ERROR: No response"

    @abstractmethod
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        pass


class Mistral7BFree(OpenRouterModelType):
    # Free model provided on OpenRouter via openrouter api and OpenAI package
    def __init__(self) -> None:
        super().__init__("mistral")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="mistralai/mistral-7b-instruct:free",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Llama318BFree(OpenRouterModelType):
    # Free model provided on OpenRouter via openrouter api and OpenAI package
    def __init__(self) -> None:
        super().__init__("llama")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Llama318B(OpenRouterModelType):
    # Free model provided on OpenRouter via openrouter api and OpenAI package
    def __init__(self) -> None:
        super().__init__("llama")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Gemini15Pro(OpenRouterModelType):
    # Premium model provided on OpenRouter via openrouter api and OpenAI package
    # API reference: https://openrouter.ai/google/gemini-pro-1.5-exp/api
    def __init__(self) -> None:
        super().__init__("gemini")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="google/gemini-pro-1.5",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }],
            )

            # DEBUGGING: Print API response to check its structure
            #print("API Response:", completion)

            if not completion.choices:
                raise ValueError("No choices returned from API.")

            if not completion.choices[0].message:
                raise ValueError("No message returned in first choice.")

            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class GPT4oMini(OpenRouterModelType):
    # Premium model provided on OpenRouter via openrouter api and OpenAI package
    # API reference: https://openrouter.ai/openai/gpt-4o-mini/api
    def __init__(self) -> None:
        super().__init__("gpt")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }],
            )

            # DEBUGGING: Print API response to check its structure
            #print("API Response:", completion)

            if not completion.choices:
                raise ValueError("No choices returned from API.")

            if not completion.choices[0].message:
                raise ValueError("No message returned in first choice.")

            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"
