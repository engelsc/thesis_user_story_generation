from openai.types.chat import ChatCompletion
from ModelTypes import ModelType
from abc import abstractmethod
from os import getenv
from typing import override, Any
from openai import AsyncOpenAI
import asyncio


MAX_SEMAPHORES: int = 10
'''Maximum amount of simultaneous requests to the API.'''

# Retrieve the API key from the environment variable
OPENROUTER_API_KEY = getenv("OPEN_ROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError(
        "OpenRouter API key not found. Please set the OPEN_ROUTER_API_KEY environment variable."
    )


class OpenRouterModelType(ModelType):
    '''
    Base class for all OpenRouter model implementations:\n
    async handling of multiple prompts via the OpenRouterAPI,
    instantiates OpenAI client, handles concurrent post requests
    '''

    _semaphore: asyncio.Semaphore = asyncio.Semaphore(MAX_SEMAPHORES)
    '''
    Add semaphore to limit concurrent API requests.
    Rate limits on OpenRouter are dependant on model type and credits remaining.
    Source: https://openrouter.ai/docs/api-reference/limits
    '''

    @override
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
    '''
    Free model provided on OpenRouter via openrouter api and OpenAI package\n
    Created May 27, 2024 | 8,192 context |  $0/M input tokens | $0/M output tokens
    API reference: https://openrouter.ai/mistralai/mistral-7b-instruct:free/api
    '''

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
                max_tokens=1000,
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Mistral7B(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created May 27, 2024 | 32,768 context | $0.03/M input tokens | $0.055/M output tokens
    API reference: https://openrouter.ai/mistralai/mistral-7b-instruct/api
    '''
    def __init__(self) -> None:
        super().__init__("mistral")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="mistralai/mistral-7b-instruct",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                max_tokens=1000,
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Ministral8B(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created Oct 17, 2024 | 128,000 context | $0.1/M input tokens | $0.1/M output tokens
    API reference: https://openrouter.ai/mistralai/ministral-8b/api
    '''
    def __init__(self) -> None:
        super().__init__("mistral")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="mistralai/ministral-8b",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                max_tokens=1000,
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Llama318BFree(OpenRouterModelType):
    '''
    Free model provided via OpenRouter API and OpenAI package\n
    defunct - do not use!
    API reference: https://openrouter.ai/meta-llama/llama-3.1-8b-instruct:free/api
    '''
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
                max_tokens=1000,
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Llama318B(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created Jul 23, 2024 | 131,072 context | $0.02/M input tokens | $0.05/M output tokens
    API reference: https://openrouter.ai/meta-llama/llama-3.1-8b-instruct/api
    '''
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
                max_tokens=1000,
                temperature=0.9, # trying to combat runaway generation by reducing creativity
                frequency_penalty=0.1, # trying to combat runaway generation by punishing more repititions
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Llama323B(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Largest model in the Llama 3.2 family\n
    Created Sep 25, 2024 | 131,000 context | $0.015/M input tokens | $0.025/M output tokens
    API reference: https://openrouter.ai/meta-llama/llama-3.2-3b-instruct/api
    '''
    def __init__(self) -> None:
        super().__init__("llama")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                model="meta-llama/llama-3.2-3b-instruct",
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                max_tokens=1000,
            )
            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Gemini15Pro(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created Apr 9, 2024 | 2,000,000 context | $1.25/M input tokens | $5/M output tokens
    API reference: https://openrouter.ai/google/gemini-pro-1.5-exp/api
    '''
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
                max_tokens=1000,
            )

            # DEBUGGING: Print API response to check its structure
            print("API Response:", completion)

            if not completion.choices:
                raise ValueError("No choices returned from API.")

            if not completion.choices[0].message:
                raise ValueError("No message returned in first choice.")

            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class Gemini15Flash8B(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created Oct 3, 2024 | 1,000,000 context | $0.0375/M input tokens | $0.15/M output tokens
    API reference: https://openrouter.ai/google/gemini-flash-1.5-8b/api
    '''
    def __init__(self) -> None:
        super().__init__("gemini")

    @override
    async def post_request(self, prompt: str, client: AsyncOpenAI) -> str:
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                extra_headers={},
                extra_body={},
                model="google/gemini-flash-1.5-8b",
                messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
                ],
                max_tokens=1000,
            )

            # DEBUGGING: Print API response to check its structure
            print("API Response:", completion)

            if not completion.choices:
                raise ValueError("No choices returned from API.")

            if not completion.choices[0].message:
                raise ValueError("No message returned in first choice.")

            return str(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error in API call: {e}")
            return "ERROR: No response"


class GPT4oMini(OpenRouterModelType):
    '''
    Premium model provided via OpenRouter API and OpenAI package\n
    Created Jul 18, 2024 | 128,000 context | $0.15/M input tokens | $0.6/M output tokens
    API reference: https://openrouter.ai/openai/gpt-4o-mini/api
    '''
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
                max_tokens=1000,
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
