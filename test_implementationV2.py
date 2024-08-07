# %% imports
from ModelTypes import TestImplementation
import pandas as pd
import aqusahandler
import data_helper as dh
import asyncio

# %% Initializing model and setting variables to be used for generation
model = TestImplementation()
generated_stories_count: int = 2
generate_debug_output: bool = False


# %% Loading natural language requirement descriptions and user prompt

raw_requirements = dh.load_requirements("dummy_requirements.csv")

print(raw_requirements)
user_prompt = dh.load_prompt(model.get_id(), "prompts.csv")
print(user_prompt)


# %% Hand over raw_data and user_prompt to model to handle async requests
###
### TODO: Test how to make this async for dealing with API delay
###
async def get_response(raw_data: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
    return await model.generate_responses(
        raw_data, user_prompt, generated_stories_count
    )


response_data = await model.generate_responses(
    raw_requirements, user_prompt, generated_stories_count
)

if generate_debug_output:
    response_data.to_csv("_temp/response_data.csv")


# %% Extract user stories into txt-fole and hand over to AQUSA-core
aqusahandler.prepare_user_stories(
    response_data, "input/user_stories.txt", generated_stories_count
)

aqusahandler.run_aqusacore("user_stories.txt", "user_stories_evaluated", "html")

# %% Extract story output from aqusa_core, merge with requirements data and save as JSON

user_stories_parsed = aqusahandler.parse_user_stories_html(
    "output/user_stories_evaluated.html"
)
if generate_debug_output:
    user_stories_parsed.to_csv("_temp/user_stories_parsed.csv")

merged_data = dh.merge_data(response_data, user_stories_parsed, generated_stories_count)
json_data = dh.save_to_json(
    merged_data, f"merged_requirements_data_{model.get_id()}.json"
)
# print(json_data)
