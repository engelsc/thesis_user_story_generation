# %% imports
# import asyncio
from ModelTypes import ModelType
from ModelsOpenRouter import Mistral7BFree, Llama318BFree
import data_helper as dh
import pandas as pd
import aqusahandler

# %% build requisites
#model: ModelType = Mistral7BFree()
model: ModelType = Llama318BFree()
generated_stories_count: int = 3

raw_requirements: pd.DataFrame = dh.load_requirements("dummy_requirements.csv")
user_prompt: str = dh.load_prompt(model.get_id(), "prompts.csv")

# %% truncate requirements
requirements_subset: pd.DataFrame = raw_requirements.iloc[:4]
# print(requirements_subset)


# %% hand over prompt to model
async def get_responses() -> pd.DataFrame:
    response_data = await model.generate_responses(
        requirements_subset, user_prompt, generated_stories_count
    )
    return response_data


# HACK: Doesn't work outside of notebook running. Comment out when not running as notebook
response_data: pd.DataFrame = await get_responses()
#

# %% hand over to aqusa_handler

user_stories_parsed = aqusahandler.process_with_aqusacore(
    response_data,
    generated_stories_count
)
print(user_stories_parsed)

# %%
merged_data = dh.merge_data(response_data, user_stories_parsed, generated_stories_count)
json_data = dh.save_to_json(
    merged_data, f"merged_requirements_data_{model.get_id()}.json"
)
