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
print(response_data)

# %% Extract user stories into txt-fole
aqusahandler.prepare_user_stories(
    response_data, "input/user_stories.txt", generated_stories_count
)

# %% hand over to AQUSA-core
# aqusahandler.run_aqusacore("user_stories.txt", "user_stories_evaluated", "html")
aqusahandler.run_aqusacore("user_stories.txt", "user_stories_evaluated", "txt")

# %% Extract story output from aqusa_core merge with requirements data and save as JSON

# user_stories_parsed = aqusahandler.parse_user_stories_html(
#    "output/user_stories_evaluated.html"
# )
# print(user_stories_parsed)

user_stories_parsed = aqusahandler.parse_user_stories_txt(
    "output/user_stories_evaluated.txt"
)
print(user_stories_parsed)

# %%
merged_data = dh.merge_data(response_data, user_stories_parsed, generated_stories_count)
json_data = dh.save_to_json(
    merged_data, f"merged_requirements_data_{model.get_id()}.json"
)
