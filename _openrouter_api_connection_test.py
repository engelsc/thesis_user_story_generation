## DEPRECATED!

# %% run everything
from ModelTypes import ModelType
from ModelsOpenRouter import Mistral7BFree
import data_helper as dh
import aqusahandler

# %% run second
model: ModelType = Mistral7BFree()
generated_stories_count: int = 1

raw_requirements = dh.load_requirements("dummy_requirements.csv")
user_prompt: str = dh.load_prompt(model.get_id(), "prompts.csv")

single_requirement: str = raw_requirements.iloc[4]["text_description"]

# build prompt

full_prompt = user_prompt + "\n\nREQUIREMENT:\n" + single_requirement
print(full_prompt)

# %% hand over prompt to model

response: str = model.post_request(full_prompt)
print(response)

# %%
with open("input/user_stories.txt", "w") as file:
    file.write(response)

aqusahandler.run_aqusacore("user_stories.txt", "user_stories_evaluated", "txt")
