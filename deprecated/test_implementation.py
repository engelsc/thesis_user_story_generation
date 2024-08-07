# %% imports
from ModelTypes import TestImplementation, ResponseDict
import csv
import asyncio
import os
import aqusahandler
## run each block or line with CTRL+SHIFT+ENTER

# %% Initializing model to be used for generation
model = TestImplementation()

# %% Loading (and cleaning) the csv containing natural language requirement descriptions
raw_data: dict[int, ResponseDict] = {}
with open("dummy_requirements.csv", newline="") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    next(reader)  # skip header
    i: int = 0
    for row in reader:
        raw_data[i] = {"requirement": row[1], "source": row[2]}
        i += 1
# print(raw_data)

# %% Load prompts file and extract prompt for currently loaded model
user_prompt: str = ""
with open("prompts.csv", newline="") as prompts:
    reader = csv.reader(prompts, delimiter=",")
    for row in reader:
        if row[0] == model.get_id():
            user_prompt = row[1]
# print(user_prompt)

# %% Hand over raw_data and user_prompt to model to handle async requests

generated_stories_count: int = 1


def get_response(raw_data: dict[int, ResponseDict], user_prompt: str) -> dict:
    return model.generate_responses(raw_data, user_prompt, generated_stories_count)


# TODO: Test how to make this async for dealing with API delay
response_data = model.generate_responses(raw_data, user_prompt, generated_stories_count)

# print(response_data)

# %% Prearing generated User Stories for AQUSA-core tool
file_path: str = "input/user_stories.txt"
if os.path.exists(file_path):
    os.remove(file_path)
file = open(file_path, "w")

for i, entry in response_data.items():
    if generated_stories_count >= 1:
        # story: str = entry["response_1"]
        # print(story)
        file.write(entry["response_1"] + "\n")
    if generated_stories_count >= 2:
        file.write(entry["response_2"])
    if generated_stories_count == 3:
        file.write(entry["response_3"])
file.close()

# %% Pipe file into aqusa

aqusahandler.run_aqusacore(
    "user_stories.txt",
    "user_stories_evaluated",
    "html",
)
