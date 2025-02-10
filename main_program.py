# %% imports, constants and functions
from itertools import product
import pandas as pd
import asyncio
import data_helper
import aqusahandler
import json
import os
from typing import Dict, Any, List
from ModelsOpenRouter import GPT4oMini, Gemini15Pro, Llama318B, Mistral7B, Llama318BFree, Mistral7BFree, Llama323B


TEST_RUN = True
TEST_STORIES = 30

# Define models analogous to experiment_multiple.py
MODELS = ["mistral", "llama32", "gemini", "gpt"]
PROMPT_LEVELS = [1, 2, 3, 4]
#MODELS = ["mistral"]
#PROMPT_LEVELS = [1]
DATA_SETS_PATH = "data_sources/sample_sets/"
PROMPTS_LOCATION = "prompts.yaml"
TEMP_GEN_OUTPUT_FOLDER = "_temp/" if not TEST_RUN else "_tmp/"
TEMP_GEN_OUTPUT_PREFIX = "testgen_output_" if not TEST_RUN else ""

# Dynamically assign dataset IDs to (model, prompt) combinations based on MODELS and PROMPT_LEVELS
dataset_mappings = {
    dataset_id: (model, prompt_level)  # Ensures dataset IDs are zero-padded
    for dataset_id, (model, prompt_level) in enumerate(product(MODELS, PROMPT_LEVELS), start=1)
}
print(dataset_mappings)


async def query_llm(model_id: str, formatted_prompts: List[str]) -> List[str]:
    model = (
        Mistral7BFree() if model_id == "mistralfree" else
        Llama318BFree() if model_id == "llamafree" else
        Mistral7B() if model_id == "mistral" else
        Llama318B() if model_id == "llama31" else
        Llama323B() if model_id == "llama32" else
        Gemini15Pro() if model_id == "gemini" else
        GPT4oMini() if model_id == "gpt" else
        None  # Extend for other models
        )
    if not model:
        raise ValueError(f"Model {model_id} not implemented yet.")

    responses = await model.generate_responses(formatted_prompts)
    return responses


async def run_generation(model_name: str, batch_size: int = 25) -> None:
	"""Generates user stories and saves results per model-prompt combination"""
	prompts : Dict[int, str] = data_helper.load_prompts(PROMPTS_LOCATION)

	for dataset_id, (model, prompt_level) in dataset_mappings.items():
		if model == model_name:
			reviews_df : pd.DataFrame = data_helper.load_app_reviews(dataset_id, DATA_SETS_PATH)
			reviews_list : List[str] = reviews_df["review_description"].tolist()

			# Trim list for test runs
			if TEST_RUN:
				reviews_list = reviews_list[:TEST_STORIES]

			formatted_prompts: List[str] = [prompts[prompt_level].format(review=item) for item in reviews_list]

			output_file = f"{TEMP_GEN_OUTPUT_FOLDER}{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"

			# Generate responses in batches for API safety
			first_batch = True # flag for overwriting on first batch

			for i in range(0, len(formatted_prompts), batch_size):
				batch_prompts = formatted_prompts[i: i + batch_size]
				batch_responses = await query_llm(model, batch_prompts)

				save_results({f"{model}_prompt_{prompt_level}": batch_responses}, output_file, append=not first_batch)
				first_batch = False


def save_results(results: Dict[str, Any], output_file: str, append: bool = False):
	"""Saves results to JSON, either appending or overwriting based on mode."""
	# Ensure the output directory exists
	directory = os.path.dirname(TEMP_GEN_OUTPUT_FOLDER)
	if directory:
		os.makedirs(directory, exist_ok=True)

	if append and os.path.exists(output_file):
		# Load existing data if appending
		with open(output_file, "r", encoding="utf-8") as file:
			existing_data = json.load(file)
	else:
		existing_data = {}

	# Append new responses to existing data
	for key, new_responses in results.items():
		existing_data.setdefault(key, []).extend(new_responses)

	with open(output_file, "w", encoding="utf-8") as file:
		json.dump(existing_data, file, indent=4)

	#print(f"{'Appended' if append else 'Saved'}: {output_file}")


def clean_output_files() -> None:
	for model in MODELS:
		for prompt_level in PROMPT_LEVELS:
			output_path = f"{TEMP_GEN_OUTPUT_FOLDER}{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
			cleaned_path = f"{TEMP_GEN_OUTPUT_FOLDER}clean_{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
			cleaned_stories = data_helper.clean_model_outputs(output_path)
			save_results(cleaned_stories, cleaned_path)


def run_aqusa_on_cleaned() -> None:
    for model in MODELS:
        for prompt_level in PROMPT_LEVELS:
            cleaned_path = f"{TEMP_GEN_OUTPUT_FOLDER}clean_{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
            final_output_path = f"{TEMP_GEN_OUTPUT_FOLDER}final_{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
            cleaned_stories = pd.read_json(cleaned_path)
            aqusa_results = aqusahandler.process_with_aqusacore(cleaned_stories)
            merged_data = data_helper.merge_data(cleaned_stories, aqusa_results)
            data_helper.save_to_json(merged_data, final_output_path)
            #print("AQUSA results:\n",aqusa_results)


# Main execution
async def main() -> None:
	for model in MODELS:
		await run_generation(model)

	clean_output_files()
	run_aqusa_on_cleaned()

# %% generate stories
#for model in MODELS:
#	await run_generation(model, 5) # type: ignore # noqa

# %% clean user stories
#clean_output_files()
# %% run aqusa on cleaned stories
#run_aqusa_on_cleaned()


# %% Main call: DO NOT RUN
# Run on execution
if __name__ == "__main__":
	asyncio.run(main())
