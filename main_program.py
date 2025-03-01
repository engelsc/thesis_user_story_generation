# %% imports, constants and functions
from itertools import product
import pandas as pd
import asyncio
import data_helper
import aqusahandler
from typing import Dict, List
from ModelsOpenRouter import GPT4oMini, Gemini15Pro, Llama318B, Mistral7B, Mistral7BFree, Llama323B, Llama318BFree, Ministral8B, Gemini15Flash8B
from data_helper import save_results


TEST_RUN = False
'''Toggles test run behaviour.\nChanges output folder and file name. Also limits amount of data to generate.'''
TEST_STORIES = 15
'''Sets amount of reviews to use from sample set. Only applies during TEST_RUN.
Max value: 200 (max reviews per sample set)'''
FIX_UNIFORM_ISSUES = True
'''Runs find-replace logic on all cleaned stories to fix the main culprits of uniform-uniform defects. Malformed "I want to" and "so that" chunks.'''

# Define models analogous to experiment_multiple.py
MODELS = ["ministral", "llama31", "geminiflash", "gpt"]
PROMPT_LEVELS = [1, 2, 3, 4]
DATA_SETS_PATH = "data_sources/sample_sets/"
PROMPTS_LOCATION = "prompts.yaml"
TEMP_GEN_OUTPUT_FOLDER = "_temp/" if TEST_RUN else "_tmp/"
TEMP_GEN_OUTPUT_PREFIX = "testgen_output_" if TEST_RUN else ""
GEN_CHUNK_SIZE = 20
'''Sets amount of LLM responses to generate before writing them to file.'''

dataset_mappings = {
    dataset_id: (model, prompt_level)
    for dataset_id, (model, prompt_level) in enumerate(product(MODELS, PROMPT_LEVELS), start=1)
}
'''Dynamically assigns dataset IDs to (model, prompt) combinations based on MODELS and PROMPT_LEVELS'''
print(dataset_mappings)


async def query_llm(model_id: str, formatted_prompts: List[str]) -> List[str]:
    '''Chooses model based on string model_id and calls generate_responses() with given list of prompts.
    :return: All LLM responses as List[str]'''
    model = (
        Mistral7BFree() if model_id == "mistralfree" else
        Llama318BFree() if model_id == "llamafree" else # defunct
        Mistral7B() if model_id == "mistral" else
        Ministral8B() if model_id == "ministral" else
        Llama318B() if model_id == "llama31" else
        Llama323B() if model_id == "llama32" else
        Gemini15Pro() if model_id == "gemini" else
        Gemini15Flash8B() if model_id == "geminiflash" else
        GPT4oMini() if model_id == "gpt" else
        None  # Extend for other models
        )
    if not model:
        raise ValueError(f"Model {model_id} not implemented yet.")

    responses = await model.generate_responses(formatted_prompts)
    return responses


async def run_generation(model_name: str, batch_size: int = 25) -> None:
	'''Generates user stories and saves results per model-prompt combination'''
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

				save_results(TEMP_GEN_OUTPUT_FOLDER, {f"{model}_prompt_{prompt_level}": batch_responses}, output_file, append=not first_batch)
				first_batch = False


def clean_output_files() -> None:
	for model in MODELS:
		for prompt_level in PROMPT_LEVELS:
			output_path = f"{TEMP_GEN_OUTPUT_FOLDER}{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
			cleaned_path = f"{TEMP_GEN_OUTPUT_FOLDER}clean_{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
			cleaned_stories = data_helper.clean_model_outputs(output_path)
			save_results(TEMP_GEN_OUTPUT_FOLDER, cleaned_stories, cleaned_path)


def replace_uniform_issues() -> None:
	for model in MODELS:
		for prompt_level in PROMPT_LEVELS:
			cleaned_path = f"{TEMP_GEN_OUTPUT_FOLDER}clean_{TEMP_GEN_OUTPUT_PREFIX}{model}_prompt{prompt_level}.json"
			find_strings = ["I want the", "so I"]
			replace_strings = ["I want to the", "so that I"]
			replaced_stories = data_helper.find_replace(find_strings, replace_strings, cleaned_path)
			save_results(TEMP_GEN_OUTPUT_FOLDER, replaced_stories, cleaned_path)


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
		await run_generation(model, GEN_CHUNK_SIZE)

	clean_output_files()
	if FIX_UNIFORM_ISSUES:
		replace_uniform_issues() # OPTIONAL
	run_aqusa_on_cleaned()

# Run on execution
if __name__ == "__main__":
	asyncio.run(main())


######################################
# Only used for testing/debugging
######################################

# %% generate stories
#for model in MODELS:
#	await run_generation(model, GEN_CHUNK_SIZE) # type: ignore # noqa

# %% clean user stories
clean_output_files()

# %% run find-replace to fix uniform-uniform issues
replace_uniform_issues()


# %% run aqusa on cleaned stories
run_aqusa_on_cleaned()
