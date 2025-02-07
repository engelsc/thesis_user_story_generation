import pandas as pd
import json
import asyncio
from typing import Any, List, Dict
from ModelsOpenRouter import Mistral7BFree, Llama318BFree, Gemini15Pro, GPT4oMini
#from data_helper import save_to_json, load_requirements

# Experiment Constants
MODEL_TYPES = ["mistral", "llama", "gpt", "gemini"]  # LLMs to evaluate
#MODEL_TYPES = ["gemini"]
NUMBER_OF_RUNS = 2  # Number of times each model runs the experiment
DATASETS = ["input/experiment_stories1.csv", "input/experiment_stories2.csv"]  # Review sets

# Sample Prompt Template
PROMPT_TEMPLATE = (
	"\"\"\""
    "Below is a collection of app reviews. Extract the minimal set of user stories that "
    "captures all requirements expressed in these reviews. Output only the user stories, "
    "each on a new line."
    "\"\"\"\n\n"
    "REVIEWS:\n"
    "{reviews}"  # Placeholder for concatenated reviews
)

# Load reviews dataset
def load_reviews(file_path: str) -> List[str]:
    df = pd.read_csv(file_path)
    return df["text_description"].tolist()

# Prepare the prompt for batch processing
def create_batch_prompt(reviews: List[str]) -> str:
    return PROMPT_TEMPLATE.format(reviews=" ".join(reviews))

# Query LLMs asynchronously
async def query_llm(model_id: str, formatted_prompt: str) -> List[str]:
    model = (
            Mistral7BFree() if model_id == "mistral" else
            Llama318BFree() if model_id == "llama" else
            Gemini15Pro() if model_id == "gemini" else
            GPT4oMini() if model_id == "gpt" else
            None  # Extend for other models
        )
    if not model:
        raise ValueError(f"Model {model_id} not implemented yet.")

    responses = await model.generate_responses([formatted_prompt])
    return responses

# Run the experiment across all LLMs and datasets
async def run_experiment() -> Dict[str, Any]:
    results = {}

    for dataset in DATASETS:
        reviews = load_reviews(dataset)
        prompt = create_batch_prompt(reviews)
        #print(prompt) # DEBUG print

        for model in MODEL_TYPES:
            model_results = []
            for _ in range(NUMBER_OF_RUNS):
                responses = await query_llm(model, prompt)
                model_results.append(responses)

            results[f"{model}_{dataset}"] = model_results

    return results

# Save results to JSON
def save_results(results: Dict[str, Any], output_file: str):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

# Main execution
async def main():
    results = await run_experiment()
    save_results(results, "output/experiment_results.json")

# Run the experiment
if __name__ == "__main__":
    asyncio.run(main())
