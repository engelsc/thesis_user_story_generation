from itertools import product
import yaml
from typing import Dict
import pandas as pd


# Define models analogous to experiment_multiple.py
MODELS = ["mistral", "llama", "gemini", "gpt"]
PROMPT_LEVELS = [1, 2, 3, 4]
DATA_SETS_PATH ="data_sources/sample_sets/"

# Dynamically assign dataset IDs to (model, prompt) combinations
dataset_mappings = {
    f"{dataset_id:02d}": (model, prompt)  # Ensures dataset IDs are zero-padded
    for dataset_id, (model, prompt) in enumerate(product(MODELS, PROMPT_LEVELS), start=1)
}

# Example usage: print dataset assignments
for dataset_id, (model, prompt) in dataset_mappings.items():
    print(f"Dataset {dataset_id}: Model={model}, Prompt Level={prompt}")


def load_prompts(filepath: str) -> Dict[int, str]:
    """Load prompt templates from a YAML file as single strings."""
    with open(filepath, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)

    if not isinstance(prompts, dict):
        raise ValueError("Invalid prompt format: Expected a dictionary of prompt levels.")

    # Ensure all levels return a single formatted string
    return {int(level): text.strip() for level, text in prompts.items()}

#print(load_prompts("prompts.yaml")[4])


def load_app_reviews(dataset_id: int, base_path: str = DATA_SETS_PATH) -> pd.DataFrame:
    """Load app reviews from a given dataset ID."""
    filepath = f"{base_path}sample_set_{dataset_id:02d}.csv"

    try:
        df = pd.read_csv(filepath)
        if "review_description" not in df.columns:
            raise ValueError(f"Missing 'review_description' column in {filepath}")

        return df[["review_description"]]  # Keep only relevant column

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return pd.DataFrame(columns=["review_description"])

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame(columns=["review_description"])
