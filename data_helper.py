import pandas as pd
import json
import re
import os
from typing import Any, Dict, List, cast
import yaml


def load_prompts(filepath: str) -> Dict[int, str]:
    """Load prompt templates from a YAML file as single strings."""
    with open(filepath, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)

    if not isinstance(prompts, dict):
        raise ValueError("Invalid prompt format: Expected a dictionary of prompt levels.")

    # Ensure all levels return a single formatted string
    return {int(level): text.strip() for level, text in prompts.items()}


def load_app_reviews(dataset_id: int, base_path: str) -> pd.DataFrame:
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


def clean_model_outputs(json_path: str) -> Dict[str, List[str]]:
	"""Loads generated outputs, extracts user stories, and structures them for AQUSA."""
	with open(json_path, "r", encoding="utf-8") as file:
		raw_data = json.load(file)

	cleaned_data : Dict[str, List[str]] = {}
	for key, responses in raw_data.items():
		cleaned_responses : List[str] = [extract_user_story(response) for response in responses]
		cleaned_data[key] = cleaned_responses

	return cleaned_data


def extract_user_story(response_text: str) -> str:
    """Extracts a clean user story from a model-generated response, handling variations and unwanted text."""

    # Normalize text: remove extra spaces, newlines, markdown symbols, and special characters
    response_text = response_text.strip().replace("\n", " ")
    response_text = re.sub(r'[\*"#`\[\]]', '', response_text).strip() # Remove markdown or special characters
    response_text = re.sub(r'\s+', ' ', response_text).strip()  # Remove remaining double spaces

    # Look for an explicit "USER STORY:" section first and extract full user story after it
    match = re.search(r'USER STORY[:\-]*\s*(As (?:a|an) .*?, I want .*?, so .*?[.!?])', response_text, re.IGNORECASE)
    if match:
        response_text = match.group(1).strip()
        return re.sub(r'USER STORY[:\-]*\s*', '', response_text)

    # If no explicit "USER STORY:" Extract a full user story format, ensuring it captures the complete "so" clause
    match = re.search(r'(As (?:a|an) .*?, I want .*?, so .*?[.!?])', response_text)
    if match:
        return match.group(1).strip()

    # Extract incomplete user story without "so" clause
    match = re.search(r'(As (?:a|an) .*?, I want .*?[.!?])', response_text)
    if match:
        return match.group(1).strip()

    # Extract very incomplete user story without "I want" or "so" clause
    match = re.search(r'(As (?:a|an) .*?[.!?])', response_text)
    if match:
        return match.group(1).strip()

    return "UNABLE_TO_EXTRACT_USER_STORY"


def merge_data(
    response_data: pd.DataFrame,
    user_stories_parsed: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Merges generated user stories with AQUSA defect results.
    Ensures all generated stories are present and defects are correctly mapped.
    """
    nested_data: List[Dict[str, Any]] = []

    for _, user_stories in response_data.items():
        for story_id, user_story in enumerate(user_stories, start=1):
            story_dict: Dict[str, Any] = {
                "story_id": story_id,
                "user_story": user_story.strip().replace('"', "").replace("'", "").replace("\n", " "),
                "defect_count": 0,  # Default to 0, will update if defects exist
                "defects": [],
            }

            # Match AQUSA defect results based on story_id
            matching_defects = user_stories_parsed[user_stories_parsed["story_id"].astype(int) == int(story_id)]

            #print(matching_defects)

            for _, defect_row in matching_defects.iterrows():
                if pd.notna(defect_row.get("defect_type", None)):
                    defect_dict = {
                        "defect_type": defect_row["defect_type"],
                        "sub_type": defect_row["sub_type"],
                        "message": defect_row["message"],
                    }
                    story_dict["defects"].append(defect_dict)

            # Update defect count
            story_dict["defect_count"] = len(story_dict["defects"])

            nested_data.append(story_dict)

    return nested_data


def save_to_json(data: list[Any], file_path: str) -> str:
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    json_data: str = json.dumps(data, indent=4)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(json_data)
    return json_data


###########################################################
#  LEGACY FUNCTIONS: for compatibility with test code only
###########################################################
def load_requirements(file_path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file_path, quotechar='"', delimiter=",")
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df


def load_prompt(model_id: str, file_path: str) -> str:
    df: pd.DataFrame = pd.read_csv(file_path, quotechar='"', delimiter=",")

    prompt_row: pd.DataFrame = df[df["model_name"] == model_id]

    if not prompt_row.empty:
        return prompt_row["prompt"].values[0]
    else:
        raise ValueError(f"No prompt found for model_id: {model_id}")


def merge_data2(
    response_data: pd.DataFrame,
    user_stories_parsed: pd.DataFrame,
    generated_stories_count: int = 1,
) -> list[Any]:
    # Initialize an empty list to store the nested data
    nested_data: list[Any] = []

    # Iterate over each requirement in response_data
    for req_id, req_row in response_data.iterrows():
        req_id = cast(int, req_id)
        requirement_dict : Dict[str, Any] = {
            "requirement_id": req_id,
            "requirement_text": str(req_row["text_description"]),
            "source": str(req_row["source"]),
            "user_stories": [],
        }
        # print(req_row)

        for i in range(1, generated_stories_count + 1):
            id: int = cast(int, requirement_dict["requirement_id"])
            story_id: int = (id * generated_stories_count) + i

            story_dict : Dict[str, Any] = {
                "story_id": story_id,
                "user_story": str(req_row[f"story_{i}"])
                .replace('"', "")
                .replace("'", "")
                .replace("\n", " "),
                "defects": [],
            }

            # Filter the user_stories_parsed DataFrame to get the relevant user story by story_id
            stories_with_defect: pd.DataFrame = user_stories_parsed[
                user_stories_parsed["story_id"].astype(int) == story_id
            ]

            if not stories_with_defect.empty:
                # Add defects if they exist
                for _, defect_row in stories_with_defect.iterrows():
                    if pd.notna(defect_row["defect_type"]):
                        defect_dict = {
                            "defect_type": defect_row["defect_type"],
                            "sub_type": defect_row["sub_type"],
                            "message": defect_row["message"],
                        }
                        story_dict["defects"].append(defect_dict)

            else:
                print(f"No defects found for story_id: {story_id}")  # Debugging line

            requirement_dict["user_stories"].append(story_dict)
        nested_data.append(requirement_dict)
    return nested_data


def merge_responses_with_data(
        responses: list[str], raw_data: pd.DataFrame, run_amount: int
    ) -> pd.DataFrame:
        response_data: pd.DataFrame = raw_data.copy()

        story_index: int = 0
        for id, _ in response_data.iterrows():
            id = cast(int, id)
            for i in range(1, run_amount + 1):
                response_data.at[id, f"story_{i}"] = responses[story_index]
                story_index += 1
        return response_data
