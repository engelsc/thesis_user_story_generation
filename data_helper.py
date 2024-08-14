import pandas as pd
import csv
import json
from typing import Any, cast


def load_requirements(file_path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(file_path, quotechar='"', delimiter=",")
    if "index" in df.columns:
        df = df.drop(columns=["index"])
    return df


def load_prompt_depr(model_id: str, file_path: str) -> str:
    with open(file_path, newline="") as prompts:
        reader = csv.reader(prompts, delimiter=",")
        for row in reader:
            if row[0] == model_id:
                return row[1]
        raise ValueError(f"No prompt found for model_id: {model_id}")


def load_prompt(model_id: str, file_path: str) -> str:
    df: pd.DataFrame = pd.read_csv(file_path, quotechar='"', delimiter=",")

    prompt_row: pd.DataFrame = df[df["model_name"] == model_id]

    if not prompt_row.empty:
        return prompt_row["prompt"].values[0]
    else:
        raise ValueError(f"No prompt found for model_id: {model_id}")


def merge_data(
    response_data: pd.DataFrame,
    user_stories_parsed: pd.DataFrame,
    generated_stories_count: int,
) -> list[Any]:
    # Initialize an empty list to store the nested data
    nested_data: list[Any] = []

    # Iterate over each requirement in response_data
    for req_id, req_row in response_data.iterrows():
        req_id = cast(int, req_id)
        requirement_dict = {
            "requirement_id": req_id,
            "requirement_text": str(req_row["text_description"]),
            "source": str(req_row["source"]),
            "user_stories": [],
        }
        # print(req_row)

        for i in range(1, generated_stories_count + 1):
            id: int = cast(int, requirement_dict["requirement_id"])
            story_id: int = (id * generated_stories_count) + i

            story_dict = {
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


def save_to_json(data: list[Any], file_path: str) -> str:
    json_data: str = json.dumps(data, indent=4)
    with open(file_path, "w") as f:
        f.write(json_data)
    return json_data
