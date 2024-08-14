import subprocess
import os
from typing import cast
from bs4 import BeautifulSoup
import pandas as pd
import re


def run_aqusacore(input_file: str, output_file: str, format: str) -> None:
    # Path to the Python 3.8 interpreter for compatibility
    python_interpreter = ".venv3.8/bin/python"

    command: list[str] = [
        python_interpreter,
        "aqusa-core/aqusacore.py",
        "-i",
        input_file,
        "-o",
        output_file,
        "-f",
        format,
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        print("Error running aqusacore.py:")
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    else:
        print("Command ran successfully:")
        print(result.stdout)


def prepare_user_stories(
    response_data: pd.DataFrame, file_path: str, count: int
) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, "w") as file:
        for _, entry in response_data.iterrows():
            cleaned_story: str = ""
            if count >= 1:
                cleaned_story = (
                    str(entry["story_1"])
                    .replace('"', "")
                    .replace("'", "")
                    .replace("\n", " ")
                )
                file.write(cleaned_story + "\n")
            if count >= 2:
                cleaned_story = (
                    str(entry["story_2"])
                    .replace('"', "")
                    .replace("'", "")
                    .replace("\n", " ")
                )
                file.write(cleaned_story + "\n")
            if count == 3:
                cleaned_story = (
                    str(entry["story_3"])
                    .replace('"', "")
                    .replace("'", "")
                    .replace("\n", " ")
                )
                file.write(cleaned_story + "\n")


def parse_user_stories_html(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    data = []
    table = soup.find("table", {"class": "sortable"})
    rows = table.find_all("tr")[1:]  # Skip header row

    for row in rows:
        cols = row.find_all("td")
        story_id = cols[0].text
        user_story = cols[1].text
        defect_type = cols[2].text
        sub_type = cols[3].text
        message = cols[4].text
        data.append([story_id, user_story, defect_type, sub_type, message])

    df = pd.DataFrame(
        data, columns=["story_id", "user_story", "defect_type", "sub_type", "message"]
    )
    return df


def parse_user_stories_txt(file_path: str) -> pd.DataFrame:
    data = []

    with open(file_path, "r") as file:
        story_id = None
        user_story = None
        defect_type = None
        sub_type = None
        message = None

        for line in file:
            # Match the story ID and user story
            story_match = re.match(r'^Story #(\d+): "(.*)"$', line.strip())
            if story_match:
                story_id = cast(int, story_match.group(1))
                user_story = str(story_match.group(2))
                continue

            # Match the defect type and sub_type
            defect_match = re.match(
                r"^\s*Defect type: ([^\.]+)\.([^ ]+)$", line.strip()
            )
            # print("defect: " + str(defect_match))
            if defect_match:
                defect_type = str(defect_match.group(1))
                sub_type = str(defect_match.group(2))
                continue

            # Match the message
            message_match = re.match(r"^\s*Message: (.*)$", line.strip())
            # print("message: " + str(message_match))
            if message_match:
                message = str(message_match.group(1))

                data.append(
                    [
                        story_id,
                        user_story,
                        defect_type,
                        sub_type,
                        message,
                    ]
                )

    # Create a DataFrame
    df = pd.DataFrame(
        data, columns=["story_id", "user_story", "defect_type", "sub_type", "message"]
    )
    return df
