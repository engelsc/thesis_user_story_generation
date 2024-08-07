import subprocess
import os
from bs4 import BeautifulSoup
import pandas as pd


def run_aqusacore(input_file: str, output_file: str, format: str) -> None:
    # Path to the Python 3.8 interpreter for compatibility
    python_interpreter = ".venv3.8/bin/python"

    command = [
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
            if count >= 1:
                file.write(entry["story_1"] + "\n")
            if count >= 2:
                file.write(entry["story_2"] + "\n")
            if count == 3:
                file.write(entry["story_3"] + "\n")


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
