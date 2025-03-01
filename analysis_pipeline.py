# %% imports and data preparation functions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import math
from typing import Tuple
from scipy.stats import kruskal #type:ignore <- disables parsing from basedpyright
from scipy.stats import chi2_contingency #type:ignore


TEST_RUN = False


DATA_PATH = "_temp/" if TEST_RUN else "_tmp/"  # USE THESE IN FINAL RUNS
DATA_PREFIX = "final_testgen_output_" if TEST_RUN else "final_" # USE THESE IN FINAL RUN
OUTPUT_PATH = "_temp/_analysis/" if TEST_RUN else "_analysis/"
OUTPUT_FILE = "merged_final_output.csv"  # Filename for merged DataFrames

def load_final_json_files(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads all final_*.json files in given folder and compiles them into two DataFrames.
    The first df contains all stories by model and prompt type with defects as lists of dictionaries.
    The second explodes and flattens the defects for defect level analysis.
    """
    all_data = []

    for filename in os.listdir(data_path):
        if filename.startswith("final_") and filename.endswith(".json"):
            file_path = os.path.join(data_path, filename)
            model_name, prompt_level = extract_model_prompt(filename)

            with open(file_path, "r", encoding="utf-8") as file:
                json_data = json.load(file)
                for story in json_data:
                    story["model"] = model_name
                    story["prompt_level"] = prompt_level
                    all_data.append(story)

    df = pd.DataFrame(all_data)

    # Reorder columns so that model, prompt_level, and story_id are first
    column_order = ["model", "prompt_level", "story_id"] + [col for col in df.columns if col not in ["model", "prompt_level", "story_id"]]
    df = df[column_order]

	# Explode defects column so each list items becomes its own row
    df_exploded = df.explode("defects")

    # Convert defect dictionaries into separate columns
    defect_details = pd.json_normalize(df_exploded["defects"]) # type: ignore
    df_exploded = df_exploded.drop(columns=["defects"]).reset_index(drop=True)
    df_exploded = pd.concat([df_exploded, defect_details], axis=1)

    #print(df_exploded.head())

    return df, df_exploded


def extract_model_prompt(filename: str) -> Tuple[str, int]:
    """
    Extracts model name and prompt level from filename.
    Expected format: {DATA_PREFIX}{model}_prompt{level}.json
    """
    parts = filename.replace(DATA_PREFIX, "").replace(".json", "").split("_prompt")
    model_name = parts[0]
    prompt_level = int(parts[1]) if len(parts) > 1 else -1
    return model_name, prompt_level

def print_header(text: str, width:int=75, char:str='#') -> None:
	print(f"\n{char*width}\n#    {text}\n{char*width}\n")

# %% Start logging

# Ensure the directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

log_file_path = "analysis_output.log"
if not TEST_RUN:
	sys.stdout = open(f"{OUTPUT_PATH}{log_file_path}", "w", encoding="utf-8")


# %% Data preparation
# Load all files into a DataFrame
df_story_level, df_defect_level = load_final_json_files(DATA_PATH)
df_story_level.to_csv(f"{OUTPUT_PATH}story_level_{OUTPUT_FILE}", index=False)
df_defect_level.to_csv(f"{OUTPUT_PATH}defect_level_{OUTPUT_FILE}", index=False)
#print(df_defect_level.head())
#print(f"Merged data saved to {OUTPUT_FILE}")

# %% Data Overview
# Check the basic structure and summary of the story-level DataFrame
print_header("Story-Level DataFrame:")
print(df_story_level.info())
#print(df_story_level.head())

# Check the basic structure and summary of the defect-level DataFrame
print_header("Defect-Level DataFrame:")
print(df_defect_level.info())
#print(df_defect_level.head())
# shows total amount of defects in info defect type non-null count


# %% Flag invalid user stories
df_story_level['valid_story'] = df_story_level['user_story'] != "UNABLE_TO_EXTRACT_USER_STORY"

# Count invalid stories by model and prompt level
invalid_counts = df_story_level[df_story_level['valid_story'] == False].groupby(['model', 'prompt_level']).size().reset_index(name='invalid_count') #type:ignore #noqa

# Count total stories per group
total_counts = df_story_level.groupby(['model', 'prompt_level']).size().reset_index(name='total_stories') #type: ignore

# Merge counts to calculate the ratio of invalid stories
summary_counts = total_counts.merge(invalid_counts, on=['model', 'prompt_level'], how='left')
summary_counts['invalid_count'] = summary_counts['invalid_count'].fillna(0).astype(int)
summary_counts['valid_stories'] = summary_counts['total_stories'] - summary_counts['invalid_count']
summary_counts['invalid_ratio'] = summary_counts['invalid_count'] / summary_counts['total_stories']

print_header("Summary of Valid and Invalid Stories by Group:")
print(summary_counts.to_string())
# %% Bar Chart for Invalid Stories by Model and Prompt Level
plt.figure()
sns.barplot(x='prompt_level', y='invalid_count', hue='model', data=summary_counts)
plt.title("Anzahl Ungültiger Stories nach Modell und Prompt Level")
plt.xlabel("Prompt Level")
plt.ylabel("Anzahl Ungültiger Stories")
plt.legend(title="Modell", bbox_to_anchor=(1, 1), loc='upper left') # Move the legend outside the plot
plt.savefig(f"{OUTPUT_PATH}invalid_stories.png")
plt.show()
# %% Filter to keep only valid stories
df_story_level_allstories = df_story_level.copy() # save all stories
df_story_level = df_story_level[df_story_level['valid_story']].copy() # keep only valid stories
df_story_level.drop(columns=['valid_story'], inplace=True) # drop the helper column

print_header("Valid and Invalid Stories Count")
print(f"Total stories: {len(df_story_level_allstories)}")
print(f"Total valid stories: {len(df_story_level)}")
# %% Rebuild defect_level DataFrame based on valid stories only
df_defect_level_allstories = df_defect_level.copy()

# Code taken from  load_final_json_files()
df_exploded = df_story_level.explode("defects")
defect_details = pd.json_normalize(df_exploded["defects"]) # type: ignore
df_exploded = df_exploded.drop(columns=["defects"]).reset_index(drop=True)
df_exploded = pd.concat([df_exploded, defect_details], axis=1)

print(f"Total defects: {len(df_defect_level_allstories)}")
df_defect_level = df_exploded
print(f"Total defects of valid stories onl: {len(df_defect_level)}")

# %%
###########################################################################
#  Story-Level Descriptive Analysis
###########################################################################
print_header("Story-Level Descriptive Analysis")

# %% Sum of total number of defects per group
total_defects = df_story_level.groupby(['model', 'prompt_level'])['defect_count'].sum().reset_index(name='total_defects')
print_header("Total Number of Defects Per Group:")
print(total_defects)
# %%
nonzero_defects = df_story_level[df_story_level['defect_count'] > 0] \
                    .groupby(['model', 'prompt_level']).size() \
                    .reset_index(name='stories_with_defects') #type: ignore

print_header("Count of Stories with at Least One Defect:")
print(nonzero_defects)


# %% Group by model and prompt level and compute count, mean, median, and std for defect_count.
# "count" is the total number of entries/user stories
group_stats = df_story_level.groupby(['model', 'prompt_level'])['defect_count']\
                .agg(count='count', mean='mean', median='median', std='std', min='min', max='max').reset_index()

print_header("Aggregated Story-Level Defect Count Statistics:")
print(group_stats)

# %% Proportion of Stories with Zero Defects
print_header("Proportion of Stories with Zero Defects")

# Count stories with defect_count == 0 for each group.
zero_defects = df_story_level[df_story_level['defect_count'] == 0]\
                .groupby(['model', 'prompt_level']).size().reset_index(name='zero_count') # type: ignore

# Count total stories per group.
total_stories = df_story_level.groupby(['model', 'prompt_level']).size().reset_index(name='total_count') # type: ignore

# Count total Zero Defects Stories across all Stories
total_story_count = total_stories["total_count"].sum()
total_zero_count = zero_defects["zero_count"].sum()

print(f"Total Stories: {total_story_count}\nTotal Stories with Zero Defects: {total_zero_count}")
print(f"Total Proportion of Zero Defects: {total_zero_count/total_story_count:.2%}\n")

# Merge the counts and calculate the ratio.
zero_defects_ratio = pd.merge(total_stories, zero_defects, on=['model', 'prompt_level'], how='left')
zero_defects_ratio['zero_count'] = zero_defects_ratio['zero_count'].fillna(0)
zero_defects_ratio['zero_defect_ratio'] = zero_defects_ratio['zero_count'] / zero_defects_ratio['total_count']

print_header("Proportion of Stories with Zero Defects over all Groups:")
print(zero_defects_ratio)

# %% Optional: Count Defect Types across models and prompt levels - VERBOSE
#defect_counts = df_defect_level.groupby(['model', 'prompt_level', 'defect_type']).size() #type: ignore
#print(defect_counts)

# %%
######################################
# Story-Level Visualizations
######################################

# %% Boxplot of Defect Count Distribution
plt.figure(figsize=(10, 5))
sns.boxplot(x='prompt_level', y='defect_count', hue='model', data=df_story_level)
plt.title("Defekt Verteilung nach Modell und Prompt Level")
plt.xlabel("Prompt Level")
plt.ylabel("Defekt Anzahl")
plt.legend(title='Modell', bbox_to_anchor=(1, 1), loc='upper left') # Move the legend outside the plot
plt.savefig(f"{OUTPUT_PATH}defect_count_distr_box.png")
plt.show()

# %% Violin Plot of Defect Count Distribution
plt.figure(figsize=(12, 5))
sns.violinplot(x='prompt_level', y='defect_count', hue='model', data=df_story_level)
plt.title("Defekt Verteilung nach Modell und Prompt Level")
plt.xlabel("Prompt Level")
plt.ylabel("Defekt Anzahl")
plt.legend(title='Modell', bbox_to_anchor=(1, 1), loc='upper left') # Move the legend outside the plot
plt.savefig(f"{OUTPUT_PATH}defect_count_distr_violin.png")
plt.show()

# %% Heatmap for Mean Defect Count by Model and Prompt Level
pivot_mean = df_story_level.pivot_table(index='model', columns='prompt_level', values='defect_count', aggfunc='mean')
plt.figure(figsize=(5, 3))
sns.heatmap(pivot_mean, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Mittelwert Defekt Anzahl pro Story")
plt.xlabel("Prompt Level")
plt.ylabel("")
plt.savefig(f"{OUTPUT_PATH}heatmap_mean_defect_count_modelprompt.png")
plt.show()
# %% more Heatmaps

# %% Proportion of Zero-Defect Stories per Model and Prompt level
# Total stories per model and prompt level.
total_stories = df_story_level.groupby(['model', 'prompt_level']).size().reset_index(name='total_count') #type:ignore #noqa

# Count of zero-defect stories per group.
zero_defects = df_story_level[df_story_level['defect_count'] == 0].groupby(['model', 'prompt_level']).size().reset_index(name='zero_count') #type:ignore #noqa

# Merge the two counts.
zero_defects_ratio = pd.merge(total_stories, zero_defects, on=['model', 'prompt_level'], how='left')
zero_defects_ratio['zero_count'] = zero_defects_ratio['zero_count'].fillna(0)
zero_defects_ratio['ratio'] = zero_defects_ratio['zero_count'] / zero_defects_ratio['total_count']

# Create a pivot table of the ratio.
pivot_ratio = zero_defects_ratio.pivot(index='model', columns='prompt_level', values='ratio')

plt.figure(figsize=(5, 3))
sns.heatmap(pivot_ratio, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Anteil Stories ohne Defekt")
plt.xlabel("Prompt Level")
plt.ylabel("")
plt.savefig(f"{OUTPUT_PATH}heatmap_proportion_zero_defects.png")
plt.show()
# %% Total Story count by Model and Prompt level
# # Create a pivot table for the total number of stories.
pivot_total = total_stories.pivot(index='model', columns='prompt_level', values='total_count')

plt.figure(figsize=(5, 3))
sns.heatmap(pivot_total, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Menge valider Stories")
plt.xlabel("Prompt Level")
plt.ylabel("")
plt.savefig(f"{OUTPUT_PATH}heatmap_total_story_count.png")
plt.show()


# %%
###########################################################################
#  Defect-Level Descriptive Analysis
###########################################################################
print_header("Defect-Level Descriptive Analysis")
# %% Main Defect Types

# %% Total Defects by Model and Prompt Level
df_defect_summary = df_defect_level.groupby(['model', 'prompt_level']).size().reset_index(name='defect_count') #type:ignore
# Print the summary DataFrame
print_header("Defect Summary (Total defects per model per prompt level):")
print(df_defect_summary)

# %% Horizontal Bar Chart for Total Defects

# Generate colors using YlGnBu colormap
colors = sns.color_palette("Blues", len(df_defect_summary))

#plt.figure(figsize=(10, 6))
plt.figure()
sns.barplot(
    data=df_defect_summary,
    x='defect_count',   # Total defects will be on the x-axis
    y='model',   		# model level on the y-axis
    hue='prompt_level',	# Bars grouped by prompt_level
    orient='h',
    #palette=colors
)

plt.title("Anzahl Defekte pro Modell und Prompt Level")
plt.xlabel("Anzahl Defekte")
plt.ylabel("")
plt.legend(title='Prompt Level', bbox_to_anchor=(1, 1), loc='upper left') # Move the legend outside the plot
plt.savefig(f"{OUTPUT_PATH}total_defect_per_modelprompt.png")
plt.show()

# %% Frequency of Defect Types by Model and Prompt Level
defect_counts = df_defect_level.groupby(['model', 'prompt_level', 'defect_type']).size().reset_index(name='count') #type:ignore
#defect_counts = df_defect_level.groupby(['model', 'prompt_level', 'defect_type']).size() #type:ignore

print_header("Defect Type Counts:")
print(defect_counts.head(10))

# %% Bar Chart for Defect Types by Model and Prompt Level
g = sns.catplot(x='prompt_level', y='count', hue='defect_type', col='model',
                data=defect_counts, kind='bar', height=4, aspect=1, col_wrap=2)
g.figure.subplots_adjust(top=0.85)
g.figure.suptitle("Häufigkeit von Defekt Typen nach Modell und Prompt Level")
g.set_axis_labels("Prompt Level", "Anzahl")
plt.savefig(f"{OUTPUT_PATH}frequency_defect_types_modelprompt.png")
plt.show()

# %% Frequency of Defect Types by Model (Aggregated Across All Prompt Levels)
defect_counts_model = df_defect_level.groupby(['model', 'defect_type']).size().reset_index(name='count') #type:ignore
print_header("Defekt Type Counts by Model:")
#print(defect_counts_model)
# Print summary table for each model.
for model in defect_counts_model['model'].unique():
    print(f"\nModel: {model}")
    print(defect_counts_model[defect_counts_model['model'] == model][['defect_type', 'count']]) #type:ignore #noqa

# %% Plotting the data
# Horizontal bar charts per model
g_model = sns.catplot(
    x='count',
    y='defect_type',
    col='model',
    data=defect_counts_model,
    kind='bar',
    height=4,
    aspect=1,
    col_wrap=2,
)

g_model.figure.subplots_adjust(top=0.85)
g_model.figure.suptitle("Häufigkeit von Defekt Typen nach Modell (aggregiert über Prompt Level)")
g_model.set_axis_labels("Anzahl", "Defekt Typ")
plt.savefig(f"{OUTPUT_PATH}frequency_defect_types_model.png")
plt.show()
# %% Frequency of Defect Types by Prompt Level (Aggregated Across All Models)
defect_counts_prompt = df_defect_level.groupby(['prompt_level', 'defect_type']).size().reset_index(name='count') #type:ignore #noqa
print_header("Defekt Type Counts by Prompt Level:")
#print(defect_counts_prompt)
# Print summary table for each model.
for prompt in defect_counts_prompt['prompt_level'].unique():
    print(f"\nPrompt Level: {prompt}")
    print(defect_counts_prompt[defect_counts_prompt['prompt_level'] == prompt][['defect_type', 'count']]) #type:ignore #noqa
# %% Plotting the data
# Horizontal bar charts per prompt level
g_prompt = sns.catplot(
    x='count',
    y='defect_type',
    col='prompt_level',
    data=defect_counts_prompt,
    kind='bar',
    height=4,
    aspect=1,
    col_wrap=2,
)

g_prompt.figure.subplots_adjust(top=0.85)
g_prompt.figure.suptitle("Häufigkeit von Defekt Typen nach Prompt Level (aggregiert über Modelle)")
g_prompt.set_axis_labels("Anzahl", "Defekt Typ")
plt.savefig(f"{OUTPUT_PATH}frequency_defect_types_prompt.png")
plt.show()




# %% Subtypes

# %% Analysis of Subtypes by Model

# Group by model, defect_type, and sub_type, and count the occurrences.
df_model_summary = df_defect_level.groupby(['model', 'defect_type', 'sub_type']).size().reset_index(name='count') #type:ignore

# Create a combined column for defect type and subtype.
df_model_summary['defect_combined'] = df_model_summary['defect_type'] + " - " + df_model_summary['sub_type']

print_header("Defect Subtype Counts by Model:")
# Print the summary table for each model.
for model in df_model_summary['model'].unique():
    print(f"\nModel: {model}")
    print(df_model_summary[df_model_summary['model'] == model][['defect_combined', 'count']]) #type:ignore #noqa

# %% Bar Chart for Defects per Model

# Get the unique models.
models = df_model_summary['model'].unique()

# calculate the total number of bars (defect categories) and total figure height
n_bars_list = [len(df_model_summary[df_model_summary['model'] == m]) for m in models]
total_height = sum(n_bars_list) * 0.6

# calculate maximum count value of defects
x_max = df_model_summary['count'].max()
x_max = math.ceil(x_max / 5) * 5  # Rounds up to nearest multiple of 5

# Create subplots with one row per model and set the height ratios to the number of bars
fig, axes = plt.subplots(nrows=len(models), figsize=(8, total_height), gridspec_kw={'height_ratios': n_bars_list}
)

if len(models) == 1:
    axes = [axes] # make axes iterable for one model

# Loop over each model's subplot
for ax, model in zip(axes, models):
    # Filter and sort the data for the current model
    df_temp = df_model_summary[df_model_summary['model'] == model].sort_values(by='count', ascending=False)

    # Plot a horizontal bar chart
    sns.barplot(x='count', y='defect_combined', data=df_temp, ax=ax, orient='h')
    ax.set_title(f"Defekt Typ - Subtyp Verteilung für Modell: {model}")
    ax.set_xlabel("Anzahl")
    #ax.set_ylabel("Defekt Typ - Subtyp")
    ax.set_ylabel("")
    ax.set_xlim(0,x_max) # Set max x-axis value to the same for all charts for comparability

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}defect_distribution_models.png")
plt.show()

# %% Analysis of Subtypes by Prompt Level

# Group by prompt_level, defect_type, and sub_type, and count the occurrences.
df_prompt_summary = df_defect_level.groupby(['prompt_level', 'defect_type', 'sub_type']).size().reset_index(name='count') #type:ignore #noqa

# Create a combined column for defect type and subtype.
df_prompt_summary['defect_combined'] = df_prompt_summary['defect_type'] + " - " + df_prompt_summary['sub_type']

print_header("Defect Subtype Counts by Prompt Level:")
# Print the summary table for each prompt level.
for prompt in sorted(df_prompt_summary['prompt_level'].unique()):
    print(f"\nPrompt Level: {prompt}")
    print(df_prompt_summary[df_prompt_summary['prompt_level'] == prompt][['defect_combined', 'count']]) #type:ignore #noqa
# %% Bar Chart for Defects per Prompt Level

# Get the unique models.
prompt_levels = sorted(df_prompt_summary['prompt_level'].unique())

# calculate the total number of bars (defect categories) and total figure height
n_bars_list = [len(df_prompt_summary[df_prompt_summary['prompt_level'] == p]) for p in prompt_levels]
total_height = sum(n_bars_list) * 0.6

# calculate maximum count value of defects
x_max = df_prompt_summary['count'].max()
x_max = math.ceil(x_max / 5) * 5  # Rounds up to nearest multiple of 5

# Create subplots with one row per model and set the height ratios to the number of bars
fig, axes = plt.subplots(nrows=len(prompt_levels), figsize=(8, total_height), gridspec_kw={'height_ratios': n_bars_list}
)

if len(prompt_levels) == 1:
    axes = [axes] # make axes iterable for one model

# Loop over each model's subplot
for ax, prompt in zip(axes, prompt_levels):
    # Filter and sort the data for the current model
    df_temp = df_prompt_summary[df_prompt_summary['prompt_level'] == prompt].sort_values(by='count', ascending=False)

    # Plot a horizontal bar chart
    sns.barplot(x='count', y='defect_combined', data=df_temp, ax=ax, orient='h')
    ax.set_title(f"Defekt Typ - Subtyp Verteilung für Prompt Level: {prompt}")
    ax.set_xlabel("Anzahl")
    #ax.set_ylabel("Defekt Typ - Subtyp")
    ax.set_ylabel("")
    ax.set_xlim(0,x_max) # Set max x-axis value to the same for all charts for comparability

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}defect_distribution_prompts.png")
plt.show()
# %% Heatmaps

# %% Defect types by Prompt Level
defect_counts_prompt = df_defect_level.groupby(['prompt_level', 'defect_type']).size().reset_index(name='count') #type:ignore #noqa

# Pivot the data so that rows are defect types and columns are prompt levels.
pivot_prompt = defect_counts_prompt.pivot(index='defect_type', columns='prompt_level', values='count').fillna(0)

# Plot a heatmap.
plt.figure(figsize=(5, 4))
sns.heatmap(pivot_prompt, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap aller Defekt Typen nach Prompt Level")
plt.xlabel("Prompt Level")
plt.ylabel("")
plt.savefig(f"{OUTPUT_PATH}heatmap_defect_types_prompt.png")
plt.show()

# %% Defect types by Models
# Aggregate counts by model and defect type.
defect_counts_model = df_defect_level.groupby(['model', 'defect_type']).size().reset_index(name='count') #type:ignore #noqa

# Pivot the data so that rows are defect types and columns are models.
pivot_model = defect_counts_model.pivot(index='defect_type', columns='model', values='count').fillna(0)

# Plot the heatmap.
plt.figure(figsize=(5, 3))
sns.heatmap(pivot_model, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap aller Defekt Typen nach Modell")
plt.xlabel("")
plt.ylabel("")
plt.savefig(f"{OUTPUT_PATH}heatmap_defect_types_model.png")
plt.show()

# %% Defect types vs. Subtypes
# Aggregate counts by defect_type and sub_type.
defect_types_subtypes = df_defect_level.groupby(['defect_type', 'sub_type']).size().reset_index(name='count') #type:ignore #noqa

# Pivot the data to create a matrix where rows are defect types and columns are subtypes.
pivot_defect_subtype = defect_types_subtypes.pivot(index='defect_type', columns='sub_type', values='count').fillna(0)

# Manually set column order to better represent the order or defect types for readability
desired_order = ['conjunctions', 'brackets', 'indicator_repetition', 'punctuation', 'uniform', 'identical', 'no_means']
pivot_defect_subtype = pivot_defect_subtype.reindex(columns=desired_order)

# Plot the heatmap.
plt.figure(figsize=(5, 3))
sns.heatmap(pivot_defect_subtype, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap aller Defekt Typen und Subtypen")
plt.xlabel("Sub Typ")
plt.ylabel("Defekt Typ")
plt.savefig(f"{OUTPUT_PATH}heatmap_defect_types_subtypes.png")
plt.show() # not really helpful

# %%
# Aggregate defect counts by model, prompt_level, and defect_type.
df_group = df_defect_level.groupby(['model', 'prompt_level', 'defect_type']).size().reset_index(name='count') #type:ignore #noqa

# Get the unique models.
models = df_group['model'].unique()

# Create a heatmap for each model.
for model in models:
    df_temp = df_group[df_group['model'] == model]
    # Pivot so that rows are defect types and columns are prompt levels.
    pivot_model = df_temp.pivot(index='defect_type', columns='prompt_level', values='count').fillna(0)

    plt.figure(figsize=(5, 3))
    sns.heatmap(pivot_model, annot=True, fmt=".0f",cmap="YlGnBu")
    plt.title(f"Defekt Typen nach Prompt Level für Modell: {model}")
    plt.xlabel("Prompt Level")
    plt.ylabel("")
    plt.savefig(f"{OUTPUT_PATH}heatmap_prompts_{model}.png")
    plt.show()


# %% Kruskal–Wallis Test: Comparing defect_count across prompt levels within each model
print_header("Kruskal-Wallis Test: Defect Count across Prompt Levels within each Model:")

models = df_story_level['model'].unique()
for model in models:
    model_data = df_story_level[df_story_level['model'] == model]
    groups = [group['defect_count'].values for _, group in model_data.groupby('prompt_level')]
    group_sizes = [len(group) for group in groups]
    print(f"For model {model}, group sizes by prompt level: {group_sizes}")
    stat, p_value = kruskal(*groups)
    print(f"Kruskal–Wallis Test for model {model} across prompt levels:\nStatistic = {stat}, p-value = {p_value}\n")

# %% Chi-Square Test for Independence between defect types and model

# Create a contingency table for defect types by model
contingency_table = pd.crosstab(df_defect_level['model'], df_defect_level['defect_type'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print_header("Chi-Square Test for Independence between defect types and model:")
print("Contingency Table:")
print(contingency_table)
print("Expected Frequencies:") # below 5 is bad
print(expected)
print(f"Chi2 statistic: {chi2}")
print(f"p-value: {p}")
print(f"Degrees of freedom: {dof}")

# %% Chi-Square Test for Independence between defect types and and prompts (across all models)

# Create a contingency table: rows are prompt levels, columns are defect types
contingency_prompt = pd.crosstab(df_defect_level['prompt_level'], df_defect_level['defect_type'])

# Run the chi-square test of independence
chi2_prompt, p_prompt, dof_prompt, expected_prompt = chi2_contingency(contingency_prompt)

print_header("Chi-Square Test for Independence between defect types and prompt level:")
print("Contingency Table:")
print(contingency_prompt)
print("Expected Frequencies:")
print(expected_prompt)
print(f"Chi2 statistic: {chi2_prompt}")
print(f"p-value: {p_prompt}")
print(f"Degrees of freedom: {dof_prompt}")

# %% Chi-Square Test for Independence between Prompt Level and Defect Type Within Each Model
models = df_defect_level['model'].unique()

print_header("Chi-Square Test for Independence between Prompt Level and Defect Type Within Each Model:")
for model in models:
    subset = df_defect_level[df_defect_level['model'] == model]
    contingency_model = pd.crosstab(subset['prompt_level'], subset['defect_type'])

    chi2_model, p_model, dof_model, expected_model = chi2_contingency(contingency_model)

    print(f"\nModel: {model}")
    print("Contingency Table:")
    print(contingency_model)
    print("Expected Frequencies:")
    print(expected_model)
    print(f"Chi2 statistic: {chi2_model}")
    print(f"p-value: {p_model}")
    print(f"Degrees of freedom: {dof_model}")

# %% Between Defect Subtype and Model
# Create a contingency table: rows are models, columns are defect subtypes
contingency_subtype = pd.crosstab(df_defect_level['model'], df_defect_level['sub_type'])

chi2_subtype, p_subtype, dof_subtype, expected_subtype = chi2_contingency(contingency_subtype)

print_header("Chi-Square Test for Independence between defect subtypes and model:")
print("Contingency Table:")
print(contingency_subtype)
print("Expected Frequencies:")
print(expected_subtype)
print(f"Chi2 statistic: {chi2_subtype}")
print(f"p-value: {p_subtype}")
print(f"Degrees of freedom: {dof_subtype}")

# %% Between Defect Presence and Model
# Create a binary variable indicating whether the story has any defects
df_story_level['has_defect'] = df_story_level['defect_count'] > 0

print_header("Chi-Square Test for Independence between Between Defect Presence and Model:")
# Build a contingency table: rows are models, columns indicate whether a story has defects
contingency_defect = pd.crosstab(df_story_level['model'], df_story_level['has_defect'])
print("Contingency Table for Model vs. Defect Presence:")
print(contingency_defect)

# Perform the chi-square test
chi2_defect, p_defect, dof_defect, expected_defect = chi2_contingency(contingency_defect)

print("\nChi-Square Test for Independence between Model and Defect Presence:")
print(f"Chi2 statistic: {chi2_defect}")
print(f"p-value: {p_defect}")
print(f"Degrees of freedom: {dof_defect}")
print("Expected Frequencies:")
print(expected_defect)

# %% Defect Presence and Prompt Level
# # Build a contingency table: rows are prompt levels, columns indicate whether a story has defects
contingency_defect_prompt = pd.crosstab(df_story_level['prompt_level'], df_story_level['has_defect'])
print_header("Contingency Table for Prompt Level vs. Defect Presence:")
print(contingency_defect_prompt)

# Perform the chi-square test
chi2_defect_prompt, p_defect_prompt, dof_defect_prompt, expected_defect_prompt = chi2_contingency(contingency_defect_prompt)

print("\nChi-Square Test for Independence between Prompt Level and Defect Presence:")
print(f"Chi2 statistic: {chi2_defect_prompt}")
print(f"p-value: {p_defect_prompt}")
print(f"Degrees of freedom: {dof_defect_prompt}")
print("Expected Frequencies:")
print(expected_defect_prompt)

# %% Stop logging
if not TEST_RUN:
	sys.stdout.close() # needs to be manually closed for some reason when running as REPL
