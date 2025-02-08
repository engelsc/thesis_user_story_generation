# %% imports
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt # noqa
from langdetect import detect 	# pyright: ignore
import langid 					# pyright: ignore
from textblob import TextBlob 	# pyright: ignore

# %% load dataset
cleaned_data_path = 'threads_reviews_cleaned.csv'

raw_data = pd.read_csv('threads_reviews.csv')

raw_data = raw_data.iloc[:, :-2] # drop the last two empty columns, only necessary for downloaded file
#print(raw_data)
# %% alternative: load dataset from kaggle directly
#import kagglehub # pyright: ignore # noqa

#kaggle_path = kagglehub.dataset_download("saloni1712/threads-an-instagram-app-reviews")
#raw_data =pd.read_csv(kaggle_path + '/threads_reviews.csv')
#print(raw_data)


# %% drop all non Google Play reviews
raw_data = raw_data[raw_data['source'] == 'Google Play']
print(raw_data.shape)


# %% add helper column for word count
def word_count(text: str):
	return len(str(text).split())

raw_data['word_count'] = raw_data['review_description'].apply(word_count) # add helper col
raw_data.to_csv('raw_data_word_count', index=False)

# %% remove all short reviews
min_word_count = 15
max_word_count = 60

data_no_shorts = raw_data[raw_data['word_count'] >= min_word_count]
data_no_long_or_short = data_no_shorts[data_no_shorts['word_count'] < max_word_count]

data_no_long_or_short = data_no_long_or_short.drop(columns=['word_count']) # helper column no longer needed

print(data_no_long_or_short.shape)
#data_no_shorts.to_csv(cleaned_data_path, index=False)

# %% drop duplicates
data_no_duplicates = data_no_long_or_short.drop_duplicates(subset='review_description', keep='first')
print(data_no_duplicates.shape)
#data_no_duplicates.to_csv(cleaned_data_path, index=False)

# %% remove non-english reviews
def is_english_langdetect(text: str):
	try:
		return detect(text) == 'en'
	except: # noqa
		return False

def is_english_langid(text: str):
	try:
		lang, _ = langid.classify(text)
		return lang == 'en'
	except: # noqa
		return False


data = data_no_duplicates
data_only_english = data[data['review_description'].apply(is_english_langdetect) & data['review_description'].apply(is_english_langid)]

print(data_only_english.shape) # not fully deterministic. the algorithms always deviate slightly
#data_only_english.to_csv(cleaned_data_path, index=False)

# %% Wait for execution above

# %% drop data past 5000 entries because of data quality issues
data = data_only_english.iloc[:5000]
print(data.shape)
#data.to_csv(cleaned_data_path, index=False)

# %% sentiment analysis

# outputs a tuple of [polarity, subjectivity]
def sentiment_score(text: str):
	blob = TextBlob(text)
	return blob.sentiment.polarity # pyright: ignore

def subjectivity_score(text: str):
	blob = TextBlob(text)
	return blob.sentiment.subjectivity # pyright: ignore

data['sentiment_polarity'] = data['review_description'].apply(sentiment_score)

# %% part 2 - not used anymore
#data['sentiment_subjectivity'] = data['review_description'].apply(subjectivity_score)
#print(data)
#data.to_csv(cleaned_data_path, index=False)

# %% remove overly positive 5 star reviews
condition = ( data['sentiment_polarity'] >= 0.65 ) & ( data['rating'] == str(5) )

data_no_overly_positive = data.drop( data[ condition ].index )

print(data_no_overly_positive.shape)
#data_no_overly_positive.to_csv(cleaned_data_path, index=False)

# %% remove CEO related or hateful reviews
data = data_no_overly_positive
keywords = ['mark', 'zuck', 'zuckerberg', 'zucky', "zuck's", 'elon', "elon's", 'musk', 'treason', 'facebook',
			'free speech', 'twitter', 'useless', 'worst', 'e*on', 'm*sk', 'kerberg']

def is_polarized(text: str):
	words = text.lower().split()
	return any(word in keywords for word in words)

data_no_ceos = data[ ~data['review_description'].apply(is_polarized) ]

print(data_no_ceos.shape)
#data_no_ceos.to_csv(cleaned_data_path, index=False)
# %% drop unused columns for further use
data = data_no_ceos

columns_to_drop = ['source', 'rating','review_date', 'sentiment_polarity']
cleaned_data = data.drop(columns=columns_to_drop)

cleaned_data.to_csv(cleaned_data_path, index=False)
# %% test sampling

data = pd.read_csv(cleaned_data_path)

# test sampling for 4 LLMs and 3 prompts each -> 16 sets of non-repeating data + 200 for experiments
amount_sample_sets = 16 # changed because 17 had already been generated at this point
sample_size = 200
random_state = 42
sub_folder = 'sample_sets'

shuffled_data = data.sample(frac = 1, random_state = random_state) # shuffles the entire dataframe with random seed

sample_sets = [shuffled_data.iloc[i * sample_size : (i + 1) * sample_size] for i in range(amount_sample_sets)]

if not os.path.exists(sub_folder):
	os.makedirs(sub_folder)

for i, sample_set in enumerate(sample_sets):
	file_name = f'sample_set_{i + 1:02}.csv'
	file_path = os.path.join(sub_folder, file_name)
	sample_set.to_csv(file_path, index = False, quoting = csv.QUOTE_ALL, quotechar = '"') # making sure all strings are quoted
