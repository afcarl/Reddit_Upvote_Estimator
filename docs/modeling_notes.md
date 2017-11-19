# Modeling Notes

## 2017-11-17

Reddit

 - PRAW now requires authentication. This will make a quick start guide to this project difficult. 
 - Reviewing `from praw.models import Submission` parameters for relevant info. 
 - Meta:
   - `author` (object, will need to be unraveled)
   - `spoiler`: boolean
   - `over_18`: boolean
   - `url`: string
   - `id`: string
   - `name`: string, contains id
   - `subreddit_name_prefixed`: string
 - Popularity:
   - `score`: int
   - `ups`: int
   - `downs`: int
   - `likes`: Unknown signature
   - `num_comments`: int
 - Text
   - `title`: string
   - `selftext`: string
   
## 2017-11-18

Load

 - Including functionality in `load` to write data set to file for later review / analysis

Extract

 - Setting up minimal framework to extract loadings
 - Moving `word_to_index` UNK mapping from extract code to `resources/create_embedding_matrix`
 - Looking up how to bin respose
 - Using more consistent labels for `text` and `response`. This should help re-usability
 
Model

 - Creating minimal model
 - Rather than having `gen_model` method accept the data set, I'm just passing the necessary input / output shape info

TODO: Backlog

 - Refactor Embedding code: Simple wrapper class that:
   - Downloads necessary files
   - Accepts iterable of input strings and outputs array of list of indices
   - Created appropriate Embedding layer
   - Accepts iterable of response variables and outputs one hot encoded variables
   - Creates appropriate output layer
 - Create more applicable model 
 - Add functionality to add prediction to observations
 
Prioritized backlog: 

 - Add functionality to add prediction to observations
 - Create more applicable model
 - Refactor Embedding code: Simple wrapper class that:
