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

## 2017-11-20

Archiving

 - Goals: 
   - Archive probabilities, classification, classification as index
   - Serialize model
   - Serialize label encoder
 - Only saving highest probability, classification as text
 - Writing method to convert Kpdateeras output back to label
 - Changing csv writer to quote all, due to raw text in dataset
 - Following [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for serializing a model
 - Everything serialized to a path on my local machine: `/var/folders/c4/brlc3vgn5v1_pvdq0f57ws2m0000gn/T/reddit_fzJ5ra`
 
## 2017-11-21

More applicable model

Backlog:

 - Vary number of layers
 - Change / modify optimizer
 - Identify other common text classification architectures
 - Perform regression instead
 - Change to another subreddit
 - Custom embedding / add tags to delineate title
 
Prioritized backlog

 - Change / modify optimizer
 - Vary number of layers
 - Change to another subreddit
   - Possibly [https://www.reddit.com/r/todayilearned/][https://www.reddit.com/r/todayilearned/]
 