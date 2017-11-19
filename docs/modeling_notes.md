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
   