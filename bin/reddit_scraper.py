#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import datetime
import pprint

import praw
import sys

from praw.models import Submission

import lib


def scrape_subreddit(subreddit_name, num_days):
    # TODO Docstring

    logging.info('Beginning Reddit scraper, for subreddit: {}, and num_days: {}'.format(subreddit_name, num_days))

    # Reference variables
    parsed_submission_agg = list()

    # Create connection. For details, see https://www.reddit.com/prefs/apps/

    logging.info('Creating reddit connection')
    reddit = praw.Reddit(client_id=lib.get_conf('client_id'),
                         client_secret=lib.get_conf('client_secret'),
                         user_agent='upvote_estimator:0.0.1')

    # Find correct subreddit
    logging.info('Searching for subreddit: {}'.format(subreddit_name))
    subreddit = reddit.subreddit(subreddit_name)
    logging.debug('Searched for subreddit: {}, found subreddit: {}, {}'.format(subreddit_name, subreddit.display_name, subreddit.title))

    # Compute correct time range (current datetime - num_days to current datetime)
    end_datetime = datetime.datetime.utcnow()
    start_datetime = end_datetime - datetime.timedelta(days=num_days)
    logging.debug('Time range: {} to {}'.format(start_datetime, end_datetime))

    # Iterate through posts chronologically
    for index, submission in enumerate(subreddit.new()):
        logging.info('Working number {}, submission: '.format(index, submission))
        parsed_submission = submission_parser(submission)
        parsed_submission_agg.append(parsed_submission)


    # TODO Add info from each post to aggregator

    # TODO Create DataFrame from pulled data

    # TODO Return

    pass

def submission_parser(submission):
    # Reference variables
    agg = dict()

    fields = ['author', 'spoiler', 'over_18', 'url', 'id', 'name', 'subreddit_name_prefixed', 'score', 'ups', 'downs', 'likes', 'num_comments', 'title', 'selftext']
    for field in fields:
        value = submission.__dict__.get(field, None)

        try:
            agg[field] = unicode(value).encode('ascii', errors='ignore') if value is not None else None

        except:
            agg[field] = None
            logging.warn('Issue encoding field: {}, for submission: {}'.format(field, submission))

    return agg
