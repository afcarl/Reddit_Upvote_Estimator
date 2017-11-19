#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import lib
from reddit_scraper import scrape_subreddit


def extract():
    # TODO Docstring

    # Extract all posts for given subreddit, going back given number of days
    scrape_subreddit(lib.get_conf('subreddit'), lib.get_conf('history_num_days'))

    # TODO Load embedding matrix
    pass


def transform(embedding_matrix, posts):
    # TODO Docstring

    # TODO Bin number of upvotes

    # TODO Simple pre-processing, lemmatization, and stopword removal

    # TODO Convert text to indices

    # TODO One hot encode response (up votes)

    pass

def model(embedding_matrix, posts):
    # TODO Docstring

    # TODO Reference variables

    # TODO Create train / test split

    # TODO Architecture variables: Input and output dimmensions

    # TODO Create and compile architecture

    pass


def load(embedding_matrix, posts, model):
    # TODO Docstring

    # TODO Output observations with true labels, expected labels

    # TODO Output summary metrics
    pass


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    embedding_matrix, posts = extract()

    embedding_matrix, posts, model = transform(embedding_matrix, posts)

    load(embedding_matrix, posts, model)

    pass


# Main section
if __name__ == '__main__':
    main()
