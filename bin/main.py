#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging

import os

import lib
from reddit_scraper import scrape_subreddit


def extract():
    # TODO Docstring

    # Extract all posts for given subreddit, going back given number of days
    posts = scrape_subreddit(lib.get_conf('subreddit'), lib.get_conf('history_num_days'))

    # TODO Load embedding matrix
    embedding_matrix = None
    word_to_index = None
    lib.archive_dataset_schemas('extract', locals(), globals())
    return embedding_matrix, word_to_index, posts


def transform(embedding_matrix, word_to_index, posts):
    # TODO Docstring

    # TODO Bin number of upvotes

    # TODO Simple pre-processing, lemmatization, and stopword removal

    # TODO Convert text to indices

    # TODO One hot encode response (up votes)

    return embedding_matrix, word_to_index, posts

def model(embedding_matrix, word_to_index, posts):
    # TODO Docstring

    # TODO Reference variables

    # TODO Create train / test split

    # TODO Architecture variables: Input and output dimmensions

    # TODO Create and compile architecture

    return embedding_matrix, word_to_index, posts, None


def load(embedding_matrix, word_to_index, posts, network):
    # TODO Docstring

    # TODO Output observations with true labels, expected labels
    posts_csv_path = os.path.join(lib.get_temp_dir(), 'posts.csv')
    posts.to_csv(path_or_buf=posts_csv_path, index=False)
    logging.info('Dataset written to file: {}'.format(posts_csv_path))
    print('Dataset written to file: {}'.format(posts_csv_path))

    # TODO Output summary metrics
    pass


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    embedding_matrix, word_to_index, posts = extract()

    embedding_matrix, word_to_index, posts = transform(embedding_matrix, word_to_index, posts)

    embedding_matrix, word_to_index, posts, network = model(embedding_matrix, word_to_index, posts)

    load(embedding_matrix, word_to_index, posts, network)

    pass


# Main section
if __name__ == '__main__':
    main()
