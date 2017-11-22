#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import cPickle
import csv
import logging
import os

import numpy
from gensim.utils import simple_preprocess

import lib
import models
import resources
from reddit_scraper import scrape_subreddit


def extract():
    # TODO Docstring

    logging.info('Begin extract')

    # Extract all posts for given subreddit, going back given number of days
    logging.info('Downloading submissions from Reddit')
    observations = scrape_subreddit(lib.get_conf('subreddit'), lib.get_conf('history_num_days'))
    logging.info('Found {} submissions'.format(len(observations.index)))

    # Load embedding matrix
    resources.download_embedding()
    embedding_matrix, word_to_index = resources.create_embedding_matrix()
    logging.info('word_to_index max index: {}'.format(max(word_to_index.values())))

    logging.info('End extract')
    lib.archive_dataset_schemas('extract', locals(), globals())
    return embedding_matrix, word_to_index, observations


def transform(embedding_matrix, word_to_index, observations):
    # TODO Docstring
    logging.info('Begin transform')

    # TODO Shuffle upvotes234

    # Response: Bin number of upvotes
    bins_maxes = [0, 10, 50, 100, numpy.inf]

    observations['bin_max'] = map(lambda x: bins_maxes[x], numpy.digitize(observations['ups'].tolist(), bins=bins_maxes))
    logging.info('Bin breakdown: \n{}'.format(observations['bin_max'].value_counts()))

    # Response: One hot encode response (up votes)
    label_encoder = lib.create_label_encoder(sorted(set(observations['bin_max'])))
    observations['response'] = observations['bin_max'].apply(lambda x: label_encoder[x])

    # Input: Create text field
    observations['text'] = observations['title'] + ' ' + observations['selftext']

    # Input: Simple pre-processing
    observations['tokens'] = observations['text'].apply(simple_preprocess)

    # Input: Convert text to indices
    observations['indices'] = observations['tokens'].apply(lambda token_list: map(lambda token: word_to_index[token],
                                                                                  token_list))

    # Input: Pad indices list with zeros, so that every article's list of indices is the same length
    observations['padded_indices'] = observations['indices'].apply(lib.pad_sequence)

    # Set up modeling input
    observations['modeling_input'] = observations['padded_indices']

    logging.info('End transform')
    lib.archive_dataset_schemas('transform', locals(), globals())
    return embedding_matrix, word_to_index, observations, label_encoder

def model(embedding_matrix, word_to_index, observations, label_encoder):
    # TODO Docstring
    logging.info('Begin model')

    lib.archive_dataset_schemas('model', locals(), globals())

    # Create train and test data sets
    train_test_mask = numpy.random.random(size=len(observations.index))
    num_train = sum(train_test_mask < .8)
    num_validate = sum(train_test_mask >= .8)
    logging.info('Proceeding w/ {} train observations, and {} test observations'.format(num_train, num_validate))

    x_train = observations['modeling_input'][train_test_mask < .8].tolist()
    y_train = observations['response'][train_test_mask < .8].tolist()
    x_test = observations['modeling_input'][train_test_mask >= .8].tolist()
    y_test = observations['response'][train_test_mask >= .8].tolist()

    # Add train / validate label to observations
    observations['training_step'] = map(lambda x: 'train' if x < .8 else 'test', train_test_mask)

    # Convert x and y vectors to numpy objects
    x_train = numpy.array(x_train, dtype=object)
    y_train = numpy.array(y_train)
    x_test = numpy.array(x_test, dtype=object)
    y_test = numpy.array(y_test)

    logging.info('x_train shape: {}, y_train shape: {}, '
                 'x_test shape: {}, y_test shape: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    # If required, train model
    if lib.get_conf('train_model'):
        logging.info('Creating and training model')

        embedding_input_length = x_train.shape[1]
        output_shape = y_train.shape[1]

        classification_model = models.gen_conv_model(embedding_input_length, output_shape, embedding_matrix, word_to_index)

        # Train model
        classification_model.fit(x_train, y_train, batch_size=512, epochs=5, validation_data=(x_test, y_test))

        logging.info('Finished creating and training model')
    else:
        classification_model = None

    # TODO Validate model

    # Add model prediction to observations
    preds = classification_model.predict(numpy.array(observations['modeling_input'].tolist(), dtype=object))
    observations['max_probability'] = map(max, preds)
    observations['modeling_prediction'] = map(lambda x: lib.prop_to_label(x, label_encoder), preds)

    # Archive schema and return
    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End model')

    return embedding_matrix, word_to_index, observations, label_encoder, classification_model


def load(embedding_matrix, word_to_index, observations, label_encoder, network):
    # TODO Docstring

    # TODO Output observations with true labels, expected labels
    posts_csv_path = os.path.join(lib.get_temp_dir(), 'posts.csv')
    observations.to_csv(path_or_buf=posts_csv_path, index=False, quoting=csv.QUOTE_ALL)
    logging.info('Dataset written to file: {}'.format(posts_csv_path))
    print('Dataset written to file: {}'.format(posts_csv_path))

    # Serialize model
    model_path = os.path.join(lib.get_temp_dir(), 'keras_model.h5')
    network.save(model_path)
    logging.info('Model written to file: {}'.format(model_path))
    print('Model written to file: {}'.format(model_path))

    # Serialize label encoder
    label_encoder_path = os.path.join(lib.get_temp_dir(), 'label_encoder.pkl')
    cPickle.dump(label_encoder, open(label_encoder_path, 'w+'))


    # TODO Output summary metrics
    pass


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # Extract
    embedding_matrix, word_to_index, observations = extract()

    # Transform
    embedding_matrix, word_to_index, observations, label_encoder = transform(embedding_matrix, word_to_index, observations)

    # Model
    embedding_matrix, word_to_index, observations, label_encoder, network = model(embedding_matrix, word_to_index, observations, label_encoder)

    # Load
    load(embedding_matrix, word_to_index, observations, label_encoder, network)

    pass


# Main section
if __name__ == '__main__':
    main()
