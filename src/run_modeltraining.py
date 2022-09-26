# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""BERT NER for Document Automation
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=C0301, E0401, E0611

# Import the required libraries required for data analysis and building model
import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np  # linear algebra
from transformers import TFBertModel
from transformers import BertTokenizerFast
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from utils import process_data, tokenize, create_model

# Input data files are available in the "./data/" directory

# # **NER USING BERT**
#
# ## The goal of a named entity recognition (NER) system is to identify all
# textual mentions of the named entities. This can be broken down into two
# sub-tasks: identifying the boundaries of the NE, and identifying its type.
#
# Named entity recognition is a task that is well-suited to the type of
# classifier-based approach. In particular, a tagger can be built that labels
# each word in a sentence using the IOB format, where chunks are labelled by
# their appropriate type.
#
# The IOB Tagging system contains tags of the form:
#
# B - {CHUNK_TYPE} – for the word in the Beginning chunk
# I - {CHUNK_TYPE} – for words Inside the chunk
# O – Outside any chunk
# The IOB tags are further classified into the following classes –
#
# geo = Geographical Entity
# org = Organization
# per = Person
# gpe = Geopolitical Entity
# tim = Time indicator
# art = Artifact
# eve = Event
# nat = Natural Phenomenon

# The maximum length of the sentence used for model building
MAX_LEN = 128

# MAX_LEN = 64

# The global variable which is used to store the length of a data structure
LENGTH = 0

if __name__ == "__main__":
    # The main function body which takens the varilable number of arguments

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")

    logger.disabled = True

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=32,
                        help='batch size exampes: 30, 60, 90, 100, 120')

    parser.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for training')

    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable intel tensorflow optimizations, \
                            default is 0')

    parser.add_argument('--save_model_path',
                        '--save_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path to save the model')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()
    if FLAGS.batch_size < 0:
        logger.info("The parameter batch size value is invalid, try with valid \
            value\n")
        sys.exit(1)

    if FLAGS.dataset_file is None:
        logger.info("The dataset file is invalid, try with valid file name\n")
        sys.exit(1)

    if FLAGS.intel == 1:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    else:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if FLAGS.save_model_path is None:
        logger.info("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        if FLAGS.intel != 1:
            save_model_path = FLAGS.save_model_path + "/stock/"
        else:
            save_model_path = FLAGS.save_model_path + "/intel/"

    # Handle Exceptions for the user entries
    try:
        if not os.path.exists(FLAGS.dataset_file):
            logger.info("Dataset file path Not Found!!")
            raise FileNotFoundError
    except FileNotFoundError:
        logger.info("Please check the Path provided!")
        sys.exit()

    # Tokenizer for getting token ids and attention_masks
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # # **Loading Dataset**
    sentence, pos, tag, enc_pos, enc_tag = process_data(FLAGS.dataset_file)

    # To use the validate data for testing, using same training data
    # and validatation data during training
    '''
    total_size = len(tag)
    training_size = int(total_size * 0.9)
    testing_size = int(total_size * 0.1)
    X_train = sentence[0:training_size]
    X_test = sentence[training_size:(training_size + testing_size)]
    y_train = tag[0:training_size]
    y_test = tag[training_size:(training_size + testing_size)]
    '''
    X_train, X_test, y_train, y_test = train_test_split(sentence, tag,
                                                        random_state=42,
                                                        test_size=0.1)

    # Get the token ids and attention_masks for training data
    input_ids, attention_mask = tokenize(X_train, tokenizer, max_len=MAX_LEN)

    # Get the token ids and attention_masks for test data
    val_input_ids, val_attention_mask = tokenize(X_test, tokenizer,
                                                 max_len=MAX_LEN)

    # # Testing Padding and Truncation Length
    was = []
    LENGTH = len(input_ids)
    for index in range(LENGTH):
        was.append(len(input_ids[index]))
    set(was)

    # Train Padding
    train_tag = []
    LENGTH = len(y_train)
    for index in range(LENGTH):
        train_tag.append(np.array(
                        y_train[index] + [0] *
                        (MAX_LEN - len(y_train[index]))))

    # Checking Padding Length
    was = []
    LENGTH = len(train_tag)
    for index in range(LENGTH):
        was.append(len(train_tag[index]))
    set(was)

    # TEST: Checking Padding and Truncation length's
    # Test Padding
    test_tag = []
    LENGTH = len(y_test)
    for index in range(LENGTH):
        test_tag.append(np.array(y_test[index] + [0] *
                        (MAX_LEN-len(y_test[index]))))

    # Checking Padding Length
    was = []
    LENGTH = len(test_tag)
    for index in range(LENGTH):
        was.append(len(test_tag[index]))
    set(was)

    # # **Building BERT Model : Transfer Learning**
    base_bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    model = create_model(base_bert_model, MAX_LEN)

    model.summary()

    # # **Traning Model**
    early_stopping = EarlyStopping(mode='min', patience=5)

    t2 = time.time()
    # 25 epochs was taking more time, hence reduced number of epochs \
    # is used to verify the performanceof IntelOneAPI
    history_bert = model.fit(
                        [input_ids, attention_mask],
                        np.array(train_tag),
                        validation_data=(
                            [val_input_ids, val_attention_mask],
                            np.array(test_tag)),
                        epochs=2, batch_size=FLAGS.batch_size,
                        callbacks=early_stopping, verbose=False)
    t3 = time.time() - t2

    model.save_weights(
        save_model_path + "/model_b" + str(FLAGS.batch_size)
        + "/model_checkpoint")

    s = f"""
    {'-'*40}
    # Model Training
    # Time (in seconds): {t3}
    # Batch size: {FLAGS.batch_size}
    # Model saved path: {(save_model_path + "/model_b" + str(FLAGS.batch_size)
                    + "/model_checkpoint")}
    {'-'*40}
    """
    print(s)
    logger.info("\n")
    logger.info(history_bert.history)
    logger.info("\n")
