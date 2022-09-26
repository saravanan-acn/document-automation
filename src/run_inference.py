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
from transformers import TFBertModel
from transformers import BertTokenizerFast

from utils import process_data, tokenize, create_model, testing, batch_testing


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
                        help='dataset file for testing')

    parser.add_argument('-i',
                        '--intel',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable intel tensorflow optimizations, \
                            default is 0')

    parser.add_argument('--model_path',
                        '--model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path of the model')

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

    if FLAGS.model_path is None:
        logger.info("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        model_path = FLAGS.model_path

    # Handle Exceptions for the user entries
    try:
        if not os.path.exists(FLAGS.dataset_file):
            logger.info("Dataset file path Not Found!!")
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error("Please check the Path provided!")
        sys.exit()

    # Tokenizer for getting token ids and attention_masks
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # # **Loading Dataset**
    sentence, pos, tag, enc_pos, enc_tag = process_data(FLAGS.dataset_file)

    # X_train, X_test, y_train, y_test = train_test_split(sentence, tag,
    #                                                    random_state=42,
    #                                                    test_size=0.01)

    # To use the validation data for testing, fixing the training data
    # and validatation data during training
    total_size = len(tag)
    X_test = sentence
    y_test = tag
    print(f"Testing dataset size: {total_size}")

    # Get the token ids and attention_masks for test data
    val_input_ids, val_attention_mask = tokenize(X_test, tokenizer,
                                                 max_len=MAX_LEN)

    # # **Building BERT Model : Transfer Learning**
    base_bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    model = create_model(base_bert_model, MAX_LEN)

    model.load_weights(model_path)

    testing(model, val_input_ids[0], val_attention_mask[0], y_test[0])

    real_time_inference = testing(
                                model,
                                val_input_ids[0],
                                val_attention_mask[0],
                                y_test[0])
    # Average batch inference time for test data
    t3 = time.time()
    TOTAL_INFERENCE_TIME = 0.0
    ACCURACY = 0.0
    for idx in range(3):
        inference_time, acc = batch_testing(
                                    model,
                                    val_input_ids,
                                    val_attention_mask,
                                    y_test,
                                    FLAGS.batch_size)
        TOTAL_INFERENCE_TIME += inference_time
        ACCURACY += acc

    # print("Average batch inference time is {} for batch size {}".format(
    #            (FLAGS.batch_size * ((TOTAL_INFERENCE_TIME/3)/len(y_test))),
    #            FLAGS.batch_size))

    batch_size = FLAGS.batch_size
    total_batches = len(y_test)/batch_size
    s = f"""
    {'-'*40}
    # Model Inference details:
    # Real time inference (in seconds): {real_time_inference}
    # Average batch inference:
    #   Time (in seconds): {((TOTAL_INFERENCE_TIME/3)/total_batches)}
    #   Batch size: {batch_size}
    #   Accuracy: {ACCURACY/3}
    {'-'*40}
    """
    print(s)

    logger.info("Done.\n")
