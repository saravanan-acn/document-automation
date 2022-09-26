# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""BERT NER for Document Automation
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=C0301, E0401, E0611, E1129

# Import the required libraries required for data analysis and building model
import os
import sys
import time
import logging
import argparse
import warnings
from statistics import mean
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast

from utils import process_data, tokenize

# The maximum length of the sentence used for model building
MAX_LEN = 128


class DatasetBatch():
    """Creating Dataset class for getting Image and labels"""
    def __init__(self, sent_val_input_ids, sent_val_attention_mask,
                 actual_y_test):

        list_length = len(sent_val_input_ids)
        val_input_list = []
        val_attention_list = []
        for idx in range(list_length):
            test_input = sent_val_input_ids[idx].reshape(1, MAX_LEN)
            test_attention = sent_val_attention_mask[idx].reshape(1, MAX_LEN)
            val_input_list.append(test_input)
            val_attention_list.append(test_attention)

        np_val_input = np.array(val_input_list)
        np_val_attention = np.array(val_attention_list)
        np_val_input = np_val_input.reshape(np_val_input.shape[0], MAX_LEN)
        np_val_attention = np_val_attention.reshape(np_val_attention.shape[0],
                                                    MAX_LEN)

        '''
        #print ("shape 1: {}, shape 2 {}: ".format(np_val_input.shape,
            #np_val_input[0].shape))
        '''
        test_val_input = []
        for idx in range(list_length):
            temp = np_val_input[idx], np_val_attention[idx]
            test_val_input.append(temp)

        self.test_input = np.array(test_val_input)
        self.test_labels = actual_y_test
        # print ("shape of test_input: ", self.test_input.shape)

    def __getitem__(self, index):
        return [self.test_input[index][0], self.test_input[index][1]], \
            self.test_labels[index]

    def __len__(self):
        return len(self.test_input)


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
                        help='batch size exampes: 32, 64, 128')

    parser.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for testing')

    parser.add_argument('--model_path',
                        '--_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path of the model')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    if FLAGS.batch_size < 0:
        logger.info("The parameter batch size value is invalid, try with valid \
            value\n")
        sys.exit(1)

    if FLAGS.dataset_file is None:
        logger.info("The dataset file is invalid, try with valid file name\n")
        sys.exit(1)

    if FLAGS.model_path is None:
        logger.info("Please provide the path to quanitized model...\n")
        sys.exit(1)
    else:
        model_path = FLAGS.model_path

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

    sentence, pos, tag, enc_pos, enc_tag = process_data(FLAGS.dataset_file)

    total_size = len(tag)
    X_test = sentence
    y_test = tag
    print(f"Testing dataset size: {total_size}")

    # Get the token ids and attention_masks for test data
    val_input_ids, val_attention_mask = tokenize(X_test, tokenizer,
                                                 max_len=MAX_LEN)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            # Load the graph in graph_def
            logger.info("load graph")
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, input_map=None,
                                    return_elements=None,
                                    name="",
                                    op_dict=None,
                                    producer_op_list=None)
                l_input_1 = graph.get_tensor_by_name('input_1:0')
                l_input_2 = graph.get_tensor_by_name('input_2:0')
                l_output = graph.get_tensor_by_name('Identity:0')
                # initialize_all_variables
                tf.compat.v1.global_variables_initializer()

                dataset = DatasetBatch(
                                val_input_ids,
                                val_attention_mask,
                                y_test)
                val_input = dataset.test_input[0][0].reshape(1, MAX_LEN)
                val_attention = dataset.test_input[0][1].reshape(1, MAX_LEN)
                Session_out = sess.run(l_output, feed_dict={
                            l_input_1: val_input,
                            l_input_2: val_attention})
                # print ("len of predictions: ", Session_out.shape)
                start_time = time.time()
                Session_out = sess.run(l_output, feed_dict={
                            l_input_1: val_input,
                            l_input_2: val_attention})
                real_time_inference = time.time() - start_time

                AVG_INFERENCE_TIME = 0
                AVG_NER_ACCURACY = 0
                bsize = FLAGS.batch_size
                total_batches = int(total_size / bsize)
                for i in range(3):
                    AVG_BATCH_TIME = 0
                    BATCH_NER_ACCURACY = 0
                    for j in range(total_batches):
                        start_time = time.time()
                        input_1 = dataset.test_input[j*bsize:(j+1)*bsize, 0, :]
                        input_2 = dataset.test_input[j*bsize:(j+1)*bsize, 1, :]
                        Session_out = sess.run(
                            l_output,
                            feed_dict={
                                        l_input_1: input_1,
                                        l_input_2: input_2})
                        end_time = time.time()-start_time
                        AVG_BATCH_TIME += end_time

                        CORRECT_CLASS_COUNT = [None]*128
                        # To display the original and predicted tags
                        orig_y_test = y_test[j*bsize:(j+1)*bsize]
                        testdata_length = len(orig_y_test)
                        for tindex in range(testdata_length):
                            test_tag = np.array(
                                        orig_y_test[tindex] + [0] *
                                        (MAX_LEN-len(orig_y_test[tindex])))

                            # print ("Padded gnd truth test_tag: ", test_tag)

                            pred_with_pad = np.argmax(
                                                Session_out[tindex],
                                                axis=-1)

                            for i in range(128):
                                if CORRECT_CLASS_COUNT[i] == None:
                                    CORRECT_CLASS_COUNT[i] = 0
                                if test_tag[i] == pred_with_pad[i]:
                                    CORRECT_CLASS_COUNT[i] += 1

                        NER_ACCURACY = [None]*128
                        for i in range(len(test_tag)):
                            if NER_ACCURACY[i] == None:
                                NER_ACCURACY[i] = 0

                            NER_ACCURACY[i] += (CORRECT_CLASS_COUNT[i] / testdata_length)

                        BATCH_NER_ACCURACY += mean(NER_ACCURACY)
                    print(f"Total Time Taken for model inference with \
                          batch size {bsize} in seconds \
                              ---> {AVG_BATCH_TIME}")
                    AVG_INFERENCE_TIME += (AVG_BATCH_TIME/total_batches)
                    AVG_NER_ACCURACY += (BATCH_NER_ACCURACY/total_batches)

                s = f"""
                {'-'*40}
                # Model Inference details:
                # Real time inference (in seconds): {real_time_inference}
                # Average Inference Time (in seconds): {AVG_INFERENCE_TIME/3}
                # Accuracy: {AVG_NER_ACCURACY/3}
                {'-'*40}
                """
                print(s)
