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
from statistics import mean
import numpy as np  # linear algebra
import tensorflow as tf
from transformers import BertTokenizerFast
from neural_compressor.experimental import Quantization, common
from utils import process_data, tokenize

# The maximum length of the sentence used for model building
MAX_LEN = 128

VAL_INPUT_IDS = []
VAL_ATTENTION_MASK = []
X_TEST = []
Y_TEST = []
BATCH_SIZE = 32


# Load the graph from the frozen graph file
def load_graph(file_name):
    """Load the graph from the frozen graph file

    Args:
        file_name: Name of the frozen graph file

    Returns:
        graph: Tensorflow graph
    """

    tf.compat.v1.logging.info('Loading graph from: ' + file_name)
    with tf.io.gfile.GFile(file_name, "rb") as graph_file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_file.read())
    with tf.Graph().as_default() as lgraph:
        tf.import_graph_def(graph_def, name='')
    return lgraph


# Evaluate the model in tensorflow graph format for accuracy
def eval_func(infer_graph):
    """Evaluate the model and determine the accuracy

    Args:
        infer_graph: Tensorflow frozen graph

    Returns:
        Average accuracy for the batch
    """

    if isinstance(infer_graph, tf.compat.v1.GraphDef):
        tgraph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(infer_graph, name='')
        infer_graph = tgraph

    config = tf.compat.v1.ConfigProto()
    # config.use_per_session_threads = 1
    # config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=infer_graph, config=config)

    l_input_1 = infer_graph.get_tensor_by_name('input_1:0')  # Input Tensor
    l_input_2 = infer_graph.get_tensor_by_name('input_2:0')  # Input Tensor
    l_output = infer_graph.get_tensor_by_name('Identity:0')  # Output Tensor
    # initialize_all_variables
    tf.compat.v1.global_variables_initializer()

    # Get the values from the global variables
    val_input_ids = VAL_INPUT_IDS
    val_attention_mask = VAL_ATTENTION_MASK
    orig_y_test = Y_TEST

    test_dataset = DatasetBatch(val_input_ids, val_attention_mask, orig_y_test)
    val_input = test_dataset.test_input[0][0].reshape(1, MAX_LEN)
    val_attention = test_dataset.test_input[0][1].reshape(1, MAX_LEN)
    session_out = sess.run(l_output, feed_dict={
                l_input_1: val_input,
                l_input_2: val_attention})
    # print ("len of predictions: ", Session_out.shape)

    avg_inference_time = 0
    avg_ner_accuracy = 0
    bsize = BATCH_SIZE
    total_batches = int(len(orig_y_test)/bsize)
    # Take average of batch inference of three trails
    for i in range(3):
        avg_batch_time = 0
        BATCH_NER_ACCURACY = 0
        for j in range(total_batches):
            start_time = time.time()
            input_1 = test_dataset.test_input[j*bsize:(j+1)*bsize, 0, :]
            input_2 = test_dataset.test_input[j*bsize:(j+1)*bsize, 1, :]
            session_out = sess.run(l_output, feed_dict={
                l_input_1: input_1,
                l_input_2: input_2})
            end_time = time.time()-start_time
            avg_batch_time += end_time

            #correct_class_count = 0
            CORRECT_CLASS_COUNT = [None]*128
            # To display the original and predicted tags
            actual_y_test = orig_y_test[j*bsize:(j+1)*bsize]
            list_length = len(actual_y_test)
            for idx in range(list_length):
                test_tag = np.array(
                            actual_y_test[idx] + [0] *
                            (MAX_LEN-len(actual_y_test[idx])))

                pred_with_pad = np.argmax(session_out[idx], axis=-1)

                for k in range(128):
                    if CORRECT_CLASS_COUNT[k] == None:
                        CORRECT_CLASS_COUNT[k] = 0
                    if test_tag[k] == pred_with_pad[k]:
                        CORRECT_CLASS_COUNT[k] += 1

            NER_ACCURACY = [None]*128
            for i in range(len(test_tag)):
                if NER_ACCURACY[i] == None:
                    NER_ACCURACY[i] = 0

                NER_ACCURACY[i] += (CORRECT_CLASS_COUNT[i] / list_length)

            BATCH_NER_ACCURACY += mean(NER_ACCURACY)

        print(f"Total Time Taken for model inference with batch size  \
                {bsize} in seconds ---> {avg_batch_time}")
        avg_inference_time += (avg_batch_time/total_batches)
        avg_ner_accuracy += (BATCH_NER_ACCURACY/total_batches)

    return avg_ner_accuracy/(3)


class DatasetBatch():
    """Creating Dataset class for getting Image and labels"""
    def __init__(self, sent_val_input_ids, sent_val_attention_mask,
                 actual_y_test):

        list_length = len(sent_val_input_ids)
        val_input_list = []
        val_attention_list = []
        for idx in range(list_length):
            val_input = sent_val_input_ids[idx].reshape(1, MAX_LEN)
            val_attention = sent_val_attention_mask[idx].reshape(1, MAX_LEN)
            val_input_list.append(val_input)
            val_attention_list.append(val_attention)

        np_val_input = np.array(val_input_list)
        np_val_attention = np.array(val_attention_list)
        np_val_input = np_val_input.reshape(np_val_input.shape[0], MAX_LEN)
        np_val_attention = np_val_attention.reshape(np_val_attention.shape[0],
                                                    MAX_LEN)

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

    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='Model path trained with tensorflow saved model \
                            directory')
    parser.add_argument('-c',
                        '--config_file',
                        type=str,
                        required=False,
                        default='./deploy.yaml',
                        help='Yaml file for quantizing model, default is \
                            "./deploy.yaml"')

    parser.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for testing')

    parser.add_argument('--save_model_path',
                        '--save_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path to save the model')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

    config_file = FLAGS.config_file

    if FLAGS.batch_size < 0:
        logger.info("The parameter batch size value is invalid, try with valid \
            value\n")
        sys.exit(1)
    if FLAGS.dataset_file is None:
        logger.info("The dataset file is invalid, try with valid file name\n")
        sys.exit(1)

    if FLAGS.model_path is None:
        logger.info("Please proivide tensorflow saved model path...\n")
        sys.exit(1)
    else:
        model_path = FLAGS.model_path

    if FLAGS.save_model_path is None:
        logger.info("Please proivide path to save the quanitized model...\n")
        sys.exit(1)
    else:
        save_model_path = FLAGS.save_model_path

    # Tokenizer for getting token ids and attention_masks
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    sentence, pos, tag, enc_pos, enc_tag = process_data(FLAGS.dataset_file)

    # To use the validatiton data for testing, fixing the training data
    # and validatation data
    X_test = sentence
    y_test = tag

    # Get the token ids and attention_masks for test data
    val_input_ids, val_attention_mask = tokenize(X_test, tokenizer,
                                                 max_len=MAX_LEN)

    # Assign variable to global variables to be accessed in eval_func
    VAL_INPUT_IDS = val_input_ids
    VAL_ATTENTION_MASK = val_attention_mask
    X_TEST = X_test
    Y_TEST = y_test
    BATCH_SIZE = FLAGS.batch_size

    VAL_INPUT_IDS = np.array(VAL_INPUT_IDS)
    VAL_ATTENTION_MASK = np.array(VAL_ATTENTION_MASK)
    X_TEST = np.array(X_TEST)
    Y_TEST = np.array(Y_TEST)

    # Convert the FP32 tensorflow model to neural compressor quanitzed model
    graph = load_graph(model_path)

    # Convert the FP32 tensorflow model to neural compressor quanitzed model
    quantizer = Quantization(config_file)
    quantizer.model = model_path
    dataset = DatasetBatch(
                val_input_ids,
                val_attention_mask,
                y_test)
    quantizer.calib_dataloader = common.DataLoader(dataset)

    quantizer.eval_func = eval_func

    t1 = time.time()
    q_model = quantizer.fit()
    t2 = time.time() - t1
    q_model.save(save_model_path)

    s = f"""
    {'-'*40}
    # Model Quantization details
    # Time (in seconds): {t2}
    # Model saved path: {save_model_path}
    {'-'*40}
    """
    print(s)
