# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""BERT NER for Document Automation
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=E0401,E0611,W0108

# Import the required libraries required for data analysis and building model
import sys
import logging
import argparse
import warnings
import tensorflow as tf
from transformers import TFBertModel
from tensorflow.python.framework.convert_to_constants \
    import convert_variables_to_constants_v2

from utils import create_model

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path',
                        '--model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path of the model')

    parser.add_argument('--save_model_path',
                        '--save_model_path',
                        type=str,
                        required=True,
                        default=None,
                        help='give the directory path of to save the model')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()

    if FLAGS.model_path is None:
        logger.info("Please provide path to load the model...\n")
        sys.exit(1)
    else:
        model_path = FLAGS.model_path

    if FLAGS.save_model_path is None:
        logger.info("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        save_model_path = FLAGS.save_model_path

    # # **Building BERT Model : Transfer Learning**
    base_bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    model = create_model(base_bert_model, MAX_LEN)

    model.load_weights(model_path)

    signature_dict = {
                    "input_1": tf.TensorSpec(
                            shape=model.inputs[0].shape,
                            dtype=model.inputs[0].dtype, name="input_1"),
                    "input_2": tf.TensorSpec(
                           shape=model.inputs[1].shape,
                           dtype=model.inputs[1].dtype, name="input_2")}

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
                                            signature_dict)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    logger.info("Frozen model layers: ")
    for layer in layers:
        logger.info(layer)

    print("-" * 50)
    logger.info("Frozen model inputs: ")
    logger.info(frozen_func.inputs)
    logger.info("Frozen model outputs: ")
    logger.info(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=FLAGS.save_model_path,
                      name="frozen_graph.pb",
                      as_text=False)
