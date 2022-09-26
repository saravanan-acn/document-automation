# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""BERT NER for Document Automation
"""

"""File contains the utility and helper functions
"""

# pylint: disable=R0914

import time
from statistics import mean
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# Grouped sklearn packages
from sklearn import preprocessing
from sklearn.metrics import classification_report
import tensorflow as tf

# The maximum length of the sentence used for model building
MAX_LEN = 128


def process_data(data_path, dataset_size=100):
    """Perform the preprocessing of the data set

    Args:
        data_path: The path of the input data file
        dataset_size: The value gives the percentage of whole data set that
        should be used for building the model

    Returns:
        sentences: The sentences that need to be used for building the model
            and has corresponding POS and Tag values for each of the word or
            token in the sentence
        pos: For all the sentences used for building the model, this gives the
            parts of speech for each token in the sentence
        tag: For all the sentences considered for building the model, this
            gives the tagging for each token in the sentence

    """

    dataframe = pd.read_csv(data_path, encoding="latin-1")

    # df = pd.DataFrame()
    # dataframe = df
    print("# of sentences: ", dataframe["Sentence #"].count())
    number_sentences = dataframe["Sentence #"].count()

    # Based on the dataset_size provided, find the number of sentences to be
    # used for building the model
    required_sentences = int((number_sentences * dataset_size) / 100)

    #print("# of sentences considered for training: ", required_sentences)

    dataframe.loc[:, "Sentence #"] = \
        dataframe["Sentence #"].fillna(method="ffill")

    enc_pos_le = preprocessing.LabelEncoder()
    enc_tag_le = preprocessing.LabelEncoder()

    # Transform the categorical variables to one hot encoding so that they are
    # suitable for model training
    dataframe.loc[:, "Tag"] = enc_tag_le.fit_transform(dataframe["Tag"])
    dataframe.loc[:, "POS"] = enc_pos_le.fit_transform(dataframe["POS"])

    req_sentences = dataframe.groupby("Sentence #")["Word"].apply(list).values
    req_pos = dataframe.groupby("Sentence #")["POS"].apply(list).values
    req_tag = dataframe.groupby("Sentence #")["Tag"].apply(list).values

    # Return the sentences, their corresponding parts of speech and taggings
    # that need to be used for building the model
    return req_sentences[0:required_sentences], req_pos[0:required_sentences],\
        req_tag[0:required_sentences], enc_pos_le, enc_tag_le


def tokenize(data, tokenizer, max_len=MAX_LEN):
    """ Tokenize the given data using a tokenizer

    Args:
        data: The sentences that need to be tokenized
        max_len: This gives the maximum length of the sentence allowed

    Returns:
        sent_inputs_ids: For all sentences in data, this gives token ids for
        the tokens in each sentence sent_attention_mask: For all sentences
        in data, this gives the attention mask for respective tokens
        in each sentence
    """
    sent_input_ids = []
    sent_attention_mask = []
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens=True,
                                        max_length=max_len,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding='max_length',
                                        truncation=True, return_tensors='np')

        sent_input_ids.append(encoded['input_ids'])
        sent_attention_mask.append(encoded['attention_mask'])
    return np.vstack(sent_input_ids), np.vstack(sent_attention_mask)


def create_model(bert_model, max_len=MAX_LEN):
    """Create the deep learning model using transfer learning

    Args:
        bert_model: A pre-trained BERT base uncased model
        max_len: The size of the input data that is fed to the model

    Returns:
        model: The trained transfer learning model that can used for evaluation
    """

    sent_input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    sent_attention_masks = tf.keras.Input(shape=(max_len, ), dtype='int32')
    bert_output = bert_model(
                        sent_input_ids,
                        attention_mask=sent_attention_masks,
                        return_dict=True)
    embedding = tf.keras.layers.Dropout(0.3)(bert_output["last_hidden_state"])
    output = tf.keras.layers.Dense(17, activation='softmax')(embedding)
    tf_model = tf.keras.models.Model(
                        inputs=[sent_input_ids, sent_attention_masks],
                        outputs=[output])
    tf_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.00001),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return tf_model


def pred(model, sent_val_input_ids, sent_val_attention_mask):
    """Perform inference using the training model

    Args:
        sent_val_input_ids: The tokenized ids for sentence used for inference
        sent_val_attention_mask: The attention mask obtained from tokenizer
        for the tokens in the sent_val_input_ids

    Returns:
        predictions: The  predicted tags for the input token ids
    """
    start_time_t0 = time.time()
    predictions = model.predict([sent_val_input_ids, sent_val_attention_mask])
    print("Inference Time: ", time.time()-start_time_t0)
    return predictions


# Evaluate the model for a batch of samples input sentence from the test data
def batch_testing(model,
                  sent_val_input_ids,
                  sent_val_attention_mask,
                  actual_y_test,
                  batch_size=10):
    """Time to test the model for a batch of samples input sentence by
    comparing the original tags to the predicted tags

    Args:
        sent_val_input_ids: The tokenized ids for input sentence used for
            inference
        sent_val_attention_mask: The attention mask obtained from
            tokenizer for the tokens in the sent_val_input_ids
        input_enc_tag: Gives the tags for the given input
        actual_y_test: The actual output (tags) for the given input sentence

    Returns:
        None
    """

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

    start_time_t0 = time.time()
    predictions = model.predict([np_val_input, np_val_attention], batch_size)
    print("Batch Inference Time: ", time.time()-start_time_t0)
    inference_time = (time.time() - start_time_t0)

    # To display the original and predicted tags
    CORRECT_CLASS_COUNT = [None]*128
    for idx in range(list_length):
        test_tag = np.array(actual_y_test[idx] + [0] *
                            (MAX_LEN-len(actual_y_test[idx])))

        # print ("Padded ground truth test_tag: ", test_tag)

        pred_with_pad = np.argmax(predictions[idx], axis=-1)

        for i in range(128):
            if CORRECT_CLASS_COUNT[i] == None:
                CORRECT_CLASS_COUNT[i] = 0
            if test_tag[i] == pred_with_pad[i]:
                CORRECT_CLASS_COUNT[i] += 1

    NER_ACCURACY = [None]*128
    for i in range(len(test_tag)):
        if NER_ACCURACY[i] == None:
            NER_ACCURACY[i] = 0

        NER_ACCURACY[i] += (CORRECT_CLASS_COUNT[i] / list_length)

    model_accuracy = mean(NER_ACCURACY)
    print("Accuracy: ", model_accuracy)
    return inference_time, model_accuracy


# Evaluate the model for a sample input sentence from the test data
def testing(model, sent_val_input_ids, sent_val_attention_mask,
            actual_y_test):
    """Time to test the model for a sample input sentence by comparing
    the original tags to the predicted tags

    Args:
        sent_val_input_ids: The tokenized ids for input sentence used for
            inference
        sent_val_attention_mask: The attention mask obtained from tokenizer
            for the tokens in the sent_val_input_ids
        input_enc_tag: Gives the tags for the given input
        actual_y_test: The actual output (tags) for the given input sentence

    Returns:
        None
    """

    val_input = sent_val_input_ids.reshape(1, MAX_LEN)
    val_attention = sent_val_attention_mask.reshape(1, MAX_LEN)

    test_tag = np.array(actual_y_test + [0] *
                        (MAX_LEN-len(actual_y_test)))

    start_time = time.time()
    pred_with_pad = np.argmax(pred(model, val_input, val_attention), axis=-1)
    end_time = time.time() - start_time
    print(classification_report(test_tag, pred_with_pad[0]))

    return end_time
