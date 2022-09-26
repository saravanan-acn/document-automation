# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""BERT NER for Document Automation
"""

# !/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd

def process_data(data_path):
    """Perform the preprocessing of the data set

    Args:
        data_path: The path of the input data file

    Returns:
        None
    """

    dataframe = pd.read_csv(data_path, encoding="latin-1")

    df_new = dataframe[0:546781]
    df_new.to_csv("ner_dataset.csv", index=False)

    df_new = dataframe[0:55122]
    df_new.to_csv("ner_test_dataset.csv", index=False)

    df_new = dataframe[0:55122]
    df_new.to_csv("ner_test_quan_dataset.csv", index=False)

if __name__ == "__main__":
    # The main function body which takens the varilable number of arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_file',
                        '--dataset_file',
                        type=str,
                        required=True,
                        default=None,
                        help='dataset file for training')

    # Holds all the arguments passed to the function
    FLAGS = parser.parse_args()

    process_data(FLAGS.dataset_file)
