# this script loads and prepares data in google cloud storage

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
import urllib
# import tempfile
import zipfile
#form six.moves import urllib


DATA_DIR = os.path.join(os.getcwd(), 'word2vec_data')
TRAINING_FILE = 'text8.zip'
DATA_URL = 'http://mattmahoney.net/dc/text8.zip'


def _download_corpus(filename, url):
    filename, _ = urllib.request.urlretrieve(url, filename)


def download(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not os.path.exists(training_file_path):
        print("Downloading file to {}".format(training_file_path))
        _download_corpus(training_file_path, DATA_URL)

    return training_file_path


def load_data():
    training_file_path = download(DATA_DIR)

    print(training_file_path)
    with zipfile.ZipFile(training_file_path) as f:
        text_words = f.read(f.namelist()[0]).lower().split()
    return text_words
