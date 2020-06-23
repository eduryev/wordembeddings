
# import numpy as np
# import tensorflow as tf
import os

import newmodel.util as util
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow import summary

from google.cloud import storage

SIMILARITY_TEST_NAMES =  {
     'MC-30.tsv': 'MC',
     'men.tsv':'MEN',
     'mturk771.tsv':'MT',
     'RG-65.tsv': 'RG',
     'rw_processed.tsv':'RW',
     'simlex999.tsv':'SX',
     'simverb3500.tsv':'SV',
     'wordsim353.tsv':'WS',
     'wordsim_relatedness_goldstandard.tsv':'WSrel',
     'wordsim_similarity_goldstandard.tsv':'WSsim'
     }

ANALOGY_TEST_NAMES =  {
     'questions-words.txt': 'QW',
     }

ANALOGY_TEST_GROUPS = {
    'semantic' : ['capital-common-countries', 'capital-world', 'currency'
    , 'city-in-state', 'city-in-state'],
    'syntactic' : ['gram1-adjective-to-adverb', 'gram2-opposite'
    , 'gram3-comparative', 'gram4-superlative', 'gram5-present-participle'
    , 'gram6-nationality-adjective', 'gram7-past-tense', 'gram8-plural'
    , 'gram9-plural-verbs'],
    'combined' : ['capital-common-countries', 'capital-world', 'currency'
    , 'city-in-state', 'city-in-state', 'gram1-adjective-to-adverb', 'gram2-opposite'
    , 'gram3-comparative', 'gram4-superlative', 'gram5-present-participle'
    , 'gram6-nationality-adjective', 'gram7-past-tense', 'gram8-plural'
    , 'gram9-plural-verbs']
}


def get_similarity_tests(job_dir):
    tests_dir = os.path.join(job_dir, 'tests', 'similarity_tests')
    tests_paths = get_tests(tests_dir)
    tests_dict = {SIMILARITY_TEST_NAMES.get(os.path.basename(test_path), os.path.basename(test_path)):test_path for test_path in tests_paths}
    return tests_dict

def get_analogy_tests(job_dir):
    tests_dir = os.path.join(job_dir, 'tests', 'analogy_tests')
    tests_paths = get_tests(tests_dir)
    tests_dict = {ANALOGY_TEST_NAMES.get(os.path.basename(test_path), os.path.basename(test_path)):test_path for test_path in tests_paths}
    return tests_dict


def get_tests(tests_path):
    if tests_path[:5] == 'gs://': # make the file available in the container
        # if tests are found in gcp bucket, overwrite local tests
        # else attempt using local test AND upload them to Google Storage Bucket

        bucket_name, path_name = util.split_gs_prefix(tests_path)
        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])

        bl_list = bucket.list_blobs(prefix = path_name)
        if len(list(bl_list)) > 0: #tests found
            if os.path.exists(path_name):
                os.remove(path_name)
            os.makedirs(path_name, exist_ok = True)
            for bl in bucket.list_blobs(prefix = path_name):
                if bl.name[-4:] in ('.txt','.tsv'):
                    bl.download_to_filename(bl.name)
            tests_path = path_name
        elif os.path.exists(path_name): # TODO: handle iteration over multiple files
            util.upload_to_gs(path_name, tests_path)
            tests_path = path_name
        else:
            return {}

    if not os.path.exists(tests_path):
        return []
    tests_list = os.listdir(tests_path)
    return [os.path.join(tests_path, test) for test in tests_list if test[-4:] in ('.tsv', '.txt')]
