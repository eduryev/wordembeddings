
# import numpy as np
# import tensorflow as tf
import os

import newmodel.util as util
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.summary import create_file_writer, scalar

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


def get_similarity_tests(job_dir):
    tests_path = os.path.join(job_dir, 'similarity_tests')

    if job_dir[:5] == 'gs://': # make the file available in the container
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

    tests_list = os.listdir(tests_path)
    tests_dict = {SIMILARITY_TEST_NAMES.get(test, test):os.path.join(tests_path, test) for test in tests_list if test[-4:] in ('.tsv', '.txt')}

    return tests_dict


# TODO: use decorators to compose callback_func
# TODO: pair pearson and spearman to avoid computation duplication
def similarity_tests_callbacks(model, mode_list, metric_list, output_stat_list, tests_dict, job_dir, out_file = None):
    callback_list = []
    log_dir = os.path.join(job_dir, 'logs')

    if out_file:
        if log_dir[:5] == 'gs://':
            bucket_name, loc_path = util.split_gs_prefix(log_dir)
            out_path = os.path.join(loc_path, out_file)
        else:
            out_path = os.path.join(log_dir, out_file)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path), exist_ok = True)
        with open(out_path, 'w+') as out:
            out.write('#test_name\t')
            for output_stat in output_stat_list:
                out.write(output_stat + '\t')
            out.write('oov_pct\n')
    else:
        out_path = None

    # create a log for each (mode, metric) pair
    for mode in mode_list:
        for metric in metric_list:

            file_writer = create_file_writer(os.path.join(log_dir, f'metrics/{mode}_{metric}'))
            def callback_func(epoch, logs):
                for test_name, test_path in tests_dict.items():
                    if out_path:
                        with open(out_path, 'a') as out:
                            out.write(test_name + '\t')

                    # separate summary for each test and stat
                    for output_stat in output_stat_list:
                        res = model.similarity_test(test_path, mode=mode, metric=metric, output_stat=output_stat)
                        summary.scalar(test_name + '_' + output_stat, data = res[0], step = epoch)

                        if out_path: # record stat
                            with open(out_path, 'a') as out:
                                out.write('{0:.5f}\t'.format(res[0]))

                    if out_path: # recor oov_pct
                        with open(out_path, 'a') as out:
                            out.write('{0:.5f}\n'.format(res[-1]))

            callback = LambdaCallback(on_epoch_end = lambda epoch, logs: callback_func(epoch, log))
            callback_list.append(callback)

    return callback_list, out_path
