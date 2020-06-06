import time, datetime
import argparse
import os

import numpy as np
import tensorflow as tf
# from tensorflow.python.lib.io import file_io

import newmodel.model as model #from . import model
import newmodel.util as util #from . import util
import newmodel.tests as tests


def get_args():
    """Argument parser.

    Returns:
    Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        default = 'glove',
        help='glove, hypglove, word2vec')
    parser.add_argument(
        '--corpus-name',
        type=str,
        required=True,
        help='enwik8, enwik9, enwiki_dump')
    parser.add_argument(
        '--log-dir',
        type=str,
        required=False,
        default = 'logs',
        help='Subfolder with checkpoints to restore the model from')
    parser.add_argument(
        '--restore-folder',
        type=str,
        required=False,
        default = None,
        help='Subfolder with checkpoints to restore the model from')
    parser.add_argument(
        '--save-folder',
        type=str,
        required=False,
        default=None,
        help='Subfolder to store model results')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=1)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32768)
    parser.add_argument(
        '--learning-rate',
        type=int,
        default=1e-3)
    # parser.add_argument(
    #     '--eval-step',
    #     type=int,
    #     default = 200)
    # parser.add_argument(
    #   '--display-step',
    #   type=int,
    #   default = 1000)
    parser.add_argument(
        '--embedding-size',
        type=int,
        default = 200)
    parser.add_argument(
        '--max-vocabulary-size',
        type=int,
        default = 50000)
    parser.add_argument(
        '--min-occurrence',
        type=int,
        default = 10)
    parser.add_argument(
        '--skip-window',
        type=int,
        default = 5)
    parser.add_argument(
        '--neg-samples',
        type=int,
        default = 16)
    parser.add_argument(
        '--stored-batch-size',
        type=int,
        default = 131072)
    parser.add_argument(
        '--po',
        type=float,
        default = 0.75)
    parser.add_argument(
        '--threshold',
        type=int,
        default = 100)
    args, _ = parser.parse_known_args()
    return args


def train_model(args):
    # download and process data if does not exist

    train_file_name = utils.normalized_train_file_name(args)
    train_file_path = os.path.join(args.job_dir, 'model_data', train_file_name)

    # if this fails, pipeline won't work properly generating incompatible tails.
    assert args.stored_batch_size % args.batch_size == 0

    word2id, id2word, word_counts, id_counts, skips_paths = util.load_process_data(train_file_name, args, remove_zero = False)
    vocabulary_size = len(word2id)

    # create the dataset
    # TODO: Note this power is different in principle
    arr_counts = np.array([id_counts[i] for i in range(len(id2word))], dtype = np.float32)
    arr_counts[:] = arr_counts**args.po
    unigram = arr_counts/arr_counts.sum()

    neg_samples = 0 if args.mode in ['glove', 'hypglove'] else args.neg_samples
    dataset = utils.create_dataset_from_stored_batches(skips_paths, args.stored_batch_size, batch_size = args.batch_size, sampling_distribution = unigram, threshold = args.threshold, po = args.po, neg_samples = args.neg_samples)
    # create the model and follow additional model specific instructions (e.g. callbacks)
    if args.mode == 'glove':
        train_model = model.GloveModel(vocabulary_size, args.embedding_size, args.neg_samples, word2id = word2id, id2word = id2word)
        train_model.compile(loss = train_model.loss, optimizer = train_model.optimizer)
    elif args.mode == 'word2vec':
        train_model = model.Word2VecModel(vocabulary_size, args.embedding_size, args.neg_samples, word2id = word2id, id2word = id2word)
        train_model.compile(loss = train_model.loss, optimizer = train_model.optimizer)
    elif args.mode == 'hypglove':
        train_model = model.HypGloveModel(vocabulary_size, args.embedding_size, args.neg_samples, word2id = word2id, id2word = id2word)
        train_model.compile(loss = train_model.loss, optimizer = train_model.optimizer)
    else:
        raise NotImplementedError

    # EDIK: these callbacks need to be reworked, see tests.py if you want to reuse anything
    # prepare tests and callbacks
    similarity_tests_dict = tests.get_similarity_tests(args.job_dir)
    print('Found following similarity tests:')
    print(similarity_tests_dict)
    sim_out_file = 'sim_tests_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + '.tsv'
    similarity_tests_callbacks, sim_out_path = tests.similarity_tests_callbacks(train_model, ['target', 'context'], ['l2', 'cos'], ['spearman', 'pearson'], similarity_tests_dict, args.job_dir, out_file = sim_out_file)

    # if restore-path is given restore the model
    if args.restore_folder is not None:
        restore_path = os.path.join(args.job_dir, 'saved_models', args.restore_folder)
        print(f'Restoring model weights from {restore_path}')
        latest = tf.train.latest_checkpoint(restore_path)
        train_model.load_weights(latest)

    # train the model
    if args.save_folder is None: # save to restore_folder unless instructed otherwise
        args.save_folder = args.restore_folder

    if args.save_folder is None:
        train_model.fit(dataset, epochs = args.num_epochs, callbacks = similarity_tests_callbacks)
    else:
        ckpt_path = os.path.join(args.job_dir, 'saved_models', args.save_folder, 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path, save_weights_only = True,
                                                 verbose = 1, max_to_keep = 5, period = 1)
        train_model.fit(dataset, epochs = args.num_epochs, callbacks = [cp_callback] + similarity_tests_callbacks)

    # if working in GCP, upload similarity tests results
    util.upload_to_gs(sim_out_path, args.job_dir)


if __name__ == '__main__':
    args = get_args()
    assert args.mode in ['hypglove', 'glove', 'word2vec'] # TODO: do it natively with argparse module

# tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_model(args)
