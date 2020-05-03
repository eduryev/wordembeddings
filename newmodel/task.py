

import time
import argparse
import os

import numpy as np
import tensorflow as tf
# from tensorflow.python.lib.io import file_io

import newmodel.model as model #from . import model
import newmodel.util as util #from . import util




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
        '--corpus-name',
        type=str,
        required=True,
        help='enwik8')
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

    # TODO: replace with vocabulary formatting
    train_file_name = f'stored_{args.corpus_name}_maxsize_{args.max_vocabulary_size}_minocc_{args.min_occurrence}_window_{args.skip_window}_storedbatch_{args.stored_batch_size}.npy'
    train_file_path = os.path.join(args.job_dir, 'model_data', train_file_name)

    # if this fails, pipeline won't work properly generating incompatible tails.
    assert args.stored_batch_size % args.batch_size == 0

    word2id, id2word, word_counts, id_counts = util.load_process_data(train_file_name, args)
    vocabulary_size = len(word2id)

    # create the dataset
    arr_counts = np.array([id_counts[i] for i in range(len(id2word))], dtype = np.float32)
    # TODO: Note this power is different in principle
    arr_counts[:] = arr_counts**args.po
    unigram = arr_counts/arr_counts.sum()
    dataset = util.create_dataset_from_stored_batches(train_file_path, args.batch_size, args.stored_batch_size, args.neg_samples, unigram, args.threshold, args.po)

    # create the model
    w2v_model = model.Word2VecModel(vocabulary_size, args.embedding_size, args.neg_samples)
    w2v_model.compile(loss = model.Word2VecNEGLoss(), optimizer = w2v_model.optimizer)

    # if restore-path is given restore the model
    if args.restore_folder is not None:
        restore_path = os.path.join(args.job_dir, 'saved_models', args.restore_folder)
        print(f'Restoring model weights from {restore_path}')
        latest = tf.train.latest_checkpoint(restore_path)
        w2v_model.load_weights(latest)

    # train the model
    if args.save_folder is None: # save to restore_folder unless instructed otherwise
        args.save_folder = args.restore_folder

    if args.save_folder is None:
        w2v_model.fit(dataset, epochs = args.num_epochs)
    else:
        ckpt_path = os.path.join(args.job_dir, 'saved_models', args.save_folder, 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path, save_weights_only = True,
                                                 verbose = 1, max_to_keep = 5, period = 1)
        w2v_model.fit(dataset, epochs = args.num_epochs, callbacks = [cp_callback])

    # save terminal model



if __name__ == '__main__':
    args = get_args()
    # tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_model(args)
