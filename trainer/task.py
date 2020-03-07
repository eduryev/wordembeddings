
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import os

import numpy as np
import tensorflow as tf

from . import model
from . import util

import time
# from model import embedding, nce_weights, nce_loss

# # + id="DXg8AOLP_KWg" colab_type="code" colab={}
# # Training Parameters.
# learning_rate = 0.1
# batch_size = 128
# num_steps = 300000
# display_step = 1000
# # eval_step = 20000
#
# # Evaluation Parameters.
# # eval_words = [b'five', b'of', b'going', b'hardware', b'american', b'britain']
#
# # Word2Vec Parameters.
# embedding_size = 200 # Dimension of the embedding vector.
# max_vocabulary_size = 50000 # Total number of different words in the vocabulary.
# min_occurrence = 10 # Remove all words that does not appears at least n times.
# skip_window = 3 # How many words to consider left and right.
# num_skips = 2 # How many times to reuse an input to generate a label.
# num_sampled = 64 # Number of negative examples to sample.


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
      '--learning-rate',
      type=int,
      default=0.1)
  parser.add_argument(
      '--batch-size',
      type=int,
      default=128)
  parser.add_argument(
      '--num-steps',
      type=int,
      default=1000)
  # parser.add_argument(
  #     '--eval-step',
  #     type=int,
  #     default = 200)
  parser.add_argument(
      '--display-step',
      type=int,
      default = 100)
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
      default = 3)
  parser.add_argument(
      '--num-skips',
      type=int,
      default = 2)
  parser.add_argument(
      '--num-sampled',
      type=int,
      default = 64)
  # parser.add_argument(
  #     '--verbosity',
  #     choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
  #     default='INFO')
  args, _ = parser.parse_known_args()
  return args


def train_word2vec(args):
    text_words = util.load_data()

    # data, vocabulary_size, word2id, id2word = model.preprocess_data(text_words, args.max_vocabulary_size, args.min_occurrence, args.skip_window, args.num_skips, args.num_sampled)

    # embedding, nce_weights, nce_biases = model.define_weights(vocabulary_size, args.embedding_size)
    w2v = model.Word2Vec(text_words, args.max_vocabulary_size, args.min_occurrence, args.skip_window, args.num_skips, args.num_sampled, args.embedding_size)
    optimizer = tf.optimizers.SGD(args.learning_rate)
    for step in range(1, args.num_steps + 1):
        batch_x, batch_y = w2v.next_batch(args.batch_size, args.num_skips, args.skip_window)
        w2v.run_optimization(optimizer, batch_x, batch_y)

        if step % args.display_step == 0 or step == 1:
            loss = w2v.nce_loss(w2v.get_embedding(batch_x), batch_y)
            print("step: %i, loss: %f" % (step, loss))

if __name__ == '__main__':
    args = get_args()
    # tf.compat.v1.logging.set_verbosity(args.verbosity)

    train_word2vec(args)
    # with open(os.path.join(args.job_dir, 'results.txt'), 'w+') as out:
    with open('results.txt', 'a+') as out:
        log_str = "Job finished! {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S'))
        print("HOHO " + log_str)
        out.write(log_str)

        log_str = 'pwd ' + os.getcwd() + "\n"
        print("HEHE " + log_str)
        out.write(log_str)

        log_str = 'files: ' + str(os.listdir()) + "\n\n"
        print("HAHA " + log_str)
        out.write(log_str)
