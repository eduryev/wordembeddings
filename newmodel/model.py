
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dot, Embedding, Input, Reshape, concatenate


class Word2VecNEGLoss(tf.keras.losses.Loss):
    def __init__(self, pow = 1., thresh = 100.):
        super().__init__()

    def call(self, y_pred, posneg):
        S = tf.sigmoid(-posneg)
        L = y_pred*(-tf.math.log(1 - S[:,0]) - tf.reduce_sum(tf.math.log(S[:, 1:]), axis = -1))
        return tf.keras.backend.mean(L, axis = -1)


class Word2VecModel(tf.keras.models.Model):
    def __init__(self, vocabulary_size, embedding_size, neg_samples):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.neg_samples = neg_samples
        self.loss = Word2VecNEGLoss()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

        self.W_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')
        self.C_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')
        self.dot_layer = Dot(axes = -1)
        self.reshape_layer1 = Reshape((1, ))
        self.reshape_layer2 = Reshape((1+ neg_samples, ))

    def call_test(self, inputs):
        """
        Only to make sure performance of actual model is comparable to dummy computations
        """
        x = self.W_embedding(inputs['target'])
        y = self.C_embedding(inputs['pos'])

        x = self.dot_layer([x, y])
        out = self.reshape_layer1(x)
        return out

    # @tf.function(input_signature=[
    #                               tf.TensorSpec(shape = (None, ), dtype = tf.int32, name = 'target'),
    #                               tf.TensorSpec(shape = (None, ), dtype = tf.int32, name = 'context'),
    #                               tf.TensorSpec(shape = (None, neg_samples), dtype = tf.int32, name = 'neg')
    #                               ])
    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        pos = self.reshape_layer1(inputs['pos'])
        pn = concatenate([pos, inputs['neg']], axis = -1)

        W_emb = self.W_embedding(inputs['target'])
        C_pn = self.C_embedding(pn)

        posneg_ = self.dot_layer([W_emb, C_pn])
        posneg = self.reshape_layer2(posneg_)

        return posneg

    def get_embedding(self, id_array):
        if not isinstance(id_array, [np.ndarray, tf.Tensor]):
          id_array = np.array(id_array, dtype = np.int32)
        return self.W_embedding(np.array(word_ids, dtype = np.int32))

    def evaluate_dist(self, sample_emb):
        # TODO: check it works well with tf instead of np
        w_emb = self.W_embedding.weights[0]
        return tf.sqrt(tf.square(tf.expand_dims(w_emb, -1) - tf.expand_dims(tf.transpose(sample_emb), 0)).sum(axis = 1))

    def get_closest(self, sample_emb = None, word_ids = None, n_closest = 1):
        if sample_emb is None:
          assert word_ids # at least one argument must be supplied
          sample_emb = self.get_embedding(word_ids)

        dist = self.evaluate_dist(sample_emb).numpy()
        return np.argsort(dist, axis = 0)[1:n_closest + 1, :]
