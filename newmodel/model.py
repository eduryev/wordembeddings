
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dot, Embedding, Input, Reshape, concatenate

from scipy.stats import spearmanr, pearsonr


class BaseModel(tf.keras.models.Model):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, word2id = None, id2word = None):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.neg_samples = neg_samples
        if word2id:
            self.word2id = word2id
        if id2word:
            self.id2word = id2word

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)

        self.W_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')
        self.C_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')

    # functions for searching closes words
    def get_embedding(self, id_or_word_array, mode = 'target'):
        if isinstance(id_or_word_array, (np.ndarray, tf.Tensor)) or len(id_or_word_array) == 0:
            return self.W_embedding(id_or_word_array) if mode == 'target' else self.C_embedding(id_or_word_array)
        elif isinstance(id_or_word_array[0], int):
            id_array = np.array(id_or_word_array, dtype = np.int32)
        else: # try converting words to ids
            assert self.word2ids
            id_array = np.array([self.word2ids[w] for w in id_or_word_array], dtype = np.int32)
        return self.W_embedding(id_array) if mode == 'target' else self.C_embedding(id_array)

    # EDIK: All below is subject to revision -- common methods for models. Defined here even if some of them can be overwritten/amended for specific models
    # TODO: put 'normalize_rows' in utils
    def normalize_rows(self, sample_emb):
        return sample_emb/tf.sqrt(tf.reduce_sum(tf.square(sample_emb), axis = 1, keepdims = True))

    def validate_metric_name(self, metric):
        assert metric in ('l2', 'cos')

    def evaluate_pair_dists(self, id_array1, id_array2, mode = 'target', metric = 'l2'):
        self.validate_metric_name(metric)
        assert len(id_array1) == len(id_array2)
        emb1 = self.get_embedding(id_array1, mode = mode)
        emb2 = self.get_embedding(id_array2, mode = mode)
        if metric == 'l2':
            return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2),axis = 1)) # axis 1 is pairwise distance
        else:
            # note minus!
            return -(tf.reduce_sum(self.normalize_rows(emb1)*self.normalize_rows(emb2), axis = 1))

    def evaluate_dists(self, sample_emb, mode = 'target', metric = 'l2'):
        self.validate_metric_name(metric)
        w_emb = self.W_embedding.weights[0] if mode == 'target' else self.C_embedding.weights[0]
        if metric == 'l2':
            # this is bad, because 3d array is allocated in memory
            # return tf.sqrt(tf.square(tf.reduce_sum(tf.expand_dims(w_emb, -1) - tf.expand_dims(tf.transpose(sample_emb), 0), axis = 1)))
            dot = tf.matmul(w_emb, sample_emb, transpose_b=True)
            # d(a,b)^2 = a^2 - 2(a, b) + b^2
            return tf.reduce_sum(tf.square(w_emb), axis = 1, keepdims = True) + tf.transpose(tf.reduce_sum(tf.square(sample_emb), axis = 1, keepdims = True)) - 2*dot
        elif metric == 'cos':
            # note minus here!
            return -tf.matmul(self.normalize_rows(w_emb), self.normalize_rows(sample_emb), transpose_b=True)


    def get_closest(self, sample_emb = None, id_or_word_array = None, n_closest = 1, mode = 'target', metric = 'l2'):
        self.validate_metric_name(metric)
        if sample_emb is None:
            assert id_or_word_array # at least one argument must be supplied
            sample_emb = self.get_embedding(id_or_word_array, mode = mode)

        dist = self.evaluate_dists(sample_emb, mode = mode, metric = metric)
        return tf.argsort(dist, axis = 0)[1:n_closest + 1, :]


    def similarity_test(self, test_file_path
                      , mode = 'target', metric = 'l2', output_stat = 'spearman'
                      , delimeter = '\t'
                      ):
        self.validate_metric_name(metric)

        oov = 0
        tot = 0
        id_array1, id_array2, similarity_test_scores = [], [], []
        with open(test_file_path, 'r') as test_file:
            for line in test_file:
                if line.startswith('#'): # skip comments
                    continue
                else:
                    w1, w2, sim = line.strip().split(delimeter)
                    tot+= 1
                    try:
                        i1, i2, sim = self.word2id[w1], self.word2id[w2], float(sim)
                        id_array1.append(i1)
                        id_array2.append(i2)
                        similarity_test_scores.append(sim)
                    except KeyError: # not in dict error
                        oov+= 1
        dists = self.evaluate_pair_dists(id_array1, id_array2, mode = mode, metric = metric)
        if output_stat == 'spearman':
            res = spearmanr(similarity_test_scores, -dists).correlation
            return res, oov/tot
        elif output_stat == 'pearson':
            res = pearsonr(similarity_test_scores, -dists)[0]
            return res, oov/tot
        else: # return both
            return spearmanr(similarity_test_scores, -dists).correlation, pearsonr(similarity_test_scores, -dists)[0], oov/tot


    def analogy_test(self, test_file_path
                      , mode = 'target', metric = 'l2', output_stat = 'spearman'
                      , delimeter = ' ', ignore_sections = False
                      , ignore_triple = False, ignore_unknown = False):
        self.validate_metric_name(metric)
        res_dict = {}
        id_array = []
        oov = 0
        tot = 0

        def record_section(section_name, id_array, oov, tot):
            res_dict[section_name] = (np.array(id_array, dtype = np.int32), oov, tot)

        cur_section_name = 'ALL' if ignore_sections else None
        with open(test_file_path, 'r') as test_file:
            for line in test_file:
                if line.startswith('#'): # skip comments
                    continue
                elif line.startswith(':'): # enter here even if ignore_sections
                    if not ignore_sections:
                        if cur_section_name:
                            record_section(cur_section_name, id_array, oov, tot)
                            id_array = []
                            oov = 0
                            tot = 0
                        cur_section_name = line[1:].strip()
                else:
                    w1, w2, w3, w4 = line.strip().split(delimeter)
                    tot+= 1
                    try:
                        i1, i2, i3, i4 = self.word2id[w1], self.word2id[w2], self.word2id[w3], self.word2id[w4]
                        id_array.append([i1, i2, i3, i4])
                    except KeyError: # not in dict error
                        oov+= 1
        # record last section
        record_section(cur_section_name, id_array, oov, tot)

        out_dict = {}
        for section, (id_array, oov, tot) in res_dict.items():
            if len(id_array) == 0:
                out_dict[section] = 0, oov, tot
                continue
            sample_emb = (
                    -self.get_embedding(id_array[:, 0], mode = mode)
                    +self.get_embedding(id_array[:,1], mode = mode)
                    +self.get_embedding(id_array[:,2], mode = mode)
                )
            closest = self.get_closest(sample_emb = sample_emb,
                                        n_closest = 5 if ignore_triple else 1,
                                        mode = mode, metric = metric).numpy()

            correct = 0
            for i, l, triple in zip(self.get_embedding(id_array[:,3], mode = mode), closest, id_array[:3]):
                if i not in l:
                    continue
                other = l[:l.index(i)]
                admissible = list(triple) if ignore_triple else []
                admmimssible = [0] if ignore_unknown else []

                if len(set(other) - set(admissible)) == 0:
                    correct+= 1

            out_dict[section] =  correct, oov, tot

        return out_dict

    def group_analogy_test_results(self, out_dict, group_dict):
        res = {k:[0,0,0] for k in group_dict.keys()}
        for section, (correct, oov, tot) in out_dict.items():
            matches = [name for name, sections in group_dict.items() if section in sections]
            for match in matches:
                res[match][0]+= correct
                res[match][1]+= oov
                res[match][2]+= tot
        return res




# Word2Vec model with NEG loss
class Word2VecNEGLoss(tf.keras.losses.Loss):
    def __init__(self, pow = 1., thresh = 100.):
        super().__init__()

    def call(self, y_true, posneg):
        S = tf.sigmoid(-posneg)
        L = y_true*(-tf.math.log(1 - S[:,0]) - tf.reduce_sum(tf.math.log(S[:, 1:]), axis = -1))
        return tf.keras.backend.mean(L, axis = -1)


class Word2VecModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, word2id = word2id, id2word = id2word)

        # model specific part         
        self.loss = Word2VecNEGLoss()

        self.dot_layer = Dot(axes = -1)
        self.reshape_layer1 = Reshape((1, ))
        self.reshape_layer2 = Reshape((1+ neg_samples, ))


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


# Glove Model
class GloveLoss(tf.keras.losses.Loss):
    def __init__(self, pow = 1., thresh = 100.):
        super().__init__()

    # REQUIRED CHANGE
    def call(self, y_true, y_pred): # note that y_pred is already capped and powered
        L = y_true*tf.square(y_pred - tf.math.log(y_true))
        return tf.keras.backend.mean(L, axis = -1)

class GloveModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, word2id = word2id, id2word = id2word)

        # model specific part         
        self.loss = GloveLoss()

        self.target_bias = Embedding(vocabulary_size, 1, input_length=1, name='target_bias')
        self.context_bias = Embedding(vocabulary_size, 1, input_length=1, name='context_bias')

        self.dot_layer = Dot(axes = -1)
        self.add_layer = Add()
        self.reshape_layer = Reshape((1, ))


    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        w_target = self.W_embedding(inputs['target'])
        w_context = self.C_embedding(inputs['pos'])
        b_target = self.target_bias(inputs['target'])
        b_context = self.context_bias(inputs['pos'])

        dotted = self.dot_layer([w_target, w_context])
        pred = self.add_layer([dotted, b_context, b_target])

        return self.reshape_layer(pred)
