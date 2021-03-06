
import os
import pickle as pkl
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dot, Embedding, Input, Reshape, concatenate
from tensorflow.keras.callbacks import LambdaCallback

from scipy.stats import spearmanr, pearsonr

from newmodel.util import split_gs_prefix, upload_to_gs, download_from_gs, save_dict, load_dict


def load_custom_model(model_path, meta_path, mode = None):
    # load args
    args_, num_epochs_, optimizer_weights_ = BaseModel.load_train_config(model_path)
    word2id, id2word, word_counts, id_counts = read_corpus_metadata(meta_path)
    vocabulary_size = max(id2word.keys()) + 1 # might be slightly more accurate

    if mode is None:
        mode = model_name.split('_')[0]
    print(f'Creating new {mode} model instance...')

    # TODO: make sure those are accessible
    class_dict = {'w2v': Word2VecModel, 'glove': GloveModel, 'hypglove': HypGloveModel}
    model = class_dict[mode](vocabulary_size, args.embedding_size, args.neg_samples, word2id = word2id, id2word = id2word)
    args_, epochs_trained_ =  model.load_model(model_path)

    return model, args_, epochs_trained_


class BaseModel(tf.keras.models.Model):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__()
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.neg_samples = neg_samples
        if word2id:
            self.word2id = word2id
        if id2word:
            self.id2word = id2word

        learning_rate = 1e-3 if learning_rate is None else learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        self.T_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')
        self.C_embedding = Embedding(vocabulary_size, embedding_size
                                    , embeddings_initializer='normal')


    def save_train_config(self, save_path, epochs_trained, args):
        '''
        Save train arguments, optimizer state and train epoch
        '''
        print(f'Saving model configuration to {save_path}')

        if save_path[:5] == 'gs://':
            bucket_name, path_name = split_gs_prefix(save_path)
        else:
            path_name = save_path
        os.makedirs(path_name, exist_ok = True)

        # save train arguments
        args_dict = dict(vars(args))
        args_dict['epochs_trained'] = epochs_trained
        save_dict(args_dict, os.path.join(path_name, 'args.pkl'))

        # save optimizer state
        optimizer_weights = tf.keras.backend.batch_get_value(self.optimizer.weights)
        with open(os.path.join(path_name, 'optimizer.pkl'), 'wb') as f:
            pkl.dump(optimizer_weights, f)

        if save_path[:5] == 'gs://':
            upload_to_gs(os.path.join(path_name, 'args.pkl'), save_path)
            upload_to_gs(os.path.join(path_name, 'optimizer.pkl'), save_path)


    # note this funcion belongs to a class, not a class instance
    def load_train_config(load_path):
        '''
        Load train arguments, optimizer state and train epoch
        '''
        if load_path[:5] == 'gs://':
            download_from_gs(os.path.join(load_path, 'args.pkl'))
            load_path = download_from_gs(os.path.join(load_path, 'optimizer.pkl'))
            load_path = os.path.dirname(load_path)

        with open(os.path.join(load_path, 'args.pkl'), 'rb') as f:
            args = pkl.load(f)
        with open(os.path.join(load_path, 'optimizer.pkl'), 'rb') as f:
            optimizer_weights = pkl.load(f)

        epochs_trained = args['epochs_trained']
        del args['epochs_trained']
        args = SimpleNamespace(**args)
        return args, epochs_trained, optimizer_weights


    def load_model(self, load_path):
        '''
        Loads the model with configuration from a given path
        '''
        args_, epochs_trained_, optimizer_weights_ = BaseModel.load_train_config(load_path)
        latest = tf.train.latest_checkpoint(load_path)
        # initial_epoch = int(os.path.basename(latest)[-9:-5])
        dummy_key = {'target': np.zeros(1, dtype = np.int32), 'pos': np.zeros(1, dtype = np.int32), 'neg': np.zeros(shape = (1, self.neg_samples), dtype = np.int32)}
        dummy_val = np.zeros(1, dtype = np.float32)

        if not self._is_compiled:
            self.compile(loss = self.loss, optimizer = self.optimizer)

        self.fit(dummy_key, dummy_val, epochs=1, verbose=0)
        self.optimizer.set_weights(optimizer_weights_)
        self.load_weights(latest)
        return args_, epochs_trained_


    def get_save_callbacks(self, save_path, args, period = 1):
        if save_path is None:
            if args.save_dir is None: # don't save the model
                return []
            save_path = os.path.join(args.job_dir, 'saved_models' , args.save_dir) # default save_path

        callbacks = []
        def save_config_callback_factory(save_path, args):
            def save_train_config(epoch, logs):
                if (epoch + 1) % period == 0:
                    self.save_train_config(save_path, epoch + 1, args)   # note that epochs_trained = epoch+1
            return save_train_config


        ckpt_path = os.path.join(save_path, 'cp-{epoch:04d}.ckpt')
        # save_weights_only = False has a bug, hopefully will get resolved:
        # https://github.com/tensorflow/tensorflow/issues/39679#event-3376275799
        # at any rate, saving custom models requires some effort, so rely on load_weights
        cp_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path, save_weights_only = True,
                                                    verbose = 1, max_to_keep = 5, period = period)
        callbacks.append(cp_ckpt)
        callbacks.append(LambdaCallback(on_epoch_end = save_config_callback_factory(save_path, args)))

        return callbacks


    # functions for searching closest words
    def get_embedding(self, id_or_word_array, mode = 'target'):
        self.validate_mode_name(mode)
        if isinstance(id_or_word_array, (np.ndarray, tf.Tensor)) or len(id_or_word_array) == 0:
            pass
        elif isinstance(id_or_word_array[0], int):
            id_or_word_array = np.array(id_or_word_array, dtype = np.int32)
        else: # try converting words to ids
            assert self.word2id
            id_or_word_array = np.array([self.word2id[w] for w in id_or_word_array], dtype = np.int32)

        if mode == 'target':
            emb = self.T_embedding(id_or_word_array)
        elif mode == 'context':
            emb = self.C_embedding(id_or_word_array)
        elif mode == 'added':
            emb = self.T_embedding(id_or_word_array)+ self.C_embedding(id_or_word_array)
        elif mode == 'concat':
            emb = tf.concat([self.C_embedding(id_or_word_array), self.T_embedding(id_or_word_array)], axis = 1)

        return emb

    def normalize_rows(self, sample_emb):
        return sample_emb/tf.sqrt(tf.reduce_sum(tf.square(sample_emb), axis = 1, keepdims = True))

    def validate_metric_name(self, metric):
        assert metric in ('l2', 'cos')

    def validate_mode_name(self, mode):
        assert mode in ('context', 'target', 'added', 'concat')

    def evaluate_pair_dists(self, id_array1, id_array2, mode = 'target', metric = 'l2'):
        self.validate_metric_name(metric)
        self.validate_mode_name(mode)
        assert len(id_array1) == len(id_array2)
        emb1 = self.get_embedding(id_array1, mode = mode)
        emb2 = self.get_embedding(id_array2, mode = mode)
        if metric == 'l2':
            return tf.sqrt(tf.reduce_sum(tf.square(emb1 - emb2), axis = 1)) # axis 1 is pairwise distance
        elif metric == 'cos':
            # note minus!
            return -(tf.reduce_sum(self.normalize_rows(emb1)*self.normalize_rows(emb2), axis = 1))
        else:
            raise NotImplementedError

    def evaluate_dists(self, sample_emb, mode = 'target', metric = 'l2'):
        self.validate_metric_name(metric)
        self.validate_mode_name(mode)
        if mode == 'target':
            w_emb = self.T_embedding.weights[0]
        elif mode == 'context':
            w_emb = self.C_embedding.weights[0]
        elif mode == 'added':
            w_emb = self.T_embedding.weights[0] + self.C_embedding.weights[0]
        elif mode == 'concat':
            w_emb = tf.concat(self.C_embedding.weights[0], self.T_embedding.weights[0], axis = 1)

        if metric == 'l2':
            # this would be bad, because 3d array is allocated in memory
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
        return tf.transpose(tf.argsort(dist, axis = 0)[1:n_closest + 1, :])


    ## similariy tests functions
    # read similarity file to list of pairs (tasks), list of floats (human similarity scores), number of out-of-vocabulary tasks
    def read_similarity_file(self, test_file, ignore_oov = True, delimeter = '\t'):
        tasks, gold_sim, oov = [], [], 0
        with open(test_file, 'r') as f:
            for line_no, line in enumerate(f.readlines()):
                if line.startswith('#'):
                    continue
                else:
                    w1, w2, sim = line.strip().lower().split(delimeter)
                    try:
                        i1, i2, sim = self.word2id[w1], self.word2id[w2], float(sim)
                        tasks.append([i1, i2])
                        gold_sim.append(sim)
                    except KeyError: # not in dict error
                        oov+= 1
                        if not ignore_oov:
                            tasks.append([self.word2id.get(w1, 0), self.word2id.get(w2, 0)])
                            gold_sim.append(float(sim))
        return np.array(tasks, dtype = np.int32), gold_sim, oov


    # TODO: perhaps leave out oov and do not return oov_ratio -- move to fetch_similarity_results
    def similarity_results(self, tasks, similarity_test_scores, oov
                      , mode = 'target', metric = 'l2', output_stat = 'both'
                      , ignore_oov = True, delimeter = '\t'
                      ):
        self.validate_metric_name(metric)

        # tasks, similarity_test_scores, oov = self.read_similarity_file(test_file_path, ignore_oov = ignore_oov, delimeter = delimeter)
        dists = self.evaluate_pair_dists(tasks[:,0], tasks[:,1], mode = mode, metric = metric)

        if ignore_oov:
            oov_ratio = float(oov) / (len(similarity_test_scores) + oov) * 100
        else:
            oov_ratio = float(oov) / len(similarity_test_scores) * 100

        if output_stat == 'spearman':
            res = spearmanr(similarity_test_scores, -dists).correlation
            return res, oov_ratio
        elif output_stat == 'pearson':
            res = pearsonr(similarity_test_scores, -dists)[0]
            return res, oov_ratio
        else: # return both
            return pearsonr(similarity_test_scores, -dists)[0], spearmanr(similarity_test_scores, -dists).correlation, oov_ratio


    def fetch_similarity_results(self, tests_dict, mode, metric
                                 , verbose = False, ignore_oov = True, delimeter = '\t'):
        res = {}
        for test_name, test_file_path in tests_dict.items():
            tasks, similarity_test_scores, oov = self.read_similarity_file(test_file_path, ignore_oov = ignore_oov, delimeter = delimeter)
            p, s, o = self.similarity_results(tasks, similarity_test_scores, oov, mode, metric, output_stat = 'both', ignore_oov = ignore_oov, delimeter = delimeter)
            res[test_name] = (p, s, o)
            if verbose:
                logstr = f'{test_name}'.ljust(10) + f'Pearson {p:.3f}'.ljust(25) + f'Spearman {s:.3f}'.ljust(25) + f'OOV: {o:.2f}%'.ljust(25)
                print(logstr)
        if not verbose:
            return res


    def get_similarity_tests_callbacks(self, tests_dict, mode_list, metric_list, job_dir, log_dir = None):
        callback_list = []

        if log_dir:
                log_path = os.path.join(job_dir, 'logs', log_dir)
        else:
            print('Logging directory for similarity tests is not specified, writing to `logs/temp` directory')
            log_path = os.path.join(job_dir, 'logs', 'temp')
        os.makedirs(log_path, exist_ok=True)

        def callback_factory(mode, metric, verbose = False, ignore_oov = True, delimeter = '\t'):
            def similarity_scalar(epoch, logs):
                if verbose:
                    print('Running similarity tests...', end = '')
                res = self.fetch_similarity_results(tests_dict, mode, metric, verbose = verbose, ignore_oov = ignore_oov, delimeter = delimeter)

                if not verbose:
                    file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'metrics', metric, mode))
                    file_writer.set_as_default()

                for test_name in res.keys():
                    _, s, _ = res[test_name]
                    if verbose:
                        print(f"{test_name}: {s:.2f}. Epoch: {epoch}")
                    else:
                        tf.summary.scalar(os.path.join(f'similarity_{test_name}', metric, mode), data = s, step = epoch)
                        # TODO: add this back, need to define model.name field
                        # tf.summary.scalar(os.path.join(f'similarity_{test_name}', 'regimes', self.name), data = s, step = epoch)
                print('\r', end = '')
            return similarity_scalar

        # create a log for each (mode, metric) pair
        for mode in mode_list:
            for metric in metric_list:
                callback_func = callback_factory(mode, metric)
                callback = LambdaCallback(on_epoch_end = callback_func)
                callback_list.append(callback)

        return callback_list


    ## analogy test functions
    # read analogy tests from 'questions-words.txt', convert lists of words to 2D arrays of indices separately for semantic and syntactic tasks
    def read_analogy_file(self, test_file_path, delimeter = ' ', ignore_sections = False): # note the delimeter
        tasks_dict = {}
        id_array = []
        oov = 0
        tot = 0

        def record_section(section_name, id_array, oov, tot):
            tasks_dict[section_name] = (np.array(id_array, dtype = np.int32), oov, tot)

        cur_section_name = 'ALL' if ignore_sections else None
        with open(test_file_path, 'r') as test_file:
            for line in test_file:
                if line.startswith('#'): # skip comments
                    continue
                elif line.startswith(':'): # enter here even if ignore_sections
                    if not ignore_sections:
                        if cur_section_name: # if not ignore_sections and not the first section
                            record_section(cur_section_name, id_array, oov, tot)
                            id_array = []
                            oov = 0
                            tot = 0
                        cur_section_name = line[1:].strip()
                else:
                    w1, w2, w3, w4 = line.strip().lower().split(delimeter)
                    tot+= 1
                    try:
                        i1, i2, i3, i4 = self.word2id[w1], self.word2id[w2], self.word2id[w3], self.word2id[w4]
                        id_array.append([i1, i2, i3, i4])
                    except KeyError: # not in dict error
                        oov+= 1 # always ignore oov
        record_section(cur_section_name, id_array, oov, tot) # record last section
        return tasks_dict


    def analogy_results(self, tasks
                      , mode = 'target', metric = 'l2'
                      , ignore_triple = True, ignore_unknown = True):

        positive = 0
        batch_size = 2**9 # will split any test into batches of 512
        n_closest = 1 + (3 if ignore_triple else 0) + (1 if ignore_unknown else 0)

        analogy_dataset = tf.data.Dataset.from_tensor_slices(tasks).batch(batch_size)
        for analogy_batch in analogy_dataset:
            sample_emb = (
                    -self.get_embedding(analogy_batch[:,0], mode = mode)
                    +self.get_embedding(analogy_batch[:,1], mode = mode)
                    +self.get_embedding(analogy_batch[:,2], mode = mode)
                )

            closest = self.get_closest(sample_emb = sample_emb,
                                        n_closest = n_closest,
                                        mode = mode, metric = metric).numpy()


            for correct, preds, triple in zip(analogy_batch.numpy()[:,3], closest, analogy_batch.numpy()[:,:3]):
                if correct not in preds:
                    continue
                other = preds[:list(preds).index(correct)]
                admissible = list(triple) if ignore_triple else []
                admissible+= [0] if ignore_unknown else []

                if len(set(other) - set(admissible)) == 0:
                    positive+= 1
        return positive


    def fetch_analogy_results(self, tests_dict, mode, metric, verbose = False, delimeter = ' ', ignore_triple = True, ignore_unknown = True):
        res = {}
        for test_name, test_file_path in tests_dict.items():
            tasks_dict = self.read_analogy_file(test_file_path, delimeter = delimeter, ignore_sections = False)
            for section, (tasks, oov, tot) in tasks_dict.items():
                positive = self.analogy_results(tasks
                      , mode = mode, metric = metric
                      , ignore_triple = ignore_triple, ignore_unknown = ignore_unknown)
                res[section] = [res.get(section, [0,0,0])[0] + positive, res.get(section, [0,0,0])[1] + oov, res.get(section, [0,0,0])[2] + tot]

        if verbose:
            for section, (positive, oov, tot) in res.items():
                assert oov <= tot
                if oov < tot:
                    print(f'Analogy test for `{section}`'.ljust(40) + f'Accuracy: {positive/(tot-oov)}'.ljust(25) + f'OOV percent: {oov/tot}'.ljust(25))
                else:
                    print(f'Analogy test for `{section}`: None of the quadruples are in the dictionary')
                print(logstr)

        return res


    def group_analogy_test_results(self, out_dict, group_dict):
        res = {k:[0,0,0] for k in group_dict.keys()}
        init_keys = list(out_dict.keys())
        for section, (correct, oov, tot) in out_dict.items():
            matches = [name for name, sections in group_dict.items() if section in sections]
            for match in matches:
                res[match][0]+= correct
                res[match][1]+= oov
                res[match][2]+= tot
        return res


    def get_analogy_tests_callbacks(self, tests_dict, mode_list, metric_list, job_dir, log_dir = None, group_dict = None):
        callback_list = []

        if log_dir:
            log_path = os.path.join(job_dir, 'logs', log_dir)
        else:
            print('Logging directory for analogy tests is not specified, writing to `logs/temp` directory')
            log_path = os.path.join(job_dir, 'logs', 'temp')
        os.makedirs(log_path, exist_ok=True)

        def callback_factory(mode, metric, verbose = False, delimeter = ' ', ignore_triple = True, ignore_unknown = True, group_dict = group_dict):
            def analogy_scalar(epoch, logs):
                if verbose:
                    print('Running analogy tests...', end = '')
                res = self.fetch_analogy_results(tests_dict, mode, metric
                                                 , verbose = verbose, delimeter = delimeter
                                                 , ignore_triple = ignore_triple, ignore_unknown = ignore_unknown)

                if group_dict:
                    res = self.group_analogy_test_results(res, group_dict)

                if verbose:
                    if group_dict: # otherwise already printed stats before
                        for section, (positive, oov, tot) in res.items():
                            assert oov <= tot
                            if oov < tot:
                                print(f'Analogy test for `{section}`'.ljust(40) + f'Accuracy: {positive/(tot-oov)}'.ljust(25) + f'OOV percent: {oov/tot}'.ljust(25))
                            else:
                                print(f'Analogy test for `{section}`: None of the quadruples are in the dictionary')
                    return # do nothing else in verbose mode

                for section, (positive, oov, tot) in list(res.items()): # clean up completely unknow test sections
                    assert oov <= tot
                    if oov == tot:
                        del res[section]

                file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'metrics', metric, mode))
                file_writer.set_as_default()
                for section, (positive, oov, tot) in res.items():
                    tf.summary.scalar(os.path.join(f'analogy_{section}', metric, mode), data = positive/(tot-oov), step = epoch)
                    # TODO: add back regimes after defining model name
                    # tf.summary.scalar(f'analogy/regimes/{self.name}', data = res['both'][0], step = epoch)
            return analogy_scalar

        # create a log for each (mode, metric) pair
        for mode in mode_list:
            for metric in metric_list:
                callback_func = callback_factory(mode, metric)
                callback = LambdaCallback(on_epoch_end = callback_func)
                callback_list.append(callback)

        return callback_list


    def get_loss_callback(self, job_dir, log_dir):
        if log_dir:
            log_path = os.path.join(job_dir, 'logs', log_dir)
        else:
            print('Logging directory for loss records is not specified, writing to `logs/temp` directory')
            log_path = os.path.join(job_dir, 'logs', 'temp')
        os.makedirs(log_path, exist_ok=True)
        def loss_scalar(epoch, logs):

            file_writer = tf.summary.create_file_writer(os.path.join(log_path, 'train'))
            file_writer.set_as_default()
            tf.summary.scalar(name = 'loss', data = logs['loss'], step = epoch)
        return LambdaCallback(on_epoch_end = loss_scalar)



# Word2Vec model with NEG loss
class Word2VecNEGLoss(tf.keras.losses.Loss):

    def __init__(self, pow = 1., thresh = 100.):
        super().__init__()

    def call(self, y_true, posneg):
        p = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(posneg[:,0]), logits=posneg[:,0])
        n = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(posneg[:,1:]), logits=posneg[:,1:]), axis = -1)
        return tf.keras.backend.mean(y_true*(p+n), axis = -1)

class Word2VecModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, learning_rate = learning_rate, word2id = word2id, id2word = id2word)

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

        W_emb = self.T_embedding(inputs['target'])
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
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, learning_rate = learning_rate, word2id = word2id, id2word = id2word)


        # model specific part
        self.loss = GloveLoss()

        self.T_bias = Embedding(vocabulary_size, 1, input_length=1, name='target_bias')
        self.C_bias = Embedding(vocabulary_size, 1, input_length=1, name='context_bias')

        self.dot_layer = Dot(axes = -1)
        self.add_layer = Add()
        self.reshape_layer = Reshape((1, ))


    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        w_target = self.T_embedding(inputs['target'])
        w_context = self.C_embedding(inputs['pos'])
        b_target = self.T_bias(inputs['target'])
        b_context = self.C_bias(inputs['pos'])

        dotted = self.dot_layer([w_target, w_context])
        pred = self.add_layer([dotted, b_context, b_target])

        return self.reshape_layer(pred)


class HypGloveModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, learning_rate = learning_rate, word2id = word2id, id2word = id2word)

        # self.type = 'hypglove'
        self.loss = GloveLoss()

        self.T_bias = Embedding(vocabulary_size, 1, input_length=1, name='target_bias')
        self.T_sigma = Embedding(vocabulary_size, 1, input_length=1, embeddings_initializer = tf.keras.initializers.Constant(value=0.1), name='target_sigma')
        self.C_bias = Embedding(vocabulary_size, 1, input_length=1, name='context_bias')

        self.dot_layer = Dot(axes = -1)
        self.add_layer = Add()
        self.reshape_layer = Reshape((1, ))


    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        w_target = self.T_embedding(inputs['target'])
        w_context = self.C_embedding(inputs['pos'])
        b_target = self.T_bias(inputs['target'])
        b_context = self.C_bias(inputs['pos'])
        sigma_target = self.T_sigma(inputs['target'])

        w_scaled = (w_context - w_target)/sigma_target
        l2_scaled = self.dot_layer([w_scaled, w_scaled])/(-2)

        pred = self.add_layer([l2_scaled, b_target, b_context])

        return self.reshape_layer(pred)


    def validate_metric_name(self, metric):
        assert metric in ('l2', 'cos', 'fisher')

    def validate_emb_mode(self, emb_mode):
        assert emb_mode in ('distr', 'space', 'eval')
