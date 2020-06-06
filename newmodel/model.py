import os
import pickle as pkl
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Dot, Embedding, Input, Reshape, concatenate
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard

from scipy.stats import spearmanr, pearsonr

from newmodel.util import read_corpus_metadata


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

        self.T_embedding = Embedding(vocabulary_size, embedding_size, embeddings_initializer='uniform')
        self.C_embedding = Embedding(vocabulary_size, embedding_size, embeddings_initializer='uniform')

    # @tf.function(input_signature=[
    #                               tf.TensorSpec(shape = (None, ), dtype = tf.int32, name = 'target'),
    #                               tf.TensorSpec(shape = (None, ), dtype = tf.int32, name = 'context'),
    #                               tf.TensorSpec(shape = (None, neg_samples), dtype = tf.int32, name = 'neg')
    #                               ])


    # functions for searching closest words
    def validate_metric_name(self, metric):
        assert metric in ('l2', 'cos')

    def validate_emb_mode(self, emb_mode):
        assert emb_mode in ('combined', 'target', 'context')

    def get_embedding(self, emb_mode = 'target', id_or_word_array = None):
        self.validate_emb_mode(emb_mode)
        if id_or_word_array == None:
            if emb_mode == 'combined':
                embedding = self.T_embedding.get_weights()[0] + self.C_embedding.get_weights()[0]
            elif emb_mode == 'target':
                embedding = self.T_embedding.get_weights()[0]
            elif emb_mode == 'context':
                embedding = self.C_embedding.get_weights()[0]
            elif emb_mode == 'distr':
                embedding = np.concatenate([self.T_embedding.get_weights()[0], self.T_sigma.get_weights()[0]], axis = -1)
            elif emb_mode == 'eval':
                embedding = np.concatenate([self.T_embedding.get_weights()[0], self.T_sigma.get_weights()[0], self.C_embedding.get_weights()[0]], axis = -1)
            return embedding
        else:
            if isinstance(id_or_word_array, (np.ndarray, tf.Tensor)): #or len(id_or_word_array) == 0:
                id_array = id_or_word_array
            elif isinstance(id_or_word_array[0], int):
                id_array = np.array(id_or_word_array, dtype = np.int32)
            elif isinstance(id_or_word_array[0], str): # try converting words to ids
                assert self.word2id
                id_array = np.array([self.word2id[w] for w in id_or_word_array], dtype = np.int32)
            if emb_mode == 'combined':
                embedding = self.T_embedding(id_array).numpy() + self.C_embedding(id_array).numpy()
            elif emb_mode == 'target':
                embedding = self.T_embedding(id_array).numpy()
            elif emb_mode == 'context':
                embedding = self.C_embedding(id_array).numpy()
            elif emb_mode == 'distr':
                embedding = np.concatenate([self.T_embedding(id_array).numpy(), self.T_sigma(id_array).numpy()], axis = -1)
            elif emb_mode == 'eval':
                embedding = np.concatenate([self.T_embedding(id_array).numpy(), self.T_sigma(id_array).numpy(), self.C_embedding(id_array).numpy()], axis = -1)
            return embedding

    def evaluate_sim(self, vectors_1, vectors_2, metric):
        self.validate_metric_name(metric)
        if metric == 'cos':
            norm_vectors_1 = vectors_1/np.sqrt(np.square(vectors_1).sum(axis = 1, keepdims = True))
            norm_vectors_2 = vectors_2/np.sqrt(np.square(vectors_2).sum(axis = 1, keepdims = True))
            sim = -np.matmul(norm_vectors_1, norm_vectors_2.T)
        elif metric == 'l2':
            sim = eucl_dist(vectors_1, vectors_2)
        elif metric == 'fisher':
            emb_dim = self.embedding_size
            assert vectors_1.shape[1] == emb_dim + 1
            mu_emb, sigma_emb = vectors_1[:,:-1], vectors_1[:,-1:]
            mu_vec, sigma_vec = vectors_2[:,:-1], vectors_2[:,-1:]
            a = np.concatenate([mu_emb/np.sqrt(2*emb_dim), sigma_emb], axis = -1)
            b = np.concatenate([mu_vec/np.sqrt(2*emb_dim), -sigma_vec], axis = -1)
            c = np.concatenate([mu_vec/np.sqrt(2*emb_dim), sigma_vec], axis = -1)
            d1, d2 = eucl_dist(a,b), eucl_dist(a,c)
            sim = np.sqrt(2*emb_dim)*np.log((d1+d2)/(d1-d2))
        return sim

    def get_closest(self, embedding, vectors, word_ids, n_closest, metric):
        first_to_show = 0
        if vectors is None:
            assert word_ids is not None # at least one argument must be supplied
            first_to_show = 1
            vectors = embedding[word_ids]
        sim = self.evaluate_sim(embedding, vectors, metric)
        closest = np.argsort(sim, axis = 0)[first_to_show : first_to_show + n_closest, : ]
        return closest

    def print_closest(self, words, n_closest = 5, emb_mode = 'combined', metric = 'cos', log_header = None):
        self.validate_emb_mode(emb_mode)
        self.validate_metric_name(metric)
        embedding = self.get_embedding(emb_mode)
        word_ids = [self.word2id.get(word,0) for word in words]
        closest = self.get_closest(embedding, None, word_ids, n_closest, metric)
        if log_header:
            log_dir = "logs/" + self.name
            f = open(log_dir + '/synonyms.txt', 'a+')
            f.write(log_header + '\n')
        for i in range(len(word_ids)):
            log_str = f'Nearest to {self.id2word[word_ids[i]]}: '
            log_str += ', '.join(list(map(self.id2word.get,closest[:,i])))
            if log_header:
                f.write(log_str + '\n')
            else:
                print(log_str)
        if log_header:
            f.write('\n\n')
            f.close()

    ## analogy test functions
    # read analogy tests from 'questions-words.txt', convert lists of words to 2D=arrays of indices separately for semantic and syntactic tasks
    def read_analogy_file(self, filename):
        sem, syn, sem_oov, syn_oov = [], [], 0, 0
        with open(filename, 'r') as questions_file:
            for line in questions_file:
                if line.startswith(':'):
                    section = line[2:].replace('\n', '')
                    continue
                else:
                    task_words = line.replace('\n', '').lower().split(' ')
                    if section[:4] == 'gram':
                        try:
                            task_ids = [self.word2id[word] for word in task_words]
                            syn.append(task_ids)
                        except KeyError:
                            syn_oov += 1
                    else:
                        try:
                            task_ids = [self.word2id[word] for word in task_words]
                            sem.append(task_ids)
                        except KeyError:
                            sem_oov += 1
        return np.array(sem), np.array(syn), sem_oov, syn_oov

    # get analogy test results on given tasks (semantic, syntactic, ...)
    def analogy_results(self, tasks, emb_mode, metric, verbose = True, hyperbolic = False, ignore_triple = True, ignore_unknown = True):
        positive = 0
        batch_size = 1000
        embedding = self.get_embedding(emb_mode)
        for batch in range(math.ceil(tasks.shape[0]/batch_size)):
            print(f'\rRunning analogy tests...batch {batch+1}/{math.ceil(tasks.shape[0]/batch_size)}', end = '')
            start, end = batch*batch_size, min((batch+1)*batch_size, tasks.shape[0])
            tasks_batch = tasks[start:end]
            if hyperbolic:
                analogy_vecs = hyperbolic_trans(embedding[tasks_batch[:,0]], embedding[tasks_batch[:,1]], embedding[tasks_batch[:,2]])
            else:
                analogy_vecs = embedding[tasks_batch[:,2]] + embedding[tasks_batch[:,1]] - embedding[tasks_batch[:,0]]
                closest_ids = self.get_closest(embedding, analogy_vecs, None, 5, metric)
            for correct, preds, triple in zip(tasks_batch[:,3], closest_ids.T, tasks_batch[:,:3]):
                if correct not in preds:
                    continue
                other = preds[:list(preds).index(correct)]
                admissible = list(triple) if ignore_triple else []
                admissible += [0] if ignore_unknown else []
                if len(set(other) - set(admissible)) == 0:
                    positive += 1
            print('\r', end = '')
        total = tasks.shape[0]
        if verbose:
            logstr = f'\rScore: {positive}/{total}'.ljust(25) + f'Accuracy: {positive/total:.3f}'.ljust(25)
            print(logstr)
        else:
            return positive, total

    # gather analogy test results (semantic, syntactic and total) to a dictionary
    def fetch_analogy_results(self, test_file, emb_mode, metric, last_n = 0, verbose = False, hyperbolic = False, ignore_triple = True, ignore_unknown = True):
        res = {}
        sem, syn, sem_oov, syn_oov = self.read_analogy_file(test_file)
        sem_pos, sem_tot = self.analogy_results(sem[-last_n:], emb_mode, metric, False, hyperbolic, ignore_triple, ignore_unknown)
        syn_pos, syn_tot = self.analogy_results(syn[-last_n:], emb_mode, metric, False, hyperbolic, ignore_triple, ignore_unknown)
        res['both'] = ((sem_pos + syn_pos)/(sem_tot + syn_tot), (sem_oov + syn_oov)/(len(sem)+len(syn)))
        res['semantic'] = (sem_pos/sem_tot, sem_oov/len(sem))
        res['syntactic'] = (syn_pos/syn_tot, syn_oov/len(syn))
        if verbose:
            for test in res.keys():
                logstr = f'{test}'.ljust(40) + f'Accuracy {res[test][0]:.5f}'.ljust(25) + f'OOV {res[test][1]:.5f}'.ljust(25)
                print(logstr)
        else:
            return res

    ## similariy tests functions
    # read similarity file to list of pairs (tasks), list of floats (human similarity scores), number of out-of-vocabulary tasks
    def read_similarity_file(self, test_file, ignore_oov = True):
        tasks, gold_sim, oov = [], [], 0
        vocab = list(self.word2id.keys())
        with open(test_file, 'r') as f:
            for line_no, line in enumerate(f.readlines()):
                if line.startswith('#'):
                    continue
                else:
                    a, b, sim = line.lower().split('\t')
                    sim = float(sim)
                    if a not in vocab or b not in vocab:
                        oov += 1
                        if not ignore_oov:
                            tasks.append([0,0])
                            gold_sim.append(sim)
                    else:
                        a, b = self.word2id[a], self.word2id[b]
                        tasks.append([a,b])
                        gold_sim.append(sim)
        return tasks, gold_sim, oov

    # get results of a similarity test (from file)
    def similarity_results(self, test_file, emb_mode, metric, ignore_oov = True):
        model_sim = []
        tasks, gold_sim, oov = self.read_similarity_file(test_file, ignore_oov)
        embedding = self.get_embedding(emb_mode)
        for a,b in tasks:
            if a == 0:
                print('Strange')
                model_sim.append(0)
            else:
                a_vec, b_vec = embedding[a][np.newaxis,:], embedding[b][np.newaxis,:]
                model_sim.append(-self.evaluate_sim(a_vec, b_vec, metric)[0,0])
        spearman = stats.spearmanr(gold_sim, model_sim)
        pearson = stats.pearsonr(gold_sim, model_sim)
        if ignore_oov:
            oov_ratio = float(oov) / (len(gold_sim) + oov) * 100
        else:
            oov_ratio = float(oov) / len(gold_sim) * 100
        return pearson, spearman, oov_ratio

    # gather all similarity test results (across all files) to a dictionary
    def fetch_similarity_results(self, test_dir, test_files, emb_mode, metric, verbose = False, ignore_oov = True):
        res = {}
        for test_file in test_files.keys():
            test_name = test_files[test_file]
            p, s, o = self.similarity_results(test_dir + test_file, emb_mode, metric, ignore_oov)
            res[test_name] = (p[0], s[0], o)
            if verbose:
                logstr = f'{test_name}'.ljust(10) + f'Pearson {p[0]:.3f}'.ljust(25) + f'Spearman {s[0]:.3f}'.ljust(25) + f'OOV: {o:.2f}%'.ljust(25)
                print(logstr)
        if not verbose:
            return res

    ## defining and fetching callbacks
    # analogy
    def analogy_scalar(self, epoch, emb_mode, metric, last_n = 0, hyperbolic = False, verbose = False, ignore_triple = True, ignore_unknown = True):
        if (epoch+1)%10==0 or epoch == 0:
            test_file = analogy_test_dir + 'questions-words.txt'
            res = self.fetch_analogy_results(test_file, emb_mode, metric, last_n, False, hyperbolic, ignore_triple, ignore_unknown)
            if not os.path.exists('logs/'):
                os.mkdir('logs/')
            log_dir = 'logs/' + self.name
            method = '/hyp' if hyperbolic else ''
            if verbose:
                print(f"QWsem: {res['semantic'][0]}. Epoch: {epoch}")
                print(f"QWsyn: {res['syntactic'][0]}. Epoch: {epoch}")
                print(f"QW: {res['both'][0]}. Epoch: {epoch}")
            else:
                file_writer = tf.summary.create_file_writer(log_dir + f'/metrics/{metric}/{emb_mode}' + method)
                file_writer.set_as_default()
                tf.summary.scalar(f'analogy_semantic/{metric}/{emb_mode}' + method, data = res['semantic'][0], step = epoch)
                tf.summary.scalar(f'analogy_syntactic/{metric}/{emb_mode}' + method, data = res['syntactic'][0], step = epoch)
                tf.summary.scalar(f'analogy/{metric}/{emb_mode}' + method, data = res['both'][0], step = epoch)
                tf.summary.scalar(f'analogy_semantic/regimes/{self.name}', data = res['semantic'][0], step = epoch)
                tf.summary.scalar(f'analogy_syntactic/regimes/{self.name}', data = res['syntactic'][0], step = epoch)
                tf.summary.scalar(f'analogy/regimes/{self.name}', data = res['both'][0], step = epoch)

    # similarity
    def similarity_scalar(self, epoch, emb_mode, metric, verbose = False, ignore_oov = True):
        print('Running similarity tests...', end = '')
        test_dir = similarity_test_dir
        test_files = similarity_test_files
        res = self.fetch_similarity_results(test_dir, test_files, emb_mode, metric, False, ignore_oov)
        if not os.path.exists('logs/'):
            os.mkdir('logs/')
        log_dir = "logs/" + self.name
        if not verbose:
            file_writer = tf.summary.create_file_writer(log_dir + f'/metrics/{metric}/{emb_mode}')
            file_writer.set_as_default()
        for test_name in res.keys():
            _, s, _ = res[test_name]
            if verbose:
                print(f"{test_name}: {s:.2f}. Epoch: {epoch}")
            else:
                tf.summary.scalar(f'similarity_{test_name}/{metric}/{emb_mode}', data = s, step = epoch)
                tf.summary.scalar(f'similarity_{test_name}/regimes/{self.name}', data = s, step = epoch)
        print('\r', end = '')


    def loss_scalar(self, epoch, logs):
        print('Recording loss...', end = '')
        if not os.path.exists('logs/'):
            os.mkdir('logs/')
        log_dir = "logs/" + self.name
        file_writer = tf.summary.create_file_writer(log_dir + f"/train")
        file_writer.set_as_default()
        tf.summary.scalar(name = 'loss', data = logs['loss'], step = epoch)
        print('\r', end = '')


    def histogram_scalar(self, epoch):
        print('Updating histograms...', end = '')
        if not os.path.exists('logs/'):
            os.mkdir('logs/')
        log_dir = "logs/" + self.name
        file_writer = tf.summary.create_file_writer(log_dir + f"/train")
        file_writer.set_as_default()
        tf.summary.histogram(name = 'target_embedding', data = self.T_embedding.weights, step = epoch)
        tf.summary.histogram(name = 'context_embedding', data = self.C_embedding.weights, step = epoch)
        if self.type == 'glove' or self.type == 'hypglove':
            tf.summary.histogram(name = 'target_bias', data = self.T_bias.weights, step = epoch)
            tf.summary.histogram(name = 'context_bias', data = self.C_bias.weights, step = epoch)
        if self.type == 'hypglove':
            tf.summary.histogram(name = 'target_sigma', data = self.T_sigma.weights, step = epoch)
        print('\r', end = '')


    def save_train_config(self, save_path, epochs_trained, args):
        '''
        Save train arguments, optimizer state and train epoch
        '''
        print(f'Saving model configuration to {save_path}')

        if save_path[:5] == 'gs://':
            bucket_name, path_name = split_gs_prefix(save_path)
        else:
            path_name = save_path
        os.makedirs(os.path.basename(path_name), exist_ok = True)

        # save train arguments
        args_dict = dict(vars(args))
        args_dict['epochs_trained'] = epochs_trained
        save_dict(args_dict, os.path.join(path_name, 'args.pkl'))

        # save optimizer state
        optimizer_weights = tf.keras.backend.batch_get_value(self.optimizer.weights)
        with open(os.path.join(path_name, 'optimizer.pkl'), 'wb') as f:
            pkl.dump(optimizer_weights, f)

        if save_path[:5] == 'gs://':
            upload_to_gs(os.path.join(path_name, 'args.pkl'), bucket_name)
            upload_to_gs(os.path.join(path_name, 'optimizer.pkl'), bucket_name)

    def get_save_callbacks(self, save_path, args):
        if save_path is None:
            save_path = args.save_path

        callbacks = []
        def save_config_callback_factory(save_path, args):
            def save_train_config(epochs_trained, logs):
                 self.save_train_config(save_path, epochs_trained, args)
            return save_train_config


        ckpt_path = os.path.join(save_path, 'cp-{epoch:04d}.ckpt')
        # save_weights_only = False has a bug, hopefully will get resolved:
        # https://github.com/tensorflow/tensorflow/issues/39679#event-3376275799
        # at any rate, saving custom models requires some effort, so rely on load_weights
        cp_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path, save_weights_only = True,
                                                    verbose = 1, max_to_keep = 5, period = 3)
        callbacks.append(cp_ckpt)
        callbacks.append(LambdaCallback(on_epoch_end = save_config_callback_factory(save_path, args)))

        return callbacks

    # note, this funcion belongs to a class, not class instance
    def load_train_config(load_path):
        '''
        Load train arguments, optimizer state and train epoch
        '''
        if load_path[:5] == 'gs://':
            download_from_gs(os.path.join(load_path), 'args.pkl')
            load_path = download_from_gs(os.path.join(load_path), 'optimizer.pkl')
            load_path = os.path.basename(load_path)

        with open(os.path.join(load_path, 'args.pkl'), 'rb') as f:
            args = pkl.load(f)
        with open(os.path.join(load_path, 'optimizer.pkl'), 'rb') as f:
            optimizer_weights = pkl.load(f)

        epochs_trained = args['epochs_trained']
        del args['epochs_trained']
        args = SimpleNamespace(**args)
        return args, epochs_trained, optimizer_weights

    # TODO check that it works for GCP
    def load_model(self, load_path):
        '''
        Loads the model with configuration from a given path
        '''
        args_, epochs_trained_, optimizer_weights_ = BaseModel.load_train_config(load_path)
        latest = tf.train.latest_checkpoint(load_path)
        # initial_epoch = int(os.path.basename(latest)[-9:-5])
        dummy_key = {'target': np.zeros(1, dtype = np.int32), 'pos': np.zeros(1, dtype = np.int32), 'neg': np.zeros(1, dtype = np.int32)}
        dummy_val = np.zeros(1, dtype = np.float32)

        if not self._is_compiled:
            self.compile(loss = self.loss, optimizer = self.optimizer)

        self.fit(dummy_key, dummy_val, epochs=1, verbose=0)
        self.optimizer.set_weights(optimizer_weights_)
        self.load_weights(latest)
        return args_, epochs_trained_


    def fetch_callbacks(self, args, save_callbacks = True, similarity = True, analogy = True, synonym = True):
        callbacks = []

        # assign directories
        try:
            log_dir = os.path.join(args.job_dir, args.log_dir, self.name)
        except AttributeError:
            log_dir = os.path.join(args.job_dir, 'logs', self.name)

        try:
            save_dir = args.save_path
        except AttributeError:
            save_dir = os.path.join(args.job_dir, 'saved_models', self.name)

        os.makedirs(log_dir, exist_ok = True)
        os.makedirs(save_dir, exist_ok = True)
        # save_path = f'saved_models/{self.name}.hdf5'

        # checkpoint
        # save_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.save_custom_model(epoch, args))
        callbacks+= self.get_save_callbacks(args.save_path, args)

        # tensorboard
        tensorboard_callback = TensorBoard(log_dir, histogram_freq = 0)
        # histogram
        histogram_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.histogram_scalar(epoch))
        # loss
        loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: self.loss_scalar(epoch, logs))
        # put in callbacks
        callbacks += [tensorboard_callback, histogram_callback, loss_callback]
        # synonyms
        syn_words = ['three','third','russian','cold','sweet','large','poor','dog','tree','book','country','football','human','universe','life','love','liberty','health','faith','progress','history','cowardice','hyperbole','clandestine','serendipity']
        synonym_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.print_closest(syn_words, 9, 'context', 'cos',
                                                                    log_header = f'Epoch {epoch+1}') if (epoch+1)%5==0 else None)
        # analogy
        analogy_combined_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.analogy_scalar(epoch, 'combined', 'cos'))
        analogy_target_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.analogy_scalar(epoch, 'target', 'cos'))
        analogy_context_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.analogy_scalar(epoch, 'context', 'cos'))
        analogy_callbacks = [analogy_combined_cos_callback, analogy_target_cos_callback, analogy_context_cos_callback]
        # similarity
        sim_combined_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'combined', 'cos'))
        sim_target_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'target', 'cos'))
        sim_context_cos_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'context', 'cos'))
        sim_combined_l2_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'combined', 'l2'))
        sim_target_l2_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'target', 'l2'))
        sim_context_l2_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'context', 'l2'))
        similarity_callbacks = [sim_combined_cos_callback, sim_combined_l2_callback, sim_target_cos_callback,
                                sim_target_l2_callback, sim_context_cos_callback, sim_context_l2_callback]
        #similarity_callbacks = [sim_combined_cos_callback, sim_target_cos_callback]
        # extra analogy and similarity for hypglove
        if self.type == 'hypglove':
            # analogy for hypglove
            analogy_distr_fisher_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.analogy_scalar(epoch, 'distr', 'fisher', False))
            analogy_distr_fisher_hyp_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.analogy_scalar(epoch, 'distr', 'fisher', True))
            analogy_callbacks += [analogy_distr_fisher_callback, analogy_distr_fisher_hyp_callback]
            # similarity for hypglove
            sim_distr_fisher_callback = LambdaCallback(on_epoch_end=lambda epoch,logs: self.similarity_scalar(epoch, 'distr', 'fisher'))
            similarity_callbacks += [sim_distr_fisher_callback]
        if similarity:
            callbacks += similarity_callbacks
        if analogy:
            callbacks += analogy_callbacks
        if synonym:
            callbacks += [synonym_callback]
        return callbacks




# Word2Vec model with NEG loss
class Word2VecNEGLoss(tf.keras.losses.Loss):
    def __init__(self, pow = 1., thresh = 100.):
        super().__init__()

    def call(self, y_true, posneg):
        S = tf.sigmoid(-posneg)
        L = y_true*(-tf.math.log(1 - S[:,0]) - tf.reduce_sum(tf.math.log(S[:, 1:]), axis = -1))
        return tf.keras.backend.mean(L, axis = -1)


class Word2VecModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, learning_rate = learning_rate, word2id = word2id, id2word = id2word)

        # model specific part
        self.type = 'w2v'

        self.dot_layer = Dot(axes = -1)
        self.reshape_layer1 = Reshape((1, ))
        self.reshape_layer2 = Reshape((1+ neg_samples, ))

        self.loss = Word2VecModel.custom_loss

    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        pos = self.reshape_layer1(inputs['pos'])
        pn = concatenate([pos, inputs['neg']], axis = -1)

        T_emb = self.T_embedding(inputs['target'])
        C_pn = self.C_embedding(pn)

        posneg_ = self.dot_layer([T_emb, C_pn])
        posneg = self.reshape_layer2(posneg_)

        return posneg

    def custom_loss(y_true, posneg):
        S = tf.sigmoid(-posneg)
        L = y_true*(-tf.math.log(1 - S[:,0]) - tf.reduce_sum(tf.math.log(S[:, 1:]), axis = -1))
        return tf.keras.backend.mean(L, axis = -1)


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
        self.type = 'glove'

        self.T_bias = Embedding(vocabulary_size, 1, input_length=1, name='target_bias')
        self.C_bias = Embedding(vocabulary_size, 1, input_length=1, name='context_bias')

        self.dot_layer = Dot(axes = -1)
        self.add_layer = Add()
        self.reshape_layer = Reshape((1, ))

        self.loss = GloveModel.custom_loss


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

    # def custom_loss(self, y_true, y_pred):
    #     return tf.keras.backend.mean(y_true * tf.square(y_pred - tf.math.log(y_true)), axis=-1)

    # def custom_loss(self, y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):
    #     return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) * K.square(y_pred - K.log(y_true)), axis=-1)

    def custom_loss(y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):
        return tf.reduce_mean(tf.math.pow(tf.clip_by_value(y_true / X_MAX, 0.0, 1.0), a) * tf.square(y_pred - tf.math.log(y_true)), axis=-1)


class HypGloveModel(BaseModel):
    def __init__(self, vocabulary_size, embedding_size, neg_samples, learning_rate = None, word2id = None, id2word = None):
        super().__init__(vocabulary_size, embedding_size, neg_samples, learning_rate = learning_rate, word2id = word2id, id2word = id2word)

        # model specific part
        self.type = 'hypglove'

        self.T_bias = Embedding(vocabulary_size, 1, input_length=1, name='target_bias')
        self.T_sigma = Embedding(vocabulary_size, 1, input_length=1, embeddings_initializer = tf.keras.initializers.Constant(value=0.1), name='target_sigma')
        self.C_bias = Embedding(vocabulary_size, 1, input_length=1, name='context_bias')

        self.dot_layer = Dot(axes = -1)
        self.add_layer = Add()
        self.reshape_layer = Reshape((1, ))

        self.loss = HypGloveModel.custom_loss

    def call(self, inputs):
        """
        Returns an array of scalar products and embeddings of positive/negative samples
        """
        w_target = self.T_embedding(inputs['target'])
        w_context = self.C_embedding(inputs['pos'])
        b_target = self.T_bias(inputs['target'])
        b_context = self.C_bias(inputs['pos'])
        sigma_target = self.T_sigma(inputs['target'])

        w_target_scaled = w_target/sigma_target
        w_context_scaled = w_context/sigma_target

        context_scaled_norm = self.dot_layer([w_context_scaled, w_context_scaled])/(-2)
        target_context_scaled_dot = self.dot_layer([w_target_scaled, w_context_scaled])

        pred = self.add_layer([target_context_scaled_dot, context_scaled_norm, b_target, b_context])

        return self.reshape_layer(pred)

    # def custom_loss(self, y_true, y_pred):
    #     return tf.keras.backend.mean(y_true * tf.square(y_pred - tf.math.log(y_true)), axis=-1)

    def custom_loss(y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):
        return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) * K.square(y_pred - K.log(y_true)), axis=-1)

    def validate_metric_name(self, metric):
        assert metric in ('l2', 'cos', 'fisher')

    def validate_emb_mode(self, emb_mode):
        assert emb_mode in ('distr', 'space', 'eval')


def load_custom_model(model_path, meta_path, mode = None):
    # load args

    args_, num_epochs_, optimizer_weights_ = BaseModel.load_train_config(model_path)
    word2id, id2word, word_counts, id_counts = read_corpus_metadata(meta_path)
    vocabulary_size = max(id2word.keys()) + 1 # might be slightly more accurate

    # with open(os.path.join(), 'rb') as f:
    #     args = SimpleNamespace(**pkl.load(f))
    # # create dataset for model
    # print(f'\rCreating training dataset for model...', end = '')
    # if mode == 'alternative':
    #     print(f'\rLoading alternative training dataset from {args.corpus_name}...', end = '')
    #     corpus_dir = '/content/drive/My Drive/Word Embeddings/datasets/enwik9/'
    #     word2count, word2id, id2word = load_vocabs(corpus_dir)
    #     vocabulary_size = len(word2id)
    #     target, context, cooccurrence = load_training_data(corpus_dir)
    # elif mode == None:
    #     print(f'\rLoading training dataset from {args.corpus_name}...', end = '')
    #     train_file_name = 'stored_{corpus_name}_maxsize_{max_vocabulary_size}_minocc_{min_occurrence}_window_{skip_window}_storedbatch_{stored_batch_size}'.format(**dict(args.__dict__))
    #     train_file_path = os.path.join(args.job_dir, 'model_data', train_file_name)
    #     word2id, id2word, word_counts, id_counts = load_process_data(train_file_name, args)
    #     vocabulary_size = len(word2id)
    #     arr_counts = np.array([id_counts[i] for i in range(len(id2word))], dtype = np.float32)
    #     arr_counts[:] = arr_counts**args.po
    #     unigram = arr_counts/arr_counts.sum()
    #     dataset = create_dataset_from_stored_batches(train_file_path, args.batch_size, args.stored_batch_size, unigram, args.threshold, args.po, neg_samples = args.neg_samples)
    # # create model

    model_name = os.path.basename(model_path)
    if mode is None:
        mode = model_name.split('_')[0]
    print(f'Creating new {mode} model instance...')

    # TODO: make sure those are accessible
    class_dict = {'w2v': Word2VecModel, 'glove': GloveModel, 'hypglove': HypGloveModel}
    model = class_dict[mode](vocabulary_size, args_.embedding_size, args_.neg_samples, learning_rate = args_.learning_rate, word2id = word2id, id2word = id2word)
    args_, epochs_trained_ =  model.load_model(model_path)

    return model, args_, epochs_trained_

    # # compile model
    # fit_dict = {'w2v': {'target': np.array([0]), 'pos': np.array([0]), 'neg':np.array([0])}, 'glove': {'target': np.array([0]), 'pos': np.array([0])}, 'hypglove': {'target': np.array([0]), 'pos': np.array([0])}}
    # model.compile(loss = model.custom_loss, optimizer = model.optimizer)
    # model.fit(fit_dict[model_type], np.array([0]), epochs=1, verbose=0)
    # # load weights
    # print(f'\rLoading weights and optimizer state from {model_name}...', end = '')
    # model.load_weights(model_dir + '/weights.h5')
    # # load optimizer state
    # with open(model_dir + '/optimizer.pkl', 'rb') as f:
    #     optimizer_weights = pkl.load(f)
    # model.optimizer.set_weights(optimizer_weights)
    # # set model name and number of trained epochs
    # model._name = model_name
    # model.epochs_trained = args.epochs_trained
    # print(f'\rLoaded model {model.name}.')
    # if mode == 'alternative':
    #     return model, target, context, cooccurrence, args
    # elif mode == None:
    #     return model, dataset, args



### DEMONSTRATION FUNCTIONS ###

def load_dict(path):
    with open(path,'rb') as f:
        d = pkl.load(f)
    return d

# define custom loss function
def custom_loss(y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):
    return tf.keras.backend.K.sum(tf.keras.backend.K.pow(tf.keras.backend.K.clip(y_true / X_MAX, 0.0, 1.0), a) * tf.keras.backend.K.square(y_pred - tf.keras.backend.K.log(y_true)), axis=-1)

def load_demo_model(model_name, job_dir = '/content/drive/My Drive/shared_project'):
    if 'enwik9' in model_name:
        corpus_name = 'enwik9'
    elif 'wikidump' in model_name:
        corpus_name = 'wikidump'
    else:
        raise Exception('Unknown corpus name. Has to be enwik9 or wikidump')
    # load corpus data
    corpus_dir = os.path.join(job_dir, 'datasets', corpus_name)
    word2count = load_dict(corpus_dir + '/word2count.pkl')
    word2id = load_dict(corpus_dir + '/word2id.pkl')
    id2word = load_dict(corpus_dir + '/id2word.pkl')
    # load model
    model_path = os.path.join(job_dir, 'saved_weights', model_name + '.hdf5')
    model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
    # load info
    info_path = os.path.join(job_dir, 'saved_infos', model_name +  '_info.pkl')
    info = load_dict(info_path)
    return model, info, word2count, word2id, id2word

def print_model_info(info):
    print('Model name:'.ljust(30), info['model_name'])
    print('Trained epochs:'.ljust(30), info['epochs_trained'])
    print('Embedding dimension:'.ljust(30), info['embedding_dimension'])
    print('Corpus name:'.ljust(30), info['corpus_name'])
    print('Corpus length:'.ljust(30), info['corpus_length'])
    print('Vocabulary size:'.ljust(30), info['vocab_size'])
    print('Min word occurrence:'.ljust(30), info['min_occurrence'])
    print('Min sentence length:'.ljust(30), info['min_sentence_length'])
    print('Training samples:'.ljust(30), info['training_samples'])
    print('Optimizer:'.ljust(30), info['optimizer'],'(learning rate:', info['learning_rate'], ', beta_1:', info['beta_1'],', beta_2:', info['beta_2'], ', rho:', info['rho'],')')
    print('Batch exponent:'.ljust(30), info['batch_exponent'])
    print('Note:'.ljust(30), info['note'])

def get_embedding(model, emb_mode = 'target'):
    if emb_mode == 'combined':
        embedding = model.get_layer('target_embedding').get_weights()[0] + model.get_layer('context_embedding').get_weights()[0]
    elif emb_mode == 'target':
        embedding = model.get_layer('target_embedding').get_weights()[0]
    elif emb_mode == 'context':
        embedding = model.get_layer('context_embedding').get_weights()[0]
    elif emb_mode == 'distr':
        embedding = np.concatenate([model.get_layer('target_embedding').get_weights()[0], model.get_layer('target_sigma').get_weights()[0]], axis = -1)
    elif emb_mode == 'eval':
        embedding = np.concatenate([model.get_layer('target_embedding').get_weights()[0], model.get_layer('target_sigma').get_weights()[0], model.get_layer('context_embedding').get_weights()[0]], axis = -1)
    return embedding

def eucl_dist(W,V):
    # W is array of shape (vocab_size, emb_dim), V is array of shape (k, emb_dim)
    W_sqnorm = np.square(W).sum(axis = 1, keepdims = True)
    V_sqnorm = np.square(V).sum(axis = 1, keepdims = True)
    X = W_sqnorm + V_sqnorm.T - 2*np.matmul(W, V.T)
    X[X<0] = 0
    return np.sqrt(X)

def evaluate_sim(model, vectors_1, vectors_2, metric):
    if metric == 'cos':
        norm_vectors_1 = vectors_1/np.sqrt(np.square(vectors_1).sum(axis = 1, keepdims = True))
        norm_vectors_2 = vectors_2/np.sqrt(np.square(vectors_2).sum(axis = 1, keepdims = True))
        sim = -np.matmul(norm_vectors_1, norm_vectors_2.T)
    elif metric == 'l2':
        sim = eucl_dist(vectors_1, vectors_2)
    elif metric == 'fisher':
        emb_dim = model.info['embedding_dimension']
        assert vectors_1.shape[1] == emb_dim + 1
        mu_emb, sigma_emb = vectors_1[:,:-1], vectors_1[:,-1:]
        mu_vec, sigma_vec = vectors_2[:,:-1], vectors_2[:,-1:]
        a = np.concatenate([mu_emb/np.sqrt(2*emb_dim), sigma_emb], axis = -1)
        b = np.concatenate([mu_vec/np.sqrt(2*emb_dim), -sigma_vec], axis = -1)
        c = np.concatenate([mu_vec/np.sqrt(2*emb_dim), sigma_vec], axis = -1)
        d1, d2 = eucl_dist(a,b), eucl_dist(a,c)
        sim = np.sqrt(2*emb_dim)*np.log((d1+d2)/(d1-d2))
    return sim

def get_closest(model, embedding, vectors, word_ids, n_closest, metric):
    first_to_show = 0
    if vectors is None:
        assert word_ids is not None # at least one argument must be supplied
        first_to_show = 1
        vectors = embedding[word_ids]
    sim = evaluate_sim(model, embedding, vectors, metric)
    closest = np.argsort(sim, axis = 0)[first_to_show : first_to_show + n_closest, : ]
    return closest

def print_closest(model, words, n_closest = 5, emb_mode = 'combined', metric = 'cos', word2id = None, id2word = None, log_header = None):
    embedding = get_embedding(model, emb_mode)
    word_ids = [word2id.get(word,0) for word in words]
    closest = get_closest(model, embedding, None, word_ids, n_closest, metric)
    if log_header:
        log_dir = logs_dir + model.name
        f = open(log_dir + '/synonyms.txt', 'a+')
        f.write(log_header + '\n')
    for i in range(len(word_ids)):
        log_str = f'Nearest to {id2word[word_ids[i]]}: '
        log_str += ', '.join(list(map(id2word.get,closest[:,i])))
        if log_header:
            f.write(log_str + '\n')
        else:
            print(log_str)
    if log_header:
        f.write('\n\n')
        f.close()

# define hyperbolic translation
def hyperbolic_trans(X, Y, Z):
    Sigma_X = X[:,-1][:,np.newaxis]
    Sigma_Y = Y[:,-1][:,np.newaxis]
    K_X = Sigma_Y/(Sigma_Y-Sigma_X)
    K_Y = Sigma_X/(Sigma_X-Sigma_Y)
    Center = X*K_X + Y*K_Y
    if np.max(np.abs(Center)) > 10**6:
        print(np.sum(np.abs(Center) > 10**6))
    return Center*(Sigma_X-Sigma_Y)/Sigma_X + Z*Sigma_Y/Sigma_X

def solve_analogy(task, model, emb_mode, metric, hyperbolic = False, word2id = None, id2word = None):
    embedding = get_embedding(model, emb_mode)
    tasks = np.array([[word2id.get(word,0) for word in task]])
    if hyperbolic:
        analogy_vecs = hyperbolic_trans(embedding[tasks[:,0]], embedding[tasks[:,1]], embedding[tasks[:,2]])
    else:
        analogy_vecs = embedding[tasks[:,2]] + embedding[tasks[:,1]] - embedding[tasks[:,0]]
    closest_ids = get_closest(model, embedding, analogy_vecs, None, 10, metric)
    answers = []
    for guess in closest_ids[:,0]:
        if guess not in tasks[0] and guess != 0:
            answers.append(id2word[guess])
    print(f'Question: {task[0]} to {task[1]} is like {task[2]} to what?')
    print(f'Guesses: ' + ', '.join(answers[:5]) + '.')

def register_embedding(weights, word2id, log_dir) -> None:
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # Create a checkpoint from embedding, the filename and key are
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for word in word2id:
            f.write("{}\n".format(word))

    # Set up config
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)
