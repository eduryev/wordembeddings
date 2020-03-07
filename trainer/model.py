
import collections
import numpy as np
import random
import tensorflow as tf


class Word2Vec:

    def __init__(self, text_words, max_vocabulary_size, min_occurrence, skip_window, num_skips, num_sampled, embedding_size, data_index = 0):
        self.text_words = text_words
        self.max_vocabulary_size = max_vocabulary_size
        self.min_occurrence = min_occurrence
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size

        self.data_index = data_index

        self.preprocess_data()
        self.define_weights()

    def preprocess_data(self):
        count = [('UNK', -1)]
        # Retrieve the most common words.
        count.extend(collections.Counter(self.text_words).most_common(self.max_vocabulary_size - 1))
        # Remove samples with less than 'min_occurrence' occurrences.
        for i in range(len(count) - 1, -1, -1):
            if count[i][1] < self.min_occurrence:
                count.pop(i)
            else:
                # The collection is ordered, so stop when 'min_occurrence' is reached.
                break
        # Compute the vocabulary size.
        self.vocabulary_size = len(count)
        # Assign an id to each word.
        self.word2id = dict()
        for i, (word, _)in enumerate(count):
            self.word2id[word] = i

        self.data = list()
        unk_count = 0
        for word in self.text_words:
            # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary.
            index = self.word2id.get(word, 0)
            if index == 0:
                unk_count += 1
            self.data.append(index)
        count[0] = ('UNK', unk_count)
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))

        print("Words count:", len(self.text_words))
        print("Unique words:", len(set(self.text_words)))
        print("Vocabulary size:", self.vocabulary_size)
        print("Most common words:", count[:10])

        # return data, vocabulary_size, word2id, id2word

    def next_batch(self, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # get window size (words left and right + current one).
        span = 2 * skip_window + 1
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch.
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def define_weights(self):
        self.embedding = tf.Variable(tf.random.normal([self.vocabulary_size, self.embedding_size]))
        # Construct the variables for the NCE loss.
        self.nce_weights = tf.Variable(tf.random.normal([self.vocabulary_size, self.embedding_size]))
        self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

    def get_embedding(self, x):
        # Lookup the corresponding embedding vectors for each sample in X.
        x_embed = tf.nn.embedding_lookup(self.embedding, x)
        return x_embed

    def nce_loss(self, x_embed, y):
        # Compute the average NCE loss for the batch.
        y = tf.cast(y, tf.int64)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                            biases=self.nce_biases,
                            labels=y,
                            inputs=x_embed,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocabulary_size))
        return loss

# # Evaluation.
# def evaluate(x_embed):
#     # with tf.device('/cpu:0'):
#     # Compute the cosine similarity between input data embedding and every embedding vectors
#     x_embed = tf.cast(x_embed, tf.float32)
#     x_embed_norm = x_embed / tf.sqrt(tf.reduce_sum(tf.square(x_embed)))
#     embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True), tf.float32)
#     cosine_sim_op = tf.matmul(x_embed_norm, embedding_norm, transpose_b=True)
#     return cosine_sim_op


# Optimization process.
    def run_optimization(self, optimizer, x, y):
        with tf.GradientTape() as g:
            emb = self.get_embedding(x)
            loss = self.nce_loss(emb, y)

        # Compute gradients.
        gradients = g.gradient(loss, [self.embedding, self.nce_weights, self.nce_biases])

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, [self.embedding, self.nce_weights, self.nce_biases]))
