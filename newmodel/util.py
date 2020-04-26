
import numpy as np
import tensorflow as tf

import subprocess
import os
import urllib
import zipfile

from collections import deque, Counter
from tqdm import tqdm


def load_unpack_zip(corpus_name, job_dir):
    """
    Download zip file from Matt Mahoney's website and unpack to `job_dir`
    """
    zip_file_name = f'{corpus_name}.zip'
    zip_file_path = os.path.join(job_dir, zip_file_name)
    if os.path.exists(zip_file_path):
        pass
    else:
        print(f'Downloading corpus file {zip_file_name}...')
        url = 'http://mattmahoney.net/dc/' + zip_file_name
        urllib.request.urlretrieve(url, zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_to_unpack:
        zip_to_unpack.extractall(job_dir)


def load_raw_data(corpus_name, job_dir, perl_cleanup = True):
    # TODO: make sure this works for both perl_cleanup values
    text_file_name = f'{corpus_name}.txt'
    text_file_path = os.path.join(job_dir, text_file_name)
    print(text_file_path)
    if os.path.exists(text_file_path):
        print(f'File {text_file_name} already exists. Nothing to be done.')
        return text_file_path
    else:
        load_unpack_zip(corpus_name, job_dir)
    if perl_cleanup:
        assert os.path.exists('main_.pl')
        bash_str = f'perl main_.pl {text_file_path[:-4]} > {text_file_path}'
        print('Cleaning up the corpus...')
        subprocess.run(bash_str, shell = True)
        print(f'Done loading the corpus. File is writtten to {text_file_name}')
    else: # text8 is already processed, so just need to add .txt extenstion
        os.rename(text_file_path[:-4], text_file_path)
    return text_file_path

from collections import deque

def count_skips(id_array, skip_window=5):
    def postprocess_count_skips(skips):
        return np.array([[k, i, j] for (i, j), k in skips.items()], dtype = np.int32)

    d = dict()
    corpus_len = len(id_array)
    assert corpus_len >= skip_window

    buffer = deque(maxlen = skip_window)
    buffer.extend(id_array[:skip_window])

    for word_id, new_word_id in tqdm(zip(id_array[:-skip_window],id_array[skip_window:])):
        buffer.append(new_word_id)
        for ind, w in enumerate(buffer):
          i, j = (word_id, w) if word_id < w else (w, word_id)
          d[(i,j)]= d.get((i,j),0) + 1./(ind+1)

    tail = id_array[-skip_window:]
    for k, word_id in enumerate(tail[:-1]):
        for ind, new_word_id in enumerate(tail[k+1:]):
          i, j = (word_id, new_word_id) if word_id < new_word_id else (new_word_id, word_id)
          d[(i,j)]= d.get((i,j),0) + 1./(ind + 1)

    target, context, cooccurences = [], [], []
    # mirror_d = {}
    # # mirror_d = {(j,i): val for (i,j), val in d.items() if i!=j}
    # for key, val in d.items():
    # if key[0]!= key[1]:
    #     mirror_d[(key[1], key[0])] = val
    res = postprocess_count_skips(d)

    # concatenate with mirror version
    return np.concatenate([res, res[np.where(res[:,1]!=res[:,2])][:,[0,2,1]]], axis = 0)


# TODO: is it resilient for gigabytes of data?
def preprocess_data(text_file_path, max_vocabulary_size, min_occurrence, skip_window):

  '''
  Create from a corpus name a triple (word2id, id2word, word_counts, skips)

    'word2id' dictionary keyed on words
    'id2word' dictionary keyed on indices
    'word_counts' dictionary keyed on words
    'skips' np.array of triples returned by 'count_skips'
  '''
  with open(text_file_path) as text_file:
    word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())

  word_counts = Counter(word_array).most_common(max_vocabulary_size - 1)

  for i in range(len(word_counts) - 1, -1, -1):
    if word_counts[i][1] < min_occurrence:
      word_counts.pop()
    else:
      break

  tot = sum([w[1] for w in word_counts])
  word_counts.insert(0, ('UNK', len(word_array) - tot))

  word2id = {w[0]:i for i, w in enumerate(word_counts)}
  id2word = {val:k for k, val in word2id.items()}
  word_counts = {w:c for w, c in word_counts}
  id_counts = {i:word_counts[w] for i, w in id2word.items()}

  id_array = list(map(lambda x: word2id.get(x, 0), word_array))

  print('Counting skips. It may take some time...')
  skips = count_skips(id_array, skip_window)

  print('Done!')
  return word2id, id2word, word_counts, id_counts, skips


def check_processed_data(file_name):
    # TODO: rework to check files stored in batches
    print(f'Loading {file_name}...')
    res = np.load(file_name)
    print(f'File {file_name} contains {res.shape[0]} rows with {res[:,0].sum()} skips across {res[:,1].max()} tokens.')
    del res


def record_corpus_metadata(word2id, word_counts, meta_file_path):
    with open(meta_file_path, 'w+') as meta_file:
        for w, i in word2id.items():
            # TODO: consider integer id keys
            meta_file.write(str(i) + '\t' + w + '\t' + str(word_counts[w]) + '\n')


def read_corpus_metadata(meta_file_path):
    with open(meta_file_path, 'r') as meta_file:
        word2id = {}
        id2word = {}
        word_counts = {}
        id_counts = {}
        for line in meta_file:
            i, w, c = line.split('\t')
            i = int(i)
            c = int(c)
            word2id[w] = i
            id2word[i] = w
            word_counts[w] = c
            id_counts[i] = c
    return word2id, id2word, word_counts, id_counts


def load_process_data(file_name, args):
    # TODO: allow different stored batch sizes
    corpus_name = args.corpus_name
    job_dir = args.job_dir
    file_path = os.path.join(job_dir, file_name)

    # if file is already there check and skip processing
    if os.path.exists(file_path):
        print(f'File {file_name} already exists. Nothing to be done. Consider checking contents.')
        # check_processed_data(file_name)
        word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(job_dir, 'meta' + file_name[6:-4] + '.tsv'))
        return word2id, id2word, word_counts, id_counts

    # handles downloading and unpacking text file to job_dir
    text_file_path = load_raw_data(corpus_name, job_dir)
    assert os.path.exists(text_file_path)

    word2id, id2word, word_counts, id_counts, skips = preprocess_data(text_file_path, args.max_vocabulary_size, args.min_occurrence, args.skip_window)
    record_corpus_metadata(word2id, word_counts, os.path.join(job_dir, 'meta' + file_name[6:-4] + '.tsv'))

    # save skips in a file
    stored_batch_size = args.stored_batch_size
    r = skips.shape[0]%stored_batch_size

    ## (num_occurrences, target, context), complete the batch
    np.save(file_path, np.concatenate([skips, skips[:((stored_batch_size - r) if r!= 0 else 0),:]], axis = 0).reshape(-1, stored_batch_size, 3))

    assert os.path.exists(file_path)
    return word2id, id2word, word_counts, id_counts


# def regenerate_neg_samples(neg_file_path, neg_samples, stored_batch_size, sampling_distribution):
#     # TODO: Think of replacing 100 with something else
#     vocabulary_size = sampling_distribution.shape[0]
#     neg_np = np.random.choice(np.arange(vocabulary_size, dtype = np.int32),
#                                     stored_batch_size*neg_samples,
#                                     p = sampling_distribution).reshape(100, stored_batch_size, neg_samples)
#

def create_dataset_from_stored_batch_sizees(file_path, batch_size, stored_batch_size, neg_samples, sampling_distribution, threshold, po):
    def data_generator(data_memmap):
        return iter(data_memmap)

    numpy_data_memmap = np.load(file_path, mmap_mode='r')

    pos_dataset = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [numpy_data_memmap],
        output_types=np.int32,
        output_shapes=(stored_batch_size, 3)).unbatch().batch(batch_size)

    # TODO: add hyperparameter for the number of stored_batch_sizees generated
    period = 32
    neg_rand = np.empty(dtype = np.int32, shape = (period, stored_batch_size, neg_samples))
    def repopulate(neg_rand):
        neg_rand[:,:,:] = np.random.choice(np.arange(sampling_distribution.shape[0], dtype = np.int32),
                                        period*stored_batch_size*neg_samples,
                                        p = sampling_distribution).reshape(period, stored_batch_size, neg_samples)

    def neg_data_generator(neg_rand):
    # TODO: consider addressing resampling through explicit callbacks
    # this is slightly slower than the line below
        i = 0
        for m in neg_rand:
            if i%period == 0:
                repopulate(neg_rand)
            i+=1
            yield m

    neg_dataset = tf.data.Dataset.from_generator(
        generator=neg_data_generator,
        args = [neg_rand],
        output_types=np.int32,
        output_shapes=(stored_batch_size, neg_samples)).repeat(256*32*2**15//(stored_batch_size*period)).unbatch().batch(batch_size)


    pn_dataset = tf.data.Dataset.zip((pos_dataset, neg_dataset))

    return pn_dataset.map(lambda x, y: ({'target': x[:, 1],'pos': x[:, 2], 'neg': y}, x[:,0]))
                                             # , tf.pow(tf.clip_by_value(x[:,0]/threshold, 1., 0.), po)))
