
import numpy as np
import tensorflow as tf

import subprocess
import os
import urllib
import zipfile, bz2

from collections import deque, Counter
from tqdm import tqdm

from google.cloud import storage
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'test-ai-docker.json'

WIKI_DUMP_URLS = {0:'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p30303.bz2',
    1: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles2.xml-p30304p88444.bz2',
    # 2: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles3.xml-p88445p200509.bz2',
    # 3: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles4.xml-p200510p352689.bz2',
    # 4: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles5.xml-p352690p565313.bz2',
    # 5: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles6.xml-p565314p892912.bz2',
    # 6: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles7.xml-p892913p1268691.bz2',
    # 7: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles8.xml-p1268692p1791079.bz2',
    # 8: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles9.xml-p1791080p2336422.bz2',
    # 9: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles10.xml-p2336423p3046512.bz2',
    # 10: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles11.xml-p3046513p3926861.bz2',
    # 11: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles12.xml-p3926862p5040436.bz2',
    # 12: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles13.xml-p5040437p6197594.bz2',
    # 13: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles14.xml-p6197595p7697594.bz2',
    14: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles15.xml-p7744801p9244800.bz2'}



def load_unpack_zip(corpus_name, job_dir):
    """
    Download zip file from Matt Mahoney's website and unpack to `job_dir`
    """
    zip_file_name = f'{corpus_name}.zip'
    zip_file_path = os.path.join(job_dir, 'model_data', zip_file_name)

    if job_dir[:5] == 'gs://':
        bucket_name, path_name = split_gs_prefix(zip_file_path)

        client = storage.Client()
        # print(bucket_name[5:]) # this removes 'gs://'
        bucket = client.get_bucket(bucket_name[5:])

        if storage.Blob(bucket=bucket, name=path_name[:-4]).exists(client):
            print(f'Unzipped file {zip_file_name[:-4]} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return

        if storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'Corpus file {zip_file_name} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
        else:
            bl = bucket.blob(path_name)
            url = 'http://mattmahoney.net/dc/' + zip_file_name
            link = urllib.request.urlopen(url)
            print(f'Downloading corpus file {zip_file_name}...')
            bl.upload_from_string(link.read())

        bl = bucket.blob(path_name)
        zipbytes = io.BytesIO(bl.download_as_string())
        assert zipfile.is_zipfile(zipbytes)
        with zipfile.ZipFile(zipbytes, 'r') as zip_to_unpack:
            for content_file_name in zip_to_unpack.namelist():
                content_file = zip_to_unpack.read(content_file_name)
                bl_unzip = bucket.blob(path_name[:-len(corpus_name)-4] + content_file_name) # remove .zip extension
                bl_unzip.upload_from_string(content_file)

    else:
        if os.path.exists(zip_file_path):
            pass
        else:
            print(f'Downloading corpus file {zip_file_name}...')
            url = 'http://mattmahoney.net/dc/' + zip_file_name
            data_dir = os.path.join(job_dir, 'model_data')
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            urllib.request.urlretrieve(url, zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_to_unpack:
            zip_to_unpack.extractall(data_dir)


def load_unpack_bz2url(ind, job_dir):
    """
    Download wikidumpfiles
    """
    url = WIKI_DUMP_URLS[ind]
    out_file_name = f'enwiki_dump_{ind}'
    out_file_path = os.path.join(job_dir, 'model_data', out_file_name)

    if job_dir[:5] == 'gs://':
        bucket_name, path_name = split_gs_prefix(out_file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])
        if storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'Corpus file {out_file_name} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return
    else:
        if os.path.exists(out_file_path):
            print(f'Corpus file {out_file_name} found in locally in {job_dir}/model_data...')
            return

    raw_bz2 = 'raw_file.bz2'
    raw_txt = 'raw_file.txt'

    print(f'Downloading wikidump file {out_file_name}...')
    urllib.request(url, raw_bz2)

    stream = bz2.BZ2File(raw_bz2).readlines()
    with open(raw_txt, 'wb+') as out:
        for line in stream:
            out.write(line)

    if not os.path.exists(os.path.dirname(out_file_path)):
        os.makedirs(os.path.dirname(out_file_path), exist_ok = True)
    stream = open(raw_txt)
    with open(out_file_path, 'w+') as out:
        for line in stream:
            line = line.strip()
            if line:
                letter = line[0]
                # the line starts with a character, a number or a quote
                if letter.isalpha() or letter.isnumeric() or letter in ('"', "'"):
                    out.write(line = '\n')

    os.remove(raw_txt)
    os.remove(raw_bz2)

    if job_dir[:5] == 'gs://':
        upload_to_gs(out_file_path, job_dir)
        os.remove(out_file_path) # remove local file


def split_gs_prefix(file_path):
    ind = [i for i,l in enumerate(file_path) if l == '/'][2]
    bucket_name, path_name = file_path[:ind], file_path[ind+1:]
    return bucket_name, path_name


def download_from_gs(file_path):
    assert file_path[:5] == 'gs://'
    print(f'Accessing {file_path} in Google Cloud Storage Bucket...')
    bucket_name, path_name = split_gs_prefix(file_path)

    client = storage.Client()
    bucket = client.get_bucket(bucket_name[5:])

    bl = bucket.blob(path_name)

    ind = [i for i,l in enumerate(path_name) if l == '/'][-1]
    if not os.path.exists(path_name[:ind]):
        os.makedirs(path_name[:ind], exist_ok = True)

    bl.download_to_filename(path_name)

    return path_name


def upload_to_gs(path_to_upload, job_dir):
    # assumes that path_to_upload is a local_path (so no bucket name)
    if job_dir[:5] != 'gs://':
        print(f'File is not uploaded to Google Cloud Storage in \'upload_to_gs\' call. Address {job_dir} provided is not a cloud location')
        return # nothing to be done
    else:
        assert os.path.exists(path_to_upload)
        bucket_name, _  = split_gs_prefix(job_dir)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])

        bl = bucket.blob(path_to_upload)
        bl.upload_from_filename(filename=path_to_upload)


def load_raw_data(corpus_name, job_dir, perl_cleanup = True):
    # TODO: make sure this works for both perl_cleanup values
    text_file_name = f'{corpus_name}.txt'
    text_file_path = os.path.join(job_dir, 'model_data', text_file_name)
    print(text_file_path)

    if job_dir[:5] == 'gs://': # work in google cloud storage
        bucket_name, path_name = split_gs_prefix(text_file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])
        if storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'Corpus file {text_file_name} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return [text_file_path]
        else:
            load_unpack_zip(corpus_name, job_dir)
        if perl_cleanup:
            assert os.path.exists('main_.pl')

            bl = bucket.blob(path_name[:-4])
            bl.download_to_filename('corpus_temp')

            bash_str = f'perl main_.pl corpus_temp > corpus_temp.txt'
            print('Cleaning up the corpus...')
            subprocess.run(bash_str, shell = True)

            bl = bucket.blob(path_name)
            bl.upload_from_filename(filename='corpus_temp.txt')

            print(f'Done loading the corpus. File is written to {text_file_name}')
            os.remove('corpus_temp.txt')
            os.remove('corpus_temp')
        return [text_file_path]

    else:
        if os.path.exists(text_file_path):
            print(f'File {text_file_name} already exists. Nothing to be done.')
            return [text_file_path]
        else:
            load_unpack_zip(corpus_name, job_dir)
        if perl_cleanup:
            # print(os.listdir())
            # print(os.getcwd())
            assert os.path.exists('main_.pl')
            bash_str = f'perl main_.pl {text_file_path[:-4]} > {text_file_path}'
            print('Cleaning up the corpus...')
            subprocess.run(bash_str, shell = True)
            print(f'Done loading the corpus. File is writtten to {text_file_name}')
        else: # text8 is already processed, so just need to add .txt extenstion
            os.rename(text_file_path[:-4], text_file_path)
        return [text_file_path]


def load_dump_urls(job_dir, perl_cleanup = True):
    text_file_paths = []

    for ind in WIKI_DUMP_URLS.keys():
        out_file_name = f'enwiki_dump_{ind}'
        out_file_path = os.path.join(job_dir, 'model_data', out_file_name)
        text_file_paths +=  load_raw_data(corpus_name, job_dir, perl_cleanup = perl_cleanup)
    return text_file_paths


def count_skips(id_array, skip_window=5):
    def postprocess_count_skips(skips):
        return np.array([[k, i, j] for (i, j), k in skips.items()], dtype = np.float32)

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

    res = postprocess_count_skips(d)

    # concatenate with mirror version
    return np.concatenate([res, res[np.where(res[:,1]!=res[:,2])][:,[0,2,1]]], axis = 0)


def save_skips_shards(skips_tmp, n_shards = 10):
    n_rows = len(skips_tmp)
    n_shards = min(n_rows, n_shards)
    shard_size = n_rows//n_shards + (n_rows%shard_size > 0)

    shard_files = [file_name in os.listdir() if file_name.find('shard') > 0]
    max_shard = max(list(map(int, [shard_file[shard_file.find('_')+1:] for shard_file in shard_files]))) if shard_files else -1

    skip_shards = []
    for s in range(n_shards):
        max_shard+= 1
        shard_name = f'shard_{max_shard}.npy'
        np.save(shard_name, skips_tmp[s*shard_size:(s+1)*shard_size])
        skip_shards.append(shard_name)
    return skip_shards


def collect_skip_shards(skip_shards):
    print(f'Collecting skip shards from {skip_shards}')
    pair_counter = Counter()
    for shard in skip_shards:
        skips_tmp = np.load(shard)
        pair_counter.update({(i,j):k for k,i,j in skips_tmp})
    return np.array([[k, i, j] for (i, j), k in pair_counter.items()], dtype = np.float32)


def preprocess_data_mult(text_file_paths, max_vocabulary_size, min_occurrence, skip_window):
    '''
    Create from a corpus name a triple (word2id, id2word, word_counts, skips)

        'word2id' dictionary keyed on words
        'id2word' dictionary keyed on indices
        'word_counts' dictionary keyed on words
        'skips' np.array of triples returned by 'count_skips'
    '''
    word_counts = Counter()
    for text_file_path in text_file_paths:
        if text_file_path[:5] == 'gs://':
            text_file_path = download_from_gs(text_file_path)

        with open(text_file_path) as text_file:
            word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())
            word_counts.update(word_array)

    word_counts = Counter(word_array).most_common(max_vocabulary_size - 1) # leave one spot for 'UNK' token
    if len(text_file_paths) > 1:
        del word_array

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

    # now create skips from each file and store them in small batches:
    print('Counting skips. It may take some time...')
    if len(text_file_paths) <= 1: # word_array was kept so far
        id_array = list(map(lambda x: word2id.get(x, 0), word_array))
        skips = count_skips(id_array, skip_window)
    else:
        skip_shards = []
        for text_file_path in text_file_paths:
            if text_file_path[:5] == 'gs://':
                text_file_path = download_from_gs(text_file_path)
            with open(text_file_path) as text_file:
                word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())
                id_array = list(map(lambda x: word2id.get(x, 0), word_array))
                skips_tmp = count_skips(id_array, skip_window)
                skip_shards.append(save_skips_shards(skips_tmp, n_shards = 10))
        skips = collect_skip_shards(skip_shards)

    print('Done!')
    return word2id, id2word, word_counts, id_counts, skips


def preprocess_data(text_file_path, max_vocabulary_size, min_occurrence, skip_window):
    word2id, id2word, word_counts, id_counts, skips = preprocess_data_mult([text_file_path], max_vocabulary_size, min_occurrence, skip_window)

    return word2id, id2word, word_counts, id_counts, skips


def check_processed_data(file_name):
    # TODO: rework to check files stored in batches
    print(f'Loading {file_name}...')
    res = np.load(file_name)
    print(f'File {file_name} contains {res.shape[0]} rows with {res[:,0].sum()} skips across {res[:,1].max()} tokens.')
    del res


def record_corpus_metadata(word2id, word_counts, meta_file_path):
    if meta_file_path[:5] == 'gs://':
        bucket_name, meta_file_path_ = split_gs_prefix(meta_file_path)
    else:
        meta_file_path_ = meta_file_path

    with open(meta_file_path_, 'w+') as meta_file:
        for w, i in word2id.items():
            # TODO: consider integer id keys
            meta_file.write(str(i) + '\t' + w + '\t' + str(word_counts[w]) + '\n')

    if meta_file_path[:5] == 'gs://':
        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])

        bl = bucket.blob(meta_file_path_)
        bl.upload_from_filename(filename=meta_file_path_)

        os.remove(meta_file_path_)


def read_corpus_metadata(meta_file_path):
    if meta_file_path[:5] == 'gs://':
        meta_file_path = download_from_gs(meta_file_path)

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
    file_path = os.path.join(job_dir, 'model_data', file_name)

    if job_dir[:5] == 'gs://': # Google Cloud Storage
        bucket_name, path_name = split_gs_prefix(file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])

        if storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'File {file_name} already exists. Nothing to be done. Consider checking contents.')
            # check_processed_data(file_name)
            word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(job_dir, 'model_data', 'meta' + file_name[6:-4] + '.tsv'))
            return word2id, id2word, word_counts, id_counts

    elif os.path.exists(file_path):
        print(f'File {file_name} already exists. Nothing to be done. Consider checking contents.')
        # check_processed_data(file_name)
        word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(job_dir, 'model_data', 'meta' + file_name[6:-4] + '.tsv'))
        return word2id, id2word, word_counts, id_counts

    if args.corpus_name == 'enwiki_dump':
        load_dump_urls(job_dir)
    else:
        text_file_paths = load_raw_data(corpus_name, job_dir)

    word2id, id2word, word_counts, id_counts, skips = preprocess_data_mult(text_file_paths, args.max_vocabulary_size, args.min_occurrence, args.skip_window)
    record_corpus_metadata(word2id, word_counts, os.path.join(job_dir, 'model_data', 'meta' + file_name[6:-4] + '.tsv'))

    # save skips in a file
    stored_batch_size = args.stored_batch_size
    r = skips.shape[0]%stored_batch_size

    if job_dir[:5] == 'gs://':
        np.save('temp_skips.npy', np.concatenate([skips, skips[:((stored_batch_size - r) if r!= 0 else 0),:]], axis = 0).reshape(-1, stored_batch_size, 3))

        bl = bucket.blob(path_name)
        bl.upload_from_filename(filename='temp_skips.npy')

        assert storage.Blob(bucket=bucket, name=path_name).exists(client)
        os.remove('temp_skips.npy')
    else:
        np.save(file_path, np.concatenate([skips, skips[:((stored_batch_size - r) if r!= 0 else 0),:]], axis = 0).reshape(-1, stored_batch_size, 3))

        assert os.path.exists(file_path)
    return word2id, id2word, word_counts, id_counts


def create_dataset_from_stored_batches(file_path, batch_size, stored_batch_size, neg_samples, sampling_distribution, threshold, po):
    def data_generator(data_memmap):
        return iter(data_memmap)

    if file_path[:5] == 'gs://':
        file_path = download_from_gs(file_path)
    numpy_data_memmap = np.load(file_path, mmap_mode='r')
    #print(f'My shape is {np.load(file_path).shape}')


    pos_dataset = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [numpy_data_memmap],
        output_types=np.int32,
        output_shapes=(stored_batch_size, 3)).unbatch().batch(batch_size)

    pos_dataset = tf.data.Dataset.from_tensor_slices(np.load(file_path)).unbatch().batch(batch_size)

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


    pn_dataset = tf.data.Dataset.zip((pos_dataset, neg_dataset)).take(3)

    return pn_dataset.map(lambda x, y: ({'target': x[:, 1],'pos': x[:, 2], 'neg': y}
                                             , tf.pow(tf.clip_by_value(x[:,0]/threshold, 1., 0.), po)))
