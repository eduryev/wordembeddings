
import numpy as np
import tensorflow as tf

import subprocess
import os
import gc
import urllib
import zipfile, bz2
import pickle as pkl


from collections import deque, Counter
from tqdm import tqdm

from google.cloud import storage
import io

from retrying import retry


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'test-ai-docker.json'


WIKI_DUMP_URLS = {0:'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p30303.bz2',
    1: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles2.xml-p30304p88444.bz2',
    2: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles3.xml-p88445p200509.bz2',
    3: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles4.xml-p200510p352689.bz2',
    4: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles5.xml-p352690p565313.bz2',
    5: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles6.xml-p565314p892912.bz2',
    6: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles7.xml-p892913p1268691.bz2',
    7: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles8.xml-p1268692p1791079.bz2',
    8: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles9.xml-p1791080p2336422.bz2',
    9: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles10.xml-p2336423p3046512.bz2',
    10: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles11.xml-p3046513p3926861.bz2',
    11: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles12.xml-p3926862p5040436.bz2',
    12: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles13.xml-p5040437p6197594.bz2',
    13: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles14.xml-p6197595p7697594.bz2',
    14: 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles15.xml-p7744801p9244800.bz2'
    }


def validate_corpora_files(path_list, path_name, n_splits, suffix = None):
    res = True
    suffix = suffix if suffix else ''
    for i in range(n_splits):
        if path_name + f'_sub{i}' + suffix in path_list:
            continue
        res = False
        break
    return res


def validate_chunks_processed(text_file_paths, n_chunks):
    assert n_chunks > 0
    last_processed_file_path = None
    chunk_thresholds = []
    for text_file_path in text_file_paths[::-1]:
        if text_file_path[:5] == 'gs://':
            bucket_name, path_name = split_gs_prefix(text_file_path)
            client = storage.Client()
            bucket = client.get_bucket(bucket_name[5:])
            file_list = [os.path.basename(bl.name) for bl in bucket.list_blobs(prefix=path_name[:-4] + '_chunk')]
        else:
            file_list = os.listdir(os.path.dirname(text_file_path))

        chunk_thresholds = []
        text_file_name = os.path.basename(text_file_path)[:-4]
        for file_name in file_list:
            i, j = file_name.find(text_file_name + '_chunk_'), file_name.find('.pkl')
            if i == 0 and j >= 0:
                chunk_thresholds+= file_name[len(text_file_name + '_chunk_'):j].split('_')

        chunk_thresholds = sorted(list(set(map(int, chunk_thresholds))))
        if len(file_list) == n_chunks and len(chunk_thresholds) == n_chunks + 1:
            print(f'Found processed file {os.path.basename(text_file_path)}')
            last_processed_file_path = text_file_path
            break
        
    return last_processed_file_path, chunk_thresholds


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


def load_unpack_bz2url(ind, job_dir, n_splits = None):
    """
    Download wikidumpfiles
    """
    url = WIKI_DUMP_URLS[ind]
    out_file_name = f'enwiki_dump_{ind}'
    out_file_path = os.path.join(job_dir, 'model_data', out_file_name)

    # check whether files already exist
    path_bz2 = None
    if job_dir[:5] == 'gs://':
        bucket_name, path_name = split_gs_prefix(out_file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])
        if n_splits is None and storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'Corpus file {out_file_name} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return
        elif n_splits and validate_corpora_files([bl.name for bl in bucket.list_blobs(prefix=path_name + '_sub')], path_name, n_splits):
            print(f'{n_splits} split corpus files found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return
        elif storage.Blob(bucket=bucket, name=path_name + '.bz2').exists(client):
            path_bz2 = download_from_gs(os.path.join(bucket_name, path_name + '.bz2'))

    else:
        if n_splits is None and os.path.exists(out_file_path):
            print(f'Corpus file {out_file_name} found locally in {job_dir}/model_data...')
            return
        elif n_splits and validate_corpora_files([file_name for file_name in os.listdir(os.path.dirname(out_file_path)) if file_name.find(out_file_name + '_sub') >= 0], out_file_name, n_splits):
            print(f'{n_splits} split corpus files found locally in {job_dir}/model_data...')
            return
        elif os.path.exists(out_file_path + '.bz2'):
            path_bz2 = out_file_path + '.bz2'
        path_name = out_file_path

    # if files not found, download and process
    raw_bz2 = f'enwiki_dump_{ind}.bz2' if path_bz2 is None else path_bz2
    raw_txt = f'enwiki_dump_{ind}_unprocessed.txt'

    if path_bz2 is None:
        print(f'Downloading wikidump file {out_file_name} from {url}...')
        urllib.request.urlretrieve(url, raw_bz2)

    p = 0
    with bz2.BZ2File(raw_bz2, 'r') as stream, open(raw_txt, 'wb+') as out:
        for line in tqdm(stream):
            p+= 1
            out.write(line)

    # don't need bz2 anymore
    if path_bz2 is None:
        if job_dir[:5] == 'gs://':
            if not os.path.exists(os.path.dirname(path_name)):
                os.makedirs(os.path.dirname(path_name), exist_ok = True)
            os.rename(raw_bz2, path_name + '.bz2')
            upload_to_gs(path_name + '.bz2', job_dir)
            os.remove(path_name + '.bz2')
        else:
            os.remove(raw_bz2)

    if not os.path.exists(os.path.dirname(path_name)):
        os.makedirs(os.path.dirname(path_name), exist_ok = True)

    # split text file into several text subfiles
    path_names = []
    with open(raw_txt, 'r') as stream:
        N = p//n_splits + (p%n_splits > 0)
        cur = 0
        cur_path_name = path_name + f'_sub{cur}' if n_splits else path_name
        cur_out = open(cur_path_name , 'w+')
        for c, line in tqdm(enumerate(stream)):
            line = line.strip()
            if line:
                letter = line[0]
                # pick up only actual articles
                if letter.isalpha() or letter.isnumeric() or letter in ("'", '"'):
                    cur_out.write('<text > ' + line + ' </text>\n')   # to trick main_.pl
            if (c+1)%N == 0 or c+1 == p:
                print(f'\rRecorded {cur_path_name}')
                cur_out.close()
                cur+= 1
                path_names.append(cur_path_name)
                cur_path_name = path_name + f'_sub{cur}'
                if c+1 < p:
                    cur_out = open(cur_path_name, 'w+')
                else:
                    break

    # don't need unrpocessed text anymore
    os.remove(raw_txt)

    if job_dir[:5] == 'gs://':
        for path_name in path_names:
            print(f'Uploading {os.path.basename(path_name)} to Google Storage Bucket...')
            assert os.path.exists(path_name)
            upload_to_gs(path_name, job_dir)
            os.remove(path_name) # remove local file


def split_gs_prefix(file_path):
    ind = [i for i,l in enumerate(file_path) if l == '/'][2]
    bucket_name, path_name = file_path[:ind], file_path[ind+1:]
    return bucket_name, path_name


@retry(stop_max_attempt_number = 5, wait_random_min=1000, wait_random_max=2000)
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


def load_raw_data(corpus_name, job_dir, perl_cleanup = True, n_splits = None):
    """
    Takes in a corpus name, checks whether matching .txt file exists in job_dir,
    if not makes it available.
    Returns a list containing a single path to file processed file.
    """
    text_file_name = f'{corpus_name}.txt'
    text_file_path = os.path.join(job_dir, 'model_data', text_file_name)

    if job_dir[:5] == 'gs://': # work in google cloud storage
        bucket_name, path_name = split_gs_prefix(text_file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])
        if n_splits is None and storage.Blob(bucket=bucket, name=path_name).exists(client):
            print(f'Corpus file {text_file_name} found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            return [text_file_path]
        elif n_splits and validate_corpora_files([bl.name for bl in bucket.list_blobs(prefix=path_name[:-4] + '_sub')], path_name[:-4], n_splits, suffix = '.txt'):
            print(f'{n_splits} text {text_file_name} files found in Google Cloud Storage Bucket at {job_dir}/model_data...')
            text_file_paths = [text_file_path[:-4] + f'_sub{s}.txt' for s in range(n_splits)]
            return text_file_paths
        else:
            if corpus_name[:12] == 'enwiki_dump_':
                ind = int(corpus_name[12:])
                load_unpack_bz2url(ind, job_dir, n_splits = n_splits)
            else:
                load_unpack_zip(corpus_name, job_dir)
        if perl_cleanup:
            assert os.path.exists('main_.pl')

            text_file_paths = [text_file_path[:-4] + f'_sub{s}.txt' for s in range(n_splits)] if n_splits else [text_file_path]
            path_names = [path_name[:-4] + f'_sub{s}.txt' for s in range(n_splits)] if n_splits else [path_name]
            for path_name in path_names:
                bl = bucket.blob(path_name[:-4])
                bl.download_to_filename('corpus_temp')

                bash_str = f'perl main_.pl corpus_temp > corpus_temp.txt'
                print('Cleaning up the corpus...')
                subprocess.run(bash_str, shell = True)

                bl = bucket.blob(path_name)
                bl.upload_from_filename(filename='corpus_temp.txt')

                print(f'Done loading the corpus. File is written to {os.path.basename(path_name)}')
                os.remove('corpus_temp.txt')
                os.remove('corpus_temp')

        else:
            print('Perl clean up is mandatory in this version')
            raise
        return text_file_paths

    else:
        if n_splits is None and os.path.exists(text_file_path):
            print(f'File {text_file_name} already exists. Nothing to be done.')
            return [text_file_path]
        elif n_splits and validate_corpora_files([file_name for file_name in os.listdir(os.path.dirname(text_file_path)) if file_name.find(text_file_name[:-4] + '_sub') >= 0], text_file_name[:-4], n_splits, suffix = '.txt'):
            print(f'{n_splits} split files from {text_file_name[:-4]} already exist. Nothing to be done.')
            text_file_paths = [text_file_path[:-4] + f'_sub{s}.txt' for s in range(n_splits)]
            return text_file_paths
        else:
            # load_unpack_zip(corpus_name, job_dir)
            if corpus_name[:12] == 'enwiki_dump_':
                ind = int(corpus_name[12:])
                load_unpack_bz2url(ind, job_dir, n_splits = n_splits)
            else:
                load_unpack_zip(corpus_name, job_dir)

        if perl_cleanup:
            assert os.path.exists('main_.pl')

            text_file_names = [text_file_name[:-4] + f'_sub{s}.txt' for s in range(n_splits)] if n_splits else [text_file_name]
            text_file_paths = [text_file_path[:-4] + f'_sub{s}.txt' for s in range(n_splits)] if n_splits else [text_file_path]

            for text_file_name in text_file_names:
                bash_str = f'perl main_.pl model_data/{text_file_name[:-4]} > model_data/{text_file_name}'
                print('Cleaning up the corpus...')
                subprocess.run(bash_str, shell = True)
                os.remove(os.path.join('model_data', {text_file_name[:-4]})) # remove temporary file
                print(f'Done loading the corpus. File is written to {text_file_name}')
        else:
            print('Perl clean up is mandatory in this version')
            raise
        return text_file_paths


def load_dump_urls(job_dir, perl_cleanup = True):
    text_file_paths = []

    for ind in WIKI_DUMP_URLS.keys():
        out_file_name = f'enwiki_dump_{ind}'
        out_file_path = os.path.join(job_dir, 'model_data', out_file_name)
        text_file_paths+= load_raw_data(out_file_name, job_dir, perl_cleanup = perl_cleanup, n_splits = 2)
    return text_file_paths


def skips_to_counter(skips):
    res_counter = Counter()
    for k, i, j in tqdm(skips):
        res_counter[(int(i), int(j))]+= k
    return res_counter


def postprocess_count_skips(skips, count_threshold = 0):
    if count_threshold > 0: # note that orders are guaranteed to match
        return (np.array([k for k, v in skips.items() if v >= count_threshold], dtype = np.int32),
                np.array([v for v in skips.values() if v >= count_threshold], dtype = np.float32))
    return (np.array(list(skips.keys()), dtype = np.int32), np.array(list(skips.values()), dtype = np.float32))


def count_skips(id_array, skip_window=5, counter_format = False, count_threshold = 0):
    def mirror(d):
        mirror_d = {}
        for (i, j), k in tqdm(d.items()):
            if i != j:
                mirror_d[(j, i)] = k
        return mirror_d

    print(f'Processing skips for a text of length {len(id_array)}')
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

    if counter_format:
        return d
    else:
        return postprocess_count_skips(d, count_threshold=count_threshold)


def get_chunk_thresholds(arr, n_chunks):
    assert n_chunks > 0
    chunk_thresholds = []
    N = len(arr)
    left = N
    thr = left//n_chunks + (left%n_chunks != 0) # ceiling

    ct = Counter(arr)
    c = 0
    for i in range(max(arr)):
        c+= ct[i]
        if c >= thr and c < N: # leave last
            chunk_thresholds.append(i+1)
            n_chunks-= 1
            left -= c
            thr = left//n_chunks + (left%n_chunks != 0)
            c = 0
            if n_chunks == 1:
                break

    chunk_thresholds += [N]*(n_chunks - 1)
    return chunk_thresholds


def shuffle_stored_batches(file_path,  stored_batch_size = None):
    def data_generator(data_memmap):
        return iter(data_memmap)

    if file_path[:5] == 'gs://':
        file_path = download_from_gs(file_path)

    # get shape
    with open(file_path + '_val.npy', 'rb') as npy:
        np.lib.format.read_magic(npy)
        shp, _, _ = np.lib.format.read_array_header_1_0(npy)

    # allocate memory
    if stored_batch_size:
        assert shp[1] == stored_batch_size
    else:
        stored_batch_size = shp[1]
    res_key = np.empty((shp[0], shp[1], 2), dtype = np.int32)
    res_val = np.empty(shp, dtype = np.float32)

    shuffle_dataset_key = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [np.load(file_path + '_key.npy', mmap_mode='r')],
        output_types=np.int32,
        output_shapes=(stored_batch_size, 2)).unbatch().shuffle(100000).batch(stored_batch_size)

    shuffle_dataset_val = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [np.load(file_path + '_val.npy', mmap_mode='r')],
        output_types=np.float32,
        output_shapes=(stored_batch_size,)).unbatch().shuffle(100000).batch(stored_batch_size)

    shuffle_dataset = tf.data.Dataset.zip((shuffle_dataset_key, shuffle_dataset_val))
    for i, (batch_key, batch_val) in enumerate(shuffle_dataset):
        b = batch_key.shape[0]
        res_key[i] = batch_key
        res_val[i] = batch_val
    
    np.save(file_path + '_key.npy', res_key)
    np.save(file_path + '_val.npy', res_val)


def load_dict(path):
  with open(path,'rb') as f:
    d = pkl.load(f)
  return d

def save_dict(d, path):
  with open(path,'wb') as f:
    pkl.dump(d, f)


def update_store_chunks(skips_upd, chunk_thresholds, new_cache_path, old_cache_path = None, remove_old = False):
    print('Updating dictionary chunks...')
    for c0, c1 in zip(chunk_thresholds[:-1], chunk_thresholds[1:]):
        if old_cache_path:
            chunk_name = f'{old_cache_path}_{c0}_{c1}.pkl'
            chunk = Counter(load_dict(chunk_name))
            print(f'Loaded {chunk_name}. Number of entries: {len(chunk)}')
            if remove_old:
                os.remove(chunk_name)
        else:
            chunk = Counter()

        for (i, j), k in skips_upd.items():
            if i >= c0 and i < c1:
                chunk[(i,j)]+= k
        # chunk.update({(i, j):k for (i,j), k in skips_upd.items() if i >= c0 and i < c1})

        save_path = f'{new_cache_path}_{c0}_{c1}.pkl'
        save_dict(dict(chunk), path = save_path)
        print(f'Saved chunk {new_cache_path}_{c0}_{c1}. Chunk length: {len(chunk)/10**6} millions.')
        del chunk
        gc.collect()


def collect_skips(cache_path, chunk_thresholds, count_threshold = .5):
    skips_key = np.zeros(shape = (0,2), dtype = np.int32)
    skips_val = np.zeros(shape = (0,), dtype = np.float32)
    for c0, c1 in zip(chunk_thresholds[:-1], chunk_thresholds[1:]):
        chunk_name = f'{cache_path}_{c0}_{c1}.pkl'
        chunk = load_dict(chunk_name)
        chunk_key, chunk_val = postprocess_count_skips(chunk, count_threshold=count_threshold)

        skips_key = np.concatenate([skips_key, chunk_key], axis = 0)
        skips_val = np.concatenate([skips_val, chunk_val], axis = 0)
    return (skips_key, skips_val)


def append_mirror(skips):
    # skips are not batched
    inds = np.where(skips[0][:,0]!=skips[0][:,1])
    return (np.concatenate([skips[0], skips[0][inds][:, [1, 0]]], axis = 0), np.concatenate([skips[1], skips[1][inds]], axis = 0))


def get_word_stats(text_file_paths, max_vocabulary_size, min_occurrence):
    # TODO: heresy -- this should nicely interact with load_process_data to store meta before and not after counting skips
    found_meta = False
    if text_file_paths[0][:5] == 'gs://':
        bucket_name, path_name = split_gs_prefix(text_file_paths[0])
        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])
        meta_data_path = os.path.join(os.path.dirname(path_name), 'meta_full_v1.tsv')
        if storage.Blob(bucket=bucket, name=meta_data_path).exists(client):
            print('resroring meta')
            word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(bucket_name, meta_data_path))
            found_meta = True
    else:
        meta_data_path = 'model_data/meta_full_v1.tsv'

        if os.path.exists(meta_data_path):
            print('restoring meta')
            word2id, id2word, word_counts, id_counts = read_corpus_metadata(meta_data_path)
            found_meta = True

    if not found_meta:
        print('No meta')
        word_counts = Counter()
        tot_all = 0
        for text_file_path in text_file_paths:
            if text_file_path[:5] == 'gs://':
                text_file_path = download_from_gs(text_file_path)

            with open(text_file_path) as text_file:
                word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())
                word_counts.update(word_array)
                tot_all += len(word_array)
            if text_file_path[:5] == 'gs://':
                os.remove(text_file_path)
            print(f'Processed word counts in {text_file_path}')

        word_counts = word_counts.most_common(max_vocabulary_size - 1) # leave one spot for 'UNK' token
        if len(text_file_paths) > 1:
            del word_array

        for i in range(len(word_counts) - 1, -1, -1):
            if word_counts[i][1] < min_occurrence:
                word_counts.pop()
            else:
                break

        tot = sum([w[1] for w in word_counts])
        word_counts.insert(0, ('UNK', tot_all - tot))

        word2id = {w[0]:i for i, w in enumerate(word_counts)}
        id2word = {val:k for k, val in word2id.items()}
        word_counts = {w:c for w, c in word_counts}
        id_counts = {i:word_counts[w] for i, w in id2word.items()}

        # heresy: remove completely later, but for now it's helpful
        if text_file_paths[0][:5] == 'gs://':
            bucket_name, _ = split_gs_prefix(text_file_paths[0])
            record_corpus_metadata(word2id, word_counts, os.path.join(bucket_name, 'new_w2v_model/model_data/meta_full_v1.tsv'))
            print('Recorded temporary meta for the full corpus')
        else:
            record_corpus_metadata(word2id, word_counts, os.path.join('model_data', 'meta_full_v1.tsv'))
            print('Recorded temporary meta for the full corpus')

    # word_array = None if len(text_file_paths) > 1 else word_array
    return word2id, id2word, word_counts, id_counts


def preprocess_data_mult(text_file_paths, max_vocabulary_size, min_occurrence, skip_window):
    '''
    Create from a corpus name a triple (word2id, id2word, word_counts, skips)

        'word2id' dictionary keyed on words
        'id2word' dictionary keyed on indices
        'word_counts' dictionary keyed on words
        'skips' np.array of triples returned by 'count_skips'
    '''

    word2id, id2word, word_counts, id_counts = get_word_stats(text_file_paths, max_vocabulary_size, min_occurrence)

    # now create skips from each file and store them in small batches:
    print('Counting skips. It may take some time...')
    if len(text_file_paths) == 1: # word_array was kept so far
        if text_file_paths[0][:5] == 'gs://':
            text_file_paths[0] = download_from_gs(text_file_paths[0])

        with open(text_file_paths[0]) as text_file:
            word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())
        id_array = list(map(lambda x: word2id.get(x, 0), word_array))
        skips = count_skips(id_array, skip_window, count_threshold=.5)
    else:

        # Step 1: Process text files into chunks
        last_processed_file_path, chunk_thresholds = validate_chunks_processed(text_file_paths, n_chunks = 4)
        text_file_paths = text_file_paths[text_file_paths.index(last_processed_file_path) + 1:] if last_processed_file_path else text_file_paths

        if last_processed_file_path:
            print(f'Last processed {last_processed_file_path}')
            print('Chunk thresholds are...')
            print(chunk_thresholds)

        all_caches = []
        new_cache_path = None
        if last_processed_file_path:
            if last_processed_file_path[:5] == 'gs://':
                bucket_name, path_name = split_gs_prefix(last_processed_file_path)
                new_cache_path =  path_name[:-4] + f'_chunk'

                for c0, c1 in zip(chunk_thresholds[:-1], chunk_thresholds[1:]):
                    chunk_path = path_name[:-4] + f'_chunk_{c0}_{c1}.pkl'
                    if not os.path.exists(chunk_path):
                        download_from_gs(os.path.join(bucket_name, chunk_path))
            else:
                new_cache_path = last_processed_file_path[:-4] + f'_chunk'
            all_caches.append(new_cache_path)

        for text_file_path in text_file_paths:
            gc.collect()
            if text_file_path[:5] == 'gs://':
                bucket_name, path_name = split_gs_prefix(text_file_path)
                download_from_gs(text_file_path)
            else:
                path_name = text_file_path

            old_cache_path = new_cache_path
            new_cache_path = path_name[:-4] + f'_chunk'

            # CHUNKS HERE
            with open(path_name) as text_file:
                word_array = tf.keras.preprocessing.text.text_to_word_sequence(text_file.readline())
                id_array = list(map(lambda x: word2id.get(x, 0), word_array))

            print(f'Counting skips for {os.path.basename(path_name)}')
            skips_upd = count_skips(id_array, skip_window, counter_format=True) # note: no count_threshold
            if not chunk_thresholds:
                # CHUNKS HERE
                chunk_thresholds = [0] + get_chunk_thresholds([i for (i, j) in skips_upd.keys()], n_chunks = 4) + [len(word2id)]
                print('Computed chunk thresholds:')
                print(chunk_thresholds)

            del id_array, word_array
            print(f'Done counting skips in {os.path.basename(text_file_path)}')

            update_store_chunks(skips_upd, chunk_thresholds, new_cache_path, old_cache_path, remove_old = False)
            if text_file_path[:5] == 'gs://':
                for c0, c1 in zip(chunk_thresholds[:-1], chunk_thresholds[1:]):
                    save_path = f'{text_file_path[:-4]}_chunk_{c0}_{c1}.pkl'
                    bucket_name, local_save_path = split_gs_prefix(save_path)
                    upload_to_gs(local_save_path, save_path)

            all_caches.append(new_cache_path)
            del skips_upd

        print('Assembling skips')
        skips = collect_skips(new_cache_path, chunk_thresholds, count_threshold = .5)
        print('Celaning up cache files')
        for cache_path in all_caches:
            for file_name in os.listdir(os.path.dirname(cache_path)):
                if file_name.find(cache_path) >= 0:
                    os.remove(file_name)

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
    if not os.path.exists(os.path.dirname(meta_file_path_)):
        os.makedirs(os.path.dirname(meta_file_path_), exist_ok = True)
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

    # os.remove(meta_file_path)
    return word2id, id2word, word_counts, id_counts


# CHANGES HERE
def load_process_data(file_name, args):
    corpus_name = args.corpus_name
    job_dir = args.job_dir
    file_path = os.path.join(job_dir, 'model_data', file_name)

    if job_dir[:5] == 'gs://': # Google Cloud Storage
        bucket_name, path_name = split_gs_prefix(file_path)

        client = storage.Client()
        bucket = client.get_bucket(bucket_name[5:])

        if storage.Blob(bucket=bucket, name=path_name + '_key.npy').exists(client) and storage.Blob(bucket=bucket, name=path_name + '_val.npy').exists(client):
            print(f'Key and value files for {file_name} already exist. Nothing to be done. Consider checking contents.')
            word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(job_dir, 'model_data', file_name + '_meta.tsv'))
            return word2id, id2word, word_counts, id_counts
    # not Google Cloud Storage
    elif os.path.exists(file_path + '_key.npy') and os.path.exists(file_path + '_val.npy'):
        print(f'Key and value files for {file_name} already exist. Nothing to be done. Consider checking contents.')
        word2id, id2word, word_counts, id_counts = read_corpus_metadata(os.path.join(job_dir, 'model_data', file_name + '_meta.tsv'))
        return word2id, id2word, word_counts, id_counts
    else:
        path_name = file_path

    if args.corpus_name == 'enwiki_dump':
        text_file_paths = load_dump_urls(job_dir)
        print('Uploaded all dump urls to: ', text_file_paths)
    else:
        text_file_paths = load_raw_data(corpus_name, job_dir)

    # count skips for each text file
    word2id, id2word, word_counts, id_counts, skips = preprocess_data_mult(text_file_paths, args.max_vocabulary_size, args.min_occurrence, args.skip_window)
    record_corpus_metadata(word2id, word_counts, os.path.join(job_dir, 'model_data', file_name + '_meta.tsv'))
    # note: this skips are not mirrored and not batched
    skips = append_mirror(skips)

    # save skips in a file
    stored_batch_size = args.stored_batch_size
    r = 0 if len(skips[0]) % stored_batch_size == 0 else stored_batch_size - len(skips[0]) % stored_batch_size

    np.save(path_name + '_key.npy', np.concatenate([skips[0], skips[0][:r]], axis = 0).reshape(-1, stored_batch_size, 2))
    np.save(path_name + '_val.npy', np.concatenate([skips[1], skips[1][:r]], axis = 0).reshape(-1, stored_batch_size))

    # shuffle in respective files (changes file contents)
    shuffle_stored_batches(path_name, stored_batch_size = stored_batch_size)

    if job_dir[:5] == 'gs://':
        upload_to_gs(path_name + '_key.npy', file_path)
        upload_to_gs(path_name + '_val.npy', file_path)
 
    return word2id, id2word, word_counts, id_counts

def create_dataset_from_stored_batches(file_path, batch_size, stored_batch_size, sampling_distribution, threshold, po, neg_samples):
    def data_generator(data_memmap):
        return iter(data_memmap)

    if file_path[:5] == 'gs://':
        file_path_key = download_from_gs(file_path + '_key.npy')
        file_path_val = download_from_gs(file_path + '_val.npy')
    else:
        file_path_key = file_path + '_key.npy'
        file_path_val = file_path + '_val.npy'

    pos_dataset_key = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [np.load(file_path_key, mmap_mode='r')],
        output_types=np.int32,
        output_shapes=(stored_batch_size, 2)).unbatch().batch(batch_size)

    pos_dataset_val = tf.data.Dataset.from_generator(
        generator=data_generator,
        args = [np.load(file_path_val, mmap_mode='r')],
        output_types=np.float32,
        output_shapes=(stored_batch_size,)).unbatch().batch(batch_size)

    if neg_samples: # positive number
        # TODO: add hyperparameter for the number of stored_batch_sizes generated
        period = 32
        neg_rand = np.empty(dtype = np.float32, shape = (period, stored_batch_size, neg_samples))
        def repopulate(neg_rand):
            neg_rand[:,:,:] = np.random.choice(np.arange(sampling_distribution.shape[0], dtype = np.int32),
                                            period*stored_batch_size*neg_samples,
                                            p = sampling_distribution).reshape(period, stored_batch_size, neg_samples)

        def neg_data_generator(neg_rand):
        # TODO: consider addressing resampling through explicit callbacks
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


        pn_dataset = tf.data.Dataset.zip((pos_dataset_key, pos_dataset_val, neg_dataset)) 

        return pn_dataset.map(lambda key, val, neg: ({'target': key[:, 0], 'pos': key[:, 1], 'neg': neg}
                                                , tf.pow(tf.clip_by_value(val/threshold, 1., 0.), po)))
    else:
        pos_dataset = tf.data.Dataset.zip((pos_dataset_key, pos_dataset_val)) 
        return pos_dataset.map(lambda key, val: ({'target': key[:, 0], 'pos': key[:, 1]}
                                                , tf.pow(tf.clip_by_value(val/threshold, 1., 0.), po)))
