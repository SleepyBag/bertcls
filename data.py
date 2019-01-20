import pandas as pd
import tensorflow as tf
from functools import partial
import numpy as np
import os
import csv
import word2vec as w2v
from bert import tokenization
from tqdm import tqdm


reading_col_name = ['usr', 'prd', 'rating', 'content']
output_col_name = ['usr', 'prd', 'rating', 'content', 'doc_len']


def build_dataset(filenames, tfrecords_filenames, stats_filename, max_doc_len):
    datasets = []

    tfrecords_filenames = [i + str(max_doc_len) for i in tfrecords_filenames]
    stats = {}
    if sum([os.path.exists(i) for i in tfrecords_filenames]) < len(tfrecords_filenames) \
            or not os.path.exists(stats_filename):
        for tfrecords_filename in tfrecords_filenames:
            if os.path.exists(tfrecords_filename):
                os.remove(tfrecords_filename)
        if os.path.exists(stats_filename):
            os.remove(stats_filename)
        # read the data and transform them
        data_frames, stats['usr_cnt'], stats['prd_cnt'] = \
            read_files(filenames, max_doc_len)

        # build the dataset
        for filename, tfrecords_filename, data_frame in \
                zip(filenames, tfrecords_filenames, data_frames):
            data_frame['content'] = \
                data_frame['content'].transform(lambda x: x.tostring())

            writer = tf.python_io.TFRecordWriter(tfrecords_filename)
            for item in data_frame.iterrows():
                def int64list(value):
                    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

                def byteslist(value):
                    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

                feature = {'usr': int64list([item[1]['usr']]),
                           'prd': int64list([item[1]['prd']]),
                           'rating': int64list([item[1]['rating']]),
                           'content': byteslist([item[1]['content']])}
                feature['doc_len'] = int64list([item[1]['doc_len']])

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            stats[filename + 'len'] = len(data_frame)
            # lengths.append(len(data_frame))

        stats_file = csv.writer(open(stats_filename, 'w'))
        # print('usr_cnt: %d, prd_cnt: %d' % (usr_cnt, prd_cnt))
        for key, val in stats.items():
            stats_file.writerow([key, val])

    def transform_example(example):
        dics = {
            'usr': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'prd': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'rating': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
            'content': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None)}
        dics['doc_len'] = tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)

        ans = tf.parse_single_example(example, dics)
        ans['content'] = tf.decode_raw(ans['content'], tf.int64)
        return ans

    for key, val in csv.reader(open(stats_filename)):
        stats[key] = int(val)
    for tfrecords_filename in tfrecords_filenames:
        dataset = tf.data.TFRecordDataset(tfrecords_filename)
        dataset = dataset.map(transform_example)
        datasets.append(dataset)

    lengths = [stats[filename + 'len'] for filename in filenames]
    return datasets, lengths, stats['usr_cnt'], stats['prd_cnt']


def split_paragraph(paragraph, tokenizer):
    sentences = paragraph.split('<sssss>')
    sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    tokens = ['[CLS]']
    # tokens = []
    for sentence in sentences:
        tokens += sentence + ['[SEP]']
    return tokens


def list_to_numpy(content, length):
    ans = np.zeros(length, dtype=np.int64)
    length = min(length, len(content))
    ans[:length] = content[:length]
    return ans


def read_files(filenames, max_doc_len):
    data_frames = [pd.read_csv(filename, sep='\t\t', names=reading_col_name, engine='python')
                   for filename in filenames]
    print('Data frame loaded.')

    tokenizer = tokenization.FullTokenizer('bert/pretrained/uncased_L-12_H-768_A-12/vocab.txt')
    # count contents' length
    for df in data_frames:
        df['content'] = df['content'].transform(partial(split_paragraph, tokenizer=tokenizer))
        df['content'] = df['content'].transform(partial(tokenizer.convert_tokens_to_ids))
        df['content'] = df['content'].transform(partial(list_to_numpy, length=max_doc_len))
        # df['rating'] = df['rating'].transform(lambda r: 1 if r > 5 else 0)
        df['rating'] = df['rating'] - df['rating'].min()
        df['doc_len'] = df['content'].transform(lambda i: np.count_nonzero(i, axis=0))

    # transform users and products to indices
    total_data = pd.concat(data_frames)
    usr = total_data['usr'].unique().tolist()
    usr = {name: index for index, name in enumerate(usr)}
    prd = total_data['prd'].unique().tolist()
    prd = {name: index for index, name in enumerate(prd)}
    for df in data_frames:
        df['usr'] = df['usr'].map(usr)
        df['prd'] = df['prd'].map(prd)
    print('Users and products indexed.')

    # transform contents into indices
    print('Contents indexed.')

    return data_frames, len(usr), len(prd)
