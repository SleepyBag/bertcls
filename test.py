# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import data
from colored import fg, stylize
import math
from tensorflow.python import debug as tf_debug

# delete all flags that remained by last run
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


try:
    del_all_flags(tf.flags.FLAGS)
except:
    pass


params = {
    'debug_params': [('debug', False, 'Whether to debug or not'),
                     ('check', False, 'Whether to make a checkpoint'),
                     ('checkpoint', '4725', 'checkpoint file'),
                     ('tfdbg', 'null', 'the manner to debug(cli or tensorboard)')],
    'data_params': [('cls_cnt', 10, "Numbers of class"),
                    ('dataset', 'test', "The dataset")],
    'model_chooing': [('model', 'huapa', 'Model to train')],
    'model_hyperparam': [("emb_dim", 200, "Dimensionality of character embedding"),
                         ("hidden_size", 200, "hidden_size"),
                         ('max_sen_len', 50, 'max number of tokens per sentence'),
                         ('max_doc_len', 40, 'max number of tokens per sentence'),
                         ('sen_aspect_cnt', 1, 'max number of tokens per sentence'),
                         ('doc_aspect_cnt', 1, 'max number of tokens per sentence'),
                         ("lr", .001, "Learning rate"),
                         ("l2_rate", 0., "rate of l2 regularization"),
                         ("lambda1", .4, "proportion of the total loss"),
                         ("lambda2", .3, "proportion of the loss of user block"),
                         ("lambda3", .3, "proportion of the loss of product block"),
                         ("bilstm", True, "use biLSTM or LSTM"),
                         ("split", False, "whether to split the document by sentences or fixed length")],
    'training_params': [("batch_size", 100, "Batch Size"),
                        ("epoch_cnt", 50, "Number of training epochs"),
                        ("training_method", 'adam', 'Method chose to tune the weights')],
    'misc_params': [("allow_soft_placement", True, "Allow device soft device placement"),
                    ("log_device_placement", False, "Log placement of ops on devices")]
}

for param_collection in list(params.values()):
    for param_name, default, description in param_collection:
        param_type = type(default)
        if param_type is int:
            tf.flags.DEFINE_integer(param_name, default, description)
        elif param_type is float:
            tf.flags.DEFINE_float(param_name, default, description)
        elif param_type is str:
            tf.flags.DEFINE_string(param_name, default, description)
        elif param_type is bool:
            tf.flags.DEFINE_boolean(param_name, default, description)

flags = tf.flags.FLAGS

# save current codes
cur_time = time.time()
os.system('mkdir code_history/' + str(cur_time))
os.system('cp *.py code_history/' + str(cur_time) + '/')
localtime = time.localtime(cur_time)

_ = flags.batch_size
# print params
output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
print("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    print(("{}={}".format(attr.upper(), value.value)))
    print("{}={}".format(attr.upper(), value.value), file=output_file)
print("")
output_file.close()


graph = tf.Graph()
with tf.Graph().as_default():
    # Load data
    print("Loading data...")
    datasets = ['data/' + flags.dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']]
    tfrecords = ['data/' + flags.dataset + s for s in ['/train.tfrecord', '/dev.tfrecord', '/test.tfrecord']]
    stats_filename = 'data/' + flags.dataset + '/stats.txt'
    embeddingpath = 'data/' + flags.dataset + '/embedding' + str(flags.emb_dim) + '.txt'
    text_filename = 'data/' + flags.dataset + '/text'
    hierarchy = flags.split
    datasets, lengths, embedding, usr_cnt, prd_cnt, wrd_dict = \
        data.build_dataset(datasets, tfrecords, stats_filename, embeddingpath, flags.max_doc_len,
                           flags.max_sen_len, hierarchy, flags.emb_dim, text_filename)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    trainset = trainset.batch(flags.batch_size)
    devset = devset.batch(flags.batch_size)
    testset = testset.batch(flags.batch_size)
    print("Loading data finished...")

    # create the session
    session_config = tf.ConfigProto(
        allow_soft_placement=flags.allow_soft_placement,
        log_device_placement=flags.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)

    if flags.tfdbg == 'cli':
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess.as_default():
        # build the model
        model_params = {
            'max_sen_len': flags.max_sen_len, 'max_doc_len': flags.max_doc_len,
            'sen_aspect_cnt': flags.sen_aspect_cnt, 'doc_aspect_cnt': flags.doc_aspect_cnt,
            'cls_cnt': flags.cls_cnt, 'embedding': embedding,
            'emb_dim': flags.emb_dim, 'hidden_size': flags.hidden_size,
            'usr_cnt': usr_cnt, 'prd_cnt': prd_cnt,
            'l2_rate': flags.l2_rate, 'debug': flags.debug,
            'debug': flags.debug,
            'lambda1': flags.lambda1, 'lambda2': flags.lambda2, 'lambda3': flags.lambda3
        }
        exec('from ' + flags.model + ' import ' + flags.model.upper() + ' as model')
        model = model(model_params)

        data_iter = tf.data.Iterator.from_structure(trainset.output_types,
                                                    output_shapes=trainset.output_shapes)
        traininit = data_iter.make_initializer(trainset)
        devinit = data_iter.make_initializer(devset)
        testinit = data_iter.make_initializer(testset)

        metrics = model.build(data_iter)

        # restore the params
        saver = tf.train.Saver()
        saver.restore(sess, 'check_points/' + flags.model + '/-' + flags.checkpoint)

        wrd = 'divide0/fold/cur_wrd:0'
        usr_attention = []
        prd_attention = []

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # run a dataset
        def run_set(sess, testlen, metrics, ops=tuple()):
            global flags
            pgb = tqdm(list(range(int(math.ceil(float(testlen) / flags.batch_size)))),
                       leave=False, ncols=50)
            metrics_total = [0] * len(metrics)
            op_results = [[] for i in ops]
            for i in pgb:
                cur_metrics = sess.run(metrics + ops)
                for j in range(len(metrics)):
                    metrics_total[j] += cur_metrics[j]
                for j in range(len(ops)):
                    op_results[j].append(cur_metrics[len(metrics) + j])
            return [metrics_total] + op_results

        try:
            for epoch in range(flags.epoch_cnt):
                sess.run(traininit)
                output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
                ops = (model.teacher_output, )
                # train on trainset
                train_metrics, train_teacher_output = \
                    run_set(sess, trainlen, metrics, ops)
                info = model.output_metrics(train_metrics, trainlen)
                info = 'Trainset:' + info
                print(stylize(info, fg('yellow')))
                train_teacher_output = np.concatenate(train_teacher_output)
                np.save('data/' + flags.dataset + '/teacher/' + flags.model + '_train', train_teacher_output)

                # test on devset
                sess.run(devinit)
                dev_metrics, dev_teacher_output = \
                    run_set(sess, devlen, metrics, ops)
                info = model.output_metrics(dev_metrics, devlen)
                info = 'Devset:  ' + info
                print(stylize(info, fg('green')))
                np.save('data/' + flags.dataset + '/teacher/' + flags.model + '_dev', dev_teacher_output)

                # test on testset
                sess.run(testinit)
                test_metrics, test_teacher_output = \
                    run_set(sess, testlen, metrics, ops)
                info = model.output_metrics(test_metrics, testlen)
                info = 'Testset: ' + info
                print(stylize(info, fg('red')))
                np.save('data/' + flags.dataset + '/teacher/' + flags.model + '_test', test_teacher_output)

                # print info of this epoch
                info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
                info = 'Epoch %d finished, ' % epoch + info
                print(stylize(info, fg('white')))

        except KeyboardInterrupt:
            print('Interrupted')
            best_test_acc = model.best_test_acc
            src = 'code_history/' + str(cur_time)
            dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
            os.system('mv ' + src + ' ' + dest)
