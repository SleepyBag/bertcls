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
from bert import modeling
from bert import optimization
import collections
import re


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
                     ('check', False, 'Whether to make a checkpoint')],
    'data_params': [('cls_cnt', 10, "Numbers of class"),
                    ('dataset', 'test', "The dataset")],
    'model_chooing': [('model', 'dhuapa', 'Model to train')],
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
    'training_params': [("batch_size", 16, "Batch Size"),
                        ("epoch_cnt", 50, "Number of training epochs"),
                        ("warmup_proportion", .1, "proportion for warming up"),
                        ("init_checkpoint", 'bert/pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt', "the pretrained checkpoint"),
                        ("bert_config_file", 'bert/pretrained/uncased_L-12_H-768_A-12/bert_config.json', "the bert config file"),
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


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


# Load data
print("Loading data...")
datasets = ['data/' + flags.dataset + s for s in ['/train.ss', '/dev.ss', '/test.ss']]
tfrecords = ['data/' + flags.dataset + s for s in ['/train.tfrecord', '/dev.tfrecord', '/test.tfrecord']]
stats_filename = 'data/' + flags.dataset + '/stats.txt'
datasets, lengths, usr_cnt, prd_cnt = \
    data.build_dataset(datasets, tfrecords, stats_filename, flags.max_doc_len)
[trainset, devset, testset] = datasets
trainlen, devlen, testlen = lengths
trainset = trainset.repeat().shuffle(100).batch(flags.batch_size)
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

with sess.graph.as_default():
    # build the model
    model_params = {
        'max_sen_len': flags.max_sen_len, 'max_doc_len': flags.max_doc_len,
        'sen_aspect_cnt': flags.sen_aspect_cnt, 'doc_aspect_cnt': flags.doc_aspect_cnt,
        'cls_cnt': flags.cls_cnt,
        'emb_dim': flags.emb_dim, 'hidden_size': flags.hidden_size,
        'usr_cnt': usr_cnt, 'prd_cnt': prd_cnt,
        'l2_rate': flags.l2_rate, 'debug': flags.debug,
        'lambda1': flags.lambda1, 'lambda2': flags.lambda2, 'lambda3': flags.lambda3
    }
    exec('from ' + flags.model + ' import ' + flags.model.upper() + ' as model')
    model = model(model_params)

    data_iter = tf.data.Iterator.from_structure(trainset.output_types,
                                                output_shapes=trainset.output_shapes)
    traininit = data_iter.make_initializer(trainset)
    devinit = data_iter.make_initializer(devset)
    testinit = data_iter.make_initializer(testset)

    metrics = model.build(data_iter, flags.bert_config_file)

    # Define Training procedure
    num_train_steps = int(trainlen / flags.batch_size * flags.epoch_cnt)
    num_warmup_steps = int(num_train_steps * flags.warmup_proportion)
    train_op = optimization.create_optimizer(
        model.loss, flags.lr, num_train_steps, num_warmup_steps, False)
    # train_op = AdamWeightDecayOptimizer(
    #     learning_rate=flags.lr,
    #     weight_decay_rate=0.01,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-6,
    #     exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # if flags.training_method == 'sgd':
    #     optimizer = tf.train.GradientDescentOptimizer(flags.lr)
    # elif flags.training_method == 'adam':
    #     optimizer = tf.train.AdamOptimizer(flags.lr)
    # elif flags.training_method == 'adadelta':
    #     optimizer = tf.train.AdadeltaOptimizer(flags.lr, epsilon=1e-6)
    # train_op = model.train(optimizer, global_step)

    # merge tensorboard summary
    if flags.debug:
        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('summary/train', sess.graph)
        # dev_writer = tf.summary.FileWriter('summary/dev', sess.graph)
        # test_writer = tf.summary.FileWriter('summary/test', sess.graph)

    # sess.run(tf.global_variables_initializer())
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names
     ) = get_assignment_map_from_checkpoint(tvars, flags.init_checkpoint)
    # tf.train.init_from_checkpoint(flags.init_checkpoint, assignment_map)

    # restore the params
    var_list = {}
    checkpoint_vars = tf.train.list_variables(flags.init_checkpoint)
    global_vars = {var.name: var for var in tf.global_variables()}
    for name, shape in checkpoint_vars:
        if name + ':0' in global_vars:
            var_list[name] = global_vars[name + ':0']
    saver = tf.train.Saver(var_list=var_list)
    saver.restore(sess, flags.init_checkpoint)

    uninitialized_vars = [var for var in tf.global_variables()
                          if var.name not in initialized_variable_names.keys()]
    init_uninitialized_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_uninitialized_op)

    if flags.check:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

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
            pgb.set_description('%.2f' % cur_metrics[0])
        return [metrics_total] + op_results

    try:
        for epoch in range(flags.epoch_cnt):
            sess.run(traininit)
            output_file = open('code_history/' + str(cur_time) + '/output.txt', 'a')
            # trainlen = flags.batch_size * flags.evaluate_every
            # when debugging, summary info is needed for tensorboard
            # cur_trainlen = trainlen if model.best_test_acc < 0.530 \
            #     else flags.evaluate_every * flags.batch_size

            # train on trainset
            if flags.debug and summary is not None:
                train_metrics, step, train_summary, _ = \
                    run_set(sess, trainlen, metrics, (global_step, summary, train_op))
            else:
                train_metrics, step, _ = \
                    run_set(sess, trainlen, metrics, (global_step, train_op, ))
            info = model.output_metrics(train_metrics, trainlen)
            info = 'Trainset:' + info
            print((stylize(info, fg('yellow'))))
            print(info, file=output_file)

            if flags.debug:
                for i, s in zip(step, train_summary):
                    train_writer.add_summary(s, i)
                    train_writer.flush()

            # sess.run(traininit)
            # for i in range(100):
            #     input_x, input_mask, input_y, prediction, correct, probs, bert_output, logits =\
            #             sess.run((model.input_x, model.input_mask, model.input_y, model.prediction, model.correct, model.probs, model.bert_output, model.logits))
            #     print(input_x, input_mask, input_y, prediction, correct, bert_output, logits)

            # test on devset
            sess.run(devinit)
            dev_metrics, = run_set(sess, devlen, metrics)
            info = model.output_metrics(dev_metrics, devlen)
            info = 'Devset:  ' + info
            print((stylize(info, fg('green'))))
            print(info, file=output_file)

            # test on testset
            sess.run(testinit)
            test_metrics, = run_set(sess, testlen, metrics)
            info = model.output_metrics(test_metrics, testlen)
            info = 'Testset: ' + info
            print((stylize(info, fg('red'))))
            print(info, file=output_file)

            # print info of this epoch
            info = model.record_metrics(dev_metrics, test_metrics, devlen, testlen)
            info = 'Epoch %d finished, ' % epoch + info
            print((stylize(info, fg('white'))))
            print(info, file=output_file)

            # write a checkpoint
            if flags.check and 'NEW' in info:
                save_path = saver.save(sess, 'checkpoints/' + flags.model + '/', global_step=step[-1])
                print(('Checkpoint saved to ' + save_path))

            output_file.close()

    except KeyboardInterrupt:
        print('Interrupted')
        best_test_acc = model.best_test_acc
        src = 'code_history/' + str(cur_time)
        dest = 'code_history/' + 'acc' + str(best_test_acc) + '_' + str(cur_time)
        os.system('mv ' + src + ' ' + dest)

