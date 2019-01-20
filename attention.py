import tensorflow as tf
from tensorflow import constant as const
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.attention import attention
from layers.hop import hop
from colored import stylize, fg
from math import sqrt
import numpy as np
from bert.modeling import BertModel
from bert.modeling import BertConfig
from bert import modeling
lookup = tf.nn.embedding_lookup


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class ATTENTION(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.l2_rate = args['l2_rate']
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.weights_initializer = xavier()
        self.biases_initializer = tf.initializers.zeros()
        self.emb_initializer = xavier()

        hsize = self.hidden_size

        # embeddings in the model
        with tf.variable_scope('emb'):
            self.embeddings = {
                # 'wrd_emb': const(self.embedding, name='wrd_emb', dtype=tf.float32),
                # 'wrd_emb': tf.Variable(self.embedding, name='wrd_emb', dtype=tf.float32),
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], self.emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], self.emb_initializer)
            }

    def build(self, data_iter, bert_config_file):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['doc_len'])

            input_x = tf.cast(input_x, tf.int32)
            self.usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            self.prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            # input_x = lookup(self.embeddings['wrd_emb'], input_x, name='cur_wrd_embedding')
            input_x = tf.reshape(input_x, [-1, self.max_doc_len])
            input_mask = tf.sequence_mask(doc_len, self.max_doc_len)
            input_mask = tf.cast(input_mask, tf.int32)

        bert_config = BertConfig.from_json_file(bert_config_file)
        bert = BertModel(bert_config, is_training=True,
                         input_ids=input_x, input_mask=input_mask,
                         token_type_ids=None,
                         use_one_hot_embeddings=False)

        pooled_output = bert.get_pooled_output()
        sequence_output = bert.get_sequence_output()
        alphas = attention(sequence_output, None, self.max_doc_len, self.max_doc_len)
        sequence_output = tf.matmul(alphas, sequence_output)
        sequence_output = tf.squeeze(sequence_output, axis=1)
        bert_output = tf.concat([pooled_output, sequence_output], axis=1)

        logits = tf.layers.dense(bert_output, self.cls_cnt,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.bert_output = bert_output
        self.logits = logits

        # build the process of model
        prediction = tf.argmax(logits, 1, name='prediction')
        self.prediction = prediction

        with tf.variable_scope("loss"):
            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            log_probs = tf.nn.log_softmax(logits)
            self.probs = tf.nn.softmax(logits)
            loss = -tf.reduce_sum(tf.one_hot(input_y, self.cls_cnt, dtype=tf.float32)
                                  * log_probs, axis=-1)
            self.loss = tf.reduce_mean(loss)
            # self.loss = sce(logits=logits, labels=tf.one_hot(input_y, self.cls_cnt))
            # self.loss = tf.reduce_mean(self.loss)
            self.total_loss = tf.reduce_sum(loss)

        prediction = tf.argmax(logits, 1, name='prediction')
        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            self.correct = correct_prediction
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return self.total_loss, mse, correct_num, accuracy

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num, accuracy = metrics
        info = 'Loss = %.3f, RMSE = %.3f, Acc = %.3f' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        _dev_loss, dev_mse, dev_correct_num, dev_accuracy = dev_metrics
        _test_loss, test_mse, test_correct_num, test_accuracy = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        test_rmse = sqrt(float(test_mse) / testlen)
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            self.best_test_rmse = test_rmse
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        return train_op
