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


class NSCLA(object):

    def __init__(self, args):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
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
                'usr_emb': var('usr_emb', [self.usr_cnt, hsize], self.emb_initializer),
                'prd_emb': var('prd_emb', [self.prd_cnt, hsize], self.emb_initializer)
            }

    def nsc(self, x, max_sen_len, max_doc_len, sen_len, doc_len):
        def lstm(inputs, sequence_length, hidden_size, scope):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                              initializer=xavier())
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_size // 2, forget_bias=0.,
                                              initializer=xavier())
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=inputs,
                sequence_length=sequence_length, dtype=tf.float32, scope=scope)
            outputs = tf.concat(outputs, axis=2)
            return outputs, state

        with tf.variable_scope('sentence_layer'):
            # lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
            lstm_bkg, _state = lstm(x, sen_len, self.hidden_size, 'lstm_bkg')
            lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, self.hidden_size])
            lstm_outputs = lstm_bkg

            alphas = attention(lstm_bkg, [], sen_len, max_sen_len,
                               biases_initializer=self.biases_initializer,
                               weights_initializer=self.weights_initializer)
            sen_bkg = tf.matmul(alphas, lstm_outputs)
            sen_bkg = tf.reshape(sen_bkg, [-1, self.hidden_size], name='new_bkg')
        outputs = tf.reshape(sen_bkg, [-1, max_doc_len, self.hidden_size])

        with tf.variable_scope('document_layer'):
            # lstm_outputs, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm')
            lstm_bkg, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm_bkg')
            lstm_outputs = lstm_bkg

            alphas = attention(lstm_bkg, [], doc_len, max_doc_len,
                               biases_initializer=self.biases_initializer,
                               weights_initializer=self.weights_initializer)
            doc_bkg = tf.matmul(alphas, lstm_outputs)
            doc_bkg = tf.reshape(doc_bkg, [-1, self.hidden_size], name='new_bkg')
        outputs = doc_bkg

        with tf.variable_scope('result'):
            d_hats = tf.layers.dense(tf.concat([outputs, self.usr, self.prd], axis=1), self.cls_cnt,
                                     kernel_initializer=self.weights_initializer,
                                     bias_initializer=self.biases_initializer)

        return d_hats

    def build(self, data_iter, bert_config_file):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_map = data_iter.get_next()
            usrid, prdid, input_x, input_y, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['doc_len'])

            input_x = tf.reshape(input_x, [-1, self.max_sen_len])
            sen_len = tf.count_nonzero(input_x, axis=-1)
            doc_len = doc_len // self.max_sen_len

            input_x = tf.cast(input_x, tf.int32)
            self.usr = lookup(self.embeddings['usr_emb'], usrid, name='cur_usr_embedding')
            self.prd = lookup(self.embeddings['prd_emb'], prdid, name='cur_prd_embedding')
            input_x = tf.reshape(input_x, [-1, self.max_sen_len])
            input_mask = tf.sequence_mask(sen_len, self.max_sen_len)
            input_mask = tf.cast(input_mask, tf.int32)

        bert_config = BertConfig.from_json_file(bert_config_file)
        bert = BertModel(bert_config, is_training=False,
                         input_ids=input_x, input_mask=input_mask,
                         token_type_ids=None,
                         use_one_hot_embeddings=False)
        # input_x = bert.get_sequence_output()
        input_x = bert.get_embedding_output()

        # build the process of model
        d_hat = self.nsc(input_x, self.max_sen_len, self.max_doc_len // self.max_sen_len,
                         sen_len, doc_len)
        prediction = tf.argmax(d_hat, 1, name='prediction')

        with tf.variable_scope("loss"):
            sce = tf.nn.softmax_cross_entropy_with_logits_v2
            self.loss = sce(logits=d_hat, labels=tf.one_hot(input_y, self.cls_cnt))

            regularizer = tf.zeros(1)
            params = tf.trainable_variables()
            for param in params:
                if param not in self.embeddings.values():
                    regularizer += tf.nn.l2_loss(param)
            self.loss = tf.reduce_sum(self.loss) + self.l2_rate * regularizer

        prediction = tf.argmax(d_hat, 1, name='prediction')
        with tf.variable_scope("metrics"):
            correct_prediction = tf.equal(prediction, input_y)
            mse = tf.reduce_sum(tf.square(prediction - input_y), name="mse")
            correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"), name="accuracy")

        return self.loss, mse, correct_num, accuracy

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
