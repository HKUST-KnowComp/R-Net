import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from func import stacked_gru, pointer, dot_attention, summ, dropout


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = dropout(tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False), keep_prob=config.emb_keep_prob, is_train=self.is_train, mode="embedding")
        self.char_mat = dropout(tf.get_variable("char_mat", initializer=tf.constant(
            char_mat, dtype=tf.float32)), keep_prob=config.emb_keep_prob, is_train=self.is_train, mode="embedding")

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
            self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                _, qh_emb = stacked_gru(qh_emb, dg, num_layers=1, seq_len=self.qh_len,
                                        keep_prob=config.keep_prob, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                _, ch_emb = stacked_gru(ch_emb, dg, num_layers=1, seq_len=self.ch_len,
                                        keep_prob=config.keep_prob, is_train=self.is_train)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            c, _ = stacked_gru(c_emb, d, batch=N, num_layers=3, seq_len=self.c_len, keep_prob=config.keep_prob,
                               is_train=self.is_train)
            tf.get_variable_scope().reuse_variables()
            q, _ = stacked_gru(q_emb, d, batch=N, num_layers=3, seq_len=self.q_len, keep_prob=config.keep_prob,
                               is_train=self.is_train)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            att, _ = stacked_gru(qc_att, d, num_layers=1, seq_len=self.c_len,
                                 keep_prob=config.keep_prob, is_train=self.is_train)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            match, _ = stacked_gru(self_att, d, num_layers=1, seq_len=self.c_len,
                                   keep_prob=config.keep_prob, is_train=self.is_train)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.keep_prob, is_train=self.is_train)
            hidden = init.get_shape().as_list()[-1]
            inp, logits1 = pointer(
                match, init, d, self.c_mask, keep_prob=config.keep_prob, is_train=self.is_train)
            _, state = GRUCell(hidden).__call__(inp, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = pointer(match, state, d, self.c_mask,
                                 keep_prob=config.keep_prob, is_train=self.is_train)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
