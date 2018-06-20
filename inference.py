import tensorflow as tf
import spacy
import os
import numpy as np
import ujson as json


from func import cudnn_gru, native_gru, dot_attention, summ, ptr_net
from prepro import word_tokenize, convert_idx

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Must be consistant with training
char_limit = 16
hidden = 75
char_dim = 8
char_hidden = 100
use_cudnn = True

# File path
target_dir = "data"
save_dir = "log/model"
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
word2idx_file = os.path.join(target_dir, "word2idx.json")
char2idx_file = os.path.join(target_dir, "char2idx.json")


class InfModel(object):
    # Used to zero elements in the probability matrix that correspond to answer
    # spans that are longer than the number of tokens specified here.
    max_answer_tokens = 15

    def __init__(self, word_mat, char_mat):
        self.c = tf.placeholder(tf.int32, [1, None])
        self.q = tf.placeholder(tf.int32, [1, None])
        self.ch = tf.placeholder(tf.int32, [1, None, char_limit])
        self.qh = tf.placeholder(tf.int32, [1, None, char_limit])
        self.tokens_in_context = tf.placeholder(tf.int64)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

    def ready(self):
        N, PL, QL, CL, d, dc, dg = \
            1, self.c_maxlen, self.q_maxlen, char_limit, hidden, char_dim, \
            char_hidden
        gru = cudnn_gru if use_cudnn else native_gru

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N,
                      input_size=c_emb.get_shape().as_list()[-1])
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d)
            rnn = gru(num_layers=1, num_units=d, batch_size=N,
                      input_size=qc_att.get_shape().as_list()[-1])
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(att, att, mask=self.c_mask, hidden=d)
            rnn = gru(num_layers=1, num_units=d, batch_size=N,
                      input_size=self_att.get_shape().as_list()[-1])
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list()[-1])
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.cond(
                self.tokens_in_context < self.max_answer_tokens,
                lambda: tf.matrix_band_part(outer, 0, -1),
                lambda: tf.matrix_band_part(outer, 0, self.max_answer_tokens))
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)


class Inference(object):

    def __init__(self):
        with open(word_emb_file, "r") as fh:
            self.word_mat = np.array(json.load(fh), dtype=np.float32)
        with open(char_emb_file, "r") as fh:
            self.char_mat = np.array(json.load(fh), dtype=np.float32)
        with open(word2idx_file, "r") as fh:
            self.word2idx_dict = json.load(fh)
        with open(char2idx_file, "r") as fh:
            self.char2idx_dict = json.load(fh)
        self.model = InfModel(self.word_mat, self.char_mat)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

    def response(self, context, question):
        sess = self.sess
        model = self.model
        span, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs = \
            self.prepro(context, question)
        yp1, yp2 = \
            sess.run(
                [model.yp1, model.yp2],
                feed_dict={
                    model.c: context_idxs, model.q: ques_idxs,
                    model.ch: context_char_idxs, model.qh: ques_char_idxs,
                    model.tokens_in_context: len(span)})
        start_idx = span[yp1[0]][0]
        end_idx = span[yp2[0]][1]
        return context[start_idx: end_idx]

    def prepro(self, context, question):
        context = context.replace("''", '" ').replace("``", '" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        ques = question.replace("''", '" ').replace("``", '" ')
        ques_tokens = word_tokenize(ques)
        ques_chars = [list(token) for token in ques_tokens]

        context_idxs = np.zeros([1, len(context_tokens)], dtype=np.int32)
        context_char_idxs = np.zeros(
            [1, len(context_tokens), char_limit], dtype=np.int32)
        ques_idxs = np.zeros([1, len(ques_tokens)], dtype=np.int32)
        ques_char_idxs = np.zeros(
            [1, len(ques_tokens), char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word2idx_dict:
                    return self.word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in self.char2idx_dict:
                return self.char2idx_dict[char]
            return 1

        for i, token in enumerate(context_tokens):
            context_idxs[0, i] = _get_word(token)

        for i, token in enumerate(ques_tokens):
            ques_idxs[0, i] = _get_word(token)

        for i, token in enumerate(context_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[0, i, j] = _get_char(char)

        for i, token in enumerate(ques_chars):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[0, i, j] = _get_char(char)
        return spans, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs


# Demo, example from paper "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
if __name__ == "__main__":
    infer = Inference()
    context = "In meteorology, precipitation is any product of the condensation " \
              "of atmospheric water vapor that falls under gravity. The main forms " \
              "of precipitation include drizzle, rain, sleet, snow, graupel and hail." \
              "Precipitation forms as smaller droplets coalesce via collision with other " \
              "rain drops or ice crystals within a cloud. Short, intense periods of rain " \
              "in scattered locations are called “showers”."
    ques1 = "What causes precipitation to fall?"
    ques2 = "What is another main form of precipitation besides drizzle, rain, snow, sleet and hail?"
    ques3 = "Where do water droplets collide with ice crystals to form precipitation?"

    # Correct: gravity, Output: drizzle, rain, sleet, snow, graupel and hail
    ans1 = infer.response(context, ques1)
    print("Answer 1: {}".format(ans1))

    # Correct: graupel, Output: graupel
    ans2 = infer.response(context, ques2)
    print("Answer 2: {}".format(ans2))

    # Correct: within a cloud, Output: within a cloud
    ans3 = infer.response(context, ques3)
    print("Answer 3: {}".format(ans3))
