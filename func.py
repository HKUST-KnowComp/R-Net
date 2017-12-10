import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _Linear, RNNCell

INF = 1e30


def stacked_gru(inputs, hidden, num_layers, seq_len, batch=None, keep_prob=1.0, is_train=None, std=0.01, concat_layers=True, scope="StackedGRU"):
    with tf.variable_scope(scope):
        outputs = [inputs]
        for layer in range(num_layers):
            with tf.variable_scope("Layer_{}".format(layer)):
                with tf.variable_scope("fw"):
                    inputs_fw = dropout(
                        outputs[-1], keep_prob=keep_prob, is_train=is_train)
                    cell_fw = GRUCell(hidden)
                    init_fw = None
                    if batch is not None:
                        init_fw = tf.tile(tf.get_variable("init_fw", [
                                          1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                        cell_fw, inputs_fw, sequence_length=seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw"):
                    _inputs_bw = tf.reverse_sequence(
                        outputs[-1], seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    inputs_bw = dropout(
                        _inputs_bw, keep_prob=keep_prob, is_train=is_train)
                    cell_bw = GRUCell(hidden)
                    init_bw = None
                    if batch is not None:
                        init_bw = tf.tile(tf.get_variable("init_bw", [
                                          1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                    out_bw, state_bw = tf.nn.dynamic_rnn(
                        cell_bw, inputs_bw, sequence_length=seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        state = tf.concat([state_fw, state_bw], axis=1)
        return res, state


def dropout(args, keep_prob, is_train, mode="recurrent"):
    noise_shape = None
    shape = tf.shape(args)
    if mode == "embedding":
        noise_shape = [shape[0], 1]
    if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]]
    args = tf.cond(is_train, lambda: tf.nn.dropout(
        args, keep_prob, noise_shape=noise_shape), lambda: args)
    return args


def softmax_mask(val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1


def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * d_memory, axis=1)
        return res


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)

        JX = tf.shape(inputs)[1]
        inputs_ = tf.nn.relu(dense(d_inputs, hidden, scope="inputs"))
        memory_ = tf.nn.relu(dense(d_memory, hidden, scope="memory"))

        outputs = tf.matmul(inputs_, tf.transpose(
            memory_, [0, 2, 1])) / (hidden ** 0.5)
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        outputs = tf.matmul(logits, memory)
        res = tf.concat([inputs, outputs], axis=2)

        dim = res.get_shape().as_list()[-1]
        gate = tf.nn.sigmoid(dense(res, dim, use_bias=False, scope="gate"))
        return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


class GRUCell(RNNCell):

    def __init__(self, num_units, reuse=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        if self._gate_linear is None:
            with tf.variable_scope("gates"):
                self._gate_linear = _Linear([inputs, state], 2 * self._num_units, True,
                                            kernel_initializer=tf.orthogonal_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
        value = tf.sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with tf.variable_scope("candidate"):
                self._candidate_linear = _Linear([inputs, r_state], self._num_units, True, kernel_initializer=tf.orthogonal_initializer(
                    1.0), bias_initializer=tf.constant_initializer(-1.0))
        c = tf.nn.tanh(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h
