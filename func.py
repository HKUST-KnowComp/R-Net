import tensorflow as tf
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell

INF = 1e30


def stacked_gru(inputs, batch, hidden, num_layers, seq_len, keep_prob=1.0, is_train=None, std=0.01, concat_layers=True, dtype=tf.float32, scope="StackedGRU"):
    with tf.variable_scope(scope):
        outputs = [inputs]
        for layer in range(num_layers):
            with tf.variable_scope("Layer_{}".format(layer)):
                cell_fw = GRUCell(hidden)
                cell_bw = GRUCell(hidden)
                init_fw = tf.tile(tf.get_variable("init_fw", [
                                  1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                init_bw = tf.tile(tf.get_variable("init_bw", [
                                  1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                d_inputs = dropout(
                    outputs[-1], keep_prob=keep_prob, is_train=is_train)
                (out_fw, out_bw), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, d_inputs,
                                                                initial_state_fw=init_fw, initial_state_bw=init_bw, sequence_length=seq_len, dtype=dtype)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        return res


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        shape = args.get_shape().as_list()
        if mode == "embedding":
            noise_shape = [shape[0], 1]
        if mode == "recurrent":
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


def summ(memory, hidden, mask, scope="summ"):
    with tf.variable_scope(scope):
        s0 = tf.nn.tanh(dense(memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)

        JX = tf.shape(inputs)[1]
        inputs_ = tf.nn.relu(dense(d_inputs, hidden, scope="inputs"))
        memory_ = tf.nn.relu(dense(d_memory, hidden, scope="memory"))

        outputs = tf.matmul(inputs_, tf.transpose(
            memory_, [0, 2, 1])) / hidden ** 0.5
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        outputs = tf.matmul(logits, memory)
        res = tf.concat([inputs, outputs], axis=2)

        dim = res.get_shape().as_list()[-1]
        gate = tf.nn.sigmoid(dense(res, dim, use_bias=False, scope="gate"))
        return res * gate


def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = inputs.get_shape().as_list()
        dim = shape[-1]
        out_shape = [shape[idx] for idx in range(len(shape) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
