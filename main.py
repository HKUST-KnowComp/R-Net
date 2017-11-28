import tensorflow as tf
import json
import numpy as np
from tqdm import tqdm
import os

from model import Model
from util import create_batch, convert_tokens, evaluate


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)

    print("Building model...")
    train_batch = create_batch(config.train_record_file, config)
    dev_batch = create_batch(config.dev_record_file, config)
    with tf.variable_scope("model"):
        model_train = Model(config, train_batch, word_mat, char_mat)
        tf.get_variable_scope().reuse_variables()
        model_dev = Model(config, dev_batch, word_mat,
                          char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        sess.run(tf.assign(model_train.is_train,
                           tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model_train.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, config.num_steps + 1)):
            global_step = sess.run(model_train.global_step) + 1
            loss, train_op = sess.run([model_train.loss, model_train.train_op])
            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model_train.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                _, summ = evaluate_batch(
                    model_train, config.val_num_batches, train_eval_file, sess, "train")
                for s in summ:
                    writer.add_summary(s, global_step)

                metrics, summ = evaluate_batch(
                    model_dev, config.val_num_batches, dev_eval_file, sess, "dev")
                sess.run(tf.assign(model_train.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model_train.lr,
                                   tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)
        coord.request_stop()
        coord.join(threads)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    print("Loading model...")
    test_batch = create_batch(config.test_record_file, config, test=True)
    with tf.variable_scope("model"):
        model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        for step in tqdm(range(total // config.batch_size)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict.update(convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist()))
            losses.append(loss)
        coord.request_stop()
        coord.join(threads)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))


def evaluate_batch(model, num_batches, eval_file, sess, data_type):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2])
        answer_dict.update(convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist()))
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]
