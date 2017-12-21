import os
import tensorflow as tf

from prepro import prepro
from main import train, test

flags = tf.flags

home = os.path.expanduser("~")
train_file = os.path.join(home, "data", "squad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "squad", "dev-v1.1.json")
glove_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

target_dir = "data"
log_dir = "log/event"
save_dir = "log/model"
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_file", glove_file, "Glove source file")

flags.DEFINE_string("train_record_file", train_record_file,
                    "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file,
                    "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file,
                    "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file,
                    "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file,
                    "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")


flags.DEFINE_integer("glove_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")

flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("num_steps", 60000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000,
                     "checkpoint to save and evaluate the model")
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150,
                     "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
flags.DEFINE_float("keep_prob", 0.7, "Dropout keep prob in rnn")
flags.DEFINE_float("ptr_keep_prob", 0.7,
                   "Dropout keep prob for pointer network")
flags.DEFINE_float("grad_clip", 10.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 75, "Hidden size")
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        prepro(config)
    elif config.mode == "test":
        test(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "load":
        train(config, load=True)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
