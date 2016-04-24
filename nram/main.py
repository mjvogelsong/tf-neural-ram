import os
import pprint

import tensorflow as tf

from nram import model

flags = tf.app.flags

flags.DEFINE_integer("epoch",
    25,
    "Epoch to train [25]")
flags.DEFINE_string("checkpoint_dir",
    "ckpt",
    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir",
    "samples",
    "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train",
    True,
    "True for training, False for testing [False]")
flags.DEFINE_integer("batch_size",
    1,
    "Batch size")
flags.DEFINE_integer("num_epochs",
    100,
    "Number of training epochs")
flags.DEFINE_string("train_dir",
    "train",
    "Training directory")
flags.DEFINE_integer("T_max_timesteps",
    10,
    "Max number of time steps for individual computation")
flags.DEFINE_integer("report_interval",
    1,
    "Report results at this epoch interval")
flags.DEFINE_integer("checkpoint_interval",
    50,
    "Save a checkpoint copy of the model at this interval")
flags.DEFINE_float("alpha_learning_rate",
    0.1,
    "Alpha - the starting learning rate")
flags.DEFINE_integer("num_h1_units",
    128,
    "Number of hidden units in layer 1")
flags.DEFINE_integer("R_num_registers",
    4,
    "Number of registers")
flags.DEFINE_integer("M_num_ints",
    12,
    "Number of integers representable in memory")
flags.DEFINE_integer("Q_num_modules",
    2,
    "Number of compute modules")
flags.DEFINE_integer("max_minibatches",
    4,
    "Maximum number of minibatches to run")

FLAGS = flags.FLAGS


def main(_):
    print("Flags:")
    pprint.PrettyPrinter().pprint(FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    m = model.Model(FLAGS)

    if FLAGS.is_train:
        m.train()
    else:
        with tf.Session() as sess:
            m.load(sess, FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
