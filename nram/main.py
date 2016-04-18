import os

import tensorflow as tf

from nram import model

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    print FLAGS
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        if FLAGS.is_train:
            model.train(FLAGS)
        else:
            model.load(FLAGS.checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
