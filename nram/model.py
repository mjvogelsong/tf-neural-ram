import math
import time

import tensorflow as tf
import numpy as np

from nram import dataset, tasks

# pylint: disable=E1103
class Model(object):
    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.num_epochs = flags.num_epochs
        self.train_dir = flags.train_dir
        self.max_minibatches = flags.max_minibatches
        self.report_interval = flags.report_interval
        self.checkpoint_interval = flags.checkpoint_interval
        self.alpha_learning_rate = flags.alpha_learning_rate
        self.num_h1_units = flags.num_h1_units
        self.R_num_registers = flags.R_num_registers
        self.M_num_ints = flags.M_num_ints
        self.Q_num_modules = flags.Q_num_modules
        self.T_max_timesteps = flags.T_max_timesteps

        # This is temporary while I'm testing ...
        self.task = tasks.Access()

    def _build(self):
        """
        Initialize the memory and registers.
        """
        # Holds the temporary memory staged in the registers
        self.r_registers = tf.get_variable("r_registers",
            [self.R_num_registers, self.M_num_ints],
            initializer=tf.constant_initializer(1.0/self.M_num_ints))

        # Holds the global memory.
        # This is where we'll insert input, and read the output
        # of the learned algorithm.
        self.BIGM_memory = tf.get_variable("BIGM_memory",
            [self.M_num_ints, self.M_num_ints],
            initializer=tf.constant_initializer(1.0/self.M_num_ints))

    def train(self):
        """
        Train the neural RAM by feeding the input matrices into the
        BIGM_memory.
        """
        with tf.Graph().as_default():
            print("Building graph ...")
            self._build()
            input_pl, targets_pl = self._placeholder_inputs(self.batch_size)
            loss, f = self._controller(input_pl, targets_pl)
            train_op = self._training(loss, self.alpha_learning_rate)
            #eval_correct = self._evaluation(logits, targets_pl)
            summary_op = tf.merge_all_summaries()
            saver = tf.train.Saver()
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            summary_writer = tf.train.SummaryWriter(self.train_dir, sess.graph)
            print("Training ...")
            for step in xrange(self.max_minibatches):
                start_time = time.time()

                feed_dict = self._fill_feed_dict(self.task, input_pl, targets_pl)

                _, loss_value, mem_value, f_value = sess.run([train_op, loss, self.BIGM_memory, f], feed_dict=feed_dict)
                duration = time.time() - start_time

                if step % self.report_interval == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                    print("input_pl: %s" % feed_dict[input_pl])
                    print("targets_pl: %s" % feed_dict[targets_pl])
                    print("BIGM: %s" % mem_value)
                    print("f: %s" % f_value)
                    '''
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    '''

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % self.checkpoint_interval == 0 \
                        or (step + 1) == self.max_minibatches:
                    saver.save(sess, self.train_dir, global_step=step)
                    # Evaluate against the training set.
                    '''
                    print('Training Data Eval:')
                    self._do_eval(sess,
                            eval_correct,
                            input_pl,
                            targets_pl,
                            data_sets.train)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    self._do_eval(sess,
                            eval_correct,
                            input_pl,
                            targets_pl,
                            data_sets.validation)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    self._do_eval(sess,
                            eval_correct,
                            input_pl,
                            targets_pl,
                            data_sets.test)
                    '''


    def save(self, sess, save_path, global_step):
        self.saver.save(sess, save_path, global_step)

    def load(self, sess, save_path):
        self.saver.restore(sess, save_path)

    def _evaluation(self, logits, labels):
        """
        Evaluate the memory to see if the highest probability
        density is on the correct integer.
        """
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def _do_eval(self, sess,
                eval_correct,
                input_placeholder,
                targets_placeholder,
                data_set):
        """
        Run one epoch of evaluation
        """
        true_count = 0
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for _ in xrange(steps_per_epoch):
            feed_dict = self._fill_feed_dict(data_set,
                                       input_placeholder,
                                       targets_placeholder)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))

    def _fill_feed_dict(self, task, input_pl, targets_pl):
        """
        Create the feed_dict for the placeholders filled with the next
        `batch size ` examples.
        """
        input_feed, targets_feed = dataset.next_batch(self.M_num_ints,
            self.batch_size, task)
        feed_dict = {
          input_pl: input_feed[0],
          targets_pl: targets_feed[0],
        }
        return feed_dict

    def _placeholder_inputs(self, batch_size):
        """
        Initialize the placeholders for inputs and target memory traces.
        """
        input_data = tf.placeholder(tf.float32, shape=(self.M_num_ints, self.M_num_ints),
            name="input_pl")
        target_data = tf.placeholder(tf.float32, shape=(self.M_num_ints, self.M_num_ints),
            name="targets_pl")
        return input_data, target_data

    def _training(self, loss, learning_rate):
        """
        Prepare the tensorflow training Op with Adam.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate, name="adam_optimizer")
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step, name="train_op")
        return train_op

    def _intermediate_controller(self, register_output):
        """
        Trainable model with a, b, c, and f outputs.
        """
        inputs = tf.transpose(self._interpret_register(register_output),
            name="interpreted_registers")

        with tf.name_scope("h1"):
            w = tf.Variable(tf.truncated_normal([self.R_num_registers, self.num_h1_units],
                stddev=1.0 / math.sqrt(float(self.R_num_registers))),
                name="w")
            b = tf.Variable(tf.zeros([self.num_h1_units]),
                name="b")
            h1 = tf.nn.relu(tf.matmul(inputs, w) + b, name="h1")

        with tf.name_scope("a"):
            w = tf.Variable(tf.truncated_normal(
                [self.num_h1_units, (self.R_num_registers + self.Q_num_modules)],
                stddev=1.0 / math.sqrt(float(self.num_h1_units))),
                name="w")
            b = tf.Variable(tf.zeros([(self.R_num_registers + self.Q_num_modules)]),
                name="b")
            logit_a = tf.matmul(h1, w) + b

        with tf.name_scope("b"):
            w = tf.Variable(tf.truncated_normal(
                [self.num_h1_units, (self.R_num_registers + self.Q_num_modules)],
                stddev=1.0 / math.sqrt(float(self.num_h1_units))),
                name="w")
            b = tf.Variable(tf.zeros([(self.R_num_registers + self.Q_num_modules)]),
                name="b")
            logit_b = tf.matmul(h1, w) + b

        with tf.name_scope("c"):
            w = tf.Variable(tf.truncated_normal(
                [self.num_h1_units, (self.R_num_registers + self.Q_num_modules)],
                stddev=1.0 / math.sqrt(float(self.num_h1_units))),
                name="w")
            b = tf.Variable(tf.zeros([(self.R_num_registers + self.Q_num_modules)]),
                name="b")
            logit_c = tf.matmul(h1, w) + b

        with tf.name_scope("f"):
            w = tf.Variable(tf.truncated_normal(
                [self.num_h1_units, 1],
                stddev=1.0 / math.sqrt(float(self.num_h1_units))),
                name="w")
            b = tf.Variable(tf.zeros([1]),
                name="b")
            logit_x = tf.matmul(h1, w) + b
            f = tf.nn.sigmoid(logit_x)

        return logit_a, logit_b, logit_c, f

    def _controller(self, input_pl, labels_pl):
        """
        Controller module integrated into the overall
        Neural RAM.
        """
        # The inputs are placed into the Memory
        self._init_memory(input_pl)
        p = []

        for step in xrange(self.T_max_timesteps):
            local_memory = self.r_registers
            for mod_idx in xrange(self.Q_num_modules):
                with tf.variable_scope("controller") as scope:
                    scope.reuse_variables()
                    a, b, _, _ = self._intermediate_controller(self.r_registers)
                    zero_padding = tf.zeros([self.Q_num_modules - mod_idx, self.M_num_ints],
                        name="zero_padding")
                    local_memory_padded = tf.concat(0, [local_memory, zero_padding],
                        name="local_memory_padded")
                    o_i = self._module_function(mod_idx,
                        tf.matmul(tf.transpose(local_memory_padded),
                            tf.transpose(tf.nn.softmax(a)),
                            name="local_mult_a"),
                        tf.matmul(tf.transpose(local_memory_padded),
                            tf.transpose(tf.nn.softmax(b)),
                            name="local_mult_b"))
                    local_memory = tf.concat(0, [local_memory, tf.transpose(o_i)])

            for reg_idx in xrange(self.R_num_registers):
                with tf.variable_scope("controller") as scope:
                    scope.reuse_variables()
                    _, _, c, _ = self._intermediate_controller(self.r_registers)
                    r_i = tf.matmul(tf.transpose(local_memory), tf.transpose(tf.nn.softmax(c)))
                    if reg_idx == 0:
                        temp_r_registers = tf.transpose(r_i)
                    else:
                        temp_r_registers = tf.concat(0, [temp_r_registers, tf.transpose(r_i)])

            _, _, _, f = self._intermediate_controller(self.r_registers)
            self.r_registers.assign(temp_r_registers)
            with tf.name_scope("loss"):
                if step == 0:
                    p.append(f)
                    loss = -p[step] * tf.log(tf.reduce_sum(labels_pl * self.BIGM_memory))
                else:
                    p.append((1.0 - p[step-1]) * f)
                    loss = tf.sub(loss, p[step] * tf.log(tf.reduce_sum(labels_pl * self.BIGM_memory)),
                        name="loss")

            '''
            if tf.random_uniform([0.0, 1.0]) <= f:
                break
            '''
        with tf.name_scope("loss"):
            return loss, f

    def _init_memory(self, int_array):
        """
        Feed the inputs through the BIGM memory.
        """
        self.BIGM_memory = int_array

    def _module_function(self, fun_idx, a, b):
        """
        Run m_i module function based on the function index.
        """
        if fun_idx == 0:
            return self._module_READ(a, b)
        elif fun_idx == 1:
            return self._module_WRITE(a, b)

    def _interpret_register(self, registers):
        """
        Only read the first value of the register to reduce
        computational complexity.
        """
        return tf.slice(registers, [0, 0], [-1, 1])

    def _read_memory(self):
        """
        Interpret the highest density integer as the value of
        that memory location.
        """
        return tf.argmax(self.BIGM_memory, 1)

    def _module_READ(self, p_pointer, _):
        """
        READ module from paper.
        """
        return tf.matmul(tf.transpose(self.BIGM_memory), p_pointer, name="o")

    def _module_WRITE(self, p_pointer, a_value):
        """
        WRITE modules from paper.
        """
        J_ones = tf.ones([self.M_num_ints, 1])
        self.BIGM_memory = \
            tf.nn.softmax((tf.matmul(J_ones - p_pointer, tf.transpose(J_ones)) * self.BIGM_memory) + \
            tf.matmul(p_pointer, tf.transpose(a_value)))
        return tf.ones([self.M_num_ints, 1], name="o") / self.M_num_ints
