import math
import time

import tensorflow as tf

class Model(object):
    def __init__(self, flags):
        self.saver = tf.train.Saver()

        self.batch_size = flags.batch_size
        self.num_epochs = flags.num_epochs
        self.train_dir = flags.train_dir
        self.max_steps = flags.max_steps
        self.report_interval = flags.report_interval
        self.checkpoint_interval = flags.checkpoint_interval
        self.alpha_learning_rate = flags.alpha_learning_rate
        self.num_h1_units = flags.num_h1_units

        self._build(flags)

    def _build(self, flags):
        self.R_num_registers = tf.constant(flags.R_num_registers, name="R_num_registers")
        self.M_num_ints = tf.constant(flags.M_num_ints, name="M_num_ints")
        self.Q_num_modules = tf.constant(flags.Q_num_modules, name="Q_num_modules")
        self.T_max_timesteps = tf.constant(flags.T_max_timesteps, name="T_max_timesteps")

        self.r_registers = tf.get_variable("r_registers",
            [self.R_num_registers, self.M_num_ints],
            initializer=tf.constant_initializer(0.0))
        self.BIGM_memory = tf.get_variable("BIGM_memory",
            [self.M_num_ints, self.M_num_ints],
            initializer=tf.constant_initializer(0.0))
        self.f_finish_prob = tf.get_variable("f_finish_prob",
            [self.T_max_timesteps],
            initializer=tf.constant_initializer(0.0))
        self.p_cumulative_finish_prob = tf.get_variable("p_cumulative_finish_prob",
            [self.T_max_timesteps],
            initializer=tf.constant_initializer(0.0))

    def train(self):
        data_sets = data.read_data_sets(self.train_dir)
        with tf.Graph().as_default():
            input_pl, targets_pl = self._placeholder_inputs(self.batch_size)
            loss = self._controller(input_pl, targets_pl)
            train_op = self._training(loss, self.alpha_learning_rate)
            eval_correct = self._evaluation(logits, targets_pl)
            summary_op = tf.merge_all_summaries()
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                summary_writer = tf.train.SummaryWriter(self.train_dir, sess.graph)
                for step in xrange(self.max_steps):
                    start_time = time.time()
                    feed_dict = self._fill_feed_dict(data_sets.train, input_pl, targets_pl)
                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    duration = time.time() - start_time

                    if step % self.report_interval == 0:
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    # Save a checkpoint and evaluate the model periodically.
                    if (step + 1) % self.checkpoint_interval == 0 \
                            or (step + 1) == self.max_steps:
                        self.saver.save(sess, self.train_dir, global_step=step)
                        # Evaluate against the training set.
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


    def save(self, sess, save_path, global_step):
        self.saver.save(sess, save_path, global_step)

    def load(self, sess, save_path):
        self.saver.restore(sess, save_path)


    def _evaluation(self, logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def _do_eval(self, sess,
                eval_correct,
                input_placeholder,
                targets_placeholder,
                data_set):
        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
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

    def _fill_feed_dict(self, data_set, input_pl, targets_pl):
        # Create the feed_dict for the placeholders filled with the next
        # `batch size ` examples.
        input_feed, targets_feed = data_set.next_batch(self.batch_size)
        feed_dict = {
          input_pl: input_feed,
          targets_pl: targets_feed,
        }
        return feed_dict

    def _placeholder_inputs(self, batch_size):
        input_data = tf.placeholder(tf.float32, shape=(batch_size, self.M_num_ints))
        target_data = tf.placeholder(tf.float32, shape=(batch_size, self.M_num_ints))
        return input_data, target_data

    def _training(self, loss, learning_rate):
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def _intermediate_controller(self, register_output):
        inputs = self._interpret_register(register_output)

        with tf.name_scope("h1"):
            w = tf.Variable(tf.truncated_normal([self.R_num_registers, self.num_h1_units],
                stddev=1.0 / math.sqrt(float(self.R_num_registers))),
                name="w")
            b = tf.Variable(tf.zeros([self.num_h1_units]),
                name="b")
            h1 = tf.nn.relu(tf.matmul(inputs, w) + b)

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
                [self.num_h1_units, (self.R_num_registers + self.Q_num_modules)],
                stddev=1.0 / math.sqrt(float(self.num_h1_units))),
                name="w")
            b = tf.Variable(tf.zeros([(self.R_num_registers + self.Q_num_modules)]),
                name="b")
            logit_x = tf.matmul(h1, w) + b
            f = tf.nn.sigmoid(logit_x)

        return logit_a, logit_b, logit_c, f

    def _controller(self, input_pl, labels_pl):
        # The inputs are placed into the Memory
        self._init_memory(input_pl)
        local_memory = tf.Variable(tf.zeros([self.R_num_registers, self.M_num_ints]))
        p = tf.Variable(0.0, name="prev")
        loss = tf.Variable(tf.zeros([1]))

        for step in xrange(self.T_max_timesteps):
            local_memory = \
                tf.concat(0, [local_memory[:self.R_num_registers], tf.zeros([self.Q_num_modules, self.M_num_ints])])
            for mod_idx in xrange(self.Q_num_modules):
                with tf.variable_scope("module") as scope:
                    scope.reuse_variables()
                    a, b, _, _ = self._intermediate_controller(local_memory)

                    module_l = \
                        tf.matmul(tf.transpose(local_memory), tf.nn.softmax(a))
                    module_r = \
                        tf.matmul(tf.transpose(local_memory), tf.nn.softmax(b))
                    local_memory[self.R_num_registers + mod_idx] = \
                        self._module_function(mod_idx, module_l, module_r)

            temp_local_memory = tf.Variable(tf.zeros([self.R_num_registers, self.M_num_ints]))
            for reg_idx in xrange(self.R_num_registers):
                with tf.variable_scope("module") as scope:
                    _, _, c, _ = self._intermediate_controller(local_memory)
                    temp_local_memory[reg_idx] = \
                        tf.matmul(tf.transpose(local_memory), tf.nn.softmax(c))

            _, _, _, f = self._intermediate_controller(local_memory)

            local_memory = temp_local_memory

            p = (1.0 - p) * f

            loss -= p * tf.log(tf.reduce_sum(labels_pl[step] * self.BIGM_memory))

            '''
            if tf.random_uniform([0.0, 1.0]) <= f:
                break
            '''

        return loss

    def _init_memory(self, int_array):
        self.BIGM_memory = tf.zeros(size=(self.M_num_ints, self.M_num_ints))
        for i, val in enumerate(int_array):
            self.BIGM_memory[i][val] = 1.0
        return len(int_array)

    def _module_function(self, fun_idx, a, b):
        if fun_idx == 0:
            return self._module_READ(a, b)
        elif fun_idx == 1:
            return self._module_WRITE(a, b)

    def _interpret_register(self, registers):
        return tf.slice(registers, [0, 0], [-1, 1])

    def _read_memory(self):
        return tf.argmax(self.BIGM_memory, 1)

    def _module_READ(self, p_pointer, _):
        return tf.matmul(tf.transpose(self.BIGM_memory), p_pointer)

    def _module_WRITE(self, p_pointer, a_value):
        J_ones = tf.ones([self.M_num_ints])
        self.BIGM_memory = \
            (tf.matmul(J_ones - p_pointer, tf.transpose(J_ones)) * self.BIGM_memory) + \
            tf.matmul(p_pointer, tf.transpose(a_value))
        return tf.zeros([self.M_num_ints])
