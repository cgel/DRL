import tensorflow as tf
import commonOps as cops
from base_agent import BaseAgent


class Agent(BaseAgent):

    def __init__(self, config, session):
        BaseAgent.__init__(self, config, session)
        # build the net
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="action_ph")
            self.reward_ph = tf.placeholder(tf.float32, [None], name="reward_ph")
            self.terminal_ph = tf.placeholder(tf.float32, [None], name="terminal_ph")
            # Define all the ops
            with tf.variable_scope("Q"):
                self.Q = self.Q_network(self.state_ph, config, "Normal")
            with tf.variable_scope("QT"):
                self.QT = self.Q_network(
                    self.stateT_ph, config, "Target")
                tf.scalar_summary(
                    "main/next_Q_max", tf.reduce_max(self.QT), collections=["Target_summaries"])
                tf.scalar_summary(
                    "main/next_Q_0", tf.reduce_max(self.QT, 1)[0], collections=["Target_summaries"])

            self.train_op = self.train_op(self.Q, self.QT, self.action_ph, self.reward_ph, self.terminal_ph, config, "Normal")
            self.sync_QT_op = []
            for W_pair in zip(
                    tf.get_collection("Target_weights"),
                    tf.get_collection("Normal_weights")):
                self.sync_QT_op.append(W_pair[0].assign(W_pair[1]))
            # Define the summary ops
            self.Q_summary_op = tf.merge_summary(
                tf.get_collection("Normal_summaries"))
            self.QT_summary_op = tf.merge_summary(
                tf.get_collection("Target_summaries"))

    def update(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = self.RM.sample_transition_batch()
        feed_dict={self.state_ph: state_batch,
                    self.stateT_ph: next_state_batch,
                    self.action_ph: action_batch,
                    self.reward_ph: reward_batch,
                    self.terminal_ph: terminal_batch}
        if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
            _, Q_summary_str = self.sess.run(
                [self.train_op, self.Q_summary_op], feed_dict, options=self.timeout_option)
            self.summary_writter.add_summary(
                Q_summary_str, self.step_count)
        else:
            _ = self.sess.run(self.train_op, feed_dict, options=self.timeout_option)

        if self.step_count % self.config.sync_rate == 0:
            self.sess.run(self.sync_QT_op)

    def Q_network(self, input_state, config, Collection=None):
        conv_stack_shape=[(32,8,4),
                    (64,4,2),
                    (64,3,1)]
        head = tf.div(input_state, 256., name="normalized_input")
        cops.build_activation_summary(head, Collection)
        head = cops.conv_stack(head, conv_stack_shape, config, Collection)
        head = cops.flatten(head)
        head = cops.add_relu_layer(head, size=512, Collection=Collection)
        Q = cops.add_linear_layer(head, config.action_num, Collection, layer_name="Q")
        # DQN summary
        for i in range(config.action_num):
            tf.scalar_summary("DQN/action" + str(i),
                                     Q[0, i], collections=["Q_summaries"])
        return Q

    def train_op(self, Q, QT, action, reward, terminal, config, Collection):
        with tf.name_scope("loss"):
            # could be done more efficiently with gather_nd or transpose + gather
            action_one_hot = tf.one_hot(
                action, config.action_num, 1., 0., name='action_one_hot')
            acted_Q = tf.reduce_sum(
                Q * action_one_hot, reduction_indices=1, name='DQN_acted')

            QT_max_action = tf.reduce_max(QT, 1)
            Y = reward + config.gamma * \
                QT_max_action * (1 - terminal)
            Y = tf.stop_gradient(Y)

            loss_batch = cops.clipped_l2(Y, acted_Q)
            loss = tf.reduce_sum(loss_batch, name="loss")

            tf.scalar_summary("losses/loss", loss,
                              collections=[Collection + "_summaries"])
            tf.scalar_summary("losses/loss_0", loss_batch[0],
                              collections=[Collection + "_summaries"])
            tf.scalar_summary("losses/loss_max", tf.reduce_max(loss_batch),
                              collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/Y_0", Y[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/Y_max", tf.reduce_max(Y), collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/QT_max_action_0", QT_max_action[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/acted_Q_0", acted_Q[0], collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/acted_Q_max", tf.reduce_max(acted_Q), collections=[Collection + "_summaries"])
            tf.scalar_summary(
                "main/reward_max", tf.reduce_max(reward), collections=[Collection + "_summaries"])

        train_op, grads = cops.graves_rmsprop_optimizer(
            loss, config.learning_rate, 0.95, 0.01, 1)

        for grad, var in grads:
            if grad is True:
                tf.histogram_summary(var.op.name + '/gradients', grad, name=var.op.name +
                                     '/gradients', collections=[Collection + "_summaries"])
        return train_op
