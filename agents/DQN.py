import tensorflow as tf
import commonOps
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
                self.Q = commonOps.deepmind_Q(self.state_ph, config, "Normal")
            with tf.variable_scope("QT"):
                self.QT = commonOps.deepmind_Q(
                    self.stateT_ph, config, "Target")
                tf.scalar_summary(
                    "main/next_Q_max", tf.reduce_max(self.QT), collections=["Target_summaries"])
                tf.scalar_summary(
                    "main/next_Q_0", tf.reduce_max(self.QT, 1)[0], collections=["Target_summaries"])

            #def dqn_train_op(Q, QT, action, reward, terminal, config, Collection):
            self.train_op = commonOps.dqn_train_op(self.Q, self.QT, self.action_ph, self.reward_ph, self.terminal_ph, config, "Normal")
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
