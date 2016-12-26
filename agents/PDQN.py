import tensorflow as tf
import numpy as np
import commonOps
from base_agent import BaseAgent

class Agent(BaseAgent):

    def __init__(self, config, session):
        BaseAgent.__init__(self, config, session)
        self.action_modes = {"e_greedy":self.e_greedy_action, "plan_e_greedy":self.plan_e_greedy_action}
        self.default_action_mode = "e_greedy"
        self.action_mode = self.default_action_mode
        # build the net
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="action_ph")
            self.reward_ph = tf.placeholder(tf.float32, [None], name="reward_ph")
            self.terminal_ph = tf.placeholder(tf.float32, [None], name="terminal_ph")
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            # Define all the ops
            with tf.variable_scope("Q"):
                self.h_state = commonOps.state_to_hidden(self.state_ph, config, "Normal")
                self.Q = commonOps.hidden_to_Q(self.h_state, config, "Normal")
                self.predicted_reward = commonOps.hidden_to_r(self.h_state, config, "Normal")
                self.predicted_h_state = commonOps.hidden_to_hidden(self.h_state, self.action_ph, config, "Normal")
                tf.get_variable_scope().reuse_variables()
                self.predicted_next_Q = commonOps.hidden_to_Q(self.predicted_h_state, config, "Normal")
            with tf.variable_scope("QT"):
                self.QT = commonOps.deepmind_Q(
                    self.stateT_ph, config, "Target")
            self.train_op = commonOps.pdqn_train_op(self.Q, self.predicted_reward,
                                self.predicted_next_Q, self.QT, self.reward_ph,
                                self.action_ph, self.terminal_ph, config, "Normal")
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

    def plan_e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            #should be optimized
            # instead of dequeuing feed the game state
            feed_dict={self.state_ph:self.game_state, self.action_ph:[0]} # action 0
            predicted_Rs, predicted_next_Q_0 = self.sess.run([self.predicted_reward, self.predicted_next_Q], feed_dict)
            predicted_next_Qs = np.zeros(self.config.action_num)
            predicted_next_Qs[0] = np.max(predicted_next_Q_0)
            for a in range(1, self.config.action_num):
                feed_dict={self.state_ph:self.game_state, self.action_ph:[a]}
                predicted_next_Qs[a] == np.max(self.sess.run(self.predicted_next_Q, feed_dict))
            action = np.argmax(predicted_Rs + self.config.gamma * predicted_next_Qs)
        return action
