import tensorflow as tf
import numpy as np
import cv2
import random
import threading
from replayMemory import ReplayMemory
import commonOps
from base_agent import BaseAgent


class Agent(BaseAgent):

    def __init__(self, config, session):
        BaseAgent.__init__(self, config, session)
        self.enqueue_from_RM_thread = threading.Thread(
        target=self.enqueue_from_RM)
        self.enqueue_from_RM_thread.daemon = True
        self.stop_enqueuing = threading.Event()

        # build the net
        self.action_modes = {"e_greedy":self.e_greedy_action}
        self.default_action_mode = "e_greedy"
        self.action_mode = self.default_action_mode
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="Action_ph")
            self.Y_ph = tf.placeholder(tf.float32, [None], name="Y_ph")
            placeholder_list = [self.state_ph, self.action_ph, self.Y_ph]
            q = tf.FIFOQueue(2, [ph.dtype for ph in placeholder_list])
            self.enqueue_op = q.enqueue(placeholder_list)
            self.state, self.action, self.Y = q.dequeue()
            self.state.set_shape(self.state_ph.get_shape())
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            # Define all the ops
            with tf.variable_scope("Q"):
                self.Q = commonOps.deepmind_Q(self.state, config, "Normal")
            with tf.variable_scope("QT"):
                self.QT = commonOps.deepmind_Q(
                    self.stateT_ph, config, "Target")
                tf.scalar_summary(
                    "main/next_Q_max", tf.reduce_max(self.QT), collections=["Target_summaries"])
                tf.scalar_summary(
                    "main/next_Q_0", tf.reduce_max(self.QT, 1)[0], collections=["Target_summaries"])

            self.train_op = commonOps.dqn_train_op(
                self.Q, self.Y, self.action, config, "Normal")
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
        if self.step_count > self.config.steps_before_training:
            if self.enqueue_from_RM_thread.isAlive() == False:
                self.enqueue_from_RM_thread.start()

            if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
                _, Q_summary_str = self.sess.run(
                    [self.train_op, self.Q_summary_op], options=self.timeout_option)
                self.summary_writter.add_summary(
                    Q_summary_str, self.step_count)
            else:
                _ = self.sess.run(self.train_op, options=self.timeout_option)
            if self.step_count % self.config.sync_rate == 0:
                self.sess.run(self.sync_QT_op)

    def enqueue_from_RM(self):
        print("Starting enqueue thread")
        while not self.stop_enqueuing.isSet():
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = self.RM.sample_transition_batch()
            feed_dict={self.stateT_ph: next_state_batch}
            if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
                QT_np, QT_summary_str = self.sess.run([self.QT, self.QT_summary_op],
                    feed_dict, options=self.timeout_option)
                self.summary_writter.add_summary(
                    QT_summary_str, self.step_count)
                self.summary_writter.add_summary(tf.Summary(value=[tf.Summary.Value(tag="main/r_max", simple_value=int(np.max(reward_batch)))]), self.step_count)
            else:
                QT_np = self.sess.run(
                    self.QT,
                    feed_dict,
                    options=self.timeout_option)


            QT_max_action = np.max(QT_np, 1)
            Y = reward_batch + self.config.gamma * \
                QT_max_action * (1 - terminal_batch)

            feed_dict = {
                self.state_ph: state_batch,
                self.action_ph: action_batch,
                self.Y_ph: Y}
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        print("Closing enqueue thread")

    def __del__(self):
        self.stop_enqueuing.set()
        BaseAgent.__del__()
