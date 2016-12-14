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
        # build the net
        self.action_modes = {"e_greedy":self.e_greedy_action, "pan_e_greedy":self.plan_e_greedy_action}
        self.default_action_mode = "e_greedy"
        self.action_mode = self.default_action_mode
        with tf.device(config.device):
            # Create all variables and the FIFOQueue
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="Action_ph")
            self.QT_ph = tf.placeholder(tf.float32, [None], name="QT_ph")
            self.predicted_QT_ph = tf.placeholder(tf.float32, [None, config.action_num], name="predicte_QT_ph")
            self.rT_ph = tf.placeholder(tf.float32, [None], name="r_ph")
            placeholder_list = [self.state_ph, self.action_ph, self.QT_ph, self.predicted_QT_ph, self.rT_ph]
            q = tf.FIFOQueue(2, [ph.dtype for ph in placeholder_list])
            self.enqueue_op = q.enqueue(placeholder_list)
            self.state_queue, self.action_queue, self.QT_queue, self.predicted_QT_queue, self.rT_queue = q.dequeue()
            self.state_queue.set_shape(self.state_ph.get_shape())
            self.action_queue.set_shape(self.action_ph.get_shape())
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="stateT_ph")
            # Define all the ops
            with tf.variable_scope("Q"):
                self.h_state = commonOps.state_to_hidden(self.state_queue, config, "Normal")
                self.Q = commonOps.hidden_to_Q(self.h_state, config, "Normal")
                self.predicted_r = commonOps.hidden_to_r(self.h_state, config, "Normal")
                self.predicted_h_state = commonOps.hidden_to_hidden(self.h_state, self.action_queue, config, "Normal")
                tf.get_variable_scope().reuse_variables()
                self.predicted_next_Q = commonOps.hidden_to_Q(self.predicted_h_state, config, "Normal")
            with tf.variable_scope("QT"):
                self.QT = commonOps.deepmind_Q(
                    self.stateT_ph, config, "Target")
            self.train_op = commonOps.pdqn_train_op(self.Q, self.QT_queue, self.predicted_r,
                                self.rT_queue, self.predicted_next_Q,
                                self.predicted_QT_queue, self.action_queue, config, "Normal")
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

        self.enqueue_from_RM_thread = threading.Thread(
            target=self.enqueue_from_RM)
        self.enqueue_from_RM_thread.daemon = True
        self.stop_enqueuing = threading.Event()

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
            if self.config.logging and self.step_count % self.config.update_summary_rate == 0:
                QT_np, QT_summary_str = self.sess.run([self.QT, self.QT_summary_op], feed_dict={
                    self.stateT_ph: next_state_batch}, options=self.timeout_option)
                self.summary_writter.add_summary(
                    QT_summary_str, self.step_count)
            else:
                QT_np = self.sess.run(
                    self.QT,
                    feed_dict={
                        self.stateT_ph: next_state_batch},
                    options=self.timeout_option)

            # simplify with np
            QT_max_action = np.max(QT_np, 1)
            Y = reward_batch + self.config.gamma * \
                QT_max_action * (1 - terminal_batch)

            feed_dict = {
                self.state_ph: state_batch,
                self.action_ph: action_batch,
                self.QT_ph: Y,
                self.predicted_QT_ph: QT_np,
                self.rT_ph: reward_batch}
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        print("Closing enqueue thread")

    def e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            # instead of dequeuing feed the game state
            action = np.argmax(
                self.sess.run(
                    self.Q, feed_dict={
                        self.state_queue: self.game_state})[0])
        return action

    def plan_e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            #should be optimized
            # instead of dequeuing feed the game state
            feed_dict={self.state_queue:self.game_state, self.action_queue:[0]}
            predicted_Rs, predicted_next_Q_0 = self.sess.run([self.predicted_r, self.predicted_next_Q], feed_dict)
            predicted_next_Qs = np.zeros(self.config.action_num)
            predicted_next_Qs[0] = np.max(predicted_next_Q_0)
            for a in range(1, self.config.action_num):
                feed_dict={self.state_queue:self.game_state, self.action_queue:[a]}
                predicted_next_Qs[a] == np.max(self.sess.run(self.predicted_next_Q, feed_dict))
            action = np.argmax(predicted_Rs + self.config.gamma * predicted_next_Qs)
        return action

    def __del__(self):
        self.stop_enqueuing.set()
