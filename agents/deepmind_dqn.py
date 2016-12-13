import tensorflow as tf
import numpy as np
import cv2
import random
import threading
from replayMemory import ReplayMemory
import commonOps


class Agent:

    def __init__(self, config, session):
        # build the net
        self.config = config
        self.sess = session
        self.RM = ReplayMemory(config)
        self.step_count = 0
        self.episode = 0
        self.isTesting = False
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

        self.reset_game()
        self.enqueue_from_RM_thread = threading.Thread(
            target=self.enqueue_from_RM)
        self.enqueue_from_RM_thread.daemon = True
        self.stop_enqueuing = threading.Event()
        self.timeout_option = tf.RunOptions(timeout_in_ms=5000)

    def step(self, x, r):
        if not self.isTesting:
            if not self.episode_begining:
                self.RM.add(
                    self.game_state[
                        :, :, :, -1], self.game_action, self.game_reward, False)
            else:
                self.episode_begining = False
            self.game_action = self.e_greedy_action(self.epsilon())
            self.observe(x, r)
            self.update()
            self.step_count += 1
        else:
            self.game_action = self.e_greedy_action(0.01)
            self.observe(x, r)
        return self.game_action

    # Add the transition to RM and reset the internal state for the next
    # episode
    def done(self):
        if not self.isTesting:
            self.RM.add(
                self.game_state[:, :, :, -1],
                self.game_action, self.game_reward, True)
        self.reset_game()

    def observe(self, x, r):
        self.game_reward = r
        x_ = cv2.resize(x, (84, 84))
        x_ = cv2.cvtColor(x_, cv2.COLOR_RGB2GRAY)
        self.game_state = np.roll(self.game_state, -1, axis=3)
        self.game_state[0, :, :, -1] = x_

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

            QT_max_action = np.max(QT_np, 1)
            Y = reward_batch + self.config.gamma * \
                QT_max_action * (1 - terminal_batch)

            feed_dict = {
                self.state_ph: state_batch,
                self.action_ph: action_batch,
                self.Y_ph: Y}
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        print("Closing enqueue thread")

    def e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            action = np.argmax(
                self.sess.run(
                    self.Q, feed_dict={
                        self.state: self.game_state})[0])
        return action

    def testing(self, t=True):
        self.isTesting = t

    def reset_game(self):
        self.episode_begining = True
        self.game_state = np.zeros(
            (1, 84, 84, self.config.buff_size), dtype=np.uint8)

    def epsilon(self):
        if self.step_count < self.config.exploration_steps:
            return self.config.initial_epsilon - \
                ((self.config.initial_epsilon - self.config.final_epsilon) /
                 self.config.exploration_steps) * self.step_count
        else:
            return self.config.final_epsilon

    def set_summary_writer(self, summary_writter):
        self.summary_writter = summary_writter

    def __del__(self):
        self.stop_enqueuing.set()
