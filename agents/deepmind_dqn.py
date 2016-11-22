import tensorflow as tf
import numpy as np
import cv2
import random
import threading
from replayMemory import ReplayMemory
import commonOps

class Agent:
    def __init__(self, config, session, summary_writter):
        # build the net
        self.config = config
        self.sess = session
        self.RM = ReplayMemory(config)
        self.summary_writter = summary_writter
        self.step = 0
        self.episode = 0
        self.training = True
        with tf.device(config.device):
            self.state_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="state_ph")
            self.action_ph = tf.placeholder(tf.int64, [None], name="Action_ph")
            self.Y_ph = tf.placeholder(tf.float32, [None], name="Y_ph")
            placeholder_list = [self.state_ph, self.action_ph, self.Y_ph]
            q = tf.FIFOQueue(2, [ph.dtype for ph in placeholder_list])
            self.enqueue_op = q.enqueue(placeholder_list)
            self.state, self.action, self.Y = q.dequeue()
            self.stateT_ph = tf.placeholder(
                tf.float32, [None, 84, 84, 4], name="next_state_ph")
            with tf.variable_scope("Q"):
                self.Q = commonOps.deepmind_Q(self.state, config, "Q")
            with tf.variable_scope("QT"):
                self.QT = commonOps.deepmind_Q(self.stateT_ph, config, "QT")
            self.sync_QT_op = [W_pair[0].assign(W_pair[1]) for W_pair in zip(tf.get_collection("QT_weights"), tf.get_collection("Q_weights"))]
            self.train_op = commonOps.build_train_op(self.Q, self.Y, self.action, config)
            self.Q_summary_op = tf.merge_summary(
                tf.get_collection("DQN_summaries"))
            self.QT_summary_op = tf.merge_summary(
                tf.get_collection("DQNT_summaries"))

        self.reset_game()

        self.enqueue_from_RM_thread = threading.Thread(target=self.enqueue_from_RM, daemon=False)
        self.stop_enqueuing = threading.Event()


    def step(self, x, r):
        if self.training:
            if not self.episode_begining:
                self.RM.add(self.game_state[:,:,-1], self.game_action, self.game_reward, False)
            else:
                self.episode_begining = False
            self.game_action = self.e_greedy_action(self.epsilon())
            self.observe(x, r)
            self.update()
            self.step += 1
        else:
            self.game_action = self.e_greedy_action(0.01)
            self.observe(x)
        return self.game_action

    # Add the transition to RM and reset the internal state for the next episode
    def done(self):
        if self.training:
            self.RM.add(self.game_state[:,:,-1], self.action, self.reward, True)
        self.reset_game_state()

    def observe(self, x, r):
        self.game_reward = r
        x_ = cv2.resize(x, (84, 84))
        self.game_state.roll(-1, axis=3)
        self.game_state[0, :, :, -1] = x_

        x_ = cv2.resize(x, (84, 84))
        self.game_state.roll(-1, axis=3)
        self.game_state[0, :, :, -1] = x_


    timeout_option = tf.RunOptions(timeout_in_ms=5000)
    def update(self):
        if self.step > self.config.steps_before_training:
            if self.enqueue_from_RM_thread.isAlive() == False:
                self.enqueue_from_RM_thread.start()

            if self.config.logging and self.step % self.config.save_summary_rate == 0:
                _, Q_summary_str = self.sess.run(
                    [self.train_op, self.Q_summary_op], options=timeout_option)
                self.summary_writter.add_summary(Q_summary_str, self.step)
            else:
                _ = self.sess.run(self.train_op, options=self.timeout_option)

            if self.step % self.config.sync_rate == 0:
                self.sess.run(self.sync_QT_op)

    def enqueue_from_RM(self):
        print("Starting enqueue thread")
        while not self.stop_enqueuing.isSet():
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, _ = RM.sample_transition_batch()
            if self.config.logging and self.step % self.config.save_summary_rate == 0:
                QT_np, QT_summary_str = self.sess.run([self.QT, self.QT_summary_op], feed_dict={
                                                   self.stateT_ph:next_state_batch})
                self.summary_writter.add_summary(QT_summary_str, self.step)
            else:
                QT_np = self.sess.run(
                    self.QT, feed_dict={self.stateT_ph:next_state_batch})

            # simplify with np
            DQNT_max_action_batch = np.max(QT_np, 1)
            Y = []
            for i in range(state_batch.shape[0]):
                terminal = terminal_batch[i]
                if terminal:
                    Y.append(reward_batch[i])
                else:
                    Y.append(reward_batch[
                             i] + self.config.gamma * DQNT_max_action_batch[i])

            feed_dict = {self.state_ph: state_batch, self.action_ph: action_batch, self.Y_ph: Y}
            self.sess.run(self.enqueue_op, feed_dict=feed_dict)
        print("Closing enqueue thread")

    def e_greedy_action(self, epsilon):
        if np.random.uniform() < epsilon:
            action = random.randint(0, self.config.action_num - 1)
        else:
            action = np.argmax(self.sess.run(self.Q, feed_dict={self.state: self.game_state})[0])
        return action

    def training(self, t=True):
        self.training = t

    def testing(self, t=True):
        self.training = not t

    def reset_game(self):
        self.episode_begining = True
        self.game_state = np.zeros((1, 84, 84, self.config.buff_size), dtype=np.uint8)

    def epsilon(self):
        if self.step < self.config.exploration_steps:
            return self.config.initial_epsilon - ((self.config.initial_epsilon - self.config.final_epsilon) / self.config.exploration_steps) * self.step
        else:
            return self.config.final_epsilon

    def __del__(self):
        self.stop_enqueuing.set()
